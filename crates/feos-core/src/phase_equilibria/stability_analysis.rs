use super::PhaseEquilibrium;
use crate::equation_of_state::Residual;
use crate::errors::{FeosError, FeosResult};
use crate::state::{Contributions, DensityInitialization, State};
use crate::{ReferenceSystem, SolverOptions, Verbosity};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, OMatrix, OVector, U1};
use num_dual::Gradients;
use num_dual::linalg::LU;
use num_dual::linalg::smallest_ev;
use quantity::Moles;

const X_DOMINANT: f64 = 0.99;
const MINIMIZE_TOL: f64 = 1E-06;
const MIN_EIGENVAL: f64 = 1E-03;
const ETA_STEP: f64 = 0.25;
const MINIMIZE_KMAX: usize = 100;
const ZERO_TPD: f64 = -1E-08;
const DUPLICATE_TOL: f64 = 1E-03;

/// # Stability analysis
impl<E: Residual<N>, N: Gradients> State<E, N>
where
    DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    /// Determine if the state is stable, i.e. if a phase split should
    /// occur or not.
    pub fn is_stable(&self, options: SolverOptions) -> FeosResult<bool> {
        Ok(self.stability_analysis(options)?.is_empty())
    }

    /// Perform a stability analysis. The result is a list of [State]s with
    /// negative tangent plane distance (i.e. lower Gibbs energy) that can be
    /// used as initial estimates for a phase equilibrium calculation.
    /// The states are sorted by their tangent plane distance, starting with
    /// the most negative one.
    ///
    /// A trial phase whose minimization fails is still returned if its
    /// tangent plane distance is negative, because that already proves the
    /// instability. Otherwise it is skipped. An error is only returned if
    /// no trial phase could be evaluated successfully.
    pub fn stability_analysis(&self, options: SolverOptions) -> FeosResult<Vec<State<E, N>>> {
        let mut result: Vec<(f64, State<E, N>)> = Vec::new();
        let mut successful_trials = 0;
        let mut last_error = None;
        for i_trial in 0..self.eos.components() + 1 {
            let phase = if i_trial == self.eos.components() {
                "Vapor phase".to_string()
            } else {
                format!("Liquid phase {}", i_trial + 1)
            };
            let mut trial_state = match self.define_trial_state(i_trial) {
                Ok(trial_state) => trial_state,
                Err(e) => {
                    log_result!(options.verbosity, "{}: Failed ({})\n", phase, e);
                    last_error = Some(e);
                    continue;
                }
            };
            // The tpd tracked during the minimization is only an estimate, so the
            // true tpd of the final trial state is used to compare candidates.
            let (tpd, steps) = match self.minimize_tpd(&mut trial_state, options) {
                Ok((None, i)) => (None, Some(i)),
                Ok((Some(_), i)) => (Some(self.tangent_plane_distance(&trial_state)), Some(i)),
                Err(e) => {
                    let tpd = self.tangent_plane_distance(&trial_state);
                    if tpd < ZERO_TPD && !PhaseEquilibrium::is_trivial_solution(self, &trial_state)
                    {
                        (Some(tpd), None)
                    } else {
                        log_result!(options.verbosity, "{}: Failed ({})\n", phase, e);
                        last_error = Some(e);
                        continue;
                    }
                }
            };
            successful_trials += 1;
            let msg = if let Some(tpd) = tpd {
                if tpd < ZERO_TPD {
                    if result.iter().any(|(_, s)| s.is_same_phase(&trial_state)) {
                        "Found already identified minimum"
                    } else {
                        result.push((tpd, trial_state));
                        "Found candidate"
                    }
                } else {
                    "Found minimum > 0"
                }
            } else {
                "Found trivial solution"
            };
            match steps {
                Some(i) => {
                    log_result!(options.verbosity, "{}: {} in {} step(s)\n", phase, msg, i)
                }
                None => log_result!(options.verbosity, "{}: {} (not converged)\n", phase, msg),
            }
        }
        if let Some(e) = last_error
            && successful_trials == 0
        {
            return Err(e);
        }
        result.sort_by(|(tpd1, _), (tpd2, _)| tpd1.total_cmp(tpd2));
        Ok(result.into_iter().map(|(_, s)| s).collect())
    }

    /// Check if two candidates from the stability analysis describe the same phase.
    ///
    /// In contrast to [PhaseEquilibrium::is_trivial_solution], the partial densities
    /// are compared relative to the total density, so that differences in trace
    /// components (which can be large in relative terms) are ignored.
    fn is_same_phase(&self, other: &State<E, N>) -> bool {
        let rho1 = self.density.into_reduced();
        let rho2 = other.density.into_reduced();
        let max_diff = self
            .molefracs
            .zip_fold(&other.molefracs, 0.0, |acc: f64, x1, x2| {
                acc.max((x1 * rho1 - x2 * rho2).abs())
            });
        max_diff / rho1.max(rho2) < DUPLICATE_TOL
    }

    /// Tangent plane distance of the trial state with respect to this state.
    ///
    /// Components that are absent in the trial state do not contribute (w ln w -> 0).
    pub(super) fn tangent_plane_distance(&self, trial_state: &State<E, N>) -> f64 {
        let ln_phi_z = self.ln_phi();
        let ln_phi_w = trial_state.ln_phi();
        let (z, w) = (&self.molefracs, &trial_state.molefracs);
        (0..w.len())
            .filter(|&i| w[i] > 0.0)
            .map(|i| w[i] * (w[i].ln() + ln_phi_w[i] - z[i].ln() - ln_phi_z[i]))
            .sum()
    }

    fn define_trial_state(&self, dominant_component: usize) -> FeosResult<State<E, N>> {
        let x_feed = &self.molefracs;
        let (n, _) = x_feed.shape_generic();

        let (x_trial, phase) = if dominant_component == self.eos.components() {
            // try an ideal vapor phase
            let x_trial = self.ln_phi().map(f64::exp).component_mul(x_feed);
            (&x_trial / x_trial.sum(), DensityInitialization::Vapor)
        } else {
            // try each component as nearly pure phase
            let factor = (1.0 - X_DOMINANT) / (x_feed.sum() - x_feed[dominant_component]);
            (
                OVector::from_fn_generic(n, U1, |i, _| {
                    if i == dominant_component {
                        X_DOMINANT
                    } else {
                        x_feed[i] * factor
                    }
                }),
                DensityInitialization::Liquid,
            )
        };

        State::new_npt(
            &self.eos,
            self.temperature,
            self.pressure(Contributions::Total),
            Moles::from_reduced(x_trial),
            Some(phase),
        )
    }

    fn minimize_tpd(
        &self,
        trial: &mut State<E, N>,
        options: SolverOptions,
    ) -> FeosResult<(Option<f64>, usize)> {
        let (max_iter, tol, verbosity) = options.unwrap_or(MINIMIZE_KMAX, MINIMIZE_TOL);
        let mut newton = false;
        let mut scaled_tol = tol;
        let mut tpd = 1E10;
        let di = self.molefracs.map(f64::ln) + self.ln_phi();

        log_iter!(verbosity, " iter |    residual    |     tpd     | Newton");
        log_iter!(verbosity, "{:-<46}", "");

        for i in 1..=max_iter {
            let error = if !newton {
                // case: direct substitution
                let y = (&di - &trial.ln_phi()).map(f64::exp);
                let tpd_old = tpd;
                tpd = 1.0 - y.sum();
                let error = (&y / y.sum() - &trial.molefracs).map(f64::abs).sum();

                *trial = State::new_npt(
                    &trial.eos,
                    trial.temperature,
                    trial.pressure(Contributions::Total),
                    Moles::from_reduced(y),
                    Some(DensityInitialization::InitialDensity(trial.density)),
                )?;
                if (i > 4 && error > scaled_tol) || (tpd > tpd_old + 1E-05 && i > 2) {
                    newton = true; // switch to newton scheme
                }
                error
            } else {
                // case: newton step
                trial.stability_newton_step(&di, &mut tpd)?
            };
            log_iter!(
                verbosity,
                " {:4} | {:14.8e} | {:11.8} | {}",
                i,
                error,
                tpd,
                newton
            );
            if PhaseEquilibrium::is_trivial_solution(self, &*trial) {
                return Ok((None, i));
            }
            if tpd < -1E-02 {
                scaled_tol = tol * 1E01
            }
            if tpd < -1E-01 {
                scaled_tol = tol * 1E02
            }
            if tpd < -1E-01 && i > 5 {
                scaled_tol = tol * 1E03
            }
            if error < scaled_tol {
                return Ok((Some(tpd), i));
            }
        }
        Err(FeosError::NotConverged(String::from("stability analysis")))
    }

    fn stability_newton_step(&mut self, di: &OVector<f64, N>, tpd: &mut f64) -> FeosResult<f64> {
        // save old values
        let tpd_old = *tpd;
        let (n, _) = di.shape_generic();

        // calculate residual and ideal hesse matrix
        // TODO: this should not require extensive properties, but I couldn't rewrite it
        // quickly without breaking it.
        let mut hesse = self.n_dln_phi_dnj() / self.total_moles()?.into_reduced();
        let lnphi = self.ln_phi();
        let y = self.moles()?.into_reduced();
        let ln_y = y.map(|y| if y > f64::EPSILON { y.ln() } else { 0.0 });
        let sq_y = y.map(f64::sqrt);
        let gradient = (&ln_y + &lnphi - di).component_mul(&sq_y);

        let hesse_ig = OMatrix::identity_generic(n, n);
        for i in 0..self.eos.components() {
            hesse.column_mut(i).component_mul_assign(&(sq_y[i] * &sq_y));
            if y[i] > f64::EPSILON {
                hesse[(i, i)] += ln_y[i] + lnphi[i] - di[i];
            }
        }

        // !-----------------------------------------------------------------------------
        // ! use method of Murray, by adding a unity matrix to Hessian, if:
        // ! (1) H is not positive definite
        // ! (2) step size is too large
        // ! (3) objective function (tpd) does not descent
        // !-----------------------------------------------------------------------------
        let mut adjust_hessian = true;
        let mut hessian: OMatrix<f64, N, N>;
        let mut eta_h = 1.0;

        while adjust_hessian {
            adjust_hessian = false;
            hessian = &hesse + &(eta_h * &hesse_ig);

            let (min_eigenval, _) = smallest_ev(hessian.clone());
            if min_eigenval < MIN_EIGENVAL && eta_h < 20.0 {
                eta_h += 2.0 * ETA_STEP;
                adjust_hessian = true;
                continue; // continue, because of Hessian-criterion (1): H not positive definite
            }

            // solve: hessian * delta_y = gradient
            let delta_y = LU::new(hessian)?.solve(&gradient);
            if delta_y
                .iter()
                .zip(y.iter())
                .any(|(dy, y)| ((0.5 * dy).powi(2) / y).abs() > 5.0)
            {
                adjust_hessian = true;
                eta_h += 2.0 * ETA_STEP;
                continue; //  continue, because of Hessian-criterion (2): too large step-size
            }

            let y = (&sq_y - &(delta_y / 2.0)).map(|v| v.powi(2));
            let ln_y = y.map(|y| if y > f64::EPSILON { y.ln() } else { 0.0 });
            *tpd = 1.0 + y.dot(&(&ln_y + &lnphi - di.add_scalar(1.0)));
            if *tpd > tpd_old + 0.0 * 1E-03 && eta_h < 30.0 {
                eta_h += ETA_STEP;
                adjust_hessian = true;
                continue; // continue, because of Hessian-criterion (3): tpd does not descent
            }

            // accept step and update state
            *self = State::new_npt(
                &self.eos,
                self.temperature,
                self.pressure(Contributions::Total),
                Moles::from_reduced(y),
                Some(DensityInitialization::InitialDensity(self.density)),
            )?;
        }
        Ok(gradient.map(f64::abs).sum())
    }
}
