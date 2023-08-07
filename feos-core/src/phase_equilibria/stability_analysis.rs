use super::PhaseEquilibrium;
use crate::equation_of_state::Residual;
use crate::errors::{EosError, EosResult};
use crate::si::Moles;
use crate::state::{Contributions, DensityInitialization, State};
use crate::{SolverOptions, Verbosity};
use ndarray::*;
use num_dual::linalg::smallest_ev;
use num_dual::linalg::LU;
use std::f64::EPSILON;
use std::ops::MulAssign;

const X_DOMINANT: f64 = 0.99;
const MINIMIZE_TOL: f64 = 1E-06;
const MIN_EIGENVAL: f64 = 1E-03;
const ETA_STEP: f64 = 0.25;
const MINIMIZE_KMAX: usize = 100;
const ZERO_TPD: f64 = -1E-08;

/// # Stability analysis
impl<E: Residual> State<E> {
    /// Determine if the state is stable, i.e. if a phase split should
    /// occur or not.
    pub fn is_stable(&self, options: SolverOptions) -> EosResult<bool> {
        Ok(self.stability_analysis(options)?.is_empty())
    }

    /// Perform a stability analysis. The result is a list of [State]s with
    /// negative tangent plane distance (i.e. lower Gibbs energy) that can be
    /// used as initial estimates for a phase equilibrium calculation.
    pub fn stability_analysis(&self, options: SolverOptions) -> EosResult<Vec<State<E>>> {
        let mut result = Vec::new();
        for i_trial in 0..self.eos.components() + 1 {
            let phase = if i_trial == self.eos.components() {
                "Vapor phase".to_string()
            } else {
                format!("Liquid phase {}", i_trial + 1)
            };
            if let Ok(mut trial_state) = self.define_trial_state(i_trial) {
                let (tpd, i) = self.minimize_tpd(&mut trial_state, options)?;
                let msg = if let Some(tpd) = tpd {
                    if tpd < ZERO_TPD {
                        if result
                            .iter()
                            .any(|s| PhaseEquilibrium::is_trivial_solution(s, &trial_state))
                        {
                            "Found already identified minimum"
                        } else {
                            result.push(trial_state);
                            "Found candidate"
                        }
                    } else {
                        "Found minimum > 0"
                    }
                } else {
                    "Found trivial solution"
                };
                log_result!(options.verbosity, "{}: {} in {} step(s)\n", phase, msg, i);
            }
        }
        Ok(result)
    }

    fn define_trial_state(&self, dominant_component: usize) -> EosResult<State<E>> {
        let x_feed = &self.molefracs;

        let (x_trial, phase) = if dominant_component == self.eos.components() {
            // try an ideal vapor phase
            let x_trial = self.ln_phi().mapv(f64::exp) * x_feed;
            (&x_trial / x_trial.sum(), DensityInitialization::Vapor)
        } else {
            // try each component as nearly pure phase
            let factor = (1.0 - X_DOMINANT) / (x_feed.sum() - x_feed[dominant_component]);
            (
                Array1::from_shape_fn(self.eos.components(), |i| {
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
            &Moles::from_reduced(x_trial),
            phase,
        )
    }

    fn minimize_tpd(
        &self,
        trial: &mut State<E>,
        options: SolverOptions,
    ) -> EosResult<(Option<f64>, usize)> {
        let (max_iter, tol, verbosity) = options.unwrap_or(MINIMIZE_KMAX, MINIMIZE_TOL);
        let mut newton = false;
        let mut scaled_tol = tol;
        let mut tpd = 1E10;
        let di = self.molefracs.mapv(f64::ln) + self.ln_phi();

        log_iter!(verbosity, " iter |    residual    |     tpd     | Newton");
        log_iter!(verbosity, "{:-<46}", "");

        for i in 1..=max_iter {
            let error = if !newton {
                // case: direct substitution
                let y = (&di - &trial.ln_phi()).mapv(f64::exp);
                let tpd_old = tpd;
                tpd = 1.0 - y.sum();
                let error = (&y / y.sum() - &trial.molefracs).mapv(f64::abs).sum();

                *trial = State::new_npt(
                    &trial.eos,
                    trial.temperature,
                    trial.pressure(Contributions::Total),
                    &Moles::from_reduced(y),
                    DensityInitialization::InitialDensity(trial.density),
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
        Err(EosError::NotConverged(String::from("stability analysis")))
    }

    fn stability_newton_step(&mut self, di: &Array1<f64>, tpd: &mut f64) -> EosResult<f64> {
        // save old values
        let tpd_old = *tpd;

        // calculate residual and ideal hesse matrix
        let mut hesse = (self.dln_phi_dnj() * Moles::from_reduced(1.0)).into_value();
        let lnphi = self.ln_phi();
        let y = self.moles.to_reduced();
        let ln_y = Zip::from(&y).map_collect(|&y| if y > EPSILON { y.ln() } else { 0.0 });
        let sq_y = y.mapv(f64::sqrt);
        let gradient = (&ln_y + &lnphi - di) * &sq_y;

        let hesse_ig = Array2::eye(self.eos.components());
        for i in 0..self.eos.components() {
            hesse
                .index_axis_mut(Axis(0), i)
                .mul_assign(&(sq_y[i] * &sq_y));
            if y[i] > EPSILON {
                hesse[[i, i]] += ln_y[i] + lnphi[i] - di[i];
            }
        }

        // !-----------------------------------------------------------------------------
        // ! use method of Murray, by adding a unity matrix to Hessian, if:
        // ! (1) H is not positive definite
        // ! (2) step size is too large
        // ! (3) objective function (tpd) does not descent
        // !-----------------------------------------------------------------------------
        let mut adjust_hessian = true;
        let mut hessian: Array2<f64>;
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

            let y = (&sq_y - &(delta_y / 2.0)).mapv(|v| v.powi(2));
            let ln_y = Zip::from(&y).map_collect(|&y| if y > EPSILON { y.ln() } else { 0.0 });
            *tpd = 1.0 + (&y * &(&ln_y + &lnphi - di - 1.0)).sum();
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
                &Moles::from_reduced(y),
                DensityInitialization::InitialDensity(self.density),
            )?;
        }
        Ok(gradient.mapv(f64::abs).sum())
    }
}
