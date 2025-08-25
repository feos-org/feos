use super::PhaseEquilibrium;
use crate::equation_of_state::Residual;
use crate::errors::{FeosError, FeosResult};
use crate::state::{Contributions, State};
use crate::{SolverOptions, Verbosity};
use nalgebra::{DVector, Matrix3, Matrix4xX};
use num_dual::{Dual, DualNum, first_derivative};
use quantity::{Dimensionless, Moles, Pressure, Temperature};

const MAX_ITER_TP: usize = 400;
const TOL_TP: f64 = 1e-8;

/// # Flash calculations
impl<E: Residual> PhaseEquilibrium<E, 2> {
    /// Perform a Tp-flash calculation. If no initial values are
    /// given, the solution is initialized using a stability analysis.
    ///
    /// The algorithm can be use to calculate phase equilibria of systems
    /// containing non-volatile components (e.g. ions).
    pub fn tp_flash(
        eos: &E,
        temperature: Temperature,
        pressure: Pressure,
        feed: &Moles<DVector<f64>>,
        initial_state: Option<&PhaseEquilibrium<E, 2>>,
        options: SolverOptions,
        non_volatile_components: Option<Vec<usize>>,
    ) -> FeosResult<Self> {
        State::new_npt(eos, temperature, pressure, feed, None)?.tp_flash(
            initial_state,
            options,
            non_volatile_components,
        )
    }
}

/// # Flash calculations
impl<E: Residual> State<E> {
    /// Perform a Tp-flash calculation using the [State] as feed.
    /// If no initial values are given, the solution is initialized
    /// using a stability analysis.
    ///
    /// The algorithm can be use to calculate phase equilibria of systems
    /// containing non-volatile components (e.g. ions).
    pub fn tp_flash(
        &self,
        initial_state: Option<&PhaseEquilibrium<E, 2>>,
        options: SolverOptions,
        non_volatile_components: Option<Vec<usize>>,
    ) -> FeosResult<PhaseEquilibrium<E, 2>> {
        // initialization
        if let Some(init) = initial_state {
            let vle = self.tp_flash_(
                init.clone()
                    .update_pressure(self.temperature, self.pressure(Contributions::Total))?,
                options,
                non_volatile_components.clone(),
            );
            if vle.is_ok() {
                return vle;
            }
        }

        let (init1, init2) = PhaseEquilibrium::vle_init_stability(self)?;
        let vle = self.tp_flash_(init1, options, non_volatile_components.clone());
        if vle.is_ok() {
            return vle;
        }

        if let Some(init2) = init2 {
            self.tp_flash_(init2, options, non_volatile_components)
        } else {
            vle
        }
    }

    pub fn tp_flash_(
        &self,
        mut new_vle_state: PhaseEquilibrium<E, 2>,
        options: SolverOptions,
        non_volatile_components: Option<Vec<usize>>,
    ) -> FeosResult<PhaseEquilibrium<E, 2>> {
        // set options
        let (max_iter, tol, verbosity) = options.unwrap_or(MAX_ITER_TP, TOL_TP);

        log_iter!(
            verbosity,
            " iter |    residual    |  phase I mole fractions  |  phase II mole fractions  "
        );
        log_iter!(verbosity, "{:-<77}", "");
        log_iter!(
            verbosity,
            " {:4} |                | {:10.8} | {:10.8}",
            0,
            new_vle_state.vapor().molefracs,
            new_vle_state.liquid().molefracs,
        );

        let mut iter = 0;
        if non_volatile_components.is_none() {
            // 3 steps of successive substitution
            new_vle_state.successive_substitution(
                self,
                3,
                &mut iter,
                &mut None,
                tol,
                verbosity,
                &non_volatile_components,
            )?;

            // check convergence
            let beta = new_vle_state.vapor_phase_fraction();
            let tpd = [
                self.tangent_plane_distance(new_vle_state.vapor()),
                self.tangent_plane_distance(new_vle_state.liquid()),
            ];
            let dg = (1.0 - beta) * tpd[1] + beta * tpd[0];

            // fix if only tpd[1] is positive
            if tpd[0] < 0.0 && dg >= 0.0 {
                let mut k = (self.ln_phi() - new_vle_state.vapor().ln_phi()).map(f64::exp);
                // Set k = 0 for non-volatile components
                if let Some(nvc) = non_volatile_components.as_ref() {
                    nvc.iter().for_each(|&c| k[c] = 0.0);
                }
                new_vle_state.update_states(self, &k)?;
                new_vle_state.successive_substitution(
                    self,
                    1,
                    &mut iter,
                    &mut None,
                    tol,
                    verbosity,
                    &non_volatile_components,
                )?;
            }

            // fix if only tpd[0] is positive
            if tpd[1] < 0.0 && dg >= 0.0 {
                let mut k = (new_vle_state.liquid().ln_phi() - self.ln_phi()).map(f64::exp);
                // Set k = 0 for non-volatile components
                if let Some(nvc) = non_volatile_components.as_ref() {
                    nvc.iter().for_each(|&c| k[c] = 0.0);
                }
                new_vle_state.update_states(self, &k)?;
                new_vle_state.successive_substitution(
                    self,
                    1,
                    &mut iter,
                    &mut None,
                    tol,
                    verbosity,
                    &non_volatile_components,
                )?;
            }
        }

        //continue with accelerated successive subsitution
        new_vle_state.accelerated_successive_substitution(
            self,
            &mut iter,
            max_iter,
            tol,
            verbosity,
            &non_volatile_components,
        )?;

        Ok(new_vle_state)
    }

    fn tangent_plane_distance(&self, trial_state: &State<E>) -> f64 {
        let ln_phi_z = self.ln_phi();
        let ln_phi_w = trial_state.ln_phi();
        let z = &self.molefracs;
        let w = &trial_state.molefracs;
        w.dot(&(w.map(f64::ln) + ln_phi_w - z.map(f64::ln) - ln_phi_z))
    }
}

impl<E: Residual> PhaseEquilibrium<E, 2> {
    fn accelerated_successive_substitution(
        &mut self,
        feed_state: &State<E>,
        iter: &mut usize,
        max_iter: usize,
        tol: f64,
        verbosity: Verbosity,
        non_volatile_components: &Option<Vec<usize>>,
    ) -> FeosResult<()> {
        for _ in 0..max_iter {
            // do 5 successive substitution steps and check for convergence
            let mut k_vec = Matrix4xX::zeros(self.vapor().eos.components());
            // let mut k_vec = Array::zeros((4, self.vapor().eos.components()));
            if self.successive_substitution(
                feed_state,
                5,
                iter,
                &mut Some(&mut k_vec),
                tol,
                verbosity,
                non_volatile_components,
            )? {
                log_result!(
                    verbosity,
                    "Tp flash: calculation converged in {} step(s)\n",
                    iter
                );
                return Ok(());
            }

            // calculate total Gibbs energy before the extrapolation
            let gibbs = self.total_gibbs_energy();

            // extrapolate K values
            let delta_vec = k_vec.rows_range(1..) - k_vec.rows_range(..3);
            let delta = Matrix3::from_fn(|i, j| delta_vec.row(i).dot(&delta_vec.row(j)));
            let d = delta[(0, 1)] * delta[(0, 1)] - delta[(0, 0)] * delta[(1, 1)];
            let a = (delta[(0, 2)] * delta[(0, 1)] - delta[(1, 2)] * delta[(0, 0)]) / d;
            let b = (delta[(1, 2)] * delta[(0, 1)] - delta[(0, 2)] * delta[(1, 1)]) / d;

            let mut k = (k_vec.row(3)
                + ((b * delta_vec.row(1) + (a + b) * delta_vec.row(2)) / (1.0 - a - b)))
                .map(f64::exp)
                .transpose();

            // Set k = 0 for non-volatile components
            if let Some(nvc) = non_volatile_components.as_ref() {
                nvc.iter().for_each(|&c| k[c] = 0.0);
            }
            if !k.iter().all(|i| i.is_finite()) {
                continue;
            }

            // calculate new states
            let mut trial_vle_state = self.clone();
            trial_vle_state.update_states(feed_state, &k)?;
            if trial_vle_state.total_gibbs_energy() < gibbs {
                *self = trial_vle_state;
            }
        }
        Err(FeosError::NotConverged("TP flash".to_owned()))
    }

    #[expect(clippy::too_many_arguments)]
    fn successive_substitution(
        &mut self,
        feed_state: &State<E>,
        iterations: usize,
        iter: &mut usize,
        k_vec: &mut Option<&mut Matrix4xX<f64>>,
        abs_tol: f64,
        verbosity: Verbosity,
        non_volatile_components: &Option<Vec<usize>>,
    ) -> FeosResult<bool> {
        for i in 0..iterations {
            let ln_phi_v = self.vapor().ln_phi();
            let ln_phi_l = self.liquid().ln_phi();
            let mut k = (&ln_phi_l - &ln_phi_v).map(f64::exp);

            // Set k = 0 for non-volatile components
            if let Some(nvc) = non_volatile_components.as_ref() {
                nvc.iter().for_each(|&c| k[c] = 0.0);
            }

            // check for convergence
            *iter += 1;
            let mut res_vec = ln_phi_l - ln_phi_v
                + self
                    .liquid()
                    .molefracs
                    .component_div(&self.vapor().molefracs)
                    .map(|i| if i > 0.0 { i.ln() } else { 0.0 });

            // Set residuum to 0 for non-volatile components
            if let Some(nvc) = non_volatile_components.as_ref() {
                nvc.iter().for_each(|&c| res_vec[c] = 0.0);
            }
            let res = res_vec.norm();
            log_iter!(
                verbosity,
                " {:4} | {:14.8e} | {:.8} | {:.8}",
                iter,
                res,
                self.vapor().molefracs,
                self.liquid().molefracs,
            );
            if res < abs_tol {
                return Ok(true);
            }

            self.update_states(feed_state, &k)?;
            if let Some(k_vec) = k_vec
                && i >= iterations - 3
            {
                k_vec.set_row(
                    i + 3 - iterations,
                    &k.map(|ki| if ki > 0.0 { ki.ln() } else { 0.0 }).transpose(),
                );
            }
        }
        Ok(false)
    }

    fn update_states(&mut self, feed_state: &State<E>, k: &DVector<f64>) -> FeosResult<()> {
        // calculate vapor phase fraction using Rachford-Rice algorithm
        let mut beta = self.vapor_phase_fraction();
        beta = rachford_rice(&feed_state.molefracs, k, Some(beta))?;

        // update VLE
        let v = feed_state.moles.clone().component_mul(&Dimensionless::new(
            k.map(|k| beta * k / (1.0 - beta + beta * k)),
        ));
        let l = feed_state.moles.clone().component_mul(&Dimensionless::new(
            k.map(|k| (1.0 - beta) / (1.0 - beta + beta * k)),
        ));
        self.update_moles(feed_state.pressure(Contributions::Total), [&v, &l])?;
        Ok(())
    }

    fn vle_init_stability(feed_state: &State<E>) -> FeosResult<(Self, Option<Self>)> {
        let mut stable_states = feed_state.stability_analysis(SolverOptions::default())?;
        let state1 = stable_states.pop();
        let state2 = stable_states.pop();
        if let Some(s1) = state1 {
            let init1 = Self::from_states(s1.clone(), feed_state.clone());
            if let Some(s2) = state2 {
                Ok((Self::from_states(s1, s2), Some(init1)))
            } else {
                Ok((init1, None))
            }
        } else {
            Err(FeosError::NoPhaseSplit)
        }
    }
}

fn rachford_rice(feed: &DVector<f64>, k: &DVector<f64>, beta_in: Option<f64>) -> FeosResult<f64> {
    const MAX_ITER: usize = 10;
    const ABS_TOL: f64 = 1e-6;

    // check if solution exists
    let (mut beta_min, mut beta_max) = if feed.dot(k) > 1.0
        && feed
            .component_div(k)
            .iter()
            .filter(|x| !x.is_nan())
            .sum::<f64>()
            > 1.0
    {
        (0.0, 1.0)
    } else {
        return Err(FeosError::IterationFailed(String::from("rachford_rice")));
    };

    // look for tighter bounds
    for (&k, &f) in k.iter().zip(feed.iter()) {
        if k > 1.0 {
            let b = (k * f - 1.0) / (k - 1.0);
            if b > beta_min {
                beta_min = b;
            }
        }
        if k < 1.0 {
            let b = (1.0 - f) / (1.0 - k);
            if b < beta_max {
                beta_max = b;
            }
        }
    }

    // initialize
    let mut beta = 0.5 * (beta_min + beta_max);
    if let Some(b) = beta_in
        && b > beta_min
        && b < beta_max
    {
        beta = b;
    }
    let g = feed.dot(&k.map(|k| (k - 1.0) / (1.0 - beta + beta * k)));
    if g > 0.0 {
        beta_min = beta
    } else {
        beta_max = beta
    }

    // iterate
    for _ in 0..MAX_ITER {
        let (g, dg) = first_derivative(
            |beta| {
                let frac = k.map(|k| (-beta + beta * k + 1.0).recip() * (k - 1.0));
                feed.map(Dual::from).dot(&frac)
            },
            beta,
        );
        if g > 0.0 {
            beta_min = beta;
        } else {
            beta_max = beta;
        }

        let dbeta = g / dg;
        beta -= dbeta;

        if beta < beta_min || beta > beta_max {
            beta = 0.5 * (beta_min + beta_max);
        }
        if dbeta.abs() < ABS_TOL {
            return Ok(beta);
        }
    }

    Ok(beta)
}
