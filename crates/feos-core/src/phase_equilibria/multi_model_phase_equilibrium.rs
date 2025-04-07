use crate::equation_of_state::Residual;
use crate::errors::{EosError, EosResult};
use crate::state::{DensityInitialization, State};
use crate::{Contributions, ReferenceSystem, SolverOptions};
use ndarray::Array1;
use quantity::{Dimensionless, Energy, Moles, Pressure, Temperature, RGAS};
use std::fmt;
use std::fmt::Write;
use std::sync::Arc;

/// A thermodynamic equilibrium state.
///
/// The struct is parametrized over the number of phases with most features
/// being implemented for the two phase vapor/liquid or liquid/liquid case.
///
/// ## Contents
///
/// + [Bubble and dew point calculations](#bubble-and-dew-point-calculations)
/// + [Heteroazeotropes](#heteroazeotropes)
/// + [Flash calculations](#flash-calculations)
/// + [Pure component phase equilibria](#pure-component-phase-equilibria)
/// + [Utility functions](#utility-functions)
#[derive(Debug)]
pub struct PhaseEquilibriumMulti<E1, E2>(State<E1>, State<E2>);

impl<E1, E2> Clone for PhaseEquilibriumMulti<E1, E2> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1.clone())
    }
}

impl<E1: Residual, E2: Residual> fmt::Display for PhaseEquilibriumMulti<E1, E2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "phase 1: {}", self.0)?;
        writeln!(f, "phase 2: {}", self.1)
    }
}

impl<E1: Residual, E2: Residual> PhaseEquilibriumMulti<E1, E2> {
    pub fn _repr_markdown_(&self) -> String {
        if self.0.eos.components() == 1 {
            let mut res = "||temperature|density|\n|-|-|-|\n".to_string();
            writeln!(
                res,
                "|phase {}|{:.5}|{:.5}|",
                1, self.0.temperature, self.0.density
            )
            .unwrap();
            writeln!(
                res,
                "|phase {}|{:.5}|{:.5}|",
                2, self.1.temperature, self.1.density
            )
            .unwrap();
            res
        } else {
            let mut res = "||temperature|density|molefracs|\n|-|-|-|-|\n".to_string();
            writeln!(
                res,
                "|phase {}|{:.5}|{:.5}|{:.5}|",
                1, self.0.temperature, self.0.density, self.0.molefracs
            )
            .unwrap();
            writeln!(
                res,
                "|phase {}|{:.5}|{:.5}|{:.5}|",
                1, self.1.temperature, self.1.density, self.1.molefracs
            )
            .unwrap();
            res
        }
    }
}

impl<E1, E2> PhaseEquilibriumMulti<E1, E2> {
    pub fn vapor(&self) -> &State<E1> {
        &self.0
    }

    pub fn liquid(&self) -> &State<E2> {
        &self.1
    }
}

impl<E1: Residual, E2: Residual> PhaseEquilibriumMulti<E1, E2> {
    pub(super) fn from_states(state1: State<E1>, state2: State<E2>) -> Self {
        Self(state1, state2)
    }

    /// Creates a new PhaseEquilibrium that contains two states at the
    /// specified temperature, pressure and moles.
    ///
    /// The constructor can be used in custom phase equilibrium solvers or,
    /// e.g., to generate initial guesses for an actual VLE solver.
    /// In general, the two states generated are NOT in an equilibrium.
    pub fn new_npt(
        eos1: &Arc<E1>,
        eos2: &Arc<E2>,
        temperature: Temperature,
        pressure: Pressure,
        vapor_moles: &Moles<Array1<f64>>,
        liquid_moles: &Moles<Array1<f64>>,
    ) -> EosResult<Self> {
        let vapor = State::new_npt(
            eos1,
            temperature,
            pressure,
            vapor_moles,
            DensityInitialization::Vapor,
        )?;
        let liquid = State::new_npt(
            eos2,
            temperature,
            pressure,
            liquid_moles,
            DensityInitialization::Liquid,
        )?;
        Ok(Self(vapor, liquid))
    }

    pub(super) fn vapor_phase_fraction(&self) -> f64 {
        (self.vapor().total_moles / (self.vapor().total_moles + self.liquid().total_moles))
            .into_value()
    }
}

impl<E1: Residual, E2: Residual> PhaseEquilibriumMulti<E1, E2> {
    pub(super) fn update_pressure(
        mut self,
        temperature: Temperature,
        pressure: Pressure,
    ) -> EosResult<Self> {
        self.0 = State::new_npt(
            &self.0.eos,
            temperature,
            pressure,
            &self.0.moles,
            DensityInitialization::InitialDensity(self.0.density),
        )?;
        self.1 = State::new_npt(
            &self.1.eos,
            temperature,
            pressure,
            &self.1.moles,
            DensityInitialization::InitialDensity(self.1.density),
        )?;
        Ok(self)
    }

    pub(super) fn update_moles(
        &mut self,
        pressure: Pressure,
        moles: [&Moles<Array1<f64>>; 2],
    ) -> EosResult<()> {
        self.0 = State::new_npt(
            &self.0.eos,
            self.0.temperature,
            pressure,
            moles[0],
            DensityInitialization::InitialDensity(self.0.density),
        )?;
        self.1 = State::new_npt(
            &self.1.eos,
            self.1.temperature,
            pressure,
            moles[1],
            DensityInitialization::InitialDensity(self.1.density),
        )?;
        Ok(())
    }

    // Total Gibbs energy excluding the constant contribution RT sum_i N_i ln(\Lambda_i^3)
    pub(super) fn total_gibbs_energy(&self) -> Energy {
        let mut g = Energy::from_reduced(0.0);
        let s = &self.0;
        let ln_rho = s.partial_density.to_reduced().mapv(f64::ln);
        g += s.residual_helmholtz_energy()
            + s.pressure(Contributions::Total) * s.volume
            + RGAS * s.temperature * (s.moles.clone() * Dimensionless::new(ln_rho - 1.0)).sum();

        let s = &self.1;
        let ln_rho = s.partial_density.to_reduced().mapv(f64::ln);
        g += s.residual_helmholtz_energy()
            + s.pressure(Contributions::Total) * s.volume
            + RGAS * s.temperature * (s.moles.clone() * Dimensionless::new(ln_rho - 1.0)).sum();
        g
    }
}

const TRIVIAL_REL_DEVIATION: f64 = 1e-5;

/// # Utility functions
impl<E1: Residual, E2: Residual> PhaseEquilibriumMulti<E1, E2> {
    pub(super) fn check_trivial_solution(self) -> EosResult<Self> {
        if Self::is_trivial_solution(self.vapor(), self.liquid()) {
            Err(EosError::TrivialSolution)
        } else {
            Ok(self)
        }
    }

    /// Check if the two states form a trivial solution
    pub fn is_trivial_solution(state1: &State<E1>, state2: &State<E2>) -> bool {
        let rho1 = state1.partial_density.to_reduced();
        let rho2 = state2.partial_density.to_reduced();

        rho1.iter()
            .zip(rho2.iter())
            .fold(0.0, |acc, (&rho1, &rho2)| {
                (rho2 / rho1 - 1.0).abs().max(acc)
            })
            < TRIVIAL_REL_DEVIATION
    }
}


const MAX_ITER_TP: usize = 400;
const TOL_TP: f64 = 1e-8;

/// # Flash calculations
impl<E1: Residual, E2: Residual> PhaseEquilibriumMulti<E1, E2> {
    /// Perform a Tp-flash calculation. If no initial values are
    /// given, the solution is initialized using a stability analysis.
    ///
    /// The algorithm can be use to calculate phase equilibria of systems
    /// containing non-volatile components (e.g. ions).
    pub fn tp_flash(
        eos1: &Arc<E1>,
        eos2: &Arc<E2>,
        temperature: Temperature,
        pressure: Pressure,
        feed: &Moles<Array1<f64>>,
        initial_state: Option<&PhaseEquilibriumMulti<E1, E2>>,
        options: SolverOptions,
        non_volatile_components: Option<Vec<usize>>,
    ) -> EosResult<Self> {
        State::new_npt(
            eos,
            temperature,
            pressure,
            feed,
            DensityInitialization::None,
        )?
        .tp_flash(initial_state, options, non_volatile_components)
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
    ) -> EosResult<PhaseEquilibrium<E, 2>> {
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
    ) -> EosResult<PhaseEquilibrium<E, 2>> {
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
                let mut k = (self.ln_phi() - new_vle_state.vapor().ln_phi()).mapv(f64::exp);
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
                let mut k = (new_vle_state.liquid().ln_phi() - self.ln_phi()).mapv(f64::exp);
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
        (w * &(w.mapv(f64::ln) + ln_phi_w - z.mapv(f64::ln) - ln_phi_z)).sum()
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
    ) -> EosResult<()> {
        for _ in 0..max_iter {
            // do 5 successive substitution steps and check for convergence
            let mut k_vec = Array::zeros((4, self.vapor().eos.components()));
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
            let delta_vec = &k_vec.slice(s![1.., ..]) - &k_vec.slice(s![..3, ..]);
            let delta = Array::from_shape_fn((3, 3), |(i, j)| {
                (&delta_vec.index_axis(Axis(0), i) * &delta_vec.index_axis(Axis(0), j)).sum()
            });
            let d = delta[(0, 1)] * delta[(0, 1)] - delta[(0, 0)] * delta[(1, 1)];
            let a = (delta[(0, 2)] * delta[(0, 1)] - delta[(1, 2)] * delta[(0, 0)]) / d;
            let b = (delta[(1, 2)] * delta[(0, 1)] - delta[(0, 2)] * delta[(1, 1)]) / d;

            let mut k = (&k_vec.index_axis(Axis(0), 3)
                + &((b * &delta_vec.index_axis(Axis(0), 1)
                    + (a + b) * &delta_vec.index_axis(Axis(0), 2))
                    / (1.0 - a - b)))
                .mapv(f64::exp);

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
        Err(EosError::NotConverged("TP flash".to_owned()))
    }

    #[expect(clippy::too_many_arguments)]
    fn successive_substitution(
        &mut self,
        feed_state: &State<E>,
        iterations: usize,
        iter: &mut usize,
        k_vec: &mut Option<&mut Array2<f64>>,
        abs_tol: f64,
        verbosity: Verbosity,
        non_volatile_components: &Option<Vec<usize>>,
    ) -> EosResult<bool> {
        for i in 0..iterations {
            let ln_phi_v = self.vapor().ln_phi();
            let ln_phi_l = self.liquid().ln_phi();
            let mut k = (&ln_phi_l - &ln_phi_v).mapv(f64::exp);

            // Set k = 0 for non-volatile components
            if let Some(nvc) = non_volatile_components.as_ref() {
                nvc.iter().for_each(|&c| k[c] = 0.0);
            }

            // check for convergence
            *iter += 1;
            let mut res_vec = ln_phi_l - ln_phi_v
                + (&self.liquid().molefracs / &self.vapor().molefracs).map(|&i| {
                    if i > 0.0 {
                        i.ln()
                    } else {
                        0.0
                    }
                });

            // Set residuum to 0 for non-volatile components
            if let Some(nvc) = non_volatile_components.as_ref() {
                nvc.iter().for_each(|&c| res_vec[c] = 0.0);
            }
            let res = norm(&res_vec);
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
            if let Some(k_vec) = k_vec {
                if i >= iterations - 3 {
                    k_vec
                        .index_axis_mut(Axis(0), i + 3 - iterations)
                        .assign(&k.map(|ki| if *ki > 0.0 { ki.ln() } else { 0.0 }));
                }
            }
        }
        Ok(false)
    }

    fn update_states(&mut self, feed_state: &State<E>, k: &Array1<f64>) -> EosResult<()> {
        // calculate vapor phase fraction using Rachford-Rice algorithm
        let mut beta = self.vapor_phase_fraction();
        beta = rachford_rice(&feed_state.molefracs, k, Some(beta))?;

        // update VLE
        let v = feed_state.moles.clone() * Dimensionless::new(beta * k / (1.0 - beta + beta * k));
        let l =
            feed_state.moles.clone() * Dimensionless::new((1.0 - beta) / (1.0 - beta + beta * k));
        self.update_moles(feed_state.pressure(Contributions::Total), [&v, &l])?;
        Ok(())
    }

    fn vle_init_stability(feed_state: &State<E>) -> EosResult<(Self, Option<Self>)> {
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
            Err(EosError::NoPhaseSplit)
        }
    }
}

fn rachford_rice(feed: &Array1<f64>, k: &Array1<f64>, beta_in: Option<f64>) -> EosResult<f64> {
    const MAX_ITER: usize = 10;
    const ABS_TOL: f64 = 1e-6;

    // check if solution exists
    let (mut beta_min, mut beta_max) =
        if (feed * k).sum() > 1.0 && (feed / k).iter().filter(|x| !x.is_nan()).sum::<f64>() > 1.0 {
            (0.0, 1.0)
        } else {
            return Err(EosError::IterationFailed(String::from("rachford_rice")));
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
    if let Some(b) = beta_in {
        if b > beta_min && b < beta_max {
            beta = b;
        }
    }
    let g = (feed * &(k - 1.0) / (1.0 - beta + beta * k)).sum();
    if g > 0.0 {
        beta_min = beta
    } else {
        beta_max = beta
    }

    // iterate
    for _ in 0..MAX_ITER {
        let frac = (k - 1.0) / (1.0 - beta + beta * k);
        let g = (feed * &frac).sum();
        let dg = -(feed * &frac * &frac).sum();
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
