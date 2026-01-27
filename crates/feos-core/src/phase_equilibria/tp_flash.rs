use super::PhaseEquilibrium;
use crate::equation_of_state::Residual;
use crate::errors::{FeosError, FeosResult};
use crate::state::{Contributions, State};
use crate::{Composition, DensityInitialization, ReferenceSystem, SolverOptions, Verbosity};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, Matrix3, OVector, SVector, U1, U2, vector};
use num_dual::{
    Dual, Dual2Vec, DualNum, DualStruct, Gradients, first_derivative, implicit_derivative_sp,
};
use quantity::{MolarEnergy, MolarVolume, Pressure, RGAS, Temperature};

const MAX_ITER_TP: usize = 400;
const TOL_TP: f64 = 1e-8;

/// # Flash calculations
impl<E: Residual<N>, N: Gradients> PhaseEquilibrium<E, 2, N>
where
    DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    /// Perform a Tp-flash calculation. If no initial values are
    /// given, the solution is initialized using a stability analysis.
    ///
    /// The algorithm can be use to calculate phase equilibria of systems
    /// containing non-volatile components (e.g. ions).
    pub fn tp_flash<X: Composition<f64, N>>(
        eos: &E,
        temperature: Temperature,
        pressure: Pressure,
        feed: X,
        initial_state: Option<&PhaseEquilibrium<E, 2, N>>,
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

impl<E: Residual<U2, D>, D: DualNum<f64> + Copy> PhaseEquilibrium<E, 2, U2, D> {
    /// Perform a Tp-flash calculation for a binary mixture.
    /// Compared to the version of the algorithm for a generic
    /// number of components ([tp_flash](PhaseEquilibrium::tp_flash)),
    /// this can be used in combination with automatic differentiation.
    pub fn tp_flash_binary<X: Composition<D, U2>>(
        eos: &E,
        temperature: Temperature<D>,
        pressure: Pressure<D>,
        feed: X,
        options: SolverOptions,
    ) -> FeosResult<Self> {
        let (feed, total_moles) = feed.into_molefracs(eos)?;
        let z = feed[0];
        let vle_re = State::new_npt(&eos.re(), temperature.re(), pressure.re(), z.re(), None)?
            .tp_flash(None, options, None)?;

        // implicit differentiation

        // specifications
        let t = temperature.into_reduced();
        let p = pressure.into_reduced();

        // molar volume and composition of the two phases
        let variables = SVector::from([
            vle_re.liquid().density.into_reduced().recip(),
            vle_re.vapor().density.into_reduced().recip(),
            vle_re.liquid().molefracs[0],
            vle_re.vapor().molefracs[0],
        ]);

        // calculate derivatives for molar volumes and compositions (first component)
        // with respect to t, p, or z or equation of state parameters
        // using implicit differentiation of the minimum in the Gibbs energy
        let [[v_l, v_v, x, y]] = implicit_derivative_sp(
            |variables, &[t, p, z]: &[_; 3]| {
                let [[v_l, v_v, x, y]] = variables.data.0;
                let beta = (z - x) / (y - x);
                let eos = eos.lift();
                let molar_gibbs_energy = |x: Dual2Vec<_, _, _>, v| {
                    let molefracs = vector![x, -x + 1.0];
                    let a_res = eos.residual_helmholtz_energy(t, v, &molefracs);
                    let a_ig = (x * (x / v).ln() - (x - 1.0) * ((-x + 1.0) / v).ln() - 1.0) * t;
                    a_res + a_ig + v * p
                };
                // g = a + pv is the potential function for a tp flash using a Helmholtz energy model
                // see https://www.sciencedirect.com/science/article/pii/S0378381299000928
                molar_gibbs_energy(y, v_v) * beta - molar_gibbs_energy(x, v_l) * (beta - 1.0)
            },
            variables,
            &[t, p, z],
        )
        .data
        .0;
        let beta = (z - x) / (y - x);
        let state = |x: D, v| {
            let density = MolarVolume::from_reduced(v).inv();
            State::new(eos, temperature, density, x)
        };
        Ok(Self::with_vapor_phase_fraction(
            state(y, v_v)?,
            state(x, v_l)?,
            beta,
            total_moles,
        ))
    }
}

/// # Flash calculations
impl<E: Residual<N>, N: Gradients> State<E, N>
where
    DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    /// Perform a Tp-flash calculation using the [State] as feed.
    /// If no initial values are given, the solution is initialized
    /// using a stability analysis.
    ///
    /// The algorithm can be use to calculate phase equilibria of systems
    /// containing non-volatile components (e.g. ions).
    pub fn tp_flash(
        &self,
        initial_state: Option<&PhaseEquilibrium<E, 2, N>>,
        options: SolverOptions,
        non_volatile_components: Option<Vec<usize>>,
    ) -> FeosResult<PhaseEquilibrium<E, 2, N>> {
        // initialization
        if let Some(initial_state) = initial_state {
            let mut init = initial_state.clone();
            init.update_states(
                self,
                initial_state.vapor().molefracs.clone(),
                initial_state.liquid().molefracs.clone(),
                initial_state.vapor_phase_fraction(),
            )?;
            let vle = self.tp_flash_(init, options, non_volatile_components.clone());
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
        mut new_vle_state: PhaseEquilibrium<E, 2, N>,
        options: SolverOptions,
        non_volatile_components: Option<Vec<usize>>,
    ) -> FeosResult<PhaseEquilibrium<E, 2, N>> {
        // set options
        let (max_iter, tol, verbosity) = options.unwrap_or(MAX_ITER_TP, TOL_TP);

        log_iter!(
            verbosity,
            " iter |    residual    |  phase I mole fractions  |  phase II mole fractions  "
        );
        log_iter!(verbosity, "{:-<77}", "");
        log_iter!(
            verbosity,
            " {:4} |                | {:10.8?} | {:10.8?}",
            0,
            new_vle_state.vapor().molefracs.as_slice(),
            new_vle_state.liquid().molefracs.as_slice(),
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
            let tpd = [
                self.tangent_plane_distance(new_vle_state.vapor()),
                self.tangent_plane_distance(new_vle_state.liquid()),
            ];
            let b = new_vle_state.phase_fractions;
            let dg = b[0] * tpd[0] + b[1] * tpd[1];

            // fix if only tpd[1] is positive
            if tpd[0] < 0.0 && dg >= 0.0 {
                let mut k = (self.ln_phi() - new_vle_state.vapor().ln_phi()).map(f64::exp);
                // Set k = 0 for non-volatile components
                if let Some(nvc) = non_volatile_components.as_ref() {
                    nvc.iter().for_each(|&c| k[c] = 0.0);
                }
                new_vle_state.rachford_rice_inplace(self, &k)?;
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
                new_vle_state.rachford_rice_inplace(self, &k)?;
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

    fn tangent_plane_distance(&self, trial_state: &State<E, N>) -> f64 {
        let ln_phi_z = self.ln_phi();
        let ln_phi_w = trial_state.ln_phi();
        let z = &self.molefracs;
        let w = &trial_state.molefracs;
        w.dot(&(w.map(f64::ln) + ln_phi_w - z.map(f64::ln) - ln_phi_z))
    }
}

impl<E: Residual<N>, N: Gradients> PhaseEquilibrium<E, 2, N>
where
    DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    fn accelerated_successive_substitution(
        &mut self,
        feed_state: &State<E, N>,
        iter: &mut usize,
        max_iter: usize,
        tol: f64,
        verbosity: Verbosity,
        non_volatile_components: &Option<Vec<usize>>,
    ) -> FeosResult<()> {
        let (n, _) = feed_state.molefracs.shape_generic();
        for _ in 0..max_iter {
            // do 5 successive substitution steps and check for convergence
            let mut k_vec = std::array::repeat(OVector::zeros_generic(n, U1));
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
            let gibbs = self.molar_gibbs_energy();

            // extrapolate K values
            let delta_vec = [
                &k_vec[1] - &k_vec[0],
                &k_vec[2] - &k_vec[1],
                &k_vec[3] - &k_vec[2],
            ];
            let delta = Matrix3::from_fn(|i, j| delta_vec[i].dot(&delta_vec[j]));
            let d = delta[(0, 1)] * delta[(0, 1)] - delta[(0, 0)] * delta[(1, 1)];
            let a = (delta[(0, 2)] * delta[(0, 1)] - delta[(1, 2)] * delta[(0, 0)]) / d;
            let b = (delta[(1, 2)] * delta[(0, 1)] - delta[(0, 2)] * delta[(1, 1)]) / d;

            let mut k = (&k_vec[3]
                + ((b * &delta_vec[1] + (a + b) * &delta_vec[2]) / (1.0 - a - b)))
                .map(f64::exp);

            // Set k = 0 for non-volatile components
            if let Some(nvc) = non_volatile_components.as_ref() {
                nvc.iter().for_each(|&c| k[c] = 0.0);
            }
            if !k.iter().all(|i| i.is_finite()) {
                continue;
            }

            // calculate new states
            let mut trial_vle_state = self.clone();
            trial_vle_state.rachford_rice_inplace(feed_state, &k)?;
            if trial_vle_state.molar_gibbs_energy() < gibbs {
                *self = trial_vle_state;
            }
        }
        Err(FeosError::NotConverged("TP flash".to_owned()))
    }

    #[expect(clippy::too_many_arguments)]
    fn successive_substitution(
        &mut self,
        feed_state: &State<E, N>,
        iterations: usize,
        iter: &mut usize,
        k_vec: &mut Option<&mut [OVector<f64, N>; 4]>,
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
            let mut res_vec = k.component_mul(&self.liquid().molefracs) - &self.vapor().molefracs;

            // Set residuum to 0 for non-volatile components
            if let Some(nvc) = non_volatile_components.as_ref() {
                nvc.iter().for_each(|&c| res_vec[c] = 0.0);
            }

            let res = res_vec.norm();
            log_iter!(
                verbosity,
                " {:4} | {:14.8e} | {:.8?} | {:.8?}",
                iter,
                res,
                self.vapor().molefracs.as_slice(),
                self.liquid().molefracs.as_slice(),
            );
            if res < abs_tol {
                return Ok(true);
            }

            self.rachford_rice_inplace(feed_state, &k)?;
            if let Some(k_vec) = k_vec
                && i >= iterations - 3
            {
                k_vec[i + 3 - iterations] = k.map(|ki| if ki > 0.0 { ki.ln() } else { 0.0 });
            }
        }
        Ok(false)
    }

    fn rachford_rice_inplace(
        &mut self,
        feed_state: &State<E, N>,
        k: &OVector<f64, N>,
    ) -> FeosResult<()> {
        // calculate vapor phase fraction using Rachford-Rice algorithm
        let (b, [v, l]) =
            rachford_rice(&feed_state.molefracs, k, Some(self.vapor_phase_fraction()))?;
        self.update_states(feed_state, v, l, b)
    }

    fn update_states(
        &mut self,
        feed_state: &State<E, N>,
        vapor_molefracs: OVector<f64, N>,
        liquid_molefracs: OVector<f64, N>,
        beta: f64,
    ) -> FeosResult<()> {
        let vapor = State::new_npt(
            &feed_state.eos,
            feed_state.temperature,
            feed_state.pressure(Contributions::Total),
            vapor_molefracs,
            Some(DensityInitialization::InitialDensity(self.vapor().density)),
        )?;
        let liquid = State::new_npt(
            &feed_state.eos,
            feed_state.temperature,
            feed_state.pressure(Contributions::Total),
            liquid_molefracs,
            Some(DensityInitialization::InitialDensity(self.liquid().density)),
        )?;

        *self = Self::with_vapor_phase_fraction(vapor, liquid, beta, feed_state.total_moles);

        Ok(())
    }

    fn vle_init_stability(feed_state: &State<E, N>) -> FeosResult<(Self, Option<Self>)> {
        let mut stable_states = feed_state.stability_analysis(SolverOptions::default())?;
        let state1 = stable_states.pop();
        let state2 = stable_states.pop();
        if let Some(s1) = state1 {
            let init1 = if s1.density < feed_state.density {
                Self::two_phase(s1.clone(), feed_state.clone())
            } else {
                Self::two_phase(feed_state.clone(), s1.clone())
            };
            let init2 = state2.map(|s2| {
                if s1.density < s2.density {
                    Self::two_phase(s1.clone(), s2.clone())
                } else {
                    Self::two_phase(s2.clone(), s1.clone())
                }
            });
            Ok((init1, init2))
        } else {
            Err(FeosError::NoPhaseSplit)
        }
    }

    // Total molar Gibbs energy excluding the constant contribution RT sum_i x_i ln(\Lambda_i^3)
    fn molar_gibbs_energy(&self) -> MolarEnergy {
        self.states
            .iter()
            .fold(MolarEnergy::from_reduced(0.0), |acc, s| {
                let ln_rho_m1 = s.partial_density().to_reduced().map(|r| r.ln() - 1.0);
                acc + s.residual_molar_helmholtz_energy()
                    + s.pressure(Contributions::Total) * s.molar_volume
                    + RGAS * s.temperature * s.molefracs.dot(&ln_rho_m1)
            })
    }
}

fn rachford_rice<N: Dim>(
    feed: &OVector<f64, N>,
    k: &OVector<f64, N>,
    beta_in: Option<f64>,
) -> FeosResult<(f64, [OVector<f64, N>; 2])>
where
    DefaultAllocator: Allocator<N>,
{
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
            // update VLE
            let v = feed.component_mul(&k.map(|k| beta * k / (1.0 - beta + beta * k)));
            let l = feed.component_mul(&k.map(|k| (1.0 - beta) / (1.0 - beta + beta * k)));
            return Ok((beta, [v, l]));
        }
    }

    // update VLE
    let v = feed.component_mul(&k.map(|k| beta * k / (1.0 - beta + beta * k)));
    let l = feed.component_mul(&k.map(|k| (1.0 - beta) / (1.0 - beta + beta * k)));

    Ok((beta, [v, l]))
}
