use crate::errors::{FeosError, FeosResult};
use crate::phase_equilibria::PhaseEquilibrium;
use crate::state::{
    Contributions,
    DensityInitialization::{InitialDensity, Liquid, Vapor},
};
use crate::{ReferenceSystem, Residual, SolverOptions, State, Verbosity};
use nalgebra::allocator::Allocator;
use nalgebra::{DMatrix, DVector, DefaultAllocator, Dim, Dyn, OVector, U1};
#[cfg(feature = "ndarray")]
use ndarray::Array1;
use num_dual::linalg::LU;
use num_dual::{DualNum, DualStruct, Gradients};
use quantity::{Density, Dimensionless, Moles, Pressure, Quantity, RGAS, SIUnit, Temperature};
use typenum::{N1, N2, P1, Z0};

const MAX_ITER_INNER: usize = 5;
const TOL_INNER: f64 = 1e-9;
const MAX_ITER_OUTER: usize = 400;
const TOL_OUTER: f64 = 1e-10;

const MAX_TSTEP: f64 = 20.0;
const MAX_LNPSTEP: f64 = 0.1;
const NEWTON_TOL: f64 = 1e-3;

/// Trait that enables functions to be generic over their input unit.
pub trait TemperatureOrPressure<D: DualNum<f64> + Copy = f64>: Copy {
    type Other: Copy;

    const IDENTIFIER: &'static str;

    fn temperature(&self) -> Option<Temperature<D>>;
    fn pressure(&self) -> Option<Pressure<D>>;

    fn temperature_pressure(
        &self,
        tp_init: Option<Self::Other>,
    ) -> (Option<Temperature<D>>, Option<Pressure<D>>, bool);

    fn from_state<E: Residual<N, D>, N: Gradients>(state: &State<E, N, D>) -> Self::Other
    where
        DefaultAllocator: Allocator<N>;

    #[cfg(feature = "ndarray")]
    fn linspace(
        &self,
        start: Self::Other,
        end: Self::Other,
        n: usize,
    ) -> (Temperature<Array1<f64>>, Pressure<Array1<f64>>);
}

impl<D: DualNum<f64> + Copy> TemperatureOrPressure<D> for Temperature<D> {
    type Other = Pressure<D>;
    const IDENTIFIER: &'static str = "temperature";

    fn temperature(&self) -> Option<Temperature<D>> {
        Some(*self)
    }

    fn pressure(&self) -> Option<Pressure<D>> {
        None
    }

    fn temperature_pressure(
        &self,
        tp_init: Option<Self::Other>,
    ) -> (Option<Temperature<D>>, Option<Pressure<D>>, bool) {
        (Some(*self), tp_init, true)
    }

    fn from_state<E: Residual<N, D>, N: Gradients>(state: &State<E, N, D>) -> Self::Other
    where
        DefaultAllocator: Allocator<N>,
    {
        state.pressure(Contributions::Total)
    }

    #[cfg(feature = "ndarray")]
    fn linspace(
        &self,
        start: Pressure<D>,
        end: Pressure<D>,
        n: usize,
    ) -> (Temperature<Array1<f64>>, Pressure<Array1<f64>>) {
        (
            Temperature::linspace(self.re(), self.re(), n),
            Pressure::linspace(start.re(), end.re(), n),
        )
    }
}

// For some inexplicable reason this does not compile if the `Pressure` type is
// used instead of the explicit unit. Maybe the type is too complicated for the
// compiler?
impl<D: DualNum<f64> + Copy> TemperatureOrPressure<D>
    for Quantity<D, SIUnit<N2, N1, P1, Z0, Z0, Z0, Z0>>
{
    type Other = Temperature<D>;
    const IDENTIFIER: &'static str = "pressure";

    fn temperature(&self) -> Option<Temperature<D>> {
        None
    }

    fn pressure(&self) -> Option<Pressure<D>> {
        Some(*self)
    }

    fn temperature_pressure(
        &self,
        tp_init: Option<Self::Other>,
    ) -> (Option<Temperature<D>>, Option<Pressure<D>>, bool) {
        (tp_init, Some(*self), false)
    }

    fn from_state<E: Residual<N, D>, N: Dim>(state: &State<E, N, D>) -> Self::Other
    where
        DefaultAllocator: Allocator<N>,
    {
        state.temperature
    }

    #[cfg(feature = "ndarray")]
    fn linspace(
        &self,
        start: Temperature<D>,
        end: Temperature<D>,
        n: usize,
    ) -> (Temperature<Array1<f64>>, Pressure<Array1<f64>>) {
        (
            Temperature::linspace(start.re(), end.re(), n),
            Pressure::linspace(self.re(), self.re(), n),
        )
    }
}

/// # Bubble and dew point calculations
impl<E: Residual<N, D>, N: Gradients, D: DualNum<f64> + Copy> PhaseEquilibrium<E, 2, N, D>
where
    DefaultAllocator: Allocator<N> + Allocator<N, N> + Allocator<U1, N>,
{
    /// Calculate a phase equilibrium for a given temperature
    /// or pressure and composition of the liquid phase.
    pub fn bubble_point<TP: TemperatureOrPressure<D>>(
        eos: &E,
        temperature_or_pressure: TP,
        liquid_molefracs: &OVector<D, N>,
        tp_init: Option<TP::Other>,
        vapor_molefracs: Option<&OVector<f64, N>>,
        options: (SolverOptions, SolverOptions),
    ) -> FeosResult<Self>
    where
        E: Clone,
    {
        Self::bubble_dew_point(
            eos,
            temperature_or_pressure,
            liquid_molefracs,
            tp_init,
            vapor_molefracs,
            true,
            options,
        )
    }

    /// Calculate a phase equilibrium for a given temperature
    /// or pressure and composition of the vapor phase.
    pub fn dew_point<TP: TemperatureOrPressure<D>>(
        eos: &E,
        temperature_or_pressure: TP,
        vapor_molefracs: &OVector<D, N>,
        tp_init: Option<TP::Other>,
        liquid_molefracs: Option<&OVector<f64, N>>,
        options: (SolverOptions, SolverOptions),
    ) -> FeosResult<Self>
    where
        E: Clone,
    {
        Self::bubble_dew_point(
            eos,
            temperature_or_pressure,
            vapor_molefracs,
            tp_init,
            liquid_molefracs,
            false,
            options,
        )
    }

    pub(super) fn bubble_dew_point<TP: TemperatureOrPressure<D>>(
        eos: &E,
        temperature_or_pressure: TP,
        vapor_molefracs: &OVector<D, N>,
        tp_init: Option<TP::Other>,
        liquid_molefracs: Option<&OVector<f64, N>>,
        bubble: bool,
        options: (SolverOptions, SolverOptions),
    ) -> FeosResult<Self>
    where
        E: Clone,
    {
        let (temperature, pressure, iterate_p) =
            temperature_or_pressure.temperature_pressure(tp_init);
        Self::bubble_dew_point_tp(
            eos,
            temperature,
            pressure,
            vapor_molefracs,
            liquid_molefracs,
            bubble,
            iterate_p,
            options,
        )
    }

    #[expect(clippy::too_many_arguments)]
    fn bubble_dew_point_tp(
        eos: &E,
        temperature: Option<Temperature<D>>,
        pressure: Option<Pressure<D>>,
        molefracs_spec: &OVector<D, N>,
        molefracs_init: Option<&OVector<f64, N>>,
        bubble: bool,
        iterate_p: bool,
        options: (SolverOptions, SolverOptions),
    ) -> FeosResult<Self>
    where
        E: Clone,
    {
        let eos_re = eos.re();
        let mut temperature_re = temperature.map(|t| t.re());
        let mut pressure_re = pressure.map(|p| p.re());
        let molefracs_spec_re = molefracs_spec.map(|x| x.re());
        let (v1, rho2) = if iterate_p {
            // temperature is specified
            let temperature_re = temperature_re.as_mut().unwrap();

            // First use given initial pressure if applicable
            if let Some(p) = pressure_re.as_mut() {
                PhaseEquilibrium::iterate_bubble_dew(
                    &eos_re,
                    temperature_re,
                    p,
                    &molefracs_spec_re,
                    molefracs_init,
                    bubble,
                    iterate_p,
                    options,
                )?
            } else {
                // Next try to initialize with an ideal gas assumption
                let x2 = PhaseEquilibrium::starting_pressure_ideal_gas(
                    &eos_re,
                    *temperature_re,
                    &molefracs_spec_re,
                    bubble,
                )
                .and_then(|(p, x)| {
                    pressure_re = Some(p);
                    PhaseEquilibrium::iterate_bubble_dew(
                        &eos_re,
                        temperature_re,
                        pressure_re.as_mut().unwrap(),
                        &molefracs_spec_re,
                        molefracs_init.or(Some(&x)),
                        bubble,
                        iterate_p,
                        options,
                    )
                });

                // Finally use the spinodal to initialize the calculation
                x2.or_else(|_| {
                    PhaseEquilibrium::starting_pressure_spinodal(
                        &eos_re,
                        *temperature_re,
                        &molefracs_spec_re,
                    )
                    .and_then(|p| {
                        pressure_re = Some(p);
                        PhaseEquilibrium::iterate_bubble_dew(
                            &eos_re,
                            temperature_re,
                            pressure_re.as_mut().unwrap(),
                            &molefracs_spec_re,
                            molefracs_init,
                            bubble,
                            iterate_p,
                            options,
                        )
                    })
                })?
            }
        } else {
            // pressure is specified
            let pressure_re = pressure_re.as_mut().unwrap();

            let temperature_re = temperature_re.as_mut().expect("An initial temperature is required for the calculation of bubble/dew points at given pressure!");
            PhaseEquilibrium::iterate_bubble_dew(
                &eos.re(),
                temperature_re,
                pressure_re,
                &molefracs_spec_re,
                molefracs_init,
                bubble,
                iterate_p,
                options,
            )?
        };

        // implicit differentiation
        let mut t = D::from(temperature_re.unwrap().into_reduced());
        let mut p = D::from(pressure_re.unwrap().into_reduced());
        let mut molar_volume = D::from(v1);
        let mut rho2 = rho2.map(D::from);
        for _ in 0..D::NDERIV {
            if iterate_p {
                Self::newton_step_t(
                    eos,
                    t,
                    molefracs_spec,
                    &mut p,
                    &mut molar_volume,
                    &mut rho2,
                    Verbosity::None,
                )
            } else {
                Self::newton_step_p(
                    eos,
                    &mut t,
                    molefracs_spec,
                    p,
                    &mut molar_volume,
                    &mut rho2,
                    Verbosity::None,
                )
            };
        }
        let state1 = State::new_intensive(
            eos,
            Temperature::from_reduced(t),
            Density::from_reduced(molar_volume.recip()),
            molefracs_spec,
        )?;
        let rho2_total = rho2.sum();
        let x2 = rho2 / rho2_total;
        let state2 = State::new_intensive(
            eos,
            Temperature::from_reduced(t),
            Density::from_reduced(rho2_total),
            &x2,
        )?;

        Ok(PhaseEquilibrium(if bubble {
            [state2, state1]
        } else {
            [state1, state2]
        }))
    }

    fn newton_step_t(
        eos: &E,
        temperature: D,
        molefracs: &OVector<D, N>,
        pressure: &mut D,
        molar_volume: &mut D,
        partial_density_other_phase: &mut OVector<D, N>,
        verbosity: Verbosity,
    ) -> f64 {
        // calculate properties
        let (p_1, mu_res_1, dp_1, dmu_1) = eos.dmu_drho(temperature, partial_density_other_phase);
        eos.dmu_dv(
            temperature,
            partial_density_other_phase.sum().recip(),
            &(partial_density_other_phase.clone() / partial_density_other_phase.sum()),
        );
        let (p_2, mu_res_2, dp_2, dmu_2) = eos.dmu_dv(temperature, *molar_volume, molefracs);

        // calculate residual
        let n = molefracs.len();
        let f = DVector::from_fn(n + 2, |i, _| {
            if i == n {
                p_1 - *pressure
            } else if i == n + 1 {
                p_2 - *pressure
            } else {
                mu_res_1[i] - mu_res_2[i]
                    + (partial_density_other_phase[i] * *molar_volume / molefracs[i]).ln()
                        * temperature
            }
        });

        // calculate Jacobian
        let jac = DMatrix::from_fn(n + 2, n + 2, |i, j| {
            if i < n && j < n {
                dmu_1[(i, j)]
            } else if i < n && j == n {
                -dmu_2[i]
            } else if i == n && j < n {
                dp_1[j]
            } else if i == n + 1 && j == n {
                dp_2
            } else if i < n && j == n + 1 {
                //d dmu/dT
                -D::one()
            } else if i == n && j == n + 1 {
                //dp1/dT
                -D::one()
            } else if i == n + 1 && j == n + 1 {
                //dp2/dT
                -D::one()
            } else {
                D::zero()
            }
        });

        // calculate Newton step
        let dx = LU::<_, _, Dyn>::new(jac).unwrap().solve(&f);

        // apply Newton step
        for i in 0..n {
            partial_density_other_phase[i] -= dx[i];
        }
        *molar_volume -= dx[n];
        *pressure -= dx[n + 1];

        let error = f.map(|r| r.re()).norm();

        let x = partial_density_other_phase.map(|r| r.re());
        let x = &x / x.sum();
        log_iteration(
            verbosity,
            Some(error),
            Temperature::from_reduced(temperature.re()),
            Pressure::from_reduced(pressure.re()),
            x.as_slice(),
            true,
        );
        error
    }

    fn newton_step_p(
        eos: &E,
        temperature: &mut D,
        molefracs: &OVector<D, N>,
        pressure: D,
        molar_volume: &mut D,
        partial_density_other_phase: &mut OVector<D, N>,
        verbosity: Verbosity,
    ) -> f64 {
        // calculate properties
        let (p_1, mu_res_1, dp_1, dmu_1) = eos.dmu_drho(*temperature, partial_density_other_phase);
        let (p_2, mu_res_2, dp_2, dmu_2) = eos.dmu_dv(*temperature, *molar_volume, molefracs);

        // calculate residual
        let n = molefracs.len();
        let f = DVector::from_fn(n + 2, |i, _| {
            if i == n {
                p_1 - pressure
            } else if i == n + 1 {
                p_2 - pressure
            } else {
                mu_res_1[i] - mu_res_2[i]
                    + (partial_density_other_phase[i] * *molar_volume / molefracs[i]).ln()
                        * *temperature
            }
        });

        // calculate Jacobian
        let jac = DMatrix::from_fn(n + 2, n + 2, |i, j| {
            if i < n && j < n {
                dmu_1[(i, j)]
            } else if i < n && j == n {
                -dmu_2[i]
            } else if i == n && j < n {
                dp_1[j]
            } else if i == n + 1 && j == n {
                dp_2
            } else if i >= n && j == n + 1 {
                -D::one()
            } else {
                D::zero()
            }
        });

        // calculate Newton step
        let dx = LU::<_, _, Dyn>::new(jac).unwrap().solve(&f);

        // apply Newton step
        for i in 0..n {
            partial_density_other_phase[i] -= dx[i];
        }
        *molar_volume -= dx[n];
        *temperature -= dx[n + 1];

        let error = f.map(|r| r.re()).norm();

        let x = partial_density_other_phase.map(|r| r.re());
        let x = &x / x.sum();
        log_iteration(
            verbosity,
            Some(error),
            Temperature::from_reduced(temperature.re()),
            Pressure::from_reduced(pressure.re()),
            x.as_slice(),
            true,
        );
        error
    }
}

/// # Bubble and dew point calculations
impl<E: Residual<N>, N: Gradients> PhaseEquilibrium<E, 2, N>
where
    DefaultAllocator: Allocator<N> + Allocator<N, N> + Allocator<U1, N>,
{
    #[expect(clippy::too_many_arguments)]
    fn iterate_bubble_dew(
        eos: &E,
        temperature: &mut Temperature,
        pressure: &mut Pressure,
        molefracs_spec: &OVector<f64, N>,
        molefracs_init: Option<&OVector<f64, N>>,
        bubble: bool,
        iterate_p: bool,
        options: (SolverOptions, SolverOptions),
    ) -> FeosResult<(f64, OVector<f64, N>)> {
        let [mut state1, mut state2] = if bubble {
            Self::starting_x2_bubble(eos, *temperature, *pressure, molefracs_spec, molefracs_init)
        } else {
            Self::starting_x2_dew(eos, *temperature, *pressure, molefracs_spec, molefracs_init)
        }?;
        let (options_inner, options_outer) = options;

        // initialize variables
        let mut err_out = 1.0;
        let mut k_out = 0;

        if PhaseEquilibrium::is_trivial_solution(&state1, &state2) {
            log_iter!(options_outer.verbosity, "Trivial solution encountered!");
            return Err(FeosError::TrivialSolution);
        }

        log_iter!(
            options_outer.verbosity,
            "res outer loop | res inner loop |   temperature  |     pressure     | molefracs second phase",
        );
        log_iter!(options_outer.verbosity, "{:-<104}", "");
        log_iteration(
            options_outer.verbosity,
            None,
            *temperature,
            *pressure,
            state2.molefracs.as_slice(),
            false,
        );

        // Outer loop for finding x2
        for ko in 0..options_outer.max_iter.unwrap_or(MAX_ITER_OUTER) {
            // Iso-Fugacity equation
            err_out = if err_out > NEWTON_TOL {
                // Inner loop for finding T or p
                for _ in 0..options_inner.max_iter.unwrap_or(MAX_ITER_INNER) {
                    let res = if iterate_p {
                        Self::adjust_p(
                            *temperature,
                            pressure,
                            &mut state1,
                            &mut state2,
                            options_inner.verbosity,
                        )?
                    } else {
                        Self::adjust_t(
                            temperature,
                            *pressure,
                            &mut state1,
                            &mut state2,
                            options_inner.verbosity,
                        )?
                    };
                    if res < options_inner.tol.unwrap_or(TOL_INNER) {
                        break;
                    }
                }
                Self::adjust_x2(&state1, &mut state2, options_outer.verbosity)
            } else {
                let mut t = temperature.into_reduced();
                let mut p = pressure.into_reduced();
                let mut molar_volume = state1.density.into_reduced().recip();
                let mut rho2 = state2.partial_density.to_reduced();
                let err = if iterate_p {
                    Self::newton_step_t(
                        &state1.eos,
                        t,
                        &state1.molefracs,
                        &mut p,
                        &mut molar_volume,
                        &mut rho2,
                        options_outer.verbosity,
                    )
                } else {
                    Self::newton_step_p(
                        &state1.eos,
                        &mut t,
                        &state1.molefracs,
                        p,
                        &mut molar_volume,
                        &mut rho2,
                        options_outer.verbosity,
                    )
                };
                *temperature = Temperature::from_reduced(t);
                *pressure = Pressure::from_reduced(p);
                state1.density = Density::from_reduced(molar_volume.recip());
                state2.partial_density = Density::from_reduced(rho2);
                Ok(err)
            }?;

            if Self::is_trivial_solution(&state1, &state2) {
                log_iter!(options_outer.verbosity, "Trivial solution encountered!");
                return Err(FeosError::TrivialSolution);
            }

            if err_out < options_outer.tol.unwrap_or(TOL_OUTER) {
                k_out = ko + 1;
                break;
            }
        }

        if err_out < options_outer.tol.unwrap_or(TOL_OUTER) {
            log_result!(
                options_outer.verbosity,
                "Bubble/dew point: calculation converged in {} step(s)\n",
                k_out
            );
            Ok((
                state1.density.into_reduced().recip(),
                state2.partial_density.to_reduced(),
            ))
        } else {
            // not converged, return error
            Err(FeosError::NotConverged(String::from(
                "bubble-dew-iteration",
            )))
        }
    }

    fn adjust_p(
        temperature: Temperature,
        pressure: &mut Pressure,
        state1: &mut State<E, N>,
        state2: &mut State<E, N>,
        verbosity: Verbosity,
    ) -> FeosResult<f64> {
        // calculate K = phi_1/phi_2 = x_2/x_1
        let ln_phi_1 = state1.ln_phi();
        let ln_phi_2 = state2.ln_phi();
        let k = (&ln_phi_1 - &ln_phi_2).map(f64::exp);

        // calculate residual
        let xk = state1.molefracs.component_mul(&k);
        let f = xk.sum() - 1.0;

        // Derivative w.r.t. ln(pressure)
        let ln_phi_1_dp = state1.dln_phi_dp();
        let ln_phi_2_dp = state2.dln_phi_dp();
        let df = ((ln_phi_1_dp - ln_phi_2_dp) * *pressure)
            .into_value()
            .component_mul(&xk)
            .sum();
        let mut lnpstep = -f / df;

        // catch too big p-steps
        lnpstep = lnpstep.clamp(-MAX_LNPSTEP, MAX_LNPSTEP);

        // Update p
        *pressure *= lnpstep.exp();

        // update states with new temperature/pressure
        Self::adjust_states(temperature, *pressure, state1, state2, None)?;

        // log
        log_iteration(
            verbosity,
            Some(f),
            temperature,
            *pressure,
            state2.molefracs.as_slice(),
            false,
        );

        Ok(f.abs())
    }

    fn adjust_t(
        temperature: &mut Temperature,
        pressure: Pressure,
        state1: &mut State<E, N>,
        state2: &mut State<E, N>,
        verbosity: Verbosity,
    ) -> FeosResult<f64> {
        // calculate K = phi_1/phi_2 = x_2/x_1
        let ln_phi_1 = state1.ln_phi();
        let ln_phi_2 = state2.ln_phi();
        let k = (&ln_phi_1 - &ln_phi_2).map(f64::exp);

        // calculate residual
        let f = state1.molefracs.dot(&k) - 1.0;

        // Derivative w.r.t. temperature
        let ln_phi_1_dt = state1.dln_phi_dt();
        let ln_phi_2_dt = state2.dln_phi_dt();
        let df = ((ln_phi_1_dt - ln_phi_2_dt)
            .component_mul(&Dimensionless::new(state1.molefracs.component_mul(&k))))
        .sum();
        let mut tstep = -f / df;

        // catch too big t-steps
        if tstep < -Temperature::from_reduced(MAX_TSTEP) {
            tstep = -Temperature::from_reduced(MAX_TSTEP);
        } else if tstep > Temperature::from_reduced(MAX_TSTEP) {
            tstep = Temperature::from_reduced(MAX_TSTEP);
        }

        // Update t
        *temperature += tstep;

        // update states with new temperature
        Self::adjust_states(*temperature, pressure, state1, state2, None)?;

        // log
        log_iteration(
            verbosity,
            Some(f),
            *temperature,
            pressure,
            state2.molefracs.as_slice(),
            false,
        );

        Ok(f.abs())
    }

    fn starting_pressure_ideal_gas(
        eos: &E,
        temperature: Temperature,
        molefracs_spec: &OVector<f64, N>,
        bubble: bool,
    ) -> FeosResult<(Pressure, OVector<f64, N>)> {
        if bubble {
            Self::starting_pressure_ideal_gas_bubble(eos, temperature, molefracs_spec)
        } else {
            Self::starting_pressure_ideal_gas_dew(eos, temperature, molefracs_spec)
        }
    }

    pub(super) fn starting_pressure_ideal_gas_bubble(
        eos: &E,
        temperature: Temperature,
        liquid_molefracs: &OVector<f64, N>,
    ) -> FeosResult<(Pressure, OVector<f64, N>)> {
        let density = 0.75 * Density::from_reduced(eos.compute_max_density(liquid_molefracs));
        let liquid = State::new_intensive(eos, temperature, density, liquid_molefracs)?;
        let v_l = liquid.partial_molar_volume();
        let p_l = liquid.pressure(Contributions::Total);
        let mu_l = liquid.residual_chemical_potential();
        let k_i = (liquid_molefracs.clone()).component_mul(
            &((mu_l - v_l * p_l) / (RGAS * temperature))
                .into_value()
                .map(f64::exp),
        );
        let p = k_i.sum() * RGAS * temperature * density;
        let y = &k_i / k_i.sum();
        Ok((p, y))
    }

    fn starting_pressure_ideal_gas_dew(
        eos: &E,
        temperature: Temperature,
        vapor_molefracs: &OVector<f64, N>,
    ) -> FeosResult<(Pressure, OVector<f64, N>)> {
        let mut p: Option<Pressure> = None;

        let mut x = vapor_molefracs.clone();
        for _ in 0..5 {
            let density = Density::from_reduced(0.75 * eos.compute_max_density(&x));
            let liquid = State::new_intensive(eos, temperature, density, &x)?;
            let v_l = liquid.partial_molar_volume();
            let p_l = liquid.pressure(Contributions::Total);
            let mu_l = liquid.residual_chemical_potential();
            let k = vapor_molefracs.clone().component_div(
                &((mu_l - v_l * p_l) / (RGAS * temperature))
                    .into_value()
                    .map(f64::exp),
            );
            let k_sum = k.sum();
            let p_new = RGAS * temperature * density / k_sum;
            x = k / k_sum;
            if let Some(p_old) = p
                && ((p_new - p_old) / p_old).into_value().abs() < 1e-5
            {
                p = Some(p_new);
                break;
            }
            p = Some(p_new);
        }
        Ok((p.unwrap(), x))
    }

    pub(super) fn starting_pressure_spinodal(
        eos: &E,
        temperature: Temperature,
        molefracs: &OVector<f64, N>,
    ) -> FeosResult<Pressure> {
        let [sp_v, sp_l] = State::spinodal(eos, temperature, Some(molefracs), Default::default())?;
        let pv = sp_v.pressure(Contributions::Total);
        let pl = sp_l.pressure(Contributions::Total);
        Ok(0.5 * (Pressure::from_reduced(0.0).max(pl) + pv))
    }

    fn starting_x2_bubble(
        eos: &E,
        temperature: Temperature,
        pressure: Pressure,
        liquid_molefracs: &OVector<f64, N>,
        vapor_molefracs: Option<&OVector<f64, N>>,
    ) -> FeosResult<[State<E, N>; 2]> {
        let liquid_state =
            State::new_xpt(eos, temperature, pressure, liquid_molefracs, Some(Liquid))?;
        let xv = match vapor_molefracs {
            Some(xv) => xv.clone(),
            None => liquid_state
                .ln_phi()
                .map(f64::exp)
                .component_mul(liquid_molefracs),
        };
        let vapor_state = State::new_xpt(eos, temperature, pressure, &xv, Some(Vapor))?;
        Ok([liquid_state, vapor_state])
    }

    fn starting_x2_dew(
        eos: &E,
        temperature: Temperature,
        pressure: Pressure,
        vapor_molefracs: &OVector<f64, N>,
        liquid_molefracs: Option<&OVector<f64, N>>,
    ) -> FeosResult<[State<E, N>; 2]> {
        let vapor_state = State::new_npt(
            eos,
            temperature,
            pressure,
            &Moles::from_reduced(vapor_molefracs.clone()),
            Some(Vapor),
        )?;
        let xl = match liquid_molefracs {
            Some(xl) => xl.clone(),
            None => {
                let xl = vapor_state
                    .ln_phi()
                    .map(f64::exp)
                    .component_mul(vapor_molefracs);
                let liquid_state = State::new_xpt(eos, temperature, pressure, &xl, Some(Liquid))?;
                (vapor_state.ln_phi() - liquid_state.ln_phi())
                    .map(f64::exp)
                    .component_mul(vapor_molefracs)
            }
        };
        let liquid_state = State::new_xpt(eos, temperature, pressure, &xl, Some(Liquid))?;
        Ok([vapor_state, liquid_state])
    }

    fn adjust_states(
        temperature: Temperature,
        pressure: Pressure,
        state1: &mut State<E, N>,
        state2: &mut State<E, N>,
        moles_state2: Option<&Moles<OVector<f64, N>>>,
    ) -> FeosResult<()> {
        *state1 = State::new_npt(
            &state1.eos,
            temperature,
            pressure,
            &state1.moles,
            Some(InitialDensity(state1.density)),
        )?;
        *state2 = State::new_npt(
            &state2.eos,
            temperature,
            pressure,
            moles_state2.unwrap_or(&state2.moles),
            Some(InitialDensity(state2.density)),
        )?;
        Ok(())
    }

    fn adjust_x2(
        state1: &State<E, N>,
        state2: &mut State<E, N>,
        verbosity: Verbosity,
    ) -> FeosResult<f64> {
        let x1 = &state1.molefracs;
        let ln_phi_1 = state1.ln_phi();
        let ln_phi_2 = state2.ln_phi();
        let k = (ln_phi_1 - ln_phi_2).map(f64::exp);
        let kx1 = k.component_mul(x1);
        let err_out = kx1
            .component_div(&state2.molefracs)
            .map(|e| (e - 1.0).abs())
            .sum();
        let x2 = &kx1 / kx1.sum();
        log_iter!(
            verbosity,
            "{:<14.8e} | {:14} | {:14} | {:16} |",
            err_out,
            "",
            "",
            ""
        );
        *state2 = State::new_xpt(
            &state2.eos,
            state2.temperature,
            state2.pressure(Contributions::Total),
            &x2,
            Some(InitialDensity(state2.density)),
        )?;
        Ok(err_out)
    }
}

fn log_iteration(
    verbosity: Verbosity,
    error: Option<f64>,
    temperature: Temperature,
    pressure: Pressure,
    x2: &[f64],
    newton: bool,
) {
    let error = error.map_or_else(|| format!("{:14}", ""), |e| format!("{:<14.8e}", e.abs()));
    log_iter!(
        verbosity,
        "{:14} | {} | {:12.8} | {:12.8} | {:.8?} {}",
        "",
        error,
        temperature,
        pressure,
        x2,
        if newton { "NEWTON" } else { "" }
    );
}
