use super::PhaseEquilibrium;
use crate::equation_of_state::Residual;
use crate::errors::{FeosError, FeosResult};
use crate::phase_equilibria::PhaseEquilibriumGeneric;
use crate::state::{Contributions, DensityInitialization, State};
use crate::{
    HelmholtzEnergyDerivatives, ReferenceSystem, SolverOptions, StateGeneric,
    TemperatureOrPressure, Verbosity,
};
use nalgebra::allocator::Allocator;
use nalgebra::{DVector, DefaultAllocator, dvector};
use quantity::{Pressure, RGAS, Temperature};
use std::sync::Arc;

const SCALE_T_NEW: f64 = 0.7;
const MAX_ITER_PURE: usize = 50;
const TOL_PURE: f64 = 1e-12;

/// # Pure component phase equilibria
impl<E: Residual> PhaseEquilibrium<E, 2> {
    /// Calculate a phase equilibrium for a pure component.
    pub fn pure<TP: TemperatureOrPressure>(
        eos: &Arc<E>,
        temperature_or_pressure: TP,
        initial_state: Option<&Self>,
        options: SolverOptions,
    ) -> FeosResult<Self> {
        if let Some(t) = temperature_or_pressure.temperature() {
            Self::pure_t(eos, t, initial_state, options)
        } else if let Some(p) = temperature_or_pressure.pressure() {
            Self::pure_p(eos, p, initial_state, options)
        } else {
            unreachable!()
        }
    }
}

impl<E: HelmholtzEnergyDerivatives<f64>> PhaseEquilibriumGeneric<E, f64, E::Components, E::Cache, 2>
where
    DefaultAllocator: Allocator<E::Components> + Allocator<E::Components, E::Components>,
{
    fn new_pt(eos: &E, temperature: Temperature, pressure: Pressure) -> FeosResult<Self> {
        let liquid = StateGeneric::new_xpt(
            eos,
            temperature,
            pressure,
            &E::pure_molefracs(),
            DensityInitialization::Liquid,
        )?;
        let vapor = StateGeneric::new_xpt(
            eos,
            temperature,
            pressure,
            &E::pure_molefracs(),
            DensityInitialization::Vapor,
        )?;
        Ok(Self([vapor, liquid]))
    }

    /// Calculate a phase equilibrium for a pure component
    /// and given temperature.
    pub fn pure_t(
        eos: &E,
        temperature: Temperature,
        initial_state: Option<&Self>,
        options: SolverOptions,
    ) -> FeosResult<Self> {
        println!("{temperature}");
        let (max_iter, tol, verbosity) = options.unwrap_or(MAX_ITER_PURE, TOL_PURE);

        // First use given initial state if applicable
        let mut vle = initial_state.and_then(|init| {
            Self::init_pure_state(init, temperature)
                .iterate_pure_t(max_iter, tol, verbosity)
                .ok()
        });

        // Next try to initialize with an ideal gas assumption
        vle = vle.or_else(|| {
            Self::init_pure_ideal_gas(eos, temperature)
                .and_then(|vle| vle.iterate_pure_t(max_iter, tol, verbosity))
                .ok()
        });

        // Finally use the spinodal to initialize the calculation
        vle.map_or_else(
            || {
                Self::init_pure_spinodal(eos, temperature)
                    .and_then(|vle| vle.iterate_pure_t(max_iter, tol, verbosity))
            },
            Ok,
        )
    }

    fn iterate_pure_t(self, max_iter: usize, tol: f64, verbosity: Verbosity) -> FeosResult<Self> {
        let mut p_old = self.vapor().pressure(Contributions::Total);
        let [mut vapor, mut liquid] = self.0;

        log_iter!(
            verbosity,
            " iter |     residual      |     pressure     |    liquid density    |    vapor density     | Newton steps"
        );
        log_iter!(verbosity, "{:-<106}", "");
        log_iter!(
            verbosity,
            " {:4} |                   | {:12.8} | {:12.8} | {:12.8} |",
            0,
            p_old,
            liquid.density,
            vapor.density
        );

        for i in 1..=max_iter {
            // calculate the pressures and derivatives
            let (p_l, p_rho_l) = liquid.p_dpdrho();
            let (p_v, p_rho_v) = vapor.p_dpdrho();
            println!("{p_l} {p_rho_l}");
            println!("{p_v} {p_rho_v}");
            // calculate the molar Helmholtz energies (already cached)
            let a_l_res = liquid.residual_molar_helmholtz_energy();
            let a_v_res = vapor.residual_molar_helmholtz_energy();
            println!("{a_l_res} {}", vapor.density);
            println!("{a_v_res} {}", liquid.density);

            // Estimate the new pressure
            let kt = RGAS * vapor.temperature;
            let delta_v = 1.0 / vapor.density - 1.0 / liquid.density;
            let delta_a =
                a_v_res - a_l_res + kt * (vapor.density / liquid.density).into_value().ln();
            let mut p_new = -delta_a / delta_v;
            println!("{p_new}");

            // If the pressure becomes negative, assume the gas phase is ideal. The
            // resulting pressure is always positive.
            if p_new.is_sign_negative() {
                p_new = p_v
                    * ((-delta_a - p_v * vapor.volume / vapor.total_moles) / kt)
                        .into_value()
                        .exp();
            }

            // Improve the estimate by exploiting the almost ideal behavior of the gas phase
            let kt = RGAS * vapor.temperature;
            let mut newton_iter = 0;
            let newton_tol = p_old * delta_v * tol;
            for _ in 0..20 {
                let p_frac = (p_new / p_old).into_value();
                let f = p_new * delta_v + delta_a + (p_frac.ln() + 1.0 - p_frac) * kt;
                let df_dp = delta_v + (1.0 / p_new - 1.0 / p_old) * kt;
                p_new -= f / df_dp;
                println!("{p_new}");
                newton_iter += 1;
                if f.abs() < newton_tol {
                    break;
                }
            }

            // Emergency brake if the implementation of the EOS is not safe.
            if p_new.is_nan() {
                return Err(FeosError::IterationFailed("pure_t".to_owned()));
            }

            // Calculate Newton steps for the densities and update state.
            let rho_l = liquid.density + (p_new - p_l) / p_rho_l;
            let rho_v = vapor.density + (p_new - p_v) / p_rho_v;
            liquid = StateGeneric::new_pure(&liquid.eos, liquid.temperature, rho_l);
            vapor = StateGeneric::new_pure(&vapor.eos, vapor.temperature, rho_v);
            if Self::is_trivial_solution(&vapor, &liquid) {
                return Err(FeosError::TrivialSolution);
            }

            // Check for convergence
            let res = (p_new - p_old).abs();
            log_iter!(
                verbosity,
                " {:4} | {:14.8e} | {:12.8} | {:12.8} | {:12.8} | {}",
                i,
                res,
                p_new,
                liquid.density,
                vapor.density,
                newton_iter
            );
            if res < p_old * tol {
                log_result!(
                    verbosity,
                    "PhaseEquilibrium::pure_t: calculation converged in {} step(s)\n",
                    i
                );
                return Ok(Self([vapor, liquid]));
            }
            p_old = p_new;
        }
        Err(FeosError::NotConverged("pure_t".to_owned()))
    }

    fn init_pure_state(initial_state: &Self, temperature: Temperature) -> Self {
        let vapor = initial_state.vapor().update_temperature(temperature);
        let liquid = initial_state.liquid().update_temperature(temperature);
        Self([vapor, liquid])
    }

    fn init_pure_ideal_gas(eos: &E, temperature: Temperature) -> FeosResult<Self> {
        let x = E::pure_molefracs();
        let v = eos.compute_max_density(&x).recip();
        let t = temperature.into_reduced();
        let a_res = eos._residual_molar_helmholtz_energy(t, v, &x);
        let p = Pressure::from_reduced(t / v * (a_res / t).exp());
        println!("{p}");
        Self::new_pt(eos, temperature, p)?.check_trivial_solution()
    }

    fn init_pure_spinodal(eos: &E, temperature: Temperature) -> FeosResult<Self> {
        let p = Self::starting_pressure_spinodal(eos, temperature, &E::pure_molefracs())?;
        PhaseEquilibriumGeneric::new_pt(eos, temperature, p)
    }
}

impl<E: Residual> PhaseEquilibrium<E, 2> {
    /// Calculate a phase equilibrium for a pure component
    /// and given pressure.
    fn pure_p(
        eos: &Arc<E>,
        pressure: Pressure,
        initial_state: Option<&Self>,
        options: SolverOptions,
    ) -> FeosResult<Self> {
        let (max_iter, tol, verbosity) = options.unwrap_or(MAX_ITER_PURE, TOL_PURE);

        // Initialize the phase equilibrium
        let mut vle = match initial_state {
            Some(init) => init
                .clone()
                .update_pressure(init.vapor().temperature, pressure)?,
            None => PhaseEquilibrium::init_pure_p(eos, pressure)?,
        };

        log_iter!(
            verbosity,
            " iter |     residual     |   temperature   |    liquid density    |    vapor density     "
        );
        log_iter!(verbosity, "{:-<89}", "");
        log_iter!(
            verbosity,
            " {:4} |                  | {:13.8} | {:12.8} | {:12.8}",
            0,
            vle.vapor().temperature,
            vle.liquid().density,
            vle.vapor().density
        );
        for i in 1..=max_iter {
            // calculate the pressures and derivatives
            let (p_l, p_rho_l) = vle.liquid().p_dpdrho();
            let (p_v, p_rho_v) = vle.vapor().p_dpdrho();
            let p_t_l = vle.liquid().dp_dt(Contributions::Total);
            let p_t_v = vle.vapor().dp_dt(Contributions::Total);

            // calculate the residual molar entropies (already cached)
            let s_l_res = vle.liquid().residual_molar_entropy();
            let s_v_res = vle.vapor().residual_molar_entropy();

            // calculate the residual molar Helmholtz energies (already cached)
            let a_l_res = vle.liquid().residual_molar_helmholtz_energy();
            let a_v_res = vle.vapor().residual_molar_helmholtz_energy();

            // calculate the molar volumes
            let v_l = 1.0 / vle.liquid().density;
            let v_v = 1.0 / vle.vapor().density;

            // estimate the temperature steps
            let kt = RGAS * vle.vapor().temperature;
            let ln_rho = (v_l / v_v).into_value().ln();
            let delta_t = (pressure * (v_v - v_l) + (a_v_res - a_l_res + kt * ln_rho))
                / (s_v_res - s_l_res - RGAS * ln_rho);
            let t_new = vle.vapor().temperature + delta_t;

            // calculate Newton steps for the densities and update state.
            let rho_l = vle.liquid().density + (pressure - p_l - p_t_l * delta_t) / p_rho_l;
            let rho_v = vle.vapor().density + (pressure - p_v - p_t_v * delta_t) / p_rho_v;

            if rho_l.is_sign_negative()
                || rho_v.is_sign_negative()
                || delta_t.abs() > Temperature::from_reduced(1.0)
            {
                // if densities are negative or the temperature step is large use density iteration instead
                vle = vle
                    .update_pressure(t_new, pressure)?
                    .check_trivial_solution()?;
            } else {
                // update state
                vle = Self([
                    StateGeneric::new_pure(eos, t_new, rho_v),
                    StateGeneric::new_pure(eos, t_new, rho_l),
                ]);
            }

            // check for convergence
            let res = delta_t.abs();
            log_iter!(
                verbosity,
                " {:4} | {:14.8e} | {:13.8} | {:12.8} | {:12.8}",
                i,
                res,
                vle.vapor().temperature,
                vle.liquid().density,
                vle.vapor().density
            );
            if res < vle.vapor().temperature * tol {
                log_result!(
                    verbosity,
                    "PhaseEquilibrium::pure_p: calculation converged in {} step(s)\n",
                    i
                );
                return Ok(vle);
            }
        }
        Err(FeosError::NotConverged("pure_p".to_owned()))
    }

    /// Initialize a new VLE for a pure substance for a given pressure.
    fn init_pure_p(eos: &Arc<E>, pressure: Pressure) -> FeosResult<Self> {
        let trial_temperatures = [
            Temperature::from_reduced(300.0),
            Temperature::from_reduced(500.0),
            Temperature::from_reduced(200.0),
        ];
        let x = dvector![1.0];
        let mut vle = None;
        let mut t0 = Temperature::from_reduced(1.0);
        for t in trial_temperatures.iter() {
            t0 = *t;
            let _vle = PhaseEquilibrium::new_pt(eos, *t, pressure)?;
            if !Self::is_trivial_solution(_vle.vapor(), _vle.liquid()) {
                return Ok(_vle);
            }
            vle = Some(_vle);
        }

        let cp = State::critical_point(eos, None, None, SolverOptions::default())?;
        if pressure > cp.pressure(Contributions::Total) {
            return Err(FeosError::SuperCritical);
        };
        if let Some(mut e) = vle {
            if e.vapor().density < cp.density {
                for _ in 0..8 {
                    t0 *= SCALE_T_NEW;
                    e.0[1] = State::new_xpt(eos, t0, pressure, &x, DensityInitialization::Liquid)?;
                    if e.liquid().density > cp.density {
                        break;
                    }
                }
            } else {
                for _ in 0..8 {
                    t0 /= SCALE_T_NEW;
                    e.0[0] = State::new_xpt(eos, t0, pressure, &x, DensityInitialization::Vapor)?;
                    if e.vapor().density < cp.density {
                        break;
                    }
                }
            }

            for _ in 0..20 {
                let h = |s: &State<_>| s.residual_enthalpy() + s.total_moles * RGAS * s.temperature;
                t0 = (h(e.vapor()) - h(e.liquid()))
                    / (e.vapor().residual_entropy()
                        - e.liquid().residual_entropy()
                        - RGAS
                            * e.vapor().total_moles
                            * ((e.vapor().density / e.liquid().density).into_value().ln()));
                let trial_state =
                    State::new_xpt(eos, t0, pressure, &x, DensityInitialization::Vapor)?;
                if trial_state.density < cp.density {
                    e.0[0] = trial_state;
                }
                let trial_state =
                    State::new_xpt(eos, t0, pressure, &x, DensityInitialization::Liquid)?;
                if trial_state.density > cp.density {
                    e.0[1] = trial_state;
                }
                if e.liquid().temperature == e.vapor().temperature {
                    return Ok(e);
                }
            }
            Err(FeosError::IterationFailed(
                "new_init_p: could not find proper initial state".to_owned(),
            ))
        } else {
            unreachable!()
        }
    }
}

impl<E: Residual> PhaseEquilibrium<E, 2> {
    /// Calculate the pure component vapor pressures of all
    /// components in the system for the given temperature.
    pub fn vapor_pressure(eos: &Arc<E>, temperature: Temperature) -> Vec<Option<Pressure>> {
        (0..eos.components())
            .map(|i| {
                let pure_eos = Arc::new(eos.subset(&[i]));
                PhaseEquilibrium::pure_t(&pure_eos, temperature, None, SolverOptions::default())
                    .map(|vle| vle.vapor().pressure(Contributions::Total))
                    .ok()
            })
            .collect()
    }

    /// Calculate the pure component boiling temperatures of all
    /// components in the system for the given pressure.
    pub fn boiling_temperature(eos: &Arc<E>, pressure: Pressure) -> Vec<Option<Temperature>> {
        (0..eos.components())
            .map(|i| {
                let pure_eos = Arc::new(eos.subset(&[i]));
                PhaseEquilibrium::pure_p(&pure_eos, pressure, None, SolverOptions::default())
                    .map(|vle| vle.vapor().temperature)
                    .ok()
            })
            .collect()
    }

    /// Calculate the pure component phase equilibria of all
    /// components in the system.
    pub fn vle_pure_comps<TP: TemperatureOrPressure>(
        eos: &Arc<E>,
        temperature_or_pressure: TP,
    ) -> Vec<Option<PhaseEquilibrium<E, 2>>> {
        (0..eos.components())
            .map(|i| {
                let pure_eos = Arc::new(eos.subset(&[i]));
                PhaseEquilibrium::pure(
                    &pure_eos,
                    temperature_or_pressure,
                    None,
                    SolverOptions::default(),
                )
                .ok()
                .map(|vle_pure| {
                    let mut molefracs_vapor = DVector::zeros(eos.components());
                    let mut molefracs_liquid = molefracs_vapor.clone();
                    molefracs_vapor[i] = 1.0;
                    molefracs_liquid[i] = 1.0;
                    let vapor = State::new_intensive(
                        eos,
                        vle_pure.vapor().temperature,
                        vle_pure.vapor().density,
                        &molefracs_vapor,
                    );
                    let liquid = State::new_intensive(
                        eos,
                        vle_pure.liquid().temperature,
                        vle_pure.liquid().density,
                        &molefracs_liquid,
                    );
                    PhaseEquilibrium::from_states(vapor, liquid)
                })
            })
            .collect()
    }
}
