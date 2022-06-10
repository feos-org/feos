use super::{PhaseEquilibrium, SolverOptions, Verbosity};
use crate::density_iteration::pressure_spinodal;
use crate::equation_of_state::EquationOfState;
use crate::errors::{EosError, EosResult};
use crate::state::{Contributions, DensityInitialization, State, TPSpec};
use crate::EosUnit;
use ndarray::{arr1, Array1};
use quantity::{QuantityArray1, QuantityScalar};
use std::convert::TryFrom;
use std::rc::Rc;

const SCALE_T_NEW: f64 = 0.7;

const MAX_ITER_PURE: usize = 50;
const TOL_PURE: f64 = 1e-12;

/// # Pure component phase equilibria
impl<U: EosUnit, E: EquationOfState> PhaseEquilibrium<U, E, 2> {
    /// Calculate a phase equilibrium for a pure component.
    pub fn pure(
        eos: &Rc<E>,
        temperature_or_pressure: QuantityScalar<U>,
        initial_state: Option<&PhaseEquilibrium<U, E, 2>>,
        options: SolverOptions,
    ) -> EosResult<Self>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        match TPSpec::try_from(temperature_or_pressure)? {
            TPSpec::Temperature(t) => Self::pure_t(eos, t, initial_state, options),
            TPSpec::Pressure(p) => Self::pure_p(eos, p, initial_state, options),
        }
    }

    /// Calculate a phase equilibrium for a pure component
    /// and given temperature.
    fn pure_t(
        eos: &Rc<E>,
        temperature: QuantityScalar<U>,
        initial_state: Option<&PhaseEquilibrium<U, E, 2>>,
        options: SolverOptions,
    ) -> EosResult<Self>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let (max_iter, tol, verbosity) = options.unwrap_or(MAX_ITER_PURE, TOL_PURE);

        // First use given initial state if applicable
        let mut vle = initial_state.and_then(|init| {
            Self::init_pure_state(init, temperature)
                .and_then(|vle| vle.iterate_pure_t(max_iter, tol, verbosity))
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

    fn iterate_pure_t(self, max_iter: usize, tol: f64, verbosity: Verbosity) -> EosResult<Self>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let mut p_old = self.vapor().pressure(Contributions::Total);
        let [mut vapor, mut liquid] = self.0;

        log_iter!(verbosity,
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
            // calculate the molar Helmholtz energies (already cached)
            let a_l = liquid.molar_helmholtz_energy(Contributions::Total);
            let a_v = vapor.molar_helmholtz_energy(Contributions::Total);

            // Estimate the new pressure
            let delta_v = 1.0 / vapor.density - 1.0 / liquid.density;
            let delta_a = a_v - a_l;
            let mut p_new = -delta_a / delta_v;

            // If the pressure becomes negative, assume the gas phase is ideal. The
            // resulting pressure is always positive.
            if p_new.is_sign_negative() {
                let mu_v = vapor.chemical_potential(Contributions::Total).get(0);
                p_new = p_v
                    * (a_l - mu_v)
                        .to_reduced(vapor.temperature * U::gas_constant())?
                        .exp();
            }

            // Improve the estimate by exploiting the almost ideal behavior of the gas phase
            let kt = U::gas_constant() * vapor.temperature;
            let mut newton_iter = 0;
            let newton_tol = p_old * delta_v * tol;
            for _ in 0..20 {
                let p_frac = p_new.to_reduced(p_old)?;
                let f = p_new * delta_v + delta_a + (p_frac.ln() + 1.0 - p_frac) * kt;
                let df_dp = delta_v + (1.0 / p_new - 1.0 / p_old) * kt;
                p_new -= f / df_dp;
                newton_iter += 1;
                if f.abs() < newton_tol {
                    break;
                }
            }

            // Emergency brake if the implementation of the EOS is not safe.
            if p_new.is_nan() {
                return Err(EosError::IterationFailed("pure_t".to_owned()));
            }

            // Calculate Newton steps for the densities and update state.
            let rho_l = liquid.density + (p_new - p_l) / p_rho_l;
            let rho_v = vapor.density + (p_new - p_v) / p_rho_v;
            liquid = State::new_pure(&liquid.eos, liquid.temperature, rho_l)?;
            vapor = State::new_pure(&vapor.eos, vapor.temperature, rho_v)?;
            if Self::is_trivial_solution(&vapor, &liquid) {
                return Err(EosError::TrivialSolution);
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
        Err(EosError::NotConverged("pure_t".to_owned()))
    }

    /// Calculate a phase equilibrium for a pure component
    /// and given pressure.
    fn pure_p(
        eos: &Rc<E>,
        pressure: QuantityScalar<U>,
        initial_state: Option<&Self>,
        options: SolverOptions,
    ) -> EosResult<Self>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
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

            // calculate the molar entropies (already cached)
            let s_l = vle.liquid().molar_entropy(Contributions::Total);
            let s_v = vle.vapor().molar_entropy(Contributions::Total);

            // calculate the molar Helmholtz energies (already cached)
            let a_l = vle.liquid().molar_helmholtz_energy(Contributions::Total);
            let a_v = vle.vapor().molar_helmholtz_energy(Contributions::Total);

            // calculate the molar volumes
            let v_l = 1.0 / vle.liquid().density;
            let v_v = 1.0 / vle.vapor().density;

            // estimate the temperature steps
            let delta_t = (pressure * (v_v - v_l) + (a_v - a_l)) / (s_v - s_l);
            let t_new = vle.vapor().temperature + delta_t;

            // calculate Newton steps for the densities and update state.
            let rho_l = vle.liquid().density + (pressure - p_l - p_t_l * delta_t) / p_rho_l;
            let rho_v = vle.vapor().density + (pressure - p_v - p_t_v * delta_t) / p_rho_v;

            if rho_l.is_sign_negative()
                || rho_v.is_sign_negative()
                || delta_t.abs() > U::reference_temperature()
            {
                // if densities are negative or the temperature step is large use density iteration instead
                vle = vle
                    .update_pressure(t_new, pressure)?
                    .check_trivial_solution()?;
            } else {
                // update state
                vle = Self([
                    State::new_pure(eos, t_new, rho_v)?,
                    State::new_pure(eos, t_new, rho_l)?,
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
        Err(EosError::NotConverged("pure_p".to_owned()))
    }

    fn init_pure_state(initial_state: &Self, temperature: QuantityScalar<U>) -> EosResult<Self> {
        let vapor = initial_state.vapor().update_temperature(temperature)?;
        let liquid = initial_state.liquid().update_temperature(temperature)?;
        Ok(Self([vapor, liquid]))
    }

    fn init_pure_ideal_gas(eos: &Rc<E>, temperature: QuantityScalar<U>) -> EosResult<Self> {
        let m = arr1(&[1.0]) * U::reference_moles();
        let density = 0.75 * eos.max_density(None)?;
        let liquid = State::new_nvt(eos, temperature, U::reference_moles() / density, &m)?;
        let z = liquid.compressibility(Contributions::Total);
        let mu = liquid.chemical_potential(Contributions::ResidualNvt);
        let p = temperature
            * density
            * U::gas_constant()
            * (mu.get(0).to_reduced(U::gas_constant() * temperature)? - z).exp();
        PhaseEquilibrium::new_npt(eos, temperature, p, &m, &m)?.check_trivial_solution()
    }

    fn init_pure_spinodal(eos: &Rc<E>, temperature: QuantityScalar<U>) -> EosResult<Self> {
        let m = arr1(&[1.0]) * U::reference_moles();
        let spinodal = Self::spinodal(eos, temperature, &m)?;
        let pv = spinodal.vapor().pressure(Contributions::Total);
        let pl = spinodal.liquid().pressure(Contributions::Total);
        let p = 0.5 * ((0.0 * U::reference_pressure()).max(pl)? + pv);
        PhaseEquilibrium::new_npt(eos, temperature, p, &m, &m)
    }

    fn spinodal(
        eos: &Rc<E>,
        temperature: QuantityScalar<U>,
        moles: &QuantityArray1<U>,
    ) -> EosResult<Self> {
        let max_density = eos.max_density(Some(moles))?;
        let sp = pressure_spinodal(eos, temperature, max_density * 1e-5, moles)?;
        let vapor = State::new_nvt(eos, temperature, moles.get(0) / sp.rho, moles)?;
        let sp = pressure_spinodal(eos, temperature, max_density, moles)?;
        let liquid = State::new_nvt(eos, temperature, moles.get(0) / sp.rho, moles)?;
        Ok(PhaseEquilibrium([vapor, liquid]))
    }

    /// Initialize a new VLE for a pure substance for a given pressure.
    fn init_pure_p(eos: &Rc<E>, pressure: QuantityScalar<U>) -> EosResult<Self>
    where
        QuantityScalar<U>: std::fmt::Display,
    {
        let trial_temperatures = [
            300.0 * U::reference_temperature(),
            500.0 * U::reference_temperature(),
            200.0 * U::reference_temperature(),
        ];
        let m = arr1(&[1.0]) * U::reference_moles();
        let mut vle = None;
        let mut t0 = U::reference_temperature();
        for t in trial_temperatures.iter() {
            t0 = *t;
            let _vle = PhaseEquilibrium::new_npt(eos, *t, pressure, &m, &m)?;
            if !Self::is_trivial_solution(_vle.vapor(), _vle.liquid()) {
                return Ok(_vle);
            }
            vle = Some(_vle);
        }

        let cp = State::critical_point(eos, None, None, SolverOptions::default())?;
        if pressure > cp.pressure(Contributions::Total) {
            return Err(EosError::SuperCritical);
        };
        if let Some(mut e) = vle {
            if e.vapor().density < cp.density {
                for _ in 0..8 {
                    t0 = t0 * SCALE_T_NEW;
                    e.0[1] = State::new_npt(eos, t0, pressure, &m, DensityInitialization::Liquid)?;
                    if e.liquid().density > cp.density {
                        break;
                    }
                }
            } else {
                for _ in 0..8 {
                    t0 = t0 / SCALE_T_NEW;
                    e.0[0] = State::new_npt(eos, t0, pressure, &m, DensityInitialization::Vapor)?;
                    if e.vapor().density < cp.density {
                        break;
                    }
                }
            }

            for _ in 0..20 {
                t0 = (e.vapor().enthalpy(Contributions::Total)
                    - e.liquid().enthalpy(Contributions::Total))
                    / (e.vapor().entropy(Contributions::Total)
                        - e.liquid().entropy(Contributions::Total));
                let trial_state =
                    State::new_npt(eos, t0, pressure, &m, DensityInitialization::Vapor)?;
                if trial_state.density < cp.density {
                    e.0[0] = trial_state;
                }
                let trial_state =
                    State::new_npt(eos, t0, pressure, &m, DensityInitialization::Liquid)?;
                if trial_state.density > cp.density {
                    e.0[1] = trial_state;
                }
                if e.liquid().temperature == e.vapor().temperature {
                    return Ok(e);
                }
            }
            Err(EosError::IterationFailed(
                "new_init_p: could not find proper initial state".to_owned(),
            ))
        } else {
            unreachable!()
        }
    }
}

impl<U: EosUnit, E: EquationOfState> PhaseEquilibrium<U, E, 2> {
    /// Calculate the pure component vapor pressures of all
    /// components in the system for the given temperature.
    pub fn vapor_pressure(
        eos: &Rc<E>,
        temperature: QuantityScalar<U>,
    ) -> Vec<Option<QuantityScalar<U>>>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        (0..eos.components())
            .map(|i| {
                let pure_eos = Rc::new(eos.subset(&[i]));
                PhaseEquilibrium::pure_t(&pure_eos, temperature, None, SolverOptions::default())
                    .map(|vle| vle.vapor().pressure(Contributions::Total))
                    .ok()
            })
            .collect()
    }

    /// Calculate the pure component boiling temperatures of all
    /// components in the system for the given pressure.
    pub fn boiling_temperature(
        eos: &Rc<E>,
        pressure: QuantityScalar<U>,
    ) -> Vec<Option<QuantityScalar<U>>>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        (0..eos.components())
            .map(|i| {
                let pure_eos = Rc::new(eos.subset(&[i]));
                PhaseEquilibrium::pure_p(&pure_eos, pressure, None, SolverOptions::default())
                    .map(|vle| vle.vapor().temperature)
                    .ok()
            })
            .collect()
    }

    /// Calculate the pure component phase equilibria of all
    /// components in the system.
    pub fn vle_pure_comps(
        eos: &Rc<E>,
        temperature_or_pressure: QuantityScalar<U>,
    ) -> Vec<Option<PhaseEquilibrium<U, E, 2>>>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        (0..eos.components())
            .map(|i| {
                let pure_eos = Rc::new(eos.subset(&[i]));
                PhaseEquilibrium::pure(
                    &pure_eos,
                    temperature_or_pressure,
                    None,
                    SolverOptions::default(),
                )
                .ok()
                .map(|vle_pure| {
                    let mut moles_vapor = Array1::zeros(eos.components()) * U::reference_moles();
                    let mut moles_liquid = moles_vapor.clone();
                    moles_vapor
                        .try_set(i, vle_pure.vapor().total_moles)
                        .unwrap();
                    moles_liquid
                        .try_set(i, vle_pure.liquid().total_moles)
                        .unwrap();
                    let vapor = State::new_nvt(
                        eos,
                        vle_pure.vapor().temperature,
                        vle_pure.vapor().volume,
                        &moles_vapor,
                    )
                    .unwrap();
                    let liquid = State::new_nvt(
                        eos,
                        vle_pure.liquid().temperature,
                        vle_pure.liquid().volume,
                        &moles_liquid,
                    )
                    .unwrap();
                    PhaseEquilibrium::from_states(vapor, liquid)
                })
            })
            .collect()
    }
}
