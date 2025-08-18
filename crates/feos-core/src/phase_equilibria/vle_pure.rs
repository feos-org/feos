use super::{PhaseEquilibrium, TRIVIAL_REL_DEVIATION};
use crate::density_iteration::{_density_iteration, _pressure_spinodal};
use crate::equation_of_state::{Residual, Subset};
use crate::errors::{FeosError, FeosResult};
use crate::state::{Contributions, DensityInitialization, State};
use crate::{ReferenceSystem, SolverOptions, TemperatureOrPressure, Verbosity};
use nalgebra::allocator::Allocator;
use nalgebra::{DVector, DefaultAllocator, Dim, U1, dvector};
use num_dual::{DualNum, DualStruct, Gradients};
use quantity::{Density, Pressure, RGAS, Temperature};

const SCALE_T_NEW: f64 = 0.7;
const MAX_ITER_PURE: usize = 50;
const TOL_PURE: f64 = 1e-12;

/// # Pure component phase equilibria
impl<E: Residual> PhaseEquilibrium<E, 2> {
    /// Calculate a phase equilibrium for a pure component.
    pub fn pure<TP: TemperatureOrPressure>(
        eos: &E,
        temperature_or_pressure: TP,
        initial_state: Option<&Self>,
        options: SolverOptions,
    ) -> FeosResult<Self> {
        if let Some(t) = temperature_or_pressure.temperature() {
            let (_, rho) = Self::pure_t(eos, t, initial_state, options)?;
            Ok(Self(rho.map(|r| {
                State::new_intensive(eos, t, r, &dvector![1.0]).unwrap()
            })))
        } else if let Some(p) = temperature_or_pressure.pressure() {
            Self::pure_p(eos, p, initial_state, options)
        } else {
            unreachable!()
        }
    }
}

impl<E: Residual<N, D>, N: Gradients, D: DualNum<f64> + Copy> PhaseEquilibrium<E, 2, N, D>
where
    DefaultAllocator: Allocator<N> + Allocator<N, N> + Allocator<U1, N>,
{
    /// Calculate a phase equilibrium for a pure component
    /// and given temperature.
    pub fn pure_t(
        eos: &E,
        temperature: Temperature<D>,
        initial_state: Option<&Self>,
        options: SolverOptions,
    ) -> FeosResult<(Pressure<D>, [Density<D>; 2])> {
        let eos_f64 = eos.re();
        let t = temperature.into_reduced();

        // First use given initial state if applicable
        let mut vle = initial_state.and_then(|init| {
            let vle = (
                init.vapor()
                    .pressure(Contributions::Total)
                    .into_reduced()
                    .re(),
                [
                    init.vapor().density.into_reduced().re(),
                    init.liquid().density.into_reduced().re(),
                ],
            );
            iterate_pure_t(&eos_f64, t.re(), vle, options).ok()
        });

        // Next try to initialize with an ideal gas assumption
        vle = vle.or_else(|| {
            let vle = _init_pure_ideal_gas(&eos_f64, temperature.re());
            iterate_pure_t(&eos_f64, t.re(), vle, options).ok()
        });

        // Finally use the spinodal to initialize the calculation
        let (p, [rho_v, rho_l]) = vle.map_or_else(
            || {
                _init_pure_spinodal(&eos_f64, temperature.re())
                    .and_then(|vle| iterate_pure_t(&eos_f64, t.re(), vle, options))
            },
            Ok,
        )?;

        // Implicit differentiation
        let mut pressure = D::from(p);
        let mut vapor_density = D::from(rho_v);
        let mut liquid_density = D::from(rho_l);
        let x = E::pure_molefracs();
        for _ in 0..D::NDERIV {
            let v_l = liquid_density.recip();
            let v_v = vapor_density.recip();
            let (f_l, p_l, dp_l) = eos._p_dpdrho(t, liquid_density, &x);
            let (f_v, p_v, dp_v) = eos._p_dpdrho(t, vapor_density, &x);
            pressure = -(f_l * v_l - f_v * v_v + t * (v_v / v_l).ln()) / (v_l - v_v);
            liquid_density += (pressure - p_l) / dp_l;
            vapor_density += (pressure - p_v) / dp_v;
        }
        Ok((
            Pressure::from_reduced(pressure),
            [
                Density::from_reduced(vapor_density),
                Density::from_reduced(liquid_density),
            ],
        ))
    }
}

fn iterate_pure_t<E: Residual<N>, N: Dim>(
    eos: &E,
    temperature: f64,
    (mut pressure, [mut vapor_density, mut liquid_density]): (f64, [f64; 2]),
    options: SolverOptions,
) -> FeosResult<(f64, [f64; 2])>
where
    DefaultAllocator: Allocator<N>,
{
    let (max_iter, tol, verbosity) = options.unwrap_or(MAX_ITER_PURE, TOL_PURE);
    let x = E::pure_molefracs();

    log_iter!(
        verbosity,
        " iter |    residual    |     pressure     |    liquid density    |    vapor density     | Newton steps"
    );
    log_iter!(verbosity, "{:-<103}", "");
    log_iter!(
        verbosity,
        " {:4} |                | {:12.8} | {:12.8} | {:12.8} |",
        0,
        Pressure::from_reduced(pressure),
        Density::from_reduced(liquid_density),
        Density::from_reduced(vapor_density)
    );

    for i in 1..=max_iter {
        // calculate properties
        let (f_l_res, p_l, p_rho_l) = eos._p_dpdrho(temperature, liquid_density, &x);
        let (f_v_res, p_v, p_rho_v) = eos._p_dpdrho(temperature, vapor_density, &x);

        // Estimate the new pressure
        let v_v = vapor_density.recip();
        let v_l = liquid_density.recip();
        let delta_v = v_v - v_l;
        let delta_a =
            f_v_res * v_v - f_l_res * v_l + temperature * (vapor_density / liquid_density).ln();
        let mut p_new = -delta_a / delta_v;

        // If the pressure becomes negative, assume the gas phase is ideal. The
        // resulting pressure is always positive.
        if p_new.is_sign_negative() {
            p_new = p_v * ((-delta_a - p_v / vapor_density) / temperature).exp();
        }

        // Improve the estimate by exploiting the almost ideal behavior of the gas phase
        let mut newton_iter = 0;
        let newton_tol = pressure * delta_v * tol;
        for _ in 0..20 {
            let p_frac = p_new / pressure;
            let f = p_new * delta_v + delta_a + (p_frac.ln() + 1.0 - p_frac) * temperature;
            let df_dp = delta_v + (1.0 / p_new - 1.0 / pressure) * temperature;
            p_new -= f / df_dp;
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
        liquid_density += (p_new - p_l) / p_rho_l;
        vapor_density += (p_new - p_v) / p_rho_v;
        if (vapor_density / liquid_density - 1.0).abs() < TRIVIAL_REL_DEVIATION {
            return Err(FeosError::TrivialSolution);
        }

        // Check for convergence
        let res = (p_new - pressure).abs();
        log_iter!(
            verbosity,
            " {:4} | {:14.8e} | {:12.8} | {:12.8} | {:12.8} | {}",
            i,
            res,
            Pressure::from_reduced(p_new),
            Density::from_reduced(liquid_density),
            Density::from_reduced(vapor_density),
            newton_iter
        );
        if res < pressure * tol {
            log_result!(
                verbosity,
                "PhaseEquilibrium::pure_t: calculation converged in {} step(s)\n",
                i
            );
            return Ok((pressure, [vapor_density, liquid_density]));
        }
        pressure = p_new;
    }
    Err(FeosError::NotConverged("pure_t".to_owned()))
}

fn _init_pure_ideal_gas<E: Residual<N>, N: Dim>(
    eos: &E,
    temperature: Temperature,
) -> (f64, [f64; 2])
where
    DefaultAllocator: Allocator<N>,
{
    let x = E::pure_molefracs();
    let v = (0.75 * eos.compute_max_density(&x)).recip();
    let t = temperature.into_reduced();
    let a_res = eos.residual_molar_helmholtz_energy(t, v, &x);
    let p = t / v * (a_res / t - 1.0).exp();
    let rho_v = p / t;
    let rho_l = v.recip();
    (p, [rho_v, rho_l])
}

fn _init_pure_spinodal<E: Residual<N>, N: Dim>(
    eos: &E,
    temperature: Temperature,
) -> FeosResult<(f64, [f64; 2])>
where
    DefaultAllocator: Allocator<N>,
{
    let x = E::pure_molefracs();
    let maxdensity = eos.compute_max_density(&x);
    let t = temperature.into_reduced();
    let (p_l, _) = _pressure_spinodal(eos, t, 0.8 * maxdensity, &x)?;
    let (p_v, _) = _pressure_spinodal(eos, t, 0.001 * maxdensity, &x)?;
    let p = 0.5 * (0.0_f64.max(p_l) + p_v);
    let rho_l = _density_iteration(eos, t, p, &x, DensityInitialization::Liquid)?;
    let rho_v = _density_iteration(eos, t, p, &x, DensityInitialization::Vapor)?;
    Ok((p, [rho_v, rho_l]))
}

impl<E: Residual> PhaseEquilibrium<E, 2> {
    fn new_pt(eos: &E, temperature: Temperature, pressure: Pressure) -> FeosResult<Self> {
        let liquid = State::new_xpt(
            eos,
            temperature,
            pressure,
            &dvector![1.0],
            Some(DensityInitialization::Liquid),
        )?;
        let vapor = State::new_xpt(
            eos,
            temperature,
            pressure,
            &dvector![1.0],
            Some(DensityInitialization::Vapor),
        )?;
        Ok(Self([vapor, liquid]))
    }

    /// Calculate a phase equilibrium for a pure component
    /// and given pressure.
    fn pure_p(
        eos: &E,
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
        Err(FeosError::NotConverged("pure_p".to_owned()))
    }

    /// Initialize a new VLE for a pure substance for a given pressure.
    fn init_pure_p(eos: &E, pressure: Pressure) -> FeosResult<Self> {
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
                    e.0[1] =
                        State::new_xpt(eos, t0, pressure, &x, Some(DensityInitialization::Liquid))?;
                    if e.liquid().density > cp.density {
                        break;
                    }
                }
            } else {
                for _ in 0..8 {
                    t0 /= SCALE_T_NEW;
                    e.0[0] =
                        State::new_xpt(eos, t0, pressure, &x, Some(DensityInitialization::Vapor))?;
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
                    State::new_xpt(eos, t0, pressure, &x, Some(DensityInitialization::Vapor))?;
                if trial_state.density < cp.density {
                    e.0[0] = trial_state;
                }
                let trial_state =
                    State::new_xpt(eos, t0, pressure, &x, Some(DensityInitialization::Liquid))?;
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

impl<E: Residual + Subset> PhaseEquilibrium<E, 2> {
    /// Calculate the pure component vapor pressures of all
    /// components in the system for the given temperature.
    pub fn vapor_pressure(eos: &E, temperature: Temperature) -> Vec<Option<Pressure>> {
        (0..eos.components())
            .map(|i| {
                let pure_eos = eos.subset(&[i]);
                PhaseEquilibrium::pure_t(&pure_eos, temperature, None, SolverOptions::default())
                    .map(|(p, _)| p)
                    .ok()
            })
            .collect()
    }

    /// Calculate the pure component boiling temperatures of all
    /// components in the system for the given pressure.
    pub fn boiling_temperature(eos: &E, pressure: Pressure) -> Vec<Option<Temperature>> {
        (0..eos.components())
            .map(|i| {
                let pure_eos = eos.subset(&[i]);
                PhaseEquilibrium::pure_p(&pure_eos, pressure, None, SolverOptions::default())
                    .map(|vle| vle.vapor().temperature)
                    .ok()
            })
            .collect()
    }

    /// Calculate the pure component phase equilibria of all
    /// components in the system.
    pub fn vle_pure_comps<TP: TemperatureOrPressure>(
        eos: &E,
        temperature_or_pressure: TP,
    ) -> Vec<Option<PhaseEquilibrium<E, 2>>> {
        (0..eos.components())
            .map(|i| {
                let pure_eos = eos.subset(&[i]);
                PhaseEquilibrium::pure(
                    &pure_eos,
                    temperature_or_pressure,
                    None,
                    SolverOptions::default(),
                )
                .and_then(|vle_pure| {
                    let mut molefracs_vapor = DVector::zeros(eos.components());
                    let mut molefracs_liquid = molefracs_vapor.clone();
                    molefracs_vapor[i] = 1.0;
                    molefracs_liquid[i] = 1.0;
                    let vapor = State::new_intensive(
                        eos,
                        vle_pure.vapor().temperature,
                        vle_pure.vapor().density,
                        &molefracs_vapor,
                    )?;
                    let liquid = State::new_intensive(
                        eos,
                        vle_pure.liquid().temperature,
                        vle_pure.liquid().density,
                        &molefracs_liquid,
                    )?;
                    Ok(PhaseEquilibrium::from_states(vapor, liquid))
                })
                .ok()
            })
            .collect()
    }
}
