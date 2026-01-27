use super::{PhaseEquilibrium, TRIVIAL_REL_DEVIATION};
use crate::density_iteration::{_density_iteration, _pressure_spinodal};
use crate::equation_of_state::{Residual, Subset};
use crate::errors::{FeosError, FeosResult};
use crate::state::{Contributions, DensityInitialization, State};
use crate::{ReferenceSystem, SolverOptions, TemperatureOrPressure, Verbosity};
use nalgebra::allocator::Allocator;
use nalgebra::{DVector, DefaultAllocator, Dim, SVector, U1, U2};
use num_dual::{DualNum, DualStruct, Gradients, gradient, partial};
use quantity::{Density, Pressure, Temperature};

const SCALE_T_NEW: f64 = 0.7;
const MAX_ITER_PURE: usize = 50;
const TOL_PURE: f64 = 1e-12;

/// # Pure component phase equilibria
impl<E: Residual<N, D>, N: Gradients, D: DualNum<f64> + Copy> PhaseEquilibrium<E, 2, N, D>
where
    DefaultAllocator: Allocator<N> + Allocator<U1, N> + Allocator<N, N>,
{
    /// Calculate a phase equilibrium for a pure component.
    pub fn pure<TP: TemperatureOrPressure<D>>(
        eos: &E,
        temperature_or_pressure: TP,
        initial_state: Option<&Self>,
        options: SolverOptions,
    ) -> FeosResult<Self> {
        let (t, rho) = if let Some(t) = temperature_or_pressure.temperature() {
            let (_, rho) = Self::pure_t(eos, t, initial_state, options)?;
            (t, rho)
        } else if let Some(p) = temperature_or_pressure.pressure() {
            Self::pure_p(eos, p, initial_state, options)?
        } else {
            unreachable!()
        };
        Ok(Self(rho.map(|r| State::new_pure(eos, t, r).unwrap())))
    }

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
            _init_pure_ideal_gas(&eos_f64, temperature.re())
                .and_then(|vle| iterate_pure_t(&eos_f64, t.re(), vle, options))
                .ok()
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
            let (a_l, p_l, dp_l) = eos.p_dpdrho(t, liquid_density, &x);
            let (a_v, p_v, dp_v) = eos.p_dpdrho(t, vapor_density, &x);
            pressure = -(a_l - a_v + t * (v_v / v_l).ln()) / (v_l - v_v);
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
        let (a_l_res, p_l, p_rho_l) = eos.p_dpdrho(temperature, liquid_density, &x);
        let (a_v_res, p_v, p_rho_v) = eos.p_dpdrho(temperature, vapor_density, &x);

        // Estimate the new pressure
        let v_v = vapor_density.recip();
        let v_l = liquid_density.recip();
        let delta_v = v_v - v_l;
        let delta_a = a_v_res - a_l_res + temperature * (vapor_density / liquid_density).ln();
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
) -> FeosResult<(f64, [f64; 2])>
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
    let rho_v = _density_iteration(eos, t, p, &x, DensityInitialization::InitialDensity(rho_v))?;
    let rho_l = _density_iteration(eos, t, p, &x, DensityInitialization::InitialDensity(rho_l))?;
    Ok((p, [rho_v, rho_l]))
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

impl<E: Residual<N, D>, N: Gradients, D: DualNum<f64> + Copy> PhaseEquilibrium<E, 2, N, D>
where
    DefaultAllocator: Allocator<N> + Allocator<U1, N> + Allocator<N, N>,
{
    /// Calculate a phase equilibrium for a pure component
    /// and given pressure.
    pub fn pure_p(
        eos: &E,
        pressure: Pressure<D>,
        initial_state: Option<&Self>,
        options: SolverOptions,
    ) -> FeosResult<(Temperature<D>, [Density<D>; 2])> {
        let eos_f64 = eos.re();
        let p = pressure.into_reduced();

        // Initialize the phase equilibrium
        let vle = match initial_state {
            Some(init) => (
                init.vapor().temperature.into_reduced().re(),
                [
                    init.vapor().density.into_reduced().re(),
                    init.liquid().density.into_reduced().re(),
                ],
            ),
            None => init_pure_p(&eos_f64, pressure.re())?,
        };
        let (t, [rho_v, rho_l]) = iterate_pure_p(&eos_f64, p.re(), vle, options)?;

        // Implicit differentiation
        let mut temperature = D::from(t);
        let mut vapor_density = D::from(rho_v);
        let mut liquid_density = D::from(rho_l);
        let x = E::pure_molefracs();
        for _ in 0..D::NDERIV {
            let v_l = liquid_density.recip();
            let v_v = vapor_density.recip();
            let (a_l, p_l, s_l, p_rho_l, p_t_l) =
                eos.p_dpdrho_dpdt(temperature, liquid_density, &x);
            let (a_v, p_v, s_v, p_rho_v, p_t_v) = eos.p_dpdrho_dpdt(temperature, vapor_density, &x);
            let ln_rho = (v_l / v_v).ln();
            let delta_t =
                (p * (v_v - v_l) + (a_v - a_l + temperature * ln_rho)) / (s_v - s_l - ln_rho);
            temperature += delta_t;
            liquid_density += (p - p_l - p_t_l * delta_t) / p_rho_l;
            vapor_density += (p - p_v - p_t_v * delta_t) / p_rho_v;
        }
        Ok((
            Temperature::from_reduced(temperature),
            [
                Density::from_reduced(vapor_density),
                Density::from_reduced(liquid_density),
            ],
        ))
    }
}

/// Calculate a phase equilibrium for a pure component
/// and given pressure.
fn iterate_pure_p<E: Residual<N>, N: Dim>(
    eos: &E,
    pressure: f64,
    (mut temperature, [mut vapor_density, mut liquid_density]): (f64, [f64; 2]),
    options: SolverOptions,
) -> FeosResult<(f64, [f64; 2])>
where
    DefaultAllocator: Allocator<N>,
{
    let (max_iter, tol, verbosity) = options.unwrap_or(MAX_ITER_PURE, TOL_PURE);
    let x = E::pure_molefracs();

    log_iter!(
        verbosity,
        " iter |     residual     |   temperature   |    liquid density    |    vapor density     "
    );
    log_iter!(verbosity, "{:-<89}", "");
    log_iter!(
        verbosity,
        " {:4} |                  | {:13.8} | {:12.8} | {:12.8}",
        0,
        Temperature::from_reduced(temperature),
        Density::from_reduced(liquid_density),
        Density::from_reduced(vapor_density)
    );
    for i in 1..=max_iter {
        // calculate properties
        let (a_l_res, p_l, s_l_res, p_rho_l, p_t_l) =
            eos.p_dpdrho_dpdt(temperature, liquid_density, &x);
        let (a_v_res, p_v, s_v_res, p_rho_v, p_t_v) =
            eos.p_dpdrho_dpdt(temperature, vapor_density, &x);

        // calculate the molar volumes
        let v_l = liquid_density.recip();
        let v_v = vapor_density.recip();

        // estimate the temperature steps
        let ln_rho = (v_l / v_v).ln();
        let delta_t = (pressure * (v_v - v_l) + (a_v_res - a_l_res + temperature * ln_rho))
            / (s_v_res - s_l_res - ln_rho);
        temperature += delta_t;

        // calculate Newton steps for the densities and update state.
        let rho_l = liquid_density + (pressure - p_l - p_t_l * delta_t) / p_rho_l;
        let rho_v = vapor_density + (pressure - p_v - p_t_v * delta_t) / p_rho_v;

        if rho_l.is_sign_negative() || rho_v.is_sign_negative() || delta_t.abs() > 1.0 {
            // if densities are negative or the temperature step is large use density iteration instead
            liquid_density = _density_iteration(
                eos,
                temperature,
                pressure,
                &x,
                DensityInitialization::InitialDensity(liquid_density),
            )?;
            vapor_density = _density_iteration(
                eos,
                temperature,
                pressure,
                &x,
                DensityInitialization::InitialDensity(vapor_density),
            )?;
        } else {
            liquid_density = rho_l;
            vapor_density = rho_v;
        }

        // check for trivial solution
        if (vapor_density / liquid_density - 1.0).abs() < TRIVIAL_REL_DEVIATION {
            return Err(FeosError::TrivialSolution);
        }

        // check for convergence
        let res = delta_t.abs();
        log_iter!(
            verbosity,
            " {:4} | {:14.8e} | {:13.8} | {:12.8} | {:12.8}",
            i,
            res,
            Temperature::from_reduced(temperature),
            Density::from_reduced(liquid_density),
            Density::from_reduced(vapor_density)
        );
        if res < temperature * tol {
            log_result!(
                verbosity,
                "PhaseEquilibrium::pure_p: calculation converged in {} step(s)\n",
                i
            );
            return Ok((temperature, [vapor_density, liquid_density]));
        }
    }
    Err(FeosError::NotConverged("pure_p".to_owned()))
}

/// Initialize a new VLE for a pure substance for a given pressure.
fn init_pure_p<E: Residual<N>, N: Gradients>(
    eos: &E,
    pressure: Pressure,
) -> FeosResult<(f64, [f64; 2])>
where
    DefaultAllocator: Allocator<N> + Allocator<U1, N> + Allocator<N, N>,
{
    let trial_temperatures = [300.0, 500.0, 200.0];
    let p = pressure.into_reduced();
    let x = E::pure_molefracs();
    let mut vle = None;
    for t in trial_temperatures {
        let liquid_density = _density_iteration(eos, t, p, &x, DensityInitialization::Liquid)?;
        let vapor_density = _density_iteration(eos, t, p, &x, DensityInitialization::Vapor)?;
        let _vle = (t, [vapor_density, liquid_density]);
        if (vapor_density / liquid_density - 1.0).abs() >= TRIVIAL_REL_DEVIATION {
            return Ok(_vle);
        }
        vle = Some(_vle);
    }
    let Some((t0, [mut rho_v, mut rho_l])) = vle else {
        unreachable!()
    };
    let [mut t_v, mut t_l] = [t0, t0];

    let cp = State::critical_point(eos, None, None, None, SolverOptions::default())?;
    let cp_density = cp.density.into_reduced();
    if pressure > cp.pressure(Contributions::Total) {
        return Err(FeosError::SuperCritical);
    };

    if rho_v < cp_density {
        // reduce temperature of liquid phase...
        for _ in 0..8 {
            t_l *= SCALE_T_NEW;
            rho_l = _density_iteration(eos, t_l, p, &x, DensityInitialization::Liquid)?;
            if rho_l > cp_density {
                break;
            }
        }
    } else {
        // ...or increase temperature of vapor phase
        for _ in 0..8 {
            t_v /= SCALE_T_NEW;
            rho_v = _density_iteration(eos, t_v, p, &x, DensityInitialization::Vapor)?;
            if rho_v < cp_density {
                break;
            }
        }
    }

    // determine new temperatures and assign them to either the liquid or the vapor phase until
    // both phases have the same temperature
    for _ in 0..20 {
        let h_s = |t, v| {
            let (a_res, da_res) = gradient::<_, _, _, U2, _>(
                partial(
                    |t_v: SVector<_, _>, x| {
                        let [[t, v]] = t_v.data.0;
                        eos.lift().residual_molar_helmholtz_energy(t, v, x)
                    },
                    &x,
                ),
                &SVector::from([t, v]),
            );
            let [[da_res_dt, da_res_dv]] = da_res.data.0;
            (a_res - t * da_res_dt - v * da_res_dv + t, -da_res_dt)
        };
        let (h_l, s_l_res) = h_s(t_l, rho_l.recip());
        let (h_v, s_v_res) = h_s(t_v, rho_v.recip());
        let t = (h_v - h_l) / (s_v_res - s_l_res - (rho_v / rho_l).ln());
        let trial_density = _density_iteration(eos, t, p, &x, DensityInitialization::Vapor)?;
        if trial_density < cp_density {
            rho_v = trial_density;
            t_v = t;
        }
        let trial_density = _density_iteration(eos, t, p, &x, DensityInitialization::Liquid)?;
        if trial_density > cp_density {
            rho_l = trial_density;
            t_l = t;
        }
        if t_l == t_v {
            return Ok((t_l, [rho_v, rho_l]));
        }
    }
    Err(FeosError::IterationFailed(
        "new_init_p: could not find proper initial state".to_owned(),
    ))
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
                    .map(|(t, _)| t)
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
