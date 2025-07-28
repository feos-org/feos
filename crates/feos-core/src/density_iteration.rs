use crate::errors::{FeosError, FeosResult};
use crate::{DensityInitialization, HelmholtzEnergyDerivatives, ReferenceSystem, StateGeneric};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, OVector};
use quantity::{Density, Moles, Pressure, RGAS, Temperature};

pub fn density_iteration_stable<E: HelmholtzEnergyDerivatives<f64>>(
    eos: &E,
    temperature: Temperature,
    pressure: Pressure,
    molefracs: &OVector<f64, E::Components>,
    density_initialization: DensityInitialization,
) -> FeosResult<StateGeneric<E, f64, E::Components, E::Cache>>
where
    DefaultAllocator: Allocator<E::Components>,
{
    // calculate state from initial density or given phase
    match density_initialization {
        DensityInitialization::InitialDensity(rho0) => {
            return density_iteration(eos, temperature, pressure, molefracs, rho0);
        }
        DensityInitialization::Vapor => {
            return density_iteration(
                eos,
                temperature,
                pressure,
                molefracs,
                pressure / temperature / RGAS,
            );
        }
        DensityInitialization::Liquid => {
            return density_iteration(
                eos,
                temperature,
                pressure,
                molefracs,
                Density::from_reduced(eos.compute_max_density(molefracs)),
            );
        }
        DensityInitialization::None => (),
    }

    // calculate stable phase
    let max_density = Density::from_reduced(eos.compute_max_density(molefracs));
    let liquid = density_iteration(eos, temperature, pressure, molefracs, max_density);

    if pressure < max_density * temperature * RGAS {
        let vapor = density_iteration(
            eos,
            temperature,
            pressure,
            molefracs,
            pressure / temperature / RGAS,
        );
        match (&liquid, &vapor) {
            (Ok(_), Err(_)) => liquid,
            (Err(_), Ok(_)) => vapor,
            (Ok(l), Ok(v)) => {
                if l.residual_gibbs_energy() > v.residual_gibbs_energy() {
                    vapor
                } else {
                    liquid
                }
            }
            _ => Err(FeosError::UndeterminedState(String::from(
                "Density iteration did not find a solution.",
            ))),
        }
    } else {
        liquid
    }
}

pub fn density_iteration<E: HelmholtzEnergyDerivatives<f64>>(
    eos: &E,
    temperature: Temperature,
    pressure: Pressure,
    molefracs: &OVector<f64, E::Components>,
    initial_density: Density,
) -> FeosResult<StateGeneric<E, f64, E::Components, E::Cache>>
where
    DefaultAllocator: Allocator<E::Components>,
{
    let rho = Density::from_reduced(_density_iteration(
        eos,
        temperature.into_reduced(),
        pressure.into_reduced(),
        molefracs,
        initial_density.into_reduced(),
    )?);
    let total_moles = Moles::new(1.0);
    Ok(StateGeneric::new_unchecked(
        eos,
        temperature,
        rho,
        total_moles,
        molefracs,
    ))
}

fn _density_iteration<E: HelmholtzEnergyDerivatives<f64>>(
    eos: &E,
    temperature: f64,
    pressure: f64,
    molefracs: &OVector<f64, E::Components>,
    initial_density: f64,
) -> FeosResult<f64>
where
    DefaultAllocator: Allocator<E::Components>,
{
    let maxdensity = eos.compute_max_density(molefracs);
    let (abstol, reltol) = (1e-12, 1e-14);

    let mut rho = initial_density;
    if rho <= 0.0 {
        return Err(FeosError::InvalidState(
            String::from("density iteration"),
            String::from("density"),
            rho,
        ));
    }

    let maxiter = 50;
    let mut iterations = 0;
    'iteration: for k in 0..maxiter {
        iterations += 1;
        let (_, mut p, mut dp_drho) = eos._p_dpdrho(temperature, rho, molefracs);

        // attempt to correct for poor initial density rho_init
        if dp_drho.is_sign_negative() && k == 0 {
            rho = if initial_density <= 0.15 * maxdensity {
                0.05 * initial_density
            } else {
                (1.1 * initial_density).min(maxdensity)
            };
            let p_ = eos._p_dpdrho(temperature, rho, molefracs);
            p = p_.0;
            dp_drho = p_.1;
        }

        let mut error = p - pressure;

        let mut delta_rho = -error / dp_drho;
        if delta_rho.abs() > 0.075 * maxdensity {
            delta_rho = 0.075 * maxdensity * delta_rho.signum();
        };
        delta_rho = delta_rho.max(-0.95 * rho); // prevent stepping to rho < 0.0

        // correction for instable region
        if dp_drho.is_sign_negative() && k < maxiter {
            let (_, _, d2pdrho2) = eos._p_dpdrho_d2pdrho2(temperature, rho, molefracs);

            if rho > 0.85 * maxdensity {
                let (sp_p, sp_rho) =
                    pressure_spinodal(eos, temperature, initial_density, molefracs)?;
                rho = sp_rho;
                error = sp_p - pressure;
                if rho > 0.85 * maxdensity {
                    if error.is_sign_negative() {
                        return Err(FeosError::IterationFailed(String::from(
                            "density_iteration",
                        )));
                    } else {
                        rho *= 0.98
                    }
                } else if error.is_sign_positive() {
                    rho = 0.001 * maxdensity
                } else {
                    rho = (rho * 1.1).min(maxdensity)
                }
            } else if error.is_sign_positive() && d2pdrho2.is_sign_positive() {
                let (sp_p, sp_rho) =
                    pressure_spinodal(eos, temperature, initial_density, molefracs)?;
                rho = sp_rho;
                error = sp_p - pressure;
                if error.is_sign_positive() {
                    rho = 0.001 * maxdensity
                } else {
                    rho = (rho * 1.1).min(maxdensity)
                }
            } else if error.is_sign_negative() && d2pdrho2.is_sign_negative() {
                let (sp_p, sp_rho) =
                    pressure_spinodal(eos, temperature, initial_density, molefracs)?;
                rho = sp_rho;
                error = sp_p - pressure;
                if error.is_sign_negative() {
                    rho = 0.8 * maxdensity
                } else {
                    rho *= 0.8
                }
            } else if error.is_sign_negative() && d2pdrho2.is_sign_positive() {
                let (_, rho_l) = pressure_spinodal(eos, temperature, 0.8 * maxdensity, molefracs)?;
                let (sp_v_p, rho_v) =
                    pressure_spinodal(eos, temperature, 0.001 * maxdensity, molefracs)?;
                error = sp_v_p - pressure;
                if error.is_sign_positive()
                    && (initial_density - rho_v).abs() < (initial_density - rho_l).abs()
                {
                    rho = 0.8 * rho_v
                } else {
                    rho = (rho_l * 1.1).min(maxdensity)
                }
            } else if error.is_sign_positive() && d2pdrho2.is_sign_negative() {
                let (_, rho_l) = pressure_spinodal(eos, temperature, 0.8 * maxdensity, molefracs)?;
                let (sp_v_p, rho_v) =
                    pressure_spinodal(eos, temperature, 0.001 * maxdensity, molefracs)?;
                error = sp_v_p - pressure;
                if error.is_sign_negative()
                    && (initial_density - rho_v).abs() > (initial_density - rho_l).abs()
                {
                    rho = (rho_l * 1.1).min(maxdensity)
                } else {
                    rho = 0.8 * rho_v
                }
            } else {
                rho = (rho + initial_density) * 0.5;
                if (rho - initial_density).abs() < 1e-8 {
                    rho = (rho + 0.1 * maxdensity).min(maxdensity)
                }
            }
            continue 'iteration;
        }
        // Newton step
        rho += delta_rho;
        if error.abs() < f64::max(abstol, rho * reltol) {
            break 'iteration;
        }
    }
    if iterations == maxiter + 1 {
        Err(FeosError::NotConverged("density_iteration".to_owned()))
    } else {
        Ok(rho)
    }
}

fn pressure_spinodal<E: HelmholtzEnergyDerivatives<f64>>(
    eos: &E,
    temperature: f64,
    rho_init: f64,
    molefracs: &OVector<f64, E::Components>,
) -> FeosResult<(f64, f64)>
where
    DefaultAllocator: Allocator<E::Components>,
{
    let maxiter = 30;
    let abstol = 1e-8;

    let maxdensity = eos.compute_max_density(molefracs);
    let mut rho = rho_init;

    if rho <= 0.0 {
        return Err(FeosError::InvalidState(
            String::from("pressure spinodal"),
            String::from("density"),
            rho,
        ));
    }

    for _ in 0..maxiter {
        let (p, dpdrho, d2pdrho2) = eos._p_dpdrho_d2pdrho2(temperature, rho, molefracs);

        let mut delta_rho = -dpdrho / d2pdrho2;
        if delta_rho.abs() > 0.05 * maxdensity {
            delta_rho = 0.05 * maxdensity * delta_rho.signum()
        }
        delta_rho = delta_rho.max(-rho * 0.95); // prevent stepping to rho < 0.0
        delta_rho = delta_rho.min(maxdensity - rho); // prevent stepping to rho > maxdensity
        rho += delta_rho;

        if dpdrho.abs() < abstol {
            return Ok((p, rho));
        }
    }
    Err(FeosError::NotConverged("pressure_spinodal".to_owned()))
}
