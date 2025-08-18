use crate::DensityInitialization::{self, InitialDensity, Liquid, Vapor};
use crate::errors::{FeosError, FeosResult};
use crate::{ReferenceSystem, Residual};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, OVector};
use num_dual::{Dual, DualNum, first_derivative};
use quantity::{Density, Pressure, Temperature};

pub fn density_iteration<E: Residual<N, D>, N: Dim, D: DualNum<f64> + Copy>(
    eos: &E,
    temperature: Temperature<D>,
    pressure: Pressure<D>,
    molefracs: &OVector<D, N>,
    initial_density: Option<DensityInitialization>,
) -> FeosResult<Density<D>>
where
    DefaultAllocator: Allocator<N>,
{
    let eos_f64 = eos.re();
    let t = temperature.into_reduced();
    let pressure = pressure.into_reduced();
    let x = molefracs.map(|x| x.re());
    let density = if let Some(initial_density) = initial_density {
        _density_iteration(&eos_f64, t.re(), pressure.re(), &x, initial_density)
    } else {
        _density_iteration_stable(&eos_f64, t.re(), pressure.re(), &x)
    }?;

    // Implicit differentiation
    let mut density = D::from(density);
    for _ in 0..D::NDERIV {
        let (_, p, dp_drho) = eos._p_dpdrho(t, density, molefracs);
        density -= (p - pressure) / dp_drho;
    }
    Ok(Density::from_reduced(density))
}

fn _density_iteration_stable<E: Residual<N>, N: Dim>(
    eos: &E,
    temperature: f64,
    pressure: f64,
    molefracs: &OVector<f64, N>,
) -> FeosResult<f64>
where
    DefaultAllocator: Allocator<N>,
{
    // calculate stable phase
    let max_density = eos.compute_max_density(molefracs);
    let liquid = _density_iteration(eos, temperature, pressure, molefracs, Liquid);

    if pressure < max_density * temperature {
        let vapor = _density_iteration(eos, temperature, pressure, molefracs, Vapor);
        match (&liquid, &vapor) {
            (Ok(_), Err(_)) => liquid,
            (Err(_), Ok(_)) => vapor,
            (Ok(l), Ok(v)) => {
                if _chemical_potential(eos, temperature, *l, molefracs)
                    > _chemical_potential(eos, temperature, *v, molefracs)
                {
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

fn _chemical_potential<E: Residual<N, f64>, N: Dim>(
    eos: &E,
    temperature: f64,
    density: f64,
    molefracs: &OVector<f64, N>,
) -> f64
where
    DefaultAllocator: Allocator<N>,
{
    let molar_volume = density.recip();
    let t = Dual::from_re(temperature);
    let x = molefracs.map(Dual::from);
    let (a_res, da_res) = first_derivative(
        |molar_volume| {
            eos.lift()
                .residual_molar_helmholtz_energy(t, molar_volume, &x)
        },
        molar_volume,
    );
    a_res - da_res * molar_volume + temperature * density.ln()
}

pub(crate) fn _density_iteration<E: Residual<N>, N: Dim>(
    eos: &E,
    temperature: f64,
    pressure: f64,
    molefracs: &OVector<f64, N>,
    initial_density: DensityInitialization,
) -> FeosResult<f64>
where
    DefaultAllocator: Allocator<N>,
{
    let maxdensity = eos.compute_max_density(molefracs);
    let initial_density = match initial_density {
        Vapor => pressure / temperature,
        Liquid => maxdensity,
        InitialDensity(d) => d.into_reduced(),
    };
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
                    _pressure_spinodal(eos, temperature, initial_density, molefracs)?;
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
                    _pressure_spinodal(eos, temperature, initial_density, molefracs)?;
                rho = sp_rho;
                error = sp_p - pressure;
                if error.is_sign_positive() {
                    rho = 0.001 * maxdensity
                } else {
                    rho = (rho * 1.1).min(maxdensity)
                }
            } else if error.is_sign_negative() && d2pdrho2.is_sign_negative() {
                let (sp_p, sp_rho) =
                    _pressure_spinodal(eos, temperature, initial_density, molefracs)?;
                rho = sp_rho;
                error = sp_p - pressure;
                if error.is_sign_negative() {
                    rho = 0.8 * maxdensity
                } else {
                    rho *= 0.8
                }
            } else if error.is_sign_negative() && d2pdrho2.is_sign_positive() {
                let (_, rho_l) = _pressure_spinodal(eos, temperature, 0.8 * maxdensity, molefracs)?;
                let (sp_v_p, rho_v) =
                    _pressure_spinodal(eos, temperature, 0.001 * maxdensity, molefracs)?;
                error = sp_v_p - pressure;
                if error.is_sign_positive()
                    && (initial_density - rho_v).abs() < (initial_density - rho_l).abs()
                {
                    rho = 0.8 * rho_v
                } else {
                    rho = (rho_l * 1.1).min(maxdensity)
                }
            } else if error.is_sign_positive() && d2pdrho2.is_sign_negative() {
                let (_, rho_l) = _pressure_spinodal(eos, temperature, 0.8 * maxdensity, molefracs)?;
                let (sp_v_p, rho_v) =
                    _pressure_spinodal(eos, temperature, 0.001 * maxdensity, molefracs)?;
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

pub(crate) fn _pressure_spinodal<E: Residual<N>, N: Dim>(
    eos: &E,
    temperature: f64,
    rho_init: f64,
    molefracs: &OVector<f64, N>,
) -> FeosResult<(f64, f64)>
where
    DefaultAllocator: Allocator<N>,
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
