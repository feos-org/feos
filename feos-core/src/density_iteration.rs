use crate::equation_of_state::Residual;
use crate::errors::{EosError, EosResult};
use crate::si::{Density, Moles, Pressure, Temperature};
use crate::state::State;
use ndarray::Array1;
use std::sync::Arc;

pub fn density_iteration<E: Residual>(
    eos: &Arc<E>,
    temperature: Temperature,
    pressure: Pressure,
    moles: &Moles<Array1<f64>>,
    initial_density: Density,
) -> EosResult<State<E>> {
    let maxdensity = eos.max_density(Some(moles))?;
    let (abstol, reltol) = (1e-12, 1e-14);
    let n = moles.sum();

    let mut rho = initial_density;
    if rho <= Density::from_reduced(0.0) {
        return Err(EosError::InvalidState(
            String::from("density iteration"),
            String::from("density"),
            rho.to_reduced(),
        ));
    }

    let maxiter = 50;
    let mut iterations = 0;
    'iteration: for k in 0..maxiter {
        iterations += 1;
        let (mut p, mut dp_drho) = State::new_nvt(eos, temperature, n / rho, moles)?.p_dpdrho();

        // attempt to correct for poor initial density rho_init
        if dp_drho.is_sign_negative() && k == 0 {
            rho = if initial_density <= 0.15 * maxdensity {
                0.05 * initial_density
            } else {
                (1.1 * initial_density).min(maxdensity)
            };
            let p_ = State::new_nvt(eos, temperature, n / rho, moles)?.p_dpdrho();
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
            let d2pdrho2 = State::new_nvt(eos, temperature, n / rho, moles)?
                .d2pdrho2()
                .2;

            if rho > 0.85 * maxdensity {
                let (sp_p, sp_rho) = pressure_spinodal(eos, temperature, initial_density, moles)?;
                rho = sp_rho;
                error = sp_p - pressure;
                if rho > 0.85 * maxdensity {
                    if error.is_sign_negative() {
                        return Err(EosError::IterationFailed(String::from("density_iteration")));
                    } else {
                        rho *= 0.98
                    }
                } else if error.is_sign_positive() {
                    rho = 0.001 * maxdensity
                } else {
                    rho = (rho * 1.1).min(maxdensity)
                }
            } else if error.is_sign_positive() && d2pdrho2.is_sign_positive() {
                let (sp_p, sp_rho) = pressure_spinodal(eos, temperature, initial_density, moles)?;
                rho = sp_rho;
                error = sp_p - pressure;
                if error.is_sign_positive() {
                    rho = 0.001 * maxdensity
                } else {
                    rho = (rho * 1.1).min(maxdensity)
                }
            } else if error.is_sign_negative() && d2pdrho2.is_sign_negative() {
                let (sp_p, sp_rho) = pressure_spinodal(eos, temperature, initial_density, moles)?;
                rho = sp_rho;
                error = sp_p - pressure;
                if error.is_sign_negative() {
                    rho = 0.8 * maxdensity
                } else {
                    rho *= 0.8
                }
            } else if error.is_sign_negative() && d2pdrho2.is_sign_positive() {
                let (_, rho_l) = pressure_spinodal(eos, temperature, 0.8 * maxdensity, moles)?;
                let (sp_v_p, rho_v) =
                    pressure_spinodal(eos, temperature, 0.001 * maxdensity, moles)?;
                error = sp_v_p - pressure;
                if error.is_sign_positive()
                    && (initial_density - rho_v).abs() < (initial_density - rho_l).abs()
                {
                    rho = 0.8 * rho_v
                } else {
                    rho = (rho_l * 1.1).min(maxdensity)
                }
            } else if error.is_sign_positive() && d2pdrho2.is_sign_negative() {
                let (_, rho_l) = pressure_spinodal(eos, temperature, 0.8 * maxdensity, moles)?;
                let (sp_v_p, rho_v) =
                    pressure_spinodal(eos, temperature, 0.001 * maxdensity, moles)?;
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
                if (rho - initial_density).to_reduced().abs() < 1e-8 {
                    rho = (rho + 0.1 * maxdensity).min(maxdensity)
                }
            }
            continue 'iteration;
        }
        // Newton step
        rho += delta_rho;
        if error.to_reduced().abs() < f64::max(abstol, (rho * reltol).to_reduced()) {
            break 'iteration;
        }
    }
    if iterations == maxiter + 1 {
        Err(EosError::NotConverged("density_iteration".to_owned()))
    } else {
        Ok(State::new_nvt(eos, temperature, n / rho, moles)?)
    }
}

fn pressure_spinodal<E: Residual>(
    eos: &Arc<E>,
    temperature: Temperature,
    rho_init: Density,
    moles: &Moles<Array1<f64>>,
) -> EosResult<(Pressure, Density)> {
    let maxiter = 30;
    let abstol = 1e-8;

    let maxdensity = eos.max_density(Some(moles))?;
    let n = moles.sum();
    let mut rho = rho_init;

    if rho <= Density::from_reduced(0.0) {
        return Err(EosError::InvalidState(
            String::from("pressure spinodal"),
            String::from("density"),
            rho.to_reduced(),
        ));
    }

    for _ in 0..maxiter {
        let (p, dpdrho, d2pdrho2) = State::new_nvt(eos, temperature, n / rho, moles)?.d2pdrho2();

        let mut delta_rho = -dpdrho / d2pdrho2;
        if delta_rho.abs() > 0.05 * maxdensity {
            delta_rho = 0.05 * maxdensity * delta_rho.signum()
        }
        delta_rho = delta_rho.max(-rho * 0.95); // prevent stepping to rho < 0.0
        delta_rho = delta_rho.min(maxdensity - rho); // prevent stepping to rho > maxdensity
        rho += delta_rho;

        if dpdrho.to_reduced().abs() < abstol {
            return Ok((p, rho));
        }
    }
    Err(EosError::NotConverged("pressure_spinodal".to_owned()))
}
