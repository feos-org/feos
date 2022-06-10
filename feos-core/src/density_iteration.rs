use crate::equation_of_state::EquationOfState;
use crate::errors::{EosError, EosResult};
use crate::state::State;
use crate::EosUnit;
use quantity::{QuantityArray1, QuantityScalar};
use std::rc::Rc;

pub struct SpinodalPoint<U: EosUnit> {
    pub p: QuantityScalar<U>,
    pub dp_drho: QuantityScalar<U>,
    pub rho: QuantityScalar<U>,
}

pub fn density_iteration<U: EosUnit, E: EquationOfState>(
    eos: &Rc<E>,
    temperature: QuantityScalar<U>,
    pressure: QuantityScalar<U>,
    moles: &QuantityArray1<U>,
    initial_density: QuantityScalar<U>,
) -> EosResult<State<U, E>> {
    let maxdensity = eos.max_density(Some(moles))?;
    let (abstol, reltol) = (1e-12, 1e-14);
    let n = moles.sum();

    let mut rho = initial_density;
    if rho <= 0.0 * U::reference_density() {
        return Err(EosError::InvalidState(
            String::from("density iteration"),
            String::from("density"),
            rho.to_reduced(U::reference_density())?,
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
                (1.1 * initial_density).min(maxdensity)?
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
        delta_rho = delta_rho.max(-0.95 * rho)?; // prevent stepping to rho < 0.0

        // correction for instable region
        if dp_drho.is_sign_negative() && k < maxiter {
            let d2pdrho2 = State::new_nvt(eos, temperature, n / rho, moles)?
                .d2pdrho2()
                .2;

            if rho > 0.85 * maxdensity {
                let sp = pressure_spinodal(eos, temperature, initial_density, moles)?;
                rho = sp.rho;
                error = sp.p - pressure;
                if rho > 0.85 * maxdensity {
                    if error.is_sign_negative() {
                        return Err(EosError::IterationFailed(String::from("density_iteration")));
                    } else {
                        rho = rho * 0.98
                    }
                } else if error.is_sign_positive() {
                    rho = 0.001 * maxdensity
                } else {
                    rho = (rho * 1.1).min(maxdensity)?
                }
            } else if error.is_sign_positive() && d2pdrho2.is_sign_positive() {
                let sp = pressure_spinodal(eos, temperature, initial_density, moles)?;
                rho = sp.rho;
                error = sp.p - pressure;
                if error.is_sign_positive() {
                    rho = 0.001 * maxdensity
                } else {
                    rho = (rho * 1.1).min(maxdensity)?
                }
            } else if error.is_sign_negative() && d2pdrho2.is_sign_negative() {
                let sp = pressure_spinodal(eos, temperature, initial_density, moles)?;
                rho = sp.rho;
                error = sp.p - pressure;
                if error.is_sign_negative() {
                    rho = 0.8 * maxdensity
                } else {
                    rho = rho * 0.8
                }
            } else if error.is_sign_negative() && d2pdrho2.is_sign_positive() {
                let sp_l = pressure_spinodal(eos, temperature, 0.8 * maxdensity, moles)?;
                let rho_l = sp_l.rho;
                let sp_v = pressure_spinodal(eos, temperature, 0.001 * maxdensity, moles)?;
                let rho_v = sp_v.rho;
                error = sp_v.p - pressure;
                if error.is_sign_positive()
                    && (initial_density - rho_v).abs() < (initial_density - rho_l).abs()
                {
                    rho = 0.8 * rho_v
                } else {
                    rho = (rho_l * 1.1).min(maxdensity)?
                }
            } else if error.is_sign_positive() && d2pdrho2.is_sign_negative() {
                let sp_l = pressure_spinodal(eos, temperature, 0.8 * maxdensity, moles)?;
                let rho_l = sp_l.rho;
                let sp_v = pressure_spinodal(eos, temperature, 0.001 * maxdensity, moles)?;
                let rho_v = sp_v.rho;
                error = sp_v.p - pressure;
                if error.is_sign_negative()
                    && (initial_density - rho_v).abs() > (initial_density - rho_l).abs()
                {
                    rho = (rho_l * 1.1).min(maxdensity)?
                } else {
                    rho = 0.8 * rho_v
                }
            } else {
                rho = (rho + initial_density) * 0.5;
                if (rho - initial_density)
                    .to_reduced(U::reference_density())?
                    .abs()
                    < 1e-8
                {
                    rho = (rho + 0.1 * maxdensity).min(maxdensity)?
                }
            }
            continue 'iteration;
        }
        // Newton step
        rho += delta_rho;
        if error.to_reduced(U::reference_pressure())?.abs()
            < f64::max(abstol, (rho * reltol).to_reduced(U::reference_density())?)
        {
            break 'iteration;
        }
    }
    if iterations == maxiter + 1 {
        Err(EosError::NotConverged("density_iteration".to_owned()))
    } else {
        Ok(State::new_nvt(eos, temperature, n / rho, moles)?)
    }
}

pub fn pressure_spinodal<U: EosUnit, E: EquationOfState>(
    eos: &Rc<E>,
    temperature: QuantityScalar<U>,
    rho_init: QuantityScalar<U>,
    moles: &QuantityArray1<U>,
) -> EosResult<SpinodalPoint<U>> {
    let maxiter = 30;
    let abstol = 1e-8;

    let maxdensity = eos.max_density(Some(moles))?;
    let n = moles.sum();
    let mut rho = rho_init;

    if rho <= 0.0 * U::reference_density() {
        return Err(EosError::InvalidState(
            String::from("pressure spinodal"),
            String::from("density"),
            rho.to_reduced(U::reference_density())?,
        ));
    }

    for _ in 0..maxiter {
        let (p, dpdrho, d2pdrho2) = State::new_nvt(eos, temperature, n / rho, moles)?.d2pdrho2();

        let mut delta_rho = -dpdrho / d2pdrho2;
        if delta_rho.abs() > 0.05 * maxdensity {
            delta_rho = 0.05 * maxdensity * delta_rho.signum()
        }
        delta_rho = delta_rho.max(-rho * 0.95)?; // prevent stepping to rho < 0.0
        delta_rho = delta_rho.min(maxdensity - rho)?; // prevent stepping to rho > maxdensity
        rho += delta_rho;

        if dpdrho
            .to_reduced(U::reference_pressure() / U::reference_density())?
            .abs()
            < abstol
        {
            return Ok(SpinodalPoint {
                p,
                dp_drho: dpdrho,
                rho,
            });
        }
    }
    Err(EosError::NotConverged("pressure_spinodal".to_owned()))
}
