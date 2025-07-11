use crate::errors::{FeosError, FeosResult};
use crate::{DensityInitialization, ReferenceSystem, Residual, StateHD};
use ndarray::Array1;
use num_dual::{
    Dual, Dual2, Dual3, DualNum, first_derivative, second_derivative, third_derivative,
};

pub trait DensityIteration<M> {
    fn compute_max_density2(&self, moles: &M) -> f64;

    fn residual_molar_helmholtz_energy<D: DualNum<f64> + Copy>(
        &self,
        temperature: D,
        molar_volume: D,
        molefracs: &M,
    ) -> D;

    fn pressure(&self, temperature: f64, density: f64, molefracs: &M) -> (f64, f64) {
        let molar_volume = density.recip();
        let t = Dual::from_re(temperature);
        let (a_res, da) = first_derivative(
            |molar_volume| self.residual_molar_helmholtz_energy(t, molar_volume, molefracs),
            molar_volume,
        );
        (a_res, -da + temperature * density)
    }

    fn p_dpdrho(&self, temperature: f64, density: f64, molefracs: &M) -> (f64, f64, f64) {
        let molar_volume = density.recip();
        let t = Dual2::from_re(temperature);
        let (a_res, da, d2a) = second_derivative(
            |molar_volume| self.residual_molar_helmholtz_energy(t, molar_volume, molefracs),
            molar_volume,
        );
        (
            a_res,
            -da + temperature * density,
            molar_volume * molar_volume * d2a + temperature,
        )
    }

    fn p_dpdrho_d2pdrho2(&self, temperature: f64, density: f64, molefracs: &M) -> (f64, f64, f64) {
        let molar_volume = density.recip();
        let t = Dual3::from_re(temperature);
        let (_, da, d2a, d3a) = third_derivative(
            |molar_volume| self.residual_molar_helmholtz_energy(t, molar_volume, molefracs),
            molar_volume,
        );
        (
            -da + temperature * density,
            molar_volume * molar_volume * d2a + temperature,
            -molar_volume * molar_volume * molar_volume * (2.0 * d2a + molar_volume * d3a),
        )
    }

    fn residual_molar_gibbs_energy(&self, temperature: f64, density: f64, molefracs: &M) -> f64 {
        let molar_volume = density.recip();
        let t = Dual::from_re(temperature);
        let (a_res, da_res) = first_derivative(
            |molar_volume| self.residual_molar_helmholtz_energy(t, molar_volume, molefracs),
            molar_volume,
        );
        let z = 1.0 - da_res * molar_volume / temperature;
        a_res - da_res * molar_volume - temperature * z.ln()
    }

    fn density_iteration(
        &self,
        temperature: f64,
        pressure: f64,
        molefracs: &M,
        density_initialization: DensityInitialization,
    ) -> FeosResult<f64> {
        // calculate state from initial density or given phase
        match density_initialization {
            DensityInitialization::InitialDensity(rho0) => {
                return self._density_iteration(
                    temperature,
                    pressure,
                    molefracs,
                    rho0.into_reduced(),
                );
            }
            DensityInitialization::Vapor => {
                return self._density_iteration(
                    temperature,
                    pressure,
                    molefracs,
                    pressure / temperature,
                );
            }
            DensityInitialization::Liquid => {
                return self._density_iteration(
                    temperature,
                    pressure,
                    molefracs,
                    self.compute_max_density2(molefracs),
                );
            }
            DensityInitialization::None => (),
        };

        // calculate stable phase
        let max_density = self.compute_max_density2(molefracs);
        let liquid = self._density_iteration(temperature, pressure, molefracs, max_density);

        if pressure < max_density * temperature {
            let vapor =
                self._density_iteration(temperature, pressure, molefracs, pressure / temperature);
            match (&liquid, &vapor) {
                (Ok(_), Err(_)) => liquid,
                (Err(_), Ok(_)) => vapor,
                (Ok(l), Ok(v)) => {
                    if self.residual_molar_gibbs_energy(temperature, *l, molefracs)
                        > self.residual_molar_gibbs_energy(temperature, *v, molefracs)
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

    fn _density_iteration(
        &self,
        temperature: f64,
        pressure: f64,
        molefracs: &M,
        initial_density: f64,
    ) -> FeosResult<f64> {
        let maxdensity = self.compute_max_density2(molefracs);
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
            let (_, mut p, mut dp_drho) = self.p_dpdrho(temperature, rho, molefracs);

            // attempt to correct for poor initial density rho_init
            if dp_drho.is_sign_negative() && k == 0 {
                rho = if initial_density <= 0.15 * maxdensity {
                    0.05 * initial_density
                } else {
                    (1.1 * initial_density).min(maxdensity)
                };
                let p_ = self.p_dpdrho(temperature, rho, molefracs);
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
                let d2pdrho2 = self.p_dpdrho_d2pdrho2(temperature, rho, molefracs).2;

                if rho > 0.85 * maxdensity {
                    let (sp_p, sp_rho) =
                        self.pressure_spinodal(temperature, initial_density, molefracs)?;
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
                        self.pressure_spinodal(temperature, initial_density, molefracs)?;
                    rho = sp_rho;
                    error = sp_p - pressure;
                    if error.is_sign_positive() {
                        rho = 0.001 * maxdensity
                    } else {
                        rho = (rho * 1.1).min(maxdensity)
                    }
                } else if error.is_sign_negative() && d2pdrho2.is_sign_negative() {
                    let (sp_p, sp_rho) =
                        self.pressure_spinodal(temperature, initial_density, molefracs)?;
                    rho = sp_rho;
                    error = sp_p - pressure;
                    if error.is_sign_negative() {
                        rho = 0.8 * maxdensity
                    } else {
                        rho *= 0.8
                    }
                } else if error.is_sign_negative() && d2pdrho2.is_sign_positive() {
                    let (_, rho_l) =
                        self.pressure_spinodal(temperature, 0.8 * maxdensity, molefracs)?;
                    let (sp_v_p, rho_v) =
                        self.pressure_spinodal(temperature, 0.001 * maxdensity, molefracs)?;
                    error = sp_v_p - pressure;
                    if error.is_sign_positive()
                        && (initial_density - rho_v).abs() < (initial_density - rho_l).abs()
                    {
                        rho = 0.8 * rho_v
                    } else {
                        rho = (rho_l * 1.1).min(maxdensity)
                    }
                } else if error.is_sign_positive() && d2pdrho2.is_sign_negative() {
                    let (_, rho_l) =
                        self.pressure_spinodal(temperature, 0.8 * maxdensity, molefracs)?;
                    let (sp_v_p, rho_v) =
                        self.pressure_spinodal(temperature, 0.001 * maxdensity, molefracs)?;
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

    fn pressure_spinodal(
        &self,
        temperature: f64,
        rho_init: f64,
        molefracs: &M,
    ) -> FeosResult<(f64, f64)> {
        let maxiter = 30;
        let abstol = 1e-8;

        let maxdensity = self.compute_max_density2(molefracs);
        let mut rho = rho_init;

        if rho <= 0.0 {
            return Err(FeosError::InvalidState(
                String::from("pressure spinodal"),
                String::from("density"),
                rho,
            ));
        }

        for _ in 0..maxiter {
            let (p, dpdrho, d2pdrho2) = self.p_dpdrho_d2pdrho2(temperature, rho, molefracs);

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
}

impl<T: Residual> DensityIteration<Array1<f64>> for T {
    fn compute_max_density2(&self, molefracs: &Array1<f64>) -> f64 {
        self.compute_max_density(molefracs)
    }

    fn residual_molar_helmholtz_energy<D: DualNum<f64> + Copy>(
        &self,
        temperature: D,
        molar_volume: D,
        molefracs: &Array1<f64>,
    ) -> D {
        let x = molefracs.mapv(D::from);
        let state = StateHD::new(temperature, molar_volume, x);
        self.residual_helmholtz_energy(&state) * temperature
    }
}
