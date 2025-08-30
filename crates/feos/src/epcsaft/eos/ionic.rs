use crate::epcsaft::eos::permittivity::Permittivity;
use crate::epcsaft::parameters::ElectrolytePcSaftPars;
use feos_core::StateHD;
use nalgebra::DVector;
use num_dual::DualNum;
use std::f64::consts::PI;

use super::ElectrolytePcSaftVariants;

const EPSILON_0: f64 = 8.85416e-12;
const QE: f64 = 1.602176634e-19f64;
const BOLTZMANN: f64 = 1.380649e-23;

impl ElectrolytePcSaftPars {
    pub fn bjerrum_length<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
        epcsaft_variant: ElectrolytePcSaftVariants,
    ) -> D {
        // relative permittivity of water (usually function of T,p,x)
        let epsilon_r = Permittivity::new(state, self, &epcsaft_variant)
            .unwrap()
            .permittivity;

        let epsreps0 = epsilon_r * EPSILON_0;

        let qe2 = QE.powi(2);

        // Bjerrum length
        (state.temperature * 4.0 * std::f64::consts::PI * epsreps0 * BOLTZMANN).recip()
            * qe2
            * 1.0e10
    }
}

pub struct Ionic;

impl Ionic {
    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        parameters: &ElectrolytePcSaftPars,
        state: &StateHD<D>,
        diameter: &DVector<D>,
        variant: ElectrolytePcSaftVariants,
    ) -> D {
        // Extract parameters
        let p = parameters;

        // Set to zero if one of the ions is 0
        let sum_mole_fraction: f64 = p.ionic_comp.iter().map(|&i| state.molefracs[i].re()).sum();
        if sum_mole_fraction == 0. {
            return D::zero();
        }

        // Calculate Bjerrum length
        let lambda_b = p.bjerrum_length(state, variant);

        // Calculate inverse Debye length
        let mut sum_dens_z = D::zero();
        for i in 0..state.molefracs.len() {
            //let ai = p.ionic_comp[i];
            sum_dens_z += state.partial_density[i] * p.z[i].powi(2);
        }

        let kappa = (lambda_b * sum_dens_z * 4.0 * PI).sqrt();

        let chi: Vec<D> = diameter
            .iter()
            .map(|&d| {
                (kappa * d).powi(3).recip()
                    * ((kappa * d + 1.0).ln() - (kappa * d + 1.0) * 2.0
                        + (kappa * d + 1.0).powi(2) * 0.5
                        + 1.5)
            })
            .collect();

        let mut sum_rho_z_chi = D::zero();
        for (i, chi) in chi.into_iter().enumerate() {
            sum_rho_z_chi += chi * state.partial_density[i] * p.z[i].powi(2);
        }

        -kappa * lambda_b * sum_rho_z_chi
    }
}

#[cfg(test)]
mod tests {
    use super::ElectrolytePcSaftVariants::Advanced;
    use super::*;
    use crate::epcsaft::parameters::utils::{water_nacl_parameters, water_nacl_parameters_perturb};
    use crate::hard_sphere::HardSphereProperties;
    use approx::assert_relative_eq;
    use nalgebra::dvector;

    #[test]
    fn helmholtz_energy_perturb() {
        let p = ElectrolytePcSaftPars::new(&water_nacl_parameters_perturb()).unwrap();
        let t = 298.0;
        let v = 31.875;

        let s = StateHD::new(t, v, &dvector![0.9, 0.05, 0.05]);

        let d = p.hs_diameter(t);
        let a_rust = Ionic.helmholtz_energy_density(&p, &s, &d, Advanced) * v;

        assert_relative_eq!(a_rust, -0.07775796084032328, epsilon = 1e-10);
    }

    #[test]
    fn helmholtz_energy() {
        let p = ElectrolytePcSaftPars::new(&water_nacl_parameters()).unwrap();
        let t = 298.0;
        let v = 31.875;
        let s = StateHD::new(t, v, &dvector![0.9, 0.05, 0.05]);
        let d = p.hs_diameter(t);
        let a_rust = Ionic.helmholtz_energy_density(&p, &s, &d, Advanced) * v;

        assert_relative_eq!(a_rust, -0.07341337106244776, epsilon = 1e-10);
    }
}
