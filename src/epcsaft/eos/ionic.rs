use crate::epcsaft::eos::permittivity::Permittivity;
use crate::epcsaft::parameters::ElectrolytePcSaftParameters;
use feos_core::StateHD;
use ndarray::*;
use num_dual::DualNum;
use std::f64::consts::PI;
use std::fmt;
use std::sync::Arc;

use super::ElectrolytePcSaftVariants;

impl ElectrolytePcSaftParameters {
    pub fn bjerrum_length<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
        epcsaft_variant: ElectrolytePcSaftVariants,
    ) -> D {
        // permittivity in vacuum
        let epsilon_0 = 8.85416e-12;

        // relative permittivity of water (usually function of T,p,x)
        let epsilon_r = Permittivity::new(state, self, &epcsaft_variant)
            .unwrap()
            .permittivity;

        let epsreps0 = epsilon_r * epsilon_0;

        // unit charge
        let qe2 = 1.602176634e-19f64.powi(2);

        // Boltzmann constant
        let boltzmann = 1.380649e-23;

        // Bjerrum length
        (state.temperature * 4.0 * std::f64::consts::PI * epsreps0 * boltzmann).recip()
            * qe2
            * 1.0e10
    }
}

pub struct Ionic {
    pub parameters: Arc<ElectrolytePcSaftParameters>,
    pub variant: ElectrolytePcSaftVariants,
}

impl Ionic {
    pub fn helmholtz_energy<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
        diameter: &Array1<D>,
    ) -> D {
        // Extract parameters
        let p = &self.parameters;

        // Calculate Bjerrum length
        let lambda_b = p.bjerrum_length(state, self.variant);

        // Calculate inverse Debye length
        let mut sum_dens_z = D::zero();
        for i in 0..state.molefracs.len() {
            //let ai = p.ionic_comp[i];
            sum_dens_z += state.partial_density[i] * p.z[i].powi(2);
        }

        let kappa = (lambda_b * sum_dens_z * 4.0 * PI).sqrt();

        let chi: Array1<D> = diameter
            .iter()
            .map(|&d| {
                (kappa * d).powi(3).recip()
                    * ((kappa * d + 1.0).ln() - (kappa * d + 1.0) * 2.0
                        + (kappa * d + 1.0).powi(2) * 0.5
                        + 1.5)
            })
            .collect();

        let mut sum_x_z_chi = D::zero();
        for i in 0..state.molefracs.len() {
            sum_x_z_chi += chi[i] * state.molefracs[i] * p.z[i].powi(2);
        }

        // Set to zero if one of the ions is 0
        let sum_mole_fraction: f64 = p.ionic_comp.iter().map(|&i| state.molefracs[i].re()).sum();
        let mut a_ion = -kappa * lambda_b * sum_x_z_chi * state.moles.sum();
        if sum_mole_fraction == 0. {
            a_ion = D::zero();
        }

        a_ion
    }
}

impl fmt::Display for Ionic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ionic")
    }
}
