use super::ElectrolytePcSaftVariants;
use crate::epcsaft::eos::permittivity::Permittivity;
use crate::epcsaft::parameters::ElectrolytePcSaftPars;
use feos_core::StateHD;
use nalgebra::DVector;
use num_dual::DualNum;

pub struct Born;

impl Born {
    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        parameters: &ElectrolytePcSaftPars,
        state: &StateHD<D>,
        diameter: &DVector<D>,
    ) -> D {
        // Parameters
        let p = parameters;

        // Calculate Bjerrum length
        let lambda_b = p.bjerrum_length(state, ElectrolytePcSaftVariants::Advanced);

        // Calculate relative permittivity
        let epsilon_r = Permittivity::new(state, p, &ElectrolytePcSaftVariants::Advanced)
            .unwrap()
            .permittivity;

        // Calculate sum rhoi zi^2 / di
        let mut sum_rhoi_zi_ai = D::zero();
        for i in 0..state.molefracs.len() {
            sum_rhoi_zi_ai += state.partial_density[i] * p.z[i].powi(2) / diameter[i];
        }

        // Calculate born contribution
        -lambda_b * (epsilon_r - 1.) * sum_rhoi_zi_ai
    }
}

#[cfg(test)]
mod tests {
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
        let a_rust = Born.helmholtz_energy_density(&p, &s, &d) * v;

        assert_relative_eq!(a_rust, -22.51064553710294, epsilon = 1e-10);
    }

    #[test]
    fn helmholtz_energy() {
        let p = ElectrolytePcSaftPars::new(&water_nacl_parameters()).unwrap();

        let t = 298.0;
        let v = 31.875;
        let s = StateHD::new(t, v, &dvector![0.9, 0.05, 0.05]);
        let d = p.hs_diameter(t);
        let a_rust = Born.helmholtz_energy_density(&p, &s, &d) * v;

        assert_relative_eq!(a_rust, -22.525624511559244, epsilon = 1e-10);
    }
}
