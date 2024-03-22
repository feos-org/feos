use crate::epcsaft::eos::permittivity::Permittivity;
use crate::epcsaft::parameters::ElectrolytePcSaftParameters;
use feos_core::StateHD;
use ndarray::Array1;
use num_dual::DualNum;
use std::fmt;
use std::sync::Arc;

use super::ElectrolytePcSaftVariants;

pub struct Born {
    pub parameters: Arc<ElectrolytePcSaftParameters>,
}

impl Born {
    pub fn helmholtz_energy<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
        diameter: &Array1<D>,
    ) -> D {
        // Parameters
        let p = &self.parameters;

        // Calculate Bjerrum length
        let lambda_b = p.bjerrum_length(state, ElectrolytePcSaftVariants::Advanced);

        // Calculate relative permittivity
        let epsilon_r = Permittivity::new(
            state,
            &self.parameters,
            &ElectrolytePcSaftVariants::Advanced,
        )
        .unwrap()
        .permittivity;

        // Calculate sum xi zi^2 / di
        let mut sum_xi_zi_ai = D::zero();
        for i in 0..state.molefracs.len() {
            sum_xi_zi_ai += state.molefracs[i] * p.z[i].powi(2) / diameter[i];
        }

        // Calculate born contribution
        -lambda_b * (epsilon_r - 1.) * sum_xi_zi_ai * state.moles.sum()
    }
}

impl fmt::Display for Born {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Born")
    }
}
