use super::SaftVRMieParameters;
use feos_core::{Components, Residual, StateHD};
use num_dual::DualNum;
use std::sync::Arc;

mod chain;
mod monomer;

pub struct SaftVRMie {
    parameters: Arc<SaftVRMieParameters>,
}

impl Components for SaftVRMie {
    fn components(&self) -> usize {
        unimplemented!()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        unimplemented!()
    }
}

impl Residual for SaftVRMie {
    fn compute_max_density(&self, moles: &ndarray::prelude::Array1<f64>) -> f64 {
        unimplemented!()
    }

    fn molar_weight(&self) -> feos_core::si::MolarWeight<ndarray::prelude::Array1<f64>> {
        unimplemented!()
    }

    fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)> {
        unimplemented!()
    }
}
