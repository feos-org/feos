use std::sync::Arc;
pub(crate) mod attractive_perturbation;
pub(crate) mod hard_sphere;
pub(crate) mod reference_perturbation;

use crate::uvtheory::UVTheoryParameters;
use attractive_perturbation::AttractivePerturbation;
use feos_core::StateHD;
use hard_sphere::HardSphere;
use num_dual::DualNum;
use reference_perturbation::ReferencePerturbation;

pub(super) struct BarkerHenderson {
    hard_sphere: HardSphere,
    reference_perturbation: ReferencePerturbation,
    attractive_perturbation: AttractivePerturbation,
}

impl BarkerHenderson {
    pub fn new(parameters: Arc<UVTheoryParameters>) -> Self {
        let hard_sphere = HardSphere {
            parameters: parameters.clone(),
        };
        let reference_perturbation = ReferencePerturbation {
            parameters: parameters.clone(),
        };
        let attractive_perturbation = AttractivePerturbation {
            parameters: parameters.clone(),
        };
        Self {
            hard_sphere,
            reference_perturbation,
            attractive_perturbation,
        }
    }
}

impl BarkerHenderson {
    pub fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)> {
        vec![
            (
                "Hard Sphere (BH)".to_string(),
                self.hard_sphere.helmholtz_energy(state),
            ),
            (
                "Reference Perturbation (BH)".to_string(),
                self.reference_perturbation.helmholtz_energy(state),
            ),
            (
                "Attractive Perturbation (BH)".to_string(),
                self.attractive_perturbation.helmholtz_energy(state),
            ),
        ]
    }
}
