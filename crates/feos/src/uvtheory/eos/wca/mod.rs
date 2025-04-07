use std::sync::Arc;
mod attractive_perturbation;
mod attractive_perturbation_uvb3;
mod hard_sphere;
mod reference_perturbation;
mod reference_perturbation_uvb3;

use crate::uvtheory::UVTheoryParameters;
use attractive_perturbation::AttractivePerturbation;
use attractive_perturbation_uvb3::AttractivePerturbationB3;
use feos_core::StateHD;
use hard_sphere::HardSphere;
use num_dual::DualNum;
use reference_perturbation::ReferencePerturbation;
use reference_perturbation_uvb3::ReferencePerturbationB3;

pub(super) struct WeeksChandlerAndersen {
    hard_sphere: HardSphere,
    reference_perturbation: ReferencePerturbation,
    attractive_perturbation: AttractivePerturbation,
}

impl WeeksChandlerAndersen {
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

impl WeeksChandlerAndersen {
    pub fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)> {
        vec![
            (
                "Hard Sphere (WCA)".to_string(),
                self.hard_sphere.helmholtz_energy(state),
            ),
            (
                "Reference Perturbation (WCA)".to_string(),
                self.reference_perturbation.helmholtz_energy(state),
            ),
            (
                "Attractive Perturbation (WCA)".to_string(),
                self.attractive_perturbation.helmholtz_energy(state),
            ),
        ]
    }
}

pub(super) struct WeeksChandlerAndersenB3 {
    hard_sphere: HardSphere,
    reference_perturbation: ReferencePerturbationB3,
    attractive_perturbation: AttractivePerturbationB3,
}

impl WeeksChandlerAndersenB3 {
    pub fn new(parameters: Arc<UVTheoryParameters>) -> Self {
        let hard_sphere = HardSphere {
            parameters: parameters.clone(),
        };
        let reference_perturbation = ReferencePerturbationB3 {
            parameters: parameters.clone(),
        };
        let attractive_perturbation = AttractivePerturbationB3 {
            parameters: parameters.clone(),
        };
        Self {
            hard_sphere,
            reference_perturbation,
            attractive_perturbation,
        }
    }
}

impl WeeksChandlerAndersenB3 {
    pub fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)> {
        vec![
            (
                "Hard Sphere (WCA)".to_string(),
                self.hard_sphere.helmholtz_energy(state),
            ),
            (
                "Reference Perturbation (WCA B3)".to_string(),
                self.reference_perturbation.helmholtz_energy(state),
            ),
            (
                "Attractive Perturbation (WCA B3)".to_string(),
                self.attractive_perturbation.helmholtz_energy(state),
            ),
        ]
    }
}
