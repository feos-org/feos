use crate::hard_sphere::HardSphere;
use crate::uvtheory::parameters::UVTheoryPars;
use attractive_perturbation::AttractivePerturbation;
use feos_core::StateHD;
use num_dual::DualNum;
use reference_perturbation::ReferencePerturbation;

mod attractive_perturbation;
mod hard_sphere;
mod reference_perturbation;

pub struct BarkerHenderson;

impl BarkerHenderson {
    pub fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy>(
        &self,
        parameters: &UVTheoryPars,
        state: &StateHD<D>,
    ) -> Vec<(String, D)> {
        vec![
            (
                "Hard Sphere (BH)".to_string(),
                HardSphere.helmholtz_energy(parameters, state),
            ),
            (
                "Reference Perturbation (BH)".to_string(),
                ReferencePerturbation.helmholtz_energy(parameters, state),
            ),
            (
                "Attractive Perturbation (BH)".to_string(),
                AttractivePerturbation.helmholtz_energy(parameters, state),
            ),
        ]
    }
}
