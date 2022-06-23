#![allow(clippy::excessive_precision)]
#![allow(clippy::needless_range_loop)]

use super::parameters::UVParameters;
use feos_core::{parameter::Parameter, EquationOfState, HelmholtzEnergy};
use ndarray::Array1;
use std::f64::consts::FRAC_PI_6;
use std::rc::Rc;

pub(crate) mod attractive_perturbation_bh;
pub(crate) mod attractive_perturbation_wca;
pub(crate) mod hard_sphere_bh;
pub(crate) mod hard_sphere_wca;
pub(crate) mod reference_perturbation_bh;
pub(crate) mod reference_perturbation_wca;
pub(crate) mod ufraction;
use attractive_perturbation_bh::AttractivePerturbationBH;
use attractive_perturbation_wca::AttractivePerturbationWCA;
use hard_sphere_bh::HardSphere;
use hard_sphere_wca::HardSphereWCA;
use reference_perturbation_bh::ReferencePerturbationBH;
use reference_perturbation_wca::ReferencePerturbationWCA;

/// Type of perturbation.
#[derive(Clone)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub enum Perturbation {
    BarkerHenderson,
    WeeksChandlerAndersen,
}

/// Configuration options for uv-theory
#[derive(Clone)]
pub struct UVTheoryOptions {
    pub max_eta: f64,
    pub perturbation: Perturbation,
}

impl Default for UVTheoryOptions {
    fn default() -> Self {
        Self {
            max_eta: 0.5,
            perturbation: Perturbation::WeeksChandlerAndersen,
        }
    }
}

/// uv-theory equation of state
pub struct UVTheory {
    parameters: Rc<UVParameters>,
    options: UVTheoryOptions,
    contributions: Vec<Box<dyn HelmholtzEnergy>>,
}

impl UVTheory {
    /// uv-theory with default options (WCA).
    pub fn new(parameters: Rc<UVParameters>) -> Self {
        Self::with_options(parameters, UVTheoryOptions::default())
    }

    /// uv-theory with provided options.
    pub fn with_options(parameters: Rc<UVParameters>, options: UVTheoryOptions) -> Self {
        let mut contributions: Vec<Box<dyn HelmholtzEnergy>> = Vec::with_capacity(3);

        match options.perturbation {
            Perturbation::BarkerHenderson => {
                contributions.push(Box::new(HardSphere {
                    parameters: parameters.clone(),
                }));
                contributions.push(Box::new(ReferencePerturbationBH {
                    parameters: parameters.clone(),
                }));
                contributions.push(Box::new(AttractivePerturbationBH {
                    parameters: parameters.clone(),
                }));
            }
            Perturbation::WeeksChandlerAndersen => {
                contributions.push(Box::new(HardSphereWCA {
                    parameters: parameters.clone(),
                }));
                contributions.push(Box::new(ReferencePerturbationWCA {
                    parameters: parameters.clone(),
                }));
                contributions.push(Box::new(AttractivePerturbationWCA {
                    parameters: parameters.clone(),
                }));
            }
        }

        Self {
            parameters,
            options,
            contributions,
        }
    }
}

impl EquationOfState for UVTheory {
    fn components(&self) -> usize {
        self.parameters.pure_records.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            Rc::new(self.parameters.subset(component_list)),
            self.options.clone(),
        )
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * self.parameters.sigma.mapv(|v| v.powi(3)) * moles).sum()
    }

    fn residual(&self) -> &[Box<dyn HelmholtzEnergy>] {
        &self.contributions
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::uvtheory::parameters::utils::test_parameters_mixture;
    use crate::uvtheory::parameters::*;
    use approx::assert_relative_eq;
    use feos_core::parameter::{Identifier, Parameter, PureRecord};
    use feos_core::{Contributions, State};
    use ndarray::arr1;
    use quantity::si::{ANGSTROM, KELVIN, MOL, NAV, RGAS};

    #[test]
    fn helmholtz_energy_pure_wca() {
        let eps_k = 150.03;
        let sig = 3.7039;
        let r = UVRecord::new(24.0, 6.0, sig, eps_k);
        //let r = UVRecord::new(12.0, 6.0, sig, eps_k);
        let i = Identifier::new(None, None, None, None, None, None);
        let pr = PureRecord::new(i, 1.0, r, None);
        let parameters = UVParameters::new_pure(pr);
        let eos = Rc::new(UVTheory::new(Rc::new(parameters)));

        let reduced_temperature = 4.0;
        //let reduced_temperature = 1.0;
        let reduced_density = 1.0;
        //let reduced_density = 0.9;
        let temperature = reduced_temperature * eps_k * KELVIN;
        let moles = arr1(&[2.0]) * MOL;
        let volume = (sig * ANGSTROM).powi(3) / reduced_density * NAV * 2.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles).unwrap();
        let a = s
            .molar_helmholtz_energy(Contributions::ResidualNvt)
            .to_reduced(RGAS * temperature)
            .unwrap();
        assert_relative_eq!(a, 2.972986567516, max_relative = 1e-12) //wca
                                                                     //assert_relative_eq!(a, -7.0843443690046017, max_relative = 1e-12) // bh: assert_relative_eq!(a, 2.993577305779432, max_relative = 1e-12)
    }
    #[test]
    fn helmholtz_energy_pure_bh() {
        let eps_k = 150.03;
        let sig = 3.7039;
        let r = UVRecord::new(24.0, 6.0, sig, eps_k);
        let i = Identifier::new(None, None, None, None, None, None);
        let pr = PureRecord::new(i, 1.0, r, None);
        let parameters = UVParameters::new_pure(pr);

        let options = UVTheoryOptions {
            max_eta: 0.5,
            perturbation: Perturbation::BarkerHenderson,
        };
        let eos = Rc::new(UVTheory::with_options(Rc::new(parameters), options));

        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let temperature = reduced_temperature * eps_k * KELVIN;
        let moles = arr1(&[2.0]) * MOL;
        let volume = (sig * ANGSTROM).powi(3) / reduced_density * NAV * 2.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles).unwrap();
        let a = s
            .molar_helmholtz_energy(Contributions::ResidualNvt)
            .to_reduced(RGAS * temperature)
            .unwrap();

        assert_relative_eq!(a, 2.993577305779432, max_relative = 1e-12)
    }

    #[test]
    fn helmholtz_energy_mixtures_bh() {
        // Mixture of equal components --> result must be the same as fpr pure fluid ///
        // component 1
        let rep1 = 24.0;
        let eps_k1 = 150.03;
        let sig1 = 3.7039;
        let r1 = UVRecord::new(rep1, 6.0, sig1, eps_k1);
        let i = Identifier::new(None, None, None, None, None, None);
        // compontent 2
        let rep2 = 24.0;
        let eps_k2 = 150.03;
        let sig2 = 3.7039;
        let r2 = UVRecord::new(rep2, 6.0, sig2, eps_k2);
        let j = Identifier::new(None, None, None, None, None, None);
        //////////////

        let pr1 = PureRecord::new(i, 1.0, r1, None);
        let pr2 = PureRecord::new(j, 1.0, r2, None);
        let pure_records = vec![pr1, pr2];
        let uv_parameters = UVParameters::new_binary(pure_records, None);
        // state
        let reduced_temperature = 4.0;
        let eps_k_x = (eps_k1 + eps_k2) / 2.0; // Check rule!!
        let t_x = reduced_temperature * eps_k_x * KELVIN;
        let sig_x = (sig1 + sig2) / 2.0; // Check rule!!
        let reduced_density = 1.0;
        let moles = arr1(&[1.7, 0.3]) * MOL;
        let total_moles = moles.sum();
        let volume = (sig_x * ANGSTROM).powi(3) / reduced_density * NAV * total_moles;

        // EoS
        let options = UVTheoryOptions {
            max_eta: 0.5,
            perturbation: Perturbation::BarkerHenderson,
        };

        let eos_bh = Rc::new(UVTheory::with_options(Rc::new(uv_parameters), options));

        let state_bh = State::new_nvt(&eos_bh, t_x, volume, &moles).unwrap();
        let a_bh = state_bh
            .molar_helmholtz_energy(Contributions::ResidualNvt)
            .to_reduced(RGAS * t_x)
            .unwrap();

        assert_relative_eq!(a_bh, 2.993577305779432, max_relative = 1e-12);
    }
    #[test]
    fn helmholtz_energy_wca_mixture() {
        let p = test_parameters_mixture(
            arr1(&[12.0, 12.0]),
            arr1(&[6.0, 6.0]),
            arr1(&[1.0, 1.0]),
            arr1(&[1.0, 0.5]),
        );

        // state
        let reduced_temperature = 1.0;
        let t_x = reduced_temperature * p.epsilon_k[0] * KELVIN;
        let reduced_density = 0.9;
        let moles = arr1(&[0.4, 0.6]) * MOL;
        let total_moles = moles.sum();
        let volume = (p.sigma[0] * ANGSTROM).powi(3) / reduced_density * NAV * total_moles;

        // EoS
        let eos_wca = Rc::new(UVTheory::new(Rc::new(p)));
        let state_wca = State::new_nvt(&eos_wca, t_x, volume, &moles).unwrap();
        let a_wca = state_wca
            .molar_helmholtz_energy(Contributions::ResidualNvt)
            .to_reduced(RGAS * t_x)
            .unwrap();

        assert_relative_eq!(a_wca, -0.597791038364405, max_relative = 1e-5)
    }

    #[test]
    fn helmholtz_energy_wca_mixture_different_sigma() {
        let p = test_parameters_mixture(
            arr1(&[12.0, 12.0]),
            arr1(&[6.0, 6.0]),
            arr1(&[1.0, 2.0]),
            arr1(&[1.0, 0.5]),
        );

        // state
        let reduced_temperature = 1.5;
        let t_x = reduced_temperature * p.epsilon_k[0] * KELVIN;
        let sigma_x_3 = (0.4 + 0.6 * 8.0) * ANGSTROM.powi(3);
        let density = 0.52000000000000002 / sigma_x_3;
        let moles = arr1(&[0.4, 0.6]) * MOL;
        let total_moles = moles.sum();
        let volume = NAV * total_moles / density;

        // EoS
        let eos_wca = Rc::new(UVTheory::new(Rc::new(p)));
        let state_wca = State::new_nvt(&eos_wca, t_x, volume, &moles).unwrap();
        let a_wca = state_wca
            .molar_helmholtz_energy(Contributions::ResidualNvt)
            .to_reduced(RGAS * t_x)
            .unwrap();
        assert_relative_eq!(a_wca, -0.034206207363139396, max_relative = 1e-5)
    }
}
