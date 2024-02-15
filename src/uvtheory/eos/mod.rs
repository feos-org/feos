#![allow(clippy::excessive_precision)]
#![allow(clippy::needless_range_loop)]

use self::bh::BarkerHenderson;
use self::wca::{WeeksChandlerAndersen, WeeksChandlerAndersenB3};

use super::parameters::UVTheoryParameters;
use feos_core::si::{MolarWeight, GRAM, MOL};
use feos_core::{parameter::Parameter, Components, Residual};
use ndarray::Array1;
use std::f64::consts::FRAC_PI_6;
use std::sync::Arc;

mod bh;
mod wca;

/// Type of perturbation.
#[derive(Clone)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub enum Perturbation {
    BarkerHenderson,
    WeeksChandlerAndersen,
    WeeksChandlerAndersenB3,
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

/// Collection of utility structs for each implementation
enum UVTheoryImplementation {
    BarkerHenderson(BarkerHenderson),
    WeeksChandlerAndersen(WeeksChandlerAndersen),
    WeeksChandlerAndersenB3(WeeksChandlerAndersenB3),
}

/// uv-theory equation of state
pub struct UVTheory {
    parameters: Arc<UVTheoryParameters>,
    options: UVTheoryOptions,
    implementation: UVTheoryImplementation,
}

impl UVTheory {
    /// uv-theory with default options (WCA).
    pub fn new(parameters: Arc<UVTheoryParameters>) -> Self {
        Self::with_options(parameters, UVTheoryOptions::default())
    }

    /// uv-theory with provided options.
    pub fn with_options(parameters: Arc<UVTheoryParameters>, options: UVTheoryOptions) -> Self {
        let implementation = match options.perturbation {
            Perturbation::BarkerHenderson => {
                UVTheoryImplementation::BarkerHenderson(BarkerHenderson::new(parameters.clone()))
            }
            Perturbation::WeeksChandlerAndersen => UVTheoryImplementation::WeeksChandlerAndersen(
                WeeksChandlerAndersen::new(parameters.clone()),
            ),
            Perturbation::WeeksChandlerAndersenB3 => {
                UVTheoryImplementation::WeeksChandlerAndersenB3(WeeksChandlerAndersenB3::new(
                    parameters.clone(),
                ))
            }
        };

        Self {
            parameters,
            options,
            implementation,
        }
    }
}

impl Components for UVTheory {
    fn components(&self) -> usize {
        self.parameters.pure_records.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            Arc::new(self.parameters.subset(component_list)),
            self.options.clone(),
        )
    }
}

impl Residual for UVTheory {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * self.parameters.sigma.mapv(|v| v.powi(3)) * moles).sum()
    }

    fn residual_helmholtz_energy_contributions<D: num_dual::DualNum<f64> + Copy>(
        &self,
        state: &feos_core::StateHD<D>,
    ) -> Vec<(String, D)> {
        match &self.implementation {
            UVTheoryImplementation::BarkerHenderson(a) => {
                a.residual_helmholtz_energy_contributions(state)
            }
            UVTheoryImplementation::WeeksChandlerAndersen(a) => {
                a.residual_helmholtz_energy_contributions(state)
            }
            UVTheoryImplementation::WeeksChandlerAndersenB3(a) => {
                a.residual_helmholtz_energy_contributions(state)
            }
        }
    }

    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::uvtheory::parameters::utils::test_parameters_mixture;
    use crate::uvtheory::parameters::*;
    use approx::assert_relative_eq;
    use feos_core::parameter::{Identifier, Parameter, PureRecord};
    use feos_core::si::{ANGSTROM, KELVIN, MOL, NAV, RGAS};
    use feos_core::{EosResult, State};
    use ndarray::arr1;
    use typenum::P3;

    #[test]
    fn helmholtz_energy_pure_wca() -> EosResult<()> {
        let sig = 3.7039;
        let eps_k = 150.03;
        let parameters = UVTheoryParameters::new_simple(24.0, 6.0, sig, eps_k)?;
        let eos = Arc::new(UVTheory::new(Arc::new(parameters)));

        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let temperature = reduced_temperature * eps_k * KELVIN;
        let moles = arr1(&[2.0]) * MOL;
        let volume = (sig * ANGSTROM).powi::<P3>() / reduced_density * NAV * 2.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles).unwrap();
        let a = (s.residual_molar_helmholtz_energy() / (RGAS * temperature)).into_value();
        assert_relative_eq!(a, 2.972986567516, max_relative = 1e-12); //wca
        Ok(())
    }

    #[test]
    fn helmholtz_energy_pure_bh() -> EosResult<()> {
        let eps_k = 150.03;
        let sig = 3.7039;
        let rep = 24.0;
        let att = 6.0;
        let parameters = UVTheoryParameters::new_simple(rep, att, sig, eps_k)?;
        let options = UVTheoryOptions {
            max_eta: 0.5,
            perturbation: Perturbation::BarkerHenderson,
        };
        let eos = Arc::new(UVTheory::with_options(Arc::new(parameters), options));

        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let temperature = reduced_temperature * eps_k * KELVIN;
        let moles = arr1(&[2.0]) * MOL;
        let volume = (sig * ANGSTROM).powi::<P3>() / reduced_density * NAV * 2.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles).unwrap();

        let a = (s.residual_molar_helmholtz_energy() / (RGAS * temperature)).into_value();

        assert_relative_eq!(a, 2.993577305779432, max_relative = 1e-12);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_pure_uvb3() -> EosResult<()> {
        let eps_k = 150.03;
        let sig = 3.7039;
        let rep = 12.0;
        let att = 6.0;
        let parameters = UVTheoryParameters::new_simple(rep, att, sig, eps_k)?;
        let options = UVTheoryOptions {
            max_eta: 0.5,
            perturbation: Perturbation::WeeksChandlerAndersenB3,
        };
        let eos = Arc::new(UVTheory::with_options(Arc::new(parameters), options));

        let reduced_temperature = 4.0;
        let reduced_density = 0.5;
        let temperature = reduced_temperature * eps_k * KELVIN;
        let moles = arr1(&[2.0]) * MOL;
        let volume = (sig * ANGSTROM).powi::<P3>() / reduced_density * NAV * 2.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles).unwrap();
        let a = (s.residual_molar_helmholtz_energy() / (RGAS * temperature)).into_value();
        dbg!(a);
        assert_relative_eq!(a, 0.37659379124271003, max_relative = 1e-12);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_mixtures_bh() -> EosResult<()> {
        // Mixture of equal components --> result must be the same as for pure fluid ///
        // component 1
        let rep1 = 24.0;
        let eps_k1 = 150.03;
        let sig1 = 3.7039;
        let r1 = UVTheoryRecord::new(rep1, 6.0, sig1, eps_k1);
        let i = Identifier::new(None, None, None, None, None, None);
        // compontent 2
        let rep2 = 24.0;
        let eps_k2 = 150.03;
        let sig2 = 3.7039;
        let r2 = UVTheoryRecord::new(rep2, 6.0, sig2, eps_k2);
        let j = Identifier::new(None, None, None, None, None, None);
        //////////////

        let pr1 = PureRecord::new(i, 1.0, r1);
        let pr2 = PureRecord::new(j, 1.0, r2);
        let pure_records = vec![pr1, pr2];
        let uv_parameters = UVTheoryParameters::new_binary(pure_records, None)?;
        // state
        let reduced_temperature = 4.0;
        let eps_k_x = (eps_k1 + eps_k2) / 2.0; // Check rule!!
        let t_x = reduced_temperature * eps_k_x * KELVIN;
        let sig_x = (sig1 + sig2) / 2.0; // Check rule!!
        let reduced_density = 1.0;
        let moles = arr1(&[1.7, 0.3]) * MOL;
        let total_moles = moles.sum();
        let volume = (sig_x * ANGSTROM).powi::<P3>() / reduced_density * NAV * total_moles;

        // EoS
        let options = UVTheoryOptions {
            max_eta: 0.5,
            perturbation: Perturbation::BarkerHenderson,
        };

        let eos_bh = Arc::new(UVTheory::with_options(Arc::new(uv_parameters), options));

        let state_bh = State::new_nvt(&eos_bh, t_x, volume, &moles).unwrap();
        let a_bh = (state_bh.residual_molar_helmholtz_energy() / (RGAS * t_x)).into_value();

        assert_relative_eq!(a_bh, 2.993577305779432, max_relative = 1e-12);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_wca_mixture() -> EosResult<()> {
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
        let volume = (p.sigma[0] * ANGSTROM).powi::<P3>() / reduced_density * NAV * total_moles;

        // EoS
        let eos_wca = Arc::new(UVTheory::new(Arc::new(p)));
        let state_wca = State::new_nvt(&eos_wca, t_x, volume, &moles).unwrap();
        let a_wca = (state_wca.residual_helmholtz_energy() / (RGAS * t_x * state_wca.total_moles))
            .into_value();

        assert_relative_eq!(a_wca, -0.597791038364405, max_relative = 1e-5);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_wca_mixture_different_sigma() -> EosResult<()> {
        let p = test_parameters_mixture(
            arr1(&[12.0, 12.0]),
            arr1(&[6.0, 6.0]),
            arr1(&[1.0, 2.0]),
            arr1(&[1.0, 0.5]),
        );

        // state
        let reduced_temperature = 1.5;
        let t_x = reduced_temperature * p.epsilon_k[0] * KELVIN;
        let sigma_x_3 = (0.4 + 0.6 * 8.0) * ANGSTROM.powi::<P3>();
        let density = 0.52000000000000002 / sigma_x_3;
        let moles = arr1(&[0.4, 0.6]) * MOL;
        let total_moles = moles.sum();
        let volume = NAV * total_moles / density;

        // EoS
        let eos_wca = Arc::new(UVTheory::new(Arc::new(p)));
        let state_wca = State::new_nvt(&eos_wca, t_x, volume, &moles).unwrap();
        let a_wca = (state_wca.residual_molar_helmholtz_energy() / (RGAS * t_x)).into_value();
        assert_relative_eq!(a_wca, -0.034206207363139396, max_relative = 1e-5);
        Ok(())
    }
}
