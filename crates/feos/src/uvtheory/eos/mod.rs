use super::parameters::{UVTheoryParameters, UVTheoryPars};
use feos_core::{Molarweight, ResidualDyn, Subset};
use nalgebra::DVector;
use num_dual::DualNum;
use quantity::MolarWeight;
use std::f64::consts::FRAC_PI_6;

mod bh;
pub use bh::BarkerHenderson;
mod wca;
pub use wca::{WeeksChandlerAndersen, WeeksChandlerAndersenB3};

/// Type of perturbation.
#[derive(Clone, Copy, PartialEq, Debug)]
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

/// uv-theory equation of state
pub struct UVTheory {
    pub parameters: UVTheoryParameters,
    params: UVTheoryPars,
    options: UVTheoryOptions,
}

impl UVTheory {
    /// uv-theory with default options (WCA).
    pub fn new(parameters: UVTheoryParameters) -> Self {
        Self::with_options(parameters, UVTheoryOptions::default())
    }

    /// uv-theory with provided options.
    pub fn with_options(parameters: UVTheoryParameters, options: UVTheoryOptions) -> Self {
        let params = UVTheoryPars::new(&parameters, options.perturbation);

        Self {
            parameters,
            params,
            options,
        }
    }
}

impl Subset for UVTheory {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(self.parameters.subset(component_list), self.options.clone())
    }
}

impl ResidualDyn for UVTheory {
    fn components(&self) -> usize {
        self.parameters.pure.len()
    }

    fn compute_max_density<D: DualNum<f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        let sigma3 = self.params.sigma.map(|v| v.powi(3));
        (sigma3.map(D::from).dot(molefracs) * FRAC_PI_6).recip() * self.options.max_eta
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &feos_core::StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        match &self.options.perturbation {
            Perturbation::BarkerHenderson => {
                BarkerHenderson.residual_helmholtz_energy_contributions(&self.params, state)
            }
            Perturbation::WeeksChandlerAndersen => {
                WeeksChandlerAndersen.residual_helmholtz_energy_contributions(&self.params, state)
            }
            Perturbation::WeeksChandlerAndersenB3 => {
                WeeksChandlerAndersenB3.residual_helmholtz_energy_contributions(&self.params, state)
            }
        }
    }
}

impl Molarweight for UVTheory {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

#[cfg(test)]
#[expect(clippy::excessive_precision)]
mod test {
    use super::*;
    use crate::uvtheory::parameters::utils::{new_simple, test_parameters_mixture};
    use crate::uvtheory::parameters::*;
    use approx::assert_relative_eq;
    use feos_core::parameter::{Identifier, PureRecord};
    use feos_core::{FeosResult, State};
    use nalgebra::dvector;
    use quantity::{ANGSTROM, KELVIN, MOL, NAV, RGAS};

    #[test]
    fn helmholtz_energy_pure_wca() -> FeosResult<()> {
        let sig = 3.7039;
        let eps_k = 150.03;
        let parameters = new_simple(24.0, 6.0, sig, eps_k);
        let eos = &UVTheory::new(parameters);

        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let temperature = reduced_temperature * eps_k * KELVIN;
        let moles = dvector![2.0] * MOL;
        let volume = (sig * ANGSTROM).powi::<3>() / reduced_density * NAV * 2.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles).unwrap();
        let a = (s.residual_molar_helmholtz_energy() / (RGAS * temperature)).into_value();
        assert_relative_eq!(a, 2.972986567516, max_relative = 1e-12); //wca
        Ok(())
    }

    #[test]
    fn helmholtz_energy_pure_bh() -> FeosResult<()> {
        let eps_k = 150.03;
        let sig = 3.7039;
        let rep = 24.0;
        let att = 6.0;
        let parameters = new_simple(rep, att, sig, eps_k);
        let options = UVTheoryOptions {
            max_eta: 0.5,
            perturbation: Perturbation::BarkerHenderson,
        };
        let eos = &UVTheory::with_options(parameters, options);

        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let temperature = reduced_temperature * eps_k * KELVIN;
        let moles = dvector![2.0] * MOL;
        let volume = (sig * ANGSTROM).powi::<3>() / reduced_density * NAV * 2.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles).unwrap();

        let a = (s.residual_molar_helmholtz_energy() / (RGAS * temperature)).into_value();

        assert_relative_eq!(a, 2.993577305779432, max_relative = 1e-12);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_pure_uvb3() -> FeosResult<()> {
        let eps_k = 150.03;
        let sig = 3.7039;
        let rep = 12.0;
        let att = 6.0;
        let parameters = new_simple(rep, att, sig, eps_k);
        let options = UVTheoryOptions {
            max_eta: 0.5,
            perturbation: Perturbation::WeeksChandlerAndersenB3,
        };
        let eos = &UVTheory::with_options(parameters, options);

        let reduced_temperature = 4.0;
        let reduced_density = 0.5;
        let temperature = reduced_temperature * eps_k * KELVIN;
        let moles = dvector![2.0] * MOL;
        let volume = (sig * ANGSTROM).powi::<3>() / reduced_density * NAV * 2.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles).unwrap();
        let a = (s.residual_molar_helmholtz_energy() / (RGAS * temperature)).into_value();
        dbg!(a);
        assert_relative_eq!(a, 0.37659379124271003, max_relative = 1e-12);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_mixtures_bh() -> FeosResult<()> {
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
        let uv_parameters = UVTheoryParameters::new_binary([pr1, pr2], None, vec![])?;
        // state
        let reduced_temperature = 4.0;
        let eps_k_x = (eps_k1 + eps_k2) / 2.0; // Check rule!!
        let t_x = reduced_temperature * eps_k_x * KELVIN;
        let sig_x = (sig1 + sig2) / 2.0; // Check rule!!
        let reduced_density = 1.0;
        let moles = dvector![1.7, 0.3] * MOL;
        let total_moles = moles.sum();
        let volume = (sig_x * ANGSTROM).powi::<3>() / reduced_density * NAV * total_moles;

        // EoS
        let options = UVTheoryOptions {
            max_eta: 0.5,
            perturbation: Perturbation::BarkerHenderson,
        };

        let eos_bh = &UVTheory::with_options(uv_parameters, options);

        let state_bh = State::new_nvt(&eos_bh, t_x, volume, &moles).unwrap();
        let a_bh = (state_bh.residual_molar_helmholtz_energy() / (RGAS * t_x)).into_value();

        assert_relative_eq!(a_bh, 2.993577305779432, max_relative = 1e-12);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_wca_mixture() -> FeosResult<()> {
        let parameters = test_parameters_mixture(
            dvector![12.0, 12.0],
            dvector![6.0, 6.0],
            dvector![1.0, 1.0],
            dvector![1.0, 0.5],
        );
        let p = UVTheoryPars::new(&parameters, Perturbation::WeeksChandlerAndersen);

        // state
        let reduced_temperature = 1.0;
        let t_x = reduced_temperature * p.epsilon_k[0] * KELVIN;
        let reduced_density = 0.9;
        let moles = dvector![0.4, 0.6] * MOL;
        let total_moles = moles.sum();
        let volume = (p.sigma[0] * ANGSTROM).powi::<3>() / reduced_density * NAV * total_moles;

        // EoS
        let eos_wca = &UVTheory::new(parameters);
        let state_wca = State::new_nvt(&eos_wca, t_x, volume, &moles).unwrap();
        let a_wca = (state_wca.residual_helmholtz_energy() / (RGAS * t_x * state_wca.total_moles))
            .into_value();

        assert_relative_eq!(a_wca, -0.597791038364405, max_relative = 1e-5);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_wca_mixture_different_sigma() -> FeosResult<()> {
        let parameters = test_parameters_mixture(
            dvector![12.0, 12.0],
            dvector![6.0, 6.0],
            dvector![1.0, 2.0],
            dvector![1.0, 0.5],
        );
        let p = UVTheoryPars::new(&parameters, Perturbation::WeeksChandlerAndersen);

        // state
        let reduced_temperature = 1.5;
        let t_x = reduced_temperature * p.epsilon_k[0] * KELVIN;
        let sigma_x_3 = (0.4 + 0.6 * 8.0) * ANGSTROM.powi::<3>();
        let density = 0.52000000000000002 / sigma_x_3;
        let moles = dvector![0.4, 0.6] * MOL;
        let total_moles = moles.sum();
        let volume = NAV * total_moles / density;

        // EoS
        let eos_wca = &UVTheory::new(parameters);
        let state_wca = State::new_nvt(&eos_wca, t_x, volume, &moles).unwrap();
        let a_wca = (state_wca.residual_molar_helmholtz_energy() / (RGAS * t_x)).into_value();
        assert_relative_eq!(a_wca, -0.034206207363139396, max_relative = 1e-5);
        Ok(())
    }
}
