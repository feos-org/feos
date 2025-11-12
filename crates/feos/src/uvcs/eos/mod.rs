#![allow(clippy::excessive_precision)]
#![allow(clippy::needless_range_loop)]
use crate::uvcs::parameters::UVCSParameters;

use super::corresponding_states::CorrespondingParameters;
use super::parameters::UVCSPars;
use feos_core::{Molarweight, ResidualDyn, StateHD, Subset};
use nalgebra::DVector;
use quantity::MolarWeight;
use std::f64::consts::FRAC_PI_6;

mod attractive_perturbation;
mod hard_sphere;
mod reference_perturbation;
use attractive_perturbation::attractive_perturbation_helmholtz_energy_density;
use hard_sphere::hard_sphere_helmholtz_energy_density;
use reference_perturbation::reference_perturbation_helmholtz_energy_density;

/// uv-theory equation of state
pub struct UVCSTheory {
    pub parameters: UVCSParameters,
    pub params: UVCSPars,
    max_eta: f64,
}

impl UVCSTheory {
    /// uv-theory with default options (WCA).
    pub fn new(parameters: UVCSParameters) -> Self {
        Self::with_options(parameters, 0.5)
    }

    /// uv-theory with provided options.
    pub fn with_options(parameters: UVCSParameters, max_eta: f64) -> Self {
        let params = UVCSPars::new(&parameters);
        Self {
            parameters,
            params,
            max_eta,
        }
    }
}

impl Subset for UVCSTheory {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(self.parameters.subset(component_list), self.max_eta)
    }
}

impl ResidualDyn for UVCSTheory {
    fn components(&self) -> usize {
        self.parameters.pure.len()
    }

    fn compute_max_density<D: num_dual::DualNum<f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        (self.params.sigma.map(|v| D::from(v.powi(3))).dot(molefracs) * FRAC_PI_6).recip()
            * self.max_eta
    }

    fn reduced_helmholtz_energy_density_contributions<D: num_dual::DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        let parameters = CorrespondingParameters::new(&self.params, state.temperature);
        vec![
            (
                "Hard Sphere (WCA)",
                hard_sphere_helmholtz_energy_density(&parameters, state),
            ),
            (
                "Reference Perturbation (WCA)",
                reference_perturbation_helmholtz_energy_density(&parameters, state),
            ),
            (
                "Attractive Perturbation (WCA)",
                attractive_perturbation_helmholtz_energy_density(&parameters, state),
            ),
        ]
    }
}

impl Molarweight for UVCSTheory {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use super::*;

    use crate::uvcs::parameters::utils::test_parameters_mixture;
    use crate::uvcs::parameters::*;
    use approx::assert_relative_eq;
    use feos_core::{
        FeosResult, State,
        parameter::{Identifier, PureRecord},
    };
    use nalgebra::dvector;
    use quantity::{ANGSTROM, KELVIN, MOL, NAV, RGAS};
    use typenum::P3;

    #[test]
    fn helmholtz_energy_pure_wca() -> FeosResult<()> {
        let sig = 3.7039;
        let eps_k = 150.03;
        let pure_record = PureRecord::new(
            Identifier::default(),
            1.0,
            UVCSRecord {
                rep: 24.0,
                att: 6.0,
                sigma: sig,
                epsilon_k: eps_k,
                quantum_correction: None,
            },
        );
        let parameters = UVCSParameters::new_pure(pure_record)?;
        let eos = Arc::new(UVCSTheory::new(parameters));

        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let temperature = reduced_temperature * eps_k * KELVIN;
        let moles = dvector![2.0] * MOL;
        let volume = (sig * ANGSTROM).powi::<P3>() / reduced_density * NAV * 2.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles).unwrap();
        let a = (s.residual_molar_helmholtz_energy() / (RGAS * temperature)).into_value();
        assert_relative_eq!(a, 2.972986567516, max_relative = 1e-12); //wca
        Ok(())
    }

    #[test]
    fn helmholtz_energy_wca_mixture() -> FeosResult<()> {
        let _p = test_parameters_mixture(
            dvector![12.0, 12.0],
            dvector![6.0, 6.0],
            dvector![1.0, 1.0],
            dvector![1.0, 0.5],
        );
        let p = UVCSPars::new(&_p);

        // state
        let reduced_temperature = 1.0;
        let t_x = reduced_temperature * p.epsilon_k[0] * KELVIN;
        let reduced_density = 0.9;
        let moles = dvector![0.4, 0.6] * MOL;
        let total_moles = moles.sum();
        let volume = (p.sigma[0] * ANGSTROM).powi::<P3>() / reduced_density * NAV * total_moles;

        // EoS
        let eos_wca = Arc::new(UVCSTheory::new(_p));
        let state_wca = State::new_nvt(&eos_wca, t_x, volume, &moles).unwrap();
        let a_wca = (state_wca.residual_helmholtz_energy() / (RGAS * t_x * state_wca.total_moles))
            .into_value();

        assert_relative_eq!(a_wca, -0.597791038364405, max_relative = 1e-5);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_wca_mixture_different_sigma() -> FeosResult<()> {
        let _p = test_parameters_mixture(
            dvector![12.0, 12.0],
            dvector![6.0, 6.0],
            dvector![1.0, 2.0],
            dvector![1.0, 0.5],
        );
        let p = UVCSPars::new(&_p);

        // state
        let reduced_temperature = 1.5;
        let t_x = reduced_temperature * p.epsilon_k[0] * KELVIN;
        let sigma_x_3 = (0.4 + 0.6 * 8.0) * ANGSTROM.powi::<P3>();
        let density = 0.52000000000000002 / sigma_x_3;
        let moles = dvector![0.4, 0.6] * MOL;
        let total_moles = moles.sum();
        let volume = NAV * total_moles / density;

        // EoS
        let eos_wca = Arc::new(UVCSTheory::new(_p));
        let state_wca = State::new_nvt(&eos_wca, t_x, volume, &moles).unwrap();
        let a_wca = (state_wca.residual_molar_helmholtz_energy() / (RGAS * t_x)).into_value();
        assert_relative_eq!(a_wca, -0.034206207363139396, max_relative = 1e-5);
        Ok(())
    }
}
