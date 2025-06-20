use feos_core::{Components, IdealGas, Residual, StateHD};
use nalgebra::{Const, SVector, U1};
use ndarray::{arr1, Array1, ScalarOperand};
use num_dual::{Derivative, DualNum, DualVec};
use std::sync::Arc;

mod phase_equilibria;
mod residual;
mod state;
mod total;
pub use phase_equilibria::PhaseEquilibriumAD;
pub use residual::{ParametersAD, ResidualHelmholtzEnergy};
pub use state::{Eigen, StateAD};
pub use total::{EquationOfStateAD, IdealGasAD, TotalHelmholtzEnergy};

/// Used internally to implement the [Residual] and [IdealGas] traits from FeOs.
pub struct FeOsWrapper<E, const N: usize>(E);

impl<R: ParametersAD, const N: usize> Components for FeOsWrapper<R, N> {
    fn components(&self) -> usize {
        N
    }

    fn subset(&self, _: &[usize]) -> Self {
        panic!("Calculating properties of subsets of models is not possible with AD.")
    }
}

impl<R: ResidualHelmholtzEnergy<N>, const N: usize> Residual for FeOsWrapper<R, N> {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        let total_moles = moles.sum();
        let molefracs = SVector::from_fn(|i, _| moles[i] / total_moles);
        self.0.compute_max_density(&molefracs)
    }

    fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)> {
        let temperature = state.temperature;
        let volume = state.volume;
        let density = SVector::from_column_slice(state.partial_density.as_slice().unwrap());
        let parameters = self.0.params();
        let a = R::residual_helmholtz_energy_density(&parameters, temperature, &density) * volume
            / temperature;
        vec![(R::RESIDUAL.into(), a)]
    }
}

impl<E: TotalHelmholtzEnergy<N>, const N: usize> IdealGas for FeOsWrapper<E, N> {
    fn ln_lambda3<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D> {
        let parameters = self.0.params();
        arr1(&E::ln_lambda3(&parameters, temperature).data.0[0])
    }

    fn ideal_gas_model(&self) -> String {
        E::IDEAL_GAS.into()
    }
}

/// Struct that stores the reference to the equation of state together with the (possibly) dual parameters.
pub struct HelmholtzEnergyWrapper<E: ParametersAD, D: DualNum<f64> + Copy, const N: usize> {
    pub eos: Arc<FeOsWrapper<E, N>>,
    pub parameters: E::Parameters<D>,
}

impl<E: ParametersAD, const N: usize> HelmholtzEnergyWrapper<E, f64, N> {
    /// Manually set the parameters and their derivatives.
    pub fn derivatives<D: DualNum<f64> + Copy>(
        &self,
        parameters: E::Parameters<D>,
    ) -> HelmholtzEnergyWrapper<E, D, N> {
        HelmholtzEnergyWrapper {
            eos: self.eos.clone(),
            parameters,
        }
    }
}

/// Models for which derivatives with respect to individual parameters can be calculated.
pub trait NamedParameters: ParametersAD {
    /// Return a mutable reference to the parameter named by `index` from the parameter set.
    fn index_parameters_mut<'a, D: DualNum<f64> + Copy>(
        parameters: &'a mut Self::Parameters<D>,
        index: &str,
    ) -> &'a mut D;
}

impl<E: NamedParameters, const N: usize> HelmholtzEnergyWrapper<E, f64, N> {
    /// Initialize the parameters to calculate their derivatives.
    pub fn named_derivatives<const P: usize>(
        &self,
        parameters: [&str; P],
    ) -> HelmholtzEnergyWrapper<E, DualVec<f64, f64, Const<P>>, N> {
        let mut params: E::Parameters<DualVec<f64, f64, Const<P>>> = self.eos.0.params();
        for (i, p) in parameters.into_iter().enumerate() {
            E::index_parameters_mut(&mut params, p).eps =
                Derivative::derivative_generic(Const::<P>, U1, i)
        }
        self.derivatives(params)
    }
}
