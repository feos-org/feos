use std::ops::Deref;

use nalgebra::{Const, U1};
use num_dual::{Derivative, DualNum, DualSVec, DualStruct};

// mod phase_equilibria;
// mod residual;
// mod state;
// mod total;
// pub use phase_equilibria::PhaseEquilibriumAD;
// pub use residual::ResidualHelmholtzEnergy;
// pub use state::{PhaseEquilibriumAD, StateAD};
// pub use total::{EquationOfStateAD, IdealGasAD, TotalHelmholtzEnergy};

/// A model that can be evaluated with derivatives of its parameters.
pub trait ParametersAD: Sized + Deref<Target = Self::Parameters<f64>> {
    /// The type of the structure that stores the parameters internally.
    type Parameters<D: DualNum<f64> + Copy>: DualStruct<D, f64> + Clone;

    // /// Return a reference to the parameters.
    // fn params(&self) -> &Self::Parameters<f64>;

    /// Lift the parameters to the given type of dual number.
    fn params_from_inner<D: DualNum<f64> + Copy, D2: DualNum<f64, Inner = D> + Copy>(
        parameters: &Self::Parameters<D>,
    ) -> Self::Parameters<D2>;

    /// Wraps the model in the [HelmholtzEnergyWrapper] struct, so that it can be used
    /// as an argument to [StateAD](crate::StateAD) and [PhaseEquilibriumAD](crate::PhaseEquilibriumAD) constructors.
    fn wrap<'a, const N: usize>(&'a self) -> HelmholtzEnergyWrapper<'a, Self, f64, N> {
        HelmholtzEnergyWrapper {
            eos: self,
            parameters: self,
        }
    }

    /// Manually set the parameters and their derivatives.
    fn derivatives<'a, D: DualNum<f64> + Copy, const N: usize>(
        &'a self,
        parameters: &'a Self::Parameters<D>,
    ) -> HelmholtzEnergyWrapper<'a, Self, D, N> {
        HelmholtzEnergyWrapper {
            eos: self,
            parameters,
        }
    }

    /// Manually set the parameters and their derivatives.
    fn derivatives2<D: DualNum<f64> + Copy>(parameters: Self::Parameters<D>) -> Self;
}

/// Struct that stores the reference to the equation of state together with the (possibly) dual parameters.
#[derive(Copy)]
pub struct HelmholtzEnergyWrapper<'a, E: ParametersAD, D: DualNum<f64> + Copy, const N: usize> {
    pub eos: &'a E,
    pub parameters: &'a E::Parameters<D>,
}

impl<'a, E: ParametersAD, D: DualNum<f64> + Copy, const N: usize> Clone
    for HelmholtzEnergyWrapper<'a, E, D, N>
{
    fn clone(&self) -> Self {
        Self {
            eos: self.eos,
            parameters: self.parameters,
        }
    }
}

/// Models for which derivatives with respect to individual parameters can be calculated.
pub trait NamedParameters: ParametersAD + for<'a> From<&'a [f64]> {
    /// Return a mutable reference to the parameter named by `index` from the parameter set.
    fn index_parameters_mut<'a, D: DualNum<f64> + Copy>(
        parameters: &'a mut Self::Parameters<D>,
        index: &str,
    ) -> &'a mut D;

    /// Return the parameters with the appropriate derivatives.
    fn named_derivatives<const P: usize>(
        &self,
        parameters: [&str; P],
    ) -> Self::Parameters<DualSVec<f64, f64, P>> {
        let mut params: Self::Parameters<DualSVec<f64, f64, P>> = Self::params_from_inner(self);
        for (i, p) in parameters.into_iter().enumerate() {
            Self::index_parameters_mut(&mut params, p).eps =
                Derivative::derivative_generic(Const::<P>, U1, i)
        }
        params
    }
}
