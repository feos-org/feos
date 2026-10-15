use crate::solver::LocalFunction;
use crate::weight_functions::WeightFunctionInfo;
use feos_core::{FeosError, FeosResult};
use ndarray::prelude::*;
use num_dual::*;

/// Individual functional contribution that can be evaluated using generalized (hyper) dual numbers.
pub trait FunctionalContribution: Sync + Send {
    /// Return the name of the contribution.
    fn name(&self) -> &'static str;

    /// Return the weight functions required in this contribution.
    fn weight_functions<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
    ) -> WeightFunctionInfo<N>;

    /// Return the Helmholtz energy density for the given temperature and weighted densities.
    fn helmholtz_energy_density<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> FeosResult<Array1<N>>;

    fn bulk_helmholtz_energy_density<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
        density: &Array1<N>,
    ) -> N {
        let weighted_densities = self
            .weight_functions(temperature)
            .weight_constants()
            .dot(density)
            .insert_axis(Axis(1));
        self.helmholtz_energy_density(temperature, weighted_densities.view())
            .unwrap()[0]
    }
}

/// The Helmholtz energy density of a functional contribution as local function of its weighted
/// densities, which the tape differentiates.
impl<C: FunctionalContribution> LocalFunction for C {
    type Error = FeosError;

    fn evaluate<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> FeosResult<Array2<N>> {
        Ok(self
            .helmholtz_energy_density(temperature, weighted_densities)?
            .insert_axis(Axis(0)))
    }
}
