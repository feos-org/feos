use super::Property;
use crate::{FeosResult, PhaseEquilibrium, Residual};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, U1};
use num_dual::{DualNum, DualStruct, Gradients};
use quantity::{_Density, Density, KELVIN, Temperature};

/// Equilibrium liquid density of a pure component as function of temperature.
pub struct EquilibriumLiquidDensity(pub Temperature);

impl<'a> From<&'a [f64]> for EquilibriumLiquidDensity {
    fn from(value: &'a [f64]) -> Self {
        Self(value[0] * KELVIN)
    }
}

impl<N: Gradients> Property<N> for EquilibriumLiquidDensity
where
    DefaultAllocator: Allocator<N> + Allocator<U1, N> + Allocator<N, N>,
{
    type Unit = _Density;
    const REFERENCE: Density = Density::new(1000.0);

    fn evaluate<E: Residual<N, D>, D: DualNum<f64, Inner = f64> + Copy>(
        &self,
        eos: &E,
    ) -> FeosResult<Density<D>> {
        let t = Temperature::from_inner(&self.0);
        PhaseEquilibrium::pure_t(eos, t, None, Default::default()).map(|(_, [_, r])| r)
    }
}
