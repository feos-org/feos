use super::Property;
use crate::DensityInitialization::Liquid;
use crate::density_iteration::density_iteration;
use crate::{FeosResult, Residual};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim};
use num_dual::{DualNum, DualStruct};
use quantity::{_Density, Density, KELVIN, PASCAL, Pressure, Temperature};

/// Liquid density of a pure component as function of temperature and pressure.
pub struct LiquidDensity(pub Temperature, pub Pressure);

impl<'a> From<&'a [f64]> for LiquidDensity {
    fn from(value: &'a [f64]) -> Self {
        Self(value[0] * KELVIN, value[1] * PASCAL)
    }
}

impl<N: Dim> Property<N> for LiquidDensity
where
    DefaultAllocator: Allocator<N>,
{
    type Unit = _Density;
    const REFERENCE: Density = Density::new(1000.0);

    fn evaluate<E: Residual<N, D>, D: DualNum<f64, Inner = f64> + Copy>(
        &self,
        eos: &E,
    ) -> FeosResult<Density<D>> {
        let x = E::pure_molefracs();
        let t = Temperature::from_inner(&self.0);
        let p = Pressure::from_inner(&self.1);
        density_iteration(eos, t, p, &x, Some(Liquid))
    }
}
