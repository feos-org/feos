use super::PropertyAD;
use crate::DensityInitialization::Liquid;
use crate::{FeosResult, Residual, State};
use nalgebra::DefaultAllocator;
use nalgebra::allocator::Allocator;
use num_dual::{DualNum, DualStruct, Gradients};
use quantity::{_MolarEntropy, KELVIN, MolarEntropy, PASCAL, Pressure, Temperature};

/// Liquid residual isobaric heat capacity of a pure component as function of temperature
/// and pressure.
pub struct ResidualIsobaricHeatCapacity(pub Temperature, pub Pressure);

impl<'a> From<&'a [f64]> for ResidualIsobaricHeatCapacity {
    fn from(value: &'a [f64]) -> Self {
        Self(value[0] * KELVIN, value[1] * PASCAL)
    }
}

impl<N: Gradients> PropertyAD<N> for ResidualIsobaricHeatCapacity
where
    DefaultAllocator: Allocator<N>,
{
    type Unit = _MolarEntropy;
    const REFERENCE: MolarEntropy = MolarEntropy::new(1.0);

    fn evaluate<E: Residual<N, D>, D: DualNum<f64, Inner = f64> + Copy>(
        &self,
        eos: &E,
    ) -> FeosResult<MolarEntropy<D>> {
        let x = E::pure_molefracs();
        let t = Temperature::from_inner(&self.0);
        let p = Pressure::from_inner(&self.1);
        let state = State::new_npt(eos, t, p, x, Some(Liquid))?;
        Ok(state.residual_molar_isobaric_heat_capacity())
    }
}
