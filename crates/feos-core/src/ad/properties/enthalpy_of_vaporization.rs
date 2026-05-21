use super::Property;
use crate::{FeosResult, PhaseEquilibrium, Residual};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, U1};
use num_dual::{DualNum, DualStruct, Gradients};
use quantity::{_MolarEnergy, KELVIN, MolarEnergy, Temperature};

/// Enthalpy of vaporization of a pure component as function of temperature.
pub struct EnthalpyOfVaporization(pub Temperature);

impl<'a> From<&'a [f64]> for EnthalpyOfVaporization {
    fn from(value: &'a [f64]) -> Self {
        Self(value[0] * KELVIN)
    }
}

impl<N: Gradients> Property<N> for EnthalpyOfVaporization
where
    DefaultAllocator: Allocator<N> + Allocator<U1, N> + Allocator<N, N>,
{
    type Unit = _MolarEnergy;
    const REFERENCE: MolarEnergy = MolarEnergy::new(1.0);

    fn evaluate<E: Residual<N, D>, D: DualNum<f64, Inner = f64> + Copy>(
        &self,
        eos: &E,
    ) -> FeosResult<MolarEnergy<D>> {
        let t = Temperature::from_inner(&self.0);
        let vle = PhaseEquilibrium::pure(eos, t, None, Default::default())?;
        let h_v = vle.vapor().residual_molar_enthalpy();
        let h_l = vle.liquid().residual_molar_enthalpy();
        Ok(h_v - h_l)
    }
}
