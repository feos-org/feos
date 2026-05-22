use super::PropertyAD;
use crate::ad::Gradient;
use crate::{FeosResult, PhaseEquilibrium, ReferenceSystem, Residual};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, U1};
use num_dual::{DualNum, DualStruct, Gradients};
use quantity::{_Pressure, KELVIN, PASCAL, Pressure, Temperature};

/// Vapor pressure of a pure component as function of temperature.
pub struct VaporPressure(pub Temperature);

impl<'a> From<&'a [f64]> for VaporPressure {
    fn from(value: &'a [f64]) -> Self {
        Self(value[0] * KELVIN)
    }
}

impl<N: Gradients> PropertyAD<N> for VaporPressure
where
    DefaultAllocator: Allocator<N> + Allocator<U1, N> + Allocator<N, N>,
{
    type Unit = _Pressure;
    const REFERENCE: Pressure = PASCAL;

    fn evaluate<E: Residual<N, D>, D: DualNum<f64, Inner = f64> + Copy>(
        &self,
        eos: &E,
    ) -> FeosResult<Pressure<D>> {
        let t = Temperature::from_inner(&self.0);
        PhaseEquilibrium::pure_t(eos, t, None, Default::default()).map(|(p, _)| p)
    }

    fn evaluate_gradient<E: Residual<N, Gradient<P>>, const P: usize>(
        &self,
        eos: &E,
    ) -> FeosResult<Pressure<Gradient<P>>> {
        let eos_f64 = eos.re();
        let (_, [vapor_density, liquid_density]) =
            PhaseEquilibrium::pure_t(&eos_f64, self.0, None, Default::default())?;

        // implicit differentiation is implemented here instead of just calling pure_t with dual
        // numbers, because for the first derivative, we can avoid calculating density derivatives.
        let v1 = 1.0 / liquid_density.to_reduced();
        let v2 = 1.0 / vapor_density.to_reduced();
        let t = self.0.into_reduced();
        let (a1, a2) = {
            let t = Gradient::from(t);
            let v1 = Gradient::from(v1);
            let v2 = Gradient::from(v2);
            let x = E::pure_molefracs();

            let a1 = eos.residual_helmholtz_energy(t, v1, &x);
            let a2 = eos.residual_helmholtz_energy(t, v2, &x);
            (a1, a2)
        };

        let p = -(a1 - a2 + t * (v2 / v1).ln()) / (v1 - v2);
        Ok(Pressure::from_reduced(p))
    }
}
