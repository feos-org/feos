use super::Property;
use crate::ad::Gradient;
use crate::{FeosResult, PhaseEquilibrium, ReferenceSystem, Residual};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, U1};
use num_dual::{DualNum, DualStruct, Gradients, first_derivative, partial2};
use quantity::{_Temperature, KELVIN, PASCAL, Pressure, Temperature};

/// Boiling temperature of a pure component as function of pressure.
pub struct BoilingTemperature(pub Pressure);

impl<'a> From<&'a [f64]> for BoilingTemperature {
    fn from(value: &'a [f64]) -> Self {
        Self(value[0] * PASCAL)
    }
}

impl<N: Gradients> Property<N> for BoilingTemperature
where
    DefaultAllocator: Allocator<N> + Allocator<U1, N> + Allocator<N, N>,
{
    type Unit = _Temperature;
    const REFERENCE: Temperature = KELVIN;

    fn evaluate<E: Residual<N, D>, D: DualNum<f64, Inner = f64> + Copy>(
        &self,
        eos: &E,
    ) -> FeosResult<Temperature<D>> {
        let p = Pressure::from_inner(&self.0);
        PhaseEquilibrium::pure_p(eos, p, None, Default::default()).map(|(t, _)| t)
    }

    fn evaluate_gradient<E: Residual<N, Gradient<P>>, const P: usize>(
        &self,
        eos: &E,
    ) -> FeosResult<quantity::Quantity<Gradient<P>, Self::Unit>> {
        let eos_f64 = eos.re();
        let (temperature, [vapor_density, liquid_density]) =
            PhaseEquilibrium::pure_p(&eos_f64, self.0, None, Default::default())?;

        let t = temperature.into_reduced();
        let v1 = 1.0 / liquid_density.to_reduced();
        let v2 = 1.0 / vapor_density.to_reduced();
        let p = self.0.into_reduced();
        let t = Gradient::from(t);
        let t = t + {
            let v1 = Gradient::from(v1);
            let v2 = Gradient::from(v2);
            let p = Gradient::from(p);
            let x = E::pure_molefracs();

            let residual_entropy = |v| {
                let (a, s) = first_derivative(
                    partial2(
                        |t, &v, x| eos.lift().residual_helmholtz_energy(t, v, x),
                        &v,
                        &x,
                    ),
                    t,
                );
                (a, -s)
            };
            let (a1, s1) = residual_entropy(v1);
            let (a2, s2) = residual_entropy(v2);

            let ln_rho = (v1 / v2).ln();
            (p * (v2 - v1) + (a2 - a1 + t * ln_rho)) / (s2 - s1 - ln_rho)
        };
        Ok(Temperature::from_reduced(t))
    }
}
