use super::PropertyAD;
use crate::Contributions;
use crate::ad::Gradient;
use crate::{Composition, FeosResult, PhaseEquilibrium, ReferenceSystem, Residual};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, U1};
use num_dual::{DualNum, DualStruct, Gradients};
use quantity::{_Pressure, KELVIN, PASCAL, Pressure, Temperature};

/// Bubble point pressure of a binary mixture as function of temperature and
/// molefracs of the first component.
///
/// An initial value for the pressure can be passed as optional argument to
/// increase robustness and speed.
pub struct BubblePointPressure(pub Temperature, pub f64, pub Option<Pressure>);

impl<'a> From<&'a [f64]> for BubblePointPressure {
    fn from(value: &'a [f64]) -> Self {
        Self(value[0] * KELVIN, value[1], Some(value[2] * PASCAL))
    }
}

impl<N: Gradients> PropertyAD<N> for BubblePointPressure
where
    DefaultAllocator: Allocator<N> + Allocator<U1, N> + Allocator<N, N>,
    f64: Composition<f64, N>,
{
    type Unit = _Pressure;
    const REFERENCE: Pressure = PASCAL;

    fn evaluate<E: Residual<N, D>, D: DualNum<f64, Inner = f64> + Copy>(
        &self,
        eos: &E,
    ) -> FeosResult<Pressure<D>>
    where
        DefaultAllocator: Allocator<N> + Allocator<U1, N> + Allocator<N, N>,
    {
        let t = Temperature::from_inner(&self.0);
        let p = Option::from_inner(&self.2);
        let (x, _) = self.1.into_molefracs(&eos.re())?;
        let x = x.map(D::from);
        let vle = PhaseEquilibrium::bubble_point(eos, t, x, p, None, Default::default())?;
        Ok(vle.vapor().pressure(Contributions::Total))
    }

    fn evaluate_gradient<E: Residual<N, Gradient<P>>, const P: usize>(
        &self,
        eos: &E,
    ) -> FeosResult<quantity::Quantity<Gradient<P>, Self::Unit>>
    where
        DefaultAllocator: Allocator<N> + Allocator<U1, N> + Allocator<N, N>,
    {
        let eos_f64 = eos.re();
        let (liquid_molefracs, _) = self.1.into_molefracs(&eos_f64)?;
        let vle = PhaseEquilibrium::bubble_point(
            &eos_f64,
            self.0,
            &liquid_molefracs,
            self.2,
            None,
            Default::default(),
        )?;

        let v_l = 1.0 / vle.liquid().density.to_reduced();
        let v_v = 1.0 / vle.vapor().density.to_reduced();
        let y = &vle.vapor().molefracs;
        let t = self.0.into_reduced();
        let (a_l, a_v, v_l, v_v) = {
            let t = Gradient::from(t);
            let v_l = Gradient::from(v_l);
            let v_v = Gradient::from(v_v);
            let y = y.map(Gradient::from);
            let x = liquid_molefracs.map(Gradient::from);

            let a_v = eos.residual_helmholtz_energy(t, v_v, &y);
            let (p_l, mu_res_l, dp_l, dmu_l) = eos.dmu_dv(t, v_l, &x);
            let vi_l = dmu_l / dp_l;
            let v_l = vi_l.dot(&y);
            let a_l = (mu_res_l - vi_l * p_l).dot(&y);
            (a_l, a_v, v_l, v_v)
        };
        let rho_l = vle.liquid().partial_density().to_reduced();
        let rho_l = [rho_l[0], rho_l[1]];
        let rho_v = vle.vapor().partial_density().to_reduced();
        let rho_v = [rho_v[0], rho_v[1]];
        let p = -(a_v - a_l
            + t * (y[0] * (rho_v[0] / rho_l[0]).ln() + y[1] * (rho_v[1] / rho_l[1]).ln() - 1.0))
            / (v_v - v_l);
        Ok(Pressure::from_reduced(p))
    }
}
