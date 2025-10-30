use crate::DensityInitialization::Liquid;
use crate::density_iteration::density_iteration;
use crate::{FeosResult, PhaseEquilibrium, ReferenceSystem, Residual};
use nalgebra::{Const, SVector, U1, U2};
#[cfg(feature = "rayon")]
use ndarray::{Array1, Array2, ArrayView2, Zip};
use num_dual::{Derivative, DualSVec, DualStruct};
use quantity::{Density, Pressure, Temperature};
#[cfg(feature = "rayon")]
use quantity::{KELVIN, KILO, METER, MOL, PASCAL};

type Gradient<const P: usize> = DualSVec<f64, f64, P>;

/// A model that can be evaluated with derivatives of its parameters.
pub trait ParametersAD<const N: usize>: for<'a> From<&'a [f64]> + Residual<Const<N>> {
    /// Return a mutable reference to the parameter named by `index` from the parameter set.
    fn index_parameters_mut<'a, const P: usize>(
        eos: &'a mut Self::Lifted<Gradient<P>>,
        index: &str,
    ) -> &'a mut Gradient<P>;

    /// Return the parameters with the appropriate derivatives.
    fn named_derivatives<const P: usize>(
        &self,
        parameter_names: [&str; P],
    ) -> Self::Lifted<Gradient<P>> {
        let mut eos = self.lift::<Gradient<P>>();
        for (i, p) in parameter_names.into_iter().enumerate() {
            Self::index_parameters_mut(&mut eos, p).eps =
                Derivative::derivative_generic(Const::<P>, U1, i)
        }
        eos
    }
}

/// Properties that can be evaluated with derivatives of model parameters.
pub trait PropertiesAD {
    fn vapor_pressure<const P: usize>(
        &self,
        temperature: Temperature,
    ) -> FeosResult<Pressure<Gradient<P>>>
    where
        Self: Residual<U1, Gradient<P>>,
    {
        let eos_f64 = self.re();
        let (_, [vapor_density, liquid_density]) =
            PhaseEquilibrium::pure_t(&eos_f64, temperature, None, Default::default())?;

        // implicit differentiation is implemented here instead of just calling pure_t with dual
        // numbers, because for the first derivative, we can avoid calculating density derivatives.
        let v1 = 1.0 / liquid_density.to_reduced();
        let v2 = 1.0 / vapor_density.to_reduced();
        let t = temperature.into_reduced();
        let (a1, a2) = {
            let t = Gradient::from(t);
            let v1 = Gradient::from(v1);
            let v2 = Gradient::from(v2);
            let x = Self::pure_molefracs();

            let a1 = self.residual_molar_helmholtz_energy(t, v1, &x);
            let a2 = self.residual_molar_helmholtz_energy(t, v2, &x);
            (a1, a2)
        };

        let p = -(a1 - a2 + t * (v2 / v1).ln()) / (v1 - v2);
        Ok(Pressure::from_reduced(p))
    }

    fn equilibrium_liquid_density<const P: usize>(
        &self,
        temperature: Temperature,
    ) -> FeosResult<(Pressure<Gradient<P>>, Density<Gradient<P>>)>
    where
        Self: Residual<U1, Gradient<P>>,
    {
        let t = Temperature::from_inner(&temperature);
        PhaseEquilibrium::pure_t(self, t, None, Default::default()).map(|(p, [_, rho])| (p, rho))
    }

    fn liquid_density<const P: usize>(
        &self,
        temperature: Temperature,
        pressure: Pressure,
    ) -> FeosResult<Density<Gradient<P>>>
    where
        Self: Residual<U1, Gradient<P>>,
    {
        let x = Self::pure_molefracs();
        let t = Temperature::from_inner(&temperature);
        let p = Pressure::from_inner(&pressure);
        density_iteration(self, t, p, &x, Some(Liquid))
    }

    #[cfg(feature = "rayon")]
    fn vapor_pressure_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>)
    where
        Self: ParametersAD<1>,
    {
        parallelize::<_, Self, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &Self::Lifted<Gradient<P>>, inp| {
                eos.vapor_pressure(inp[0] * KELVIN)
                    .map(|p| p.convert_into(PASCAL))
            },
        )
    }

    #[cfg(feature = "rayon")]
    fn liquid_density_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>)
    where
        Self: ParametersAD<1>,
    {
        parallelize::<_, Self, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &Self::Lifted<Gradient<P>>, inp| {
                eos.liquid_density(inp[0] * KELVIN, inp[1] * PASCAL)
                    .map(|d| d.convert_into(KILO * MOL / (METER * METER * METER)))
            },
        )
    }

    #[cfg(feature = "rayon")]
    fn equilibrium_liquid_density_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>)
    where
        Self: ParametersAD<1>,
    {
        parallelize::<_, Self, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &Self::Lifted<Gradient<P>>, inp| {
                eos.equilibrium_liquid_density(inp[0] * KELVIN)
                    .map(|(_, d)| d.convert_into(KILO * MOL / (METER * METER * METER)))
            },
        )
    }

    fn bubble_point_pressure<const P: usize>(
        &self,
        temperature: Temperature,
        pressure: Option<Pressure>,
        liquid_molefracs: SVector<f64, 2>,
    ) -> FeosResult<Pressure<Gradient<P>>>
    where
        Self: Residual<U2, Gradient<P>>,
    {
        let eos_f64 = self.re();
        let vle = PhaseEquilibrium::bubble_point(
            &eos_f64,
            temperature,
            &liquid_molefracs,
            pressure,
            None,
            Default::default(),
        )?;

        // implicit differentiation is implemented here instead of just calling bubble_point with dual
        // numbers, because for the first derivative, we can avoid calculating density derivatives.
        let v_l = 1.0 / vle.liquid().density.to_reduced();
        let v_v = 1.0 / vle.vapor().density.to_reduced();
        let y = &vle.vapor().molefracs;
        let y: SVector<_, 2> = SVector::from_fn(|i, _| y[i]);
        let t = temperature.into_reduced();
        let (a_l, a_v, v_l, v_v) = {
            let t = Gradient::from(t);
            let v_l = Gradient::from(v_l);
            let v_v = Gradient::from(v_v);
            let y = y.map(Gradient::from);
            let x = liquid_molefracs.map(Gradient::from);

            let a_v = self.residual_molar_helmholtz_energy(t, v_v, &y);
            let (p_l, mu_res_l, dp_l, dmu_l) = self.dmu_dv(t, v_l, &x);
            let vi_l = dmu_l / dp_l;
            let v_l = vi_l.dot(&y);
            let a_l = (mu_res_l - vi_l * p_l).dot(&y);
            (a_l, a_v, v_l, v_v)
        };
        let rho_l = vle.liquid().partial_density.to_reduced();
        let rho_l = [rho_l[0], rho_l[1]];
        let rho_v = vle.vapor().partial_density.to_reduced();
        let rho_v = [rho_v[0], rho_v[1]];
        let p = -(a_v - a_l
            + t * (y[0] * (rho_v[0] / rho_l[0]).ln() + y[1] * (rho_v[1] / rho_l[1]).ln() - 1.0))
            / (v_v - v_l);
        Ok(Pressure::from_reduced(p))
    }

    fn dew_point_pressure<const P: usize>(
        &self,
        temperature: Temperature,
        pressure: Option<Pressure>,
        vapor_molefracs: SVector<f64, 2>,
    ) -> FeosResult<Pressure<Gradient<P>>>
    where
        Self: Residual<U2, Gradient<P>>,
    {
        let eos_f64 = self.re();
        let vle = PhaseEquilibrium::dew_point(
            &eos_f64,
            temperature,
            &vapor_molefracs,
            pressure,
            None,
            Default::default(),
        )?;

        // implicit differentiation is implemented here instead of just calling dew_point with dual
        // numbers, because for the first derivative, we can avoid calculating density derivatives.
        let v_l = 1.0 / vle.liquid().density.to_reduced();
        let v_v = 1.0 / vle.vapor().density.to_reduced();
        let x = &vle.liquid().molefracs;
        let x: SVector<_, 2> = SVector::from_fn(|i, _| x[i]);
        let t = temperature.into_reduced();
        let (a_l, a_v, v_l, v_v) = {
            let t = Gradient::from(t);
            let v_l = Gradient::from(v_l);
            let v_v = Gradient::from(v_v);
            let x = x.map(Gradient::from);
            let y = vapor_molefracs.map(Gradient::from);

            let a_l = self.residual_molar_helmholtz_energy(t, v_l, &x);
            let (p_v, mu_res_v, dp_v, dmu_v) = self.dmu_dv(t, v_v, &y);
            let vi_v = dmu_v / dp_v;
            let v_v = vi_v.dot(&x);
            let a_v = (mu_res_v - vi_v * p_v).dot(&x);
            (a_l, a_v, v_l, v_v)
        };
        let rho_l = vle.liquid().partial_density.to_reduced();
        let rho_l = [rho_l[0], rho_l[1]];
        let rho_v = vle.vapor().partial_density.to_reduced();
        let rho_v = [rho_v[0], rho_v[1]];
        let p = -(a_l - a_v
            + t * (x[0] * (rho_l[0] / rho_v[0]).ln() + x[1] * (rho_l[1] / rho_v[1]).ln() - 1.0))
            / (v_l - v_v);
        Ok(Pressure::from_reduced(p))
    }

    #[cfg(feature = "rayon")]
    fn bubble_point_pressure_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>)
    where
        Self: ParametersAD<2>,
    {
        parallelize::<_, Self, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &Self::Lifted<Gradient<P>>, inp| {
                eos.bubble_point_pressure(
                    inp[0] * KELVIN,
                    Some(inp[2] * PASCAL),
                    SVector::from([inp[1], 1.0 - inp[1]]),
                )
                .map(|p| p.convert_into(PASCAL))
            },
        )
    }

    #[cfg(feature = "rayon")]
    fn dew_point_pressure_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>)
    where
        Self: ParametersAD<2>,
    {
        parallelize::<_, Self, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &Self::Lifted<Gradient<P>>, inp| {
                eos.dew_point_pressure(
                    inp[0] * KELVIN,
                    Some(inp[2] * PASCAL),
                    SVector::from([inp[1], 1.0 - inp[1]]),
                )
                .map(|p| p.convert_into(PASCAL))
            },
        )
    }
}

impl<T> PropertiesAD for T {}

#[cfg(feature = "rayon")]
fn parallelize<F, E: ParametersAD<N>, const N: usize, const P: usize>(
    parameter_names: [String; P],
    parameters: ArrayView2<f64>,
    input: ArrayView2<f64>,
    f: F,
) -> (Array1<f64>, Array2<f64>, Array1<bool>)
where
    F: Fn(&E::Lifted<Gradient<P>>, &[f64]) -> FeosResult<Gradient<P>> + Sync,
{
    let parameter_names = parameter_names.each_ref().map(|s| s as &str);
    let value_dual = Zip::from(parameters.rows())
        .and(input.rows())
        .par_map_collect(|par, inp| {
            let par = par.as_slice().expect("Parameter array is not contiguous!");
            let inp = inp.as_slice().expect("Input array is not contiguous!");
            let eos = E::from(par).named_derivatives(parameter_names);
            f(&eos, inp)
        });
    let status = value_dual.iter().map(|p| p.is_ok()).collect();
    let value_dual: Array1<_> = value_dual.into_iter().flatten().collect();
    let mut value = Array1::zeros(value_dual.len());
    let mut grad = Array2::zeros([value_dual.len(), P]);
    Zip::from(grad.rows_mut())
        .and(&mut value)
        .and(&value_dual)
        .for_each(|mut grad, p, p_dual| {
            *p = p_dual.re;
            let eps = p_dual.eps.unwrap_generic(Const::<P>, U1).data.0[0].to_vec();
            grad.assign(&Array1::from(eps));
        });
    (value, grad, status)
}
