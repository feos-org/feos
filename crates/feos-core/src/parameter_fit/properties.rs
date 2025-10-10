use super::Gradient;
#[cfg(feature = "rayon")]
use super::ParametersAD;
use crate::density_iteration::density_iteration;
use crate::{
    DensityInitialization::Liquid, FeosResult, PhaseEquilibrium, ReferenceSystem, Residual,
};
#[cfg(feature = "rayon")]
use nalgebra::Const;
use nalgebra::{SVector, U1, U2};
#[cfg(feature = "rayon")]
use ndarray::{Array1, Array2, ArrayView2, Zip};
use num_dual::DualStruct;
use quantity::{Density, Pressure, Temperature};
#[cfg(feature = "rayon")]
use quantity::{KELVIN, KILO, METER, MOL, PASCAL};

/// A collection of functions that calculate fit properties and their derivatives
/// with respect to model parameters.
pub struct ParameterFit;

impl ParameterFit {
    pub fn vapor_pressure<E: Residual<U1, Gradient<P>>, const P: usize>(
        eos: &E,
        temperature: Temperature,
    ) -> FeosResult<Pressure<Gradient<P>>> {
        let eos_f64 = eos.re();
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
            let x = SVector::from([Gradient::from(1.0)]);

            let a1 = eos.residual_molar_helmholtz_energy(t, v1, &x);
            let a2 = eos.residual_molar_helmholtz_energy(t, v2, &x);
            (a1, a2)
        };

        let p = -(a1 - a2 + t * (v2 / v1).ln()) / (v1 - v2);
        Ok(Pressure::from_reduced(p))
    }

    pub fn equilibrium_liquid_density<E: Residual<U1, Gradient<P>>, const P: usize>(
        eos: &E,
        temperature: Temperature,
    ) -> FeosResult<(Pressure<Gradient<P>>, Density<Gradient<P>>)> {
        let t = Temperature::from_inner(&temperature);
        PhaseEquilibrium::pure_t(eos, t, None, Default::default()).map(|(p, [_, rho])| (p, rho))
    }

    pub fn liquid_density<E: Residual<U1, Gradient<P>>, const P: usize>(
        eos: &E,
        temperature: Temperature,
        pressure: Pressure,
    ) -> FeosResult<Density<Gradient<P>>> {
        let x = E::pure_molefracs();
        let t = Temperature::from_inner(&temperature);
        let p = Pressure::from_inner(&pressure);
        density_iteration(eos, t, p, &x, Some(Liquid))
    }

    #[cfg(feature = "rayon")]
    pub fn vapor_pressure_parallel<E: ParametersAD<1>, const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        parallelize::<_, E, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &E::Lifted<Gradient<P>>, inp| {
                Self::vapor_pressure(eos, inp[0] * KELVIN).map(|p| p.convert_into(PASCAL))
            },
        )
    }

    #[cfg(feature = "rayon")]
    pub fn liquid_density_parallel<E: ParametersAD<1>, const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        parallelize::<_, E, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &E::Lifted<Gradient<P>>, inp| {
                Self::liquid_density(eos, inp[0] * KELVIN, inp[1] * PASCAL)
                    .map(|d| d.convert_into(KILO * MOL / (METER * METER * METER)))
            },
        )
    }

    #[cfg(feature = "rayon")]
    pub fn equilibrium_liquid_density_parallel<E: ParametersAD<1>, const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        parallelize::<_, E, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &E::Lifted<Gradient<P>>, inp| {
                Self::equilibrium_liquid_density(eos, inp[0] * KELVIN)
                    .map(|(_, d)| d.convert_into(KILO * MOL / (METER * METER * METER)))
            },
        )
    }

    pub fn bubble_point_pressure<E: Residual<U2, Gradient<P>>, const P: usize>(
        eos: &E,
        temperature: Temperature,
        pressure: Option<Pressure>,
        liquid_molefracs: SVector<f64, 2>,
    ) -> FeosResult<Pressure<Gradient<P>>> {
        let eos_f64 = eos.re();
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

            let a_v = eos.residual_molar_helmholtz_energy(t, v_v, &y);
            let (p_l, mu_res_l, dp_l, dmu_l) = eos.dmu_dv(t, v_l, &x);
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

    pub fn dew_point_pressure<E: Residual<U2, Gradient<P>>, const P: usize>(
        eos: &E,
        temperature: Temperature,
        pressure: Option<Pressure>,
        vapor_molefracs: SVector<f64, 2>,
    ) -> FeosResult<Pressure<Gradient<P>>> {
        let eos_f64 = eos.re();
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

            let a_l = eos.residual_molar_helmholtz_energy(t, v_l, &x);
            let (p_v, mu_res_v, dp_v, dmu_v) = eos.dmu_dv(t, v_v, &y);
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
    pub fn bubble_point_pressure_parallel<E: ParametersAD<2>, const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        parallelize::<_, E, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &E::Lifted<Gradient<P>>, inp| {
                Self::bubble_point_pressure(
                    eos,
                    inp[0] * KELVIN,
                    Some(inp[2] * PASCAL),
                    SVector::from([inp[1], 1.0 - inp[1]]),
                )
                .map(|p| p.convert_into(PASCAL))
            },
        )
    }

    #[cfg(feature = "rayon")]
    pub fn dew_point_pressure_parallel<E: ParametersAD<2>, const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        parallelize::<_, E, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &E::Lifted<Gradient<P>>, inp| {
                Self::dew_point_pressure(
                    eos,
                    inp[0] * KELVIN,
                    Some(inp[2] * PASCAL),
                    SVector::from([inp[1], 1.0 - inp[1]]),
                )
                .map(|p| p.convert_into(PASCAL))
            },
        )
    }
}

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
            let eos = E::from_parameter_slice(par, parameter_names);
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
