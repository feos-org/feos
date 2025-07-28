use super::Gradient;
use crate::{HelmholtzEnergyWrapper, NamedParameters, ResidualHelmholtzEnergy};
use feos_core::FeosResult;
use nalgebra::{Const, SVector, U1};
use ndarray::{Array1, Array2, ArrayView2, Zip};
use quantity::{KELVIN, KILO, METER, MOL, PASCAL};

pub trait PureModel: ResidualHelmholtzEnergy<1> + NamedParameters {
    fn vapor_pressure_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        parallelize(
            parameter_names,
            parameters,
            input,
            |eos: &HelmholtzEnergyWrapper<Self, Gradient<P>, 1>, inp| {
                eos.vapor_pressure(inp[0] * KELVIN)
                    .map(|p| p.convert_into(PASCAL))
            },
        )
    }

    fn liquid_density_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        parallelize(
            parameter_names,
            parameters,
            input,
            |eos: &HelmholtzEnergyWrapper<Self, Gradient<P>, 1>, inp| {
                eos.liquid_density(inp[0] * KELVIN, inp[1] * PASCAL)
                    .map(|d| d.convert_into(KILO * MOL / (METER * METER * METER)))
            },
        )
    }

    fn equilibrium_liquid_density_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        parallelize(
            parameter_names,
            parameters,
            input,
            |eos: &HelmholtzEnergyWrapper<Self, Gradient<P>, 1>, inp| {
                eos.equilibrium_liquid_density(inp[0] * KELVIN)
                    .map(|(_, d)| d.convert_into(KILO * MOL / (METER * METER * METER)))
            },
        )
    }
}

impl<T: ResidualHelmholtzEnergy<1> + NamedParameters> PureModel for T {}

pub trait BinaryModel: ResidualHelmholtzEnergy<2> + NamedParameters {
    fn bubble_point_pressure_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        parallelize(
            parameter_names,
            parameters,
            input,
            |eos: &HelmholtzEnergyWrapper<Self, Gradient<P>, 2>, inp| {
                eos.bubble_point_pressure(
                    inp[0] * KELVIN,
                    Some(inp[2] * PASCAL),
                    SVector::from([inp[1], 1.0 - inp[1]]),
                )
                .map(|p| p.convert_into(PASCAL))
            },
        )
    }

    fn dew_point_pressure_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        parallelize(
            parameter_names,
            parameters,
            input,
            |eos: &HelmholtzEnergyWrapper<Self, Gradient<P>, 2>, inp| {
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

impl<T: ResidualHelmholtzEnergy<2> + NamedParameters> BinaryModel for T {}

fn parallelize<F, R: NamedParameters, const N: usize, const P: usize>(
    parameter_names: [String; P],
    parameters: ArrayView2<f64>,
    input: ArrayView2<f64>,
    f: F,
) -> (Array1<f64>, Array2<f64>, Array1<bool>)
where
    F: Fn(&HelmholtzEnergyWrapper<R, Gradient<P>, N>, &[f64]) -> FeosResult<Gradient<P>> + Sync,
{
    let parameter_names = parameter_names.each_ref().map(|s| s as &str);
    let value_dual = Zip::from(parameters.rows())
        .and(input.rows())
        .par_map_collect(|par, inp| {
            let par = par.as_slice().expect("Parameter array is not contiguous!");
            let inp = inp.as_slice().expect("Input array is not contiguous!");
            let eos = R::from(par);
            let deriv = eos.named_derivatives(parameter_names);
            let eos = eos.derivatives(&deriv);
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
