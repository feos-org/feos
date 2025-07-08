use super::{BinaryProperty, Property, PureProperty};
use crate::{HelmholtzEnergyWrapper, NamedParameters, ParametersAD, ResidualHelmholtzEnergy};
use feos_core::FeosResult;
use nalgebra::{Const, SVector, U1};
use ndarray::{Array1, Array2, ArrayView2, Zip};
use num_dual::DualVec;
use quantity::{KELVIN, KILO, METER, MOL, PASCAL};

impl Property {
    pub fn evaluate_parallel<
        Pure: ResidualHelmholtzEnergy<1> + NamedParameters,
        Binary: ResidualHelmholtzEnergy<2> + NamedParameters,
        const P: usize,
    >(
        &self,
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        let parameter_names = parameter_names.each_ref().map(|s| s as &str);
        let value_dual = Zip::from(parameters.rows())
            .and(input.rows())
            .par_map_collect(|par, inp| {
                let par = par.as_slice().expect("Parameter array is not contiguous!");
                let inp = inp.as_slice().expect("Input array is not contiguous!");
                match self {
                    Self::Pure(prop) => {
                        let eos = Pure::from(par).wrap().named_derivatives(parameter_names);
                        prop.evaluate(&eos, inp)
                    }
                    Self::Binary(prop) => {
                        let eos = Binary::from(par).wrap().named_derivatives(parameter_names);
                        prop.evaluate(&eos, inp)
                    }
                }
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
}

impl PureProperty {
    fn evaluate<E: ResidualHelmholtzEnergy<1> + ParametersAD, const P: usize>(
        &self,
        eos: &HelmholtzEnergyWrapper<E, DualVec<f64, f64, Const<P>>, 1>,
        input: &[f64],
    ) -> FeosResult<DualVec<f64, f64, Const<P>>> {
        match self {
            Self::VaporPressure => {
                Self::vapor_pressure(eos, input[0] * KELVIN).map(|p| p.convert_into(PASCAL))
            }
            Self::LiquidDensity => Self::liquid_density(eos, input[0] * KELVIN, input[1] * PASCAL)
                .map(|p| p.convert_into(KILO * MOL / (METER * METER * METER))),
            Self::EquilibriumLiquidDensity => {
                Self::equilibrium_liquid_density(eos, input[0] * KELVIN)
                    .map(|(_, r)| r.convert_into(KILO * MOL / (METER * METER * METER)))
            }
        }
    }
}

impl BinaryProperty {
    fn evaluate<E: ResidualHelmholtzEnergy<2> + ParametersAD, const P: usize>(
        &self,
        eos: &HelmholtzEnergyWrapper<E, DualVec<f64, f64, Const<P>>, 2>,
        input: &[f64],
    ) -> FeosResult<DualVec<f64, f64, Const<P>>> {
        match self {
            Self::BubblePointPressure => Self::bubble_point_pressure(
                eos,
                input[0] * KELVIN,
                Some(input[2] * PASCAL),
                SVector::from([input[1], 1.0 - input[1]]),
            )
            .map(|p| p.convert_into(PASCAL)),
            Self::DewPointPressure => Self::dew_point_pressure(
                eos,
                input[0] * KELVIN,
                Some(input[2] * PASCAL),
                SVector::from([input[1], 1.0 - input[1]]),
            )
            .map(|p| p.convert_into(PASCAL)),
        }
    }
}
