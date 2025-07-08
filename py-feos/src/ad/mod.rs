#[cfg(feature = "pcsaft")]
use feos_ad::eos::{PcSaftBinary, PcSaftPure};
use feos_ad::{
    bubble_point_pressure, dew_point_pressure, equilibrium_liquid_density, liquid_density,
    vapor_pressure, HelmholtzEnergyWrapper, NamedParameters, ParametersAD, ResidualHelmholtzEnergy,
};
use feos_core::FeosResult;
use nalgebra::{Const, SVector, U1};
use ndarray::{Array1, Array2, ArrayView2, Zip};
use num_dual::DualVec;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use quantity::{KELVIN, KILO, METER, MOL, PASCAL};
use typenum::P3;

#[pyclass(name = "Property")]
#[derive(Clone, Copy)]
pub struct PyProperty(Property);

#[pymethods]
#[expect(non_snake_case)]
impl PyProperty {
    #[classattr]
    fn VaporPressure() -> Self {
        Self(Property::Pure(PureProperty::VaporPressure))
    }

    #[classattr]
    fn LiquidDensity() -> Self {
        Self(Property::Pure(PureProperty::LiquidDensity))
    }

    #[classattr]
    fn EquilibriumLiquidDensity() -> Self {
        Self(Property::Pure(PureProperty::EquilibriumLiquidDensity))
    }

    #[classattr]
    fn BubblePointPressure() -> Self {
        Self(Property::Binary(BinaryProperty::BubblePointPressure))
    }

    #[classattr]
    fn DewPointPressure() -> Self {
        Self(Property::Binary(BinaryProperty::DewPointPressure))
    }
}

#[derive(Clone, Copy)]
enum Property {
    Pure(PureProperty),
    Binary(BinaryProperty),
}

#[derive(Clone, Copy)]
enum PureProperty {
    VaporPressure,
    LiquidDensity,
    EquilibriumLiquidDensity,
}

#[derive(Clone, Copy)]
enum BinaryProperty {
    BubblePointPressure,
    DewPointPressure,
}

#[pyclass(name = "Estimator")]
pub struct PyEstimator;

#[pymethods]
impl PyEstimator {
    #[cfg(feature = "pcsaft")]
    #[staticmethod]
    #[expect(clippy::type_complexity)]
    fn pcsaft_non_assoc<'py>(
        property: PyProperty,
        parameter_names: Bound<'py, PyAny>,
        parameters: PyReadonlyArray2<f64>,
        input: PyReadonlyArray2<f64>,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<bool>>,
    ) {
        evaluate_gradients::<PcSaftPure<4>, PcSaftBinary<4>>(
            property.0,
            parameter_names,
            parameters,
            input,
        )
    }

    #[cfg(feature = "pcsaft")]
    #[staticmethod]
    #[expect(clippy::type_complexity)]
    fn pcsaft_full<'py>(
        property: PyProperty,
        parameter_names: Bound<'py, PyAny>,
        parameters: PyReadonlyArray2<f64>,
        input: PyReadonlyArray2<f64>,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<bool>>,
    ) {
        evaluate_gradients::<PcSaftPure<8>, PcSaftBinary<8>>(
            property.0,
            parameter_names,
            parameters,
            input,
        )
    }
}

macro_rules! impl_evaluate_gradients {
    ($($p:literal),*) => {
        fn evaluate_gradients<
            'py,
            Pure: ResidualHelmholtzEnergy<1> + NamedParameters,
            Binary: ResidualHelmholtzEnergy<2> + NamedParameters,
        >(
            property: Property,
            parameter_names: Bound<'py, PyAny>,
            parameters: PyReadonlyArray2<f64>,
            input: PyReadonlyArray2<f64>,
        ) -> (
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray2<f64>>,
            Bound<'py, PyArray1<bool>>,
        ) {
            let (value, grad, status) =
            $(
            if let Ok(p) = parameter_names.extract::<[String; $p]>() {
                property.evaluate::<Pure, Binary, $p>(p, parameters.as_array(), input.as_array())
            } else)* {
                panic!("Gradients can only be evaluated for up to 15 parameters!")
            };
            (
                value.to_pyarray(parameter_names.py()),
                grad.to_pyarray(parameter_names.py()),
                status.to_pyarray(parameter_names.py()),
            )
        }
    };
}

impl_evaluate_gradients!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

impl Property {
    fn evaluate<
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
                vapor_pressure(eos, input[0] * KELVIN).map(|p| p.convert_into(PASCAL))
            }
            Self::LiquidDensity => liquid_density(eos, input[0] * KELVIN, input[1] * PASCAL)
                .map(|p| p.convert_into(KILO * MOL / METER.powi::<P3>())),
            Self::EquilibriumLiquidDensity => equilibrium_liquid_density(eos, input[0] * KELVIN)
                .map(|(_, r)| r.convert_into(KILO * MOL / METER.powi::<P3>())),
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
            Self::BubblePointPressure => bubble_point_pressure(
                eos,
                input[0] * KELVIN,
                Some(input[2] * PASCAL),
                SVector::from([input[1], 1.0 - input[1]]),
            )
            .map(|p| p.convert_into(PASCAL)),
            Self::DewPointPressure => dew_point_pressure(
                eos,
                input[0] * KELVIN,
                Some(input[2] * PASCAL),
                SVector::from([input[1], 1.0 - input[1]]),
            )
            .map(|p| p.convert_into(PASCAL)),
        }
    }
}
