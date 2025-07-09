#[cfg(feature = "pcsaft")]
use feos_ad::eos::{PcSaftBinary, PcSaftPure};
use feos_ad::{BinaryModel, NamedParameters, PureModel, ResidualHelmholtzEnergy};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use paste::paste;
use pyo3::prelude::*;

#[pyclass(name = "Model", eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyModel {
    PcSaftNonAssoc,
    PcSaftFull,
}

enum BinaryModels {
    PcSaftNonAssoc,
    PcSaftFull,
}

impl From<PyModel> for BinaryModels {
    fn from(value: PyModel) -> Self {
        match value {
            PyModel::PcSaftNonAssoc => Self::PcSaftNonAssoc,
            PyModel::PcSaftFull => Self::PcSaftFull,
        }
    }
}

#[pyclass(name = "Estimator")]
pub struct PyEstimator;

type GradResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<bool>>,
);

macro_rules! expand_models {
    ($enum:ty, $prop:ident, $($model:ident: $type:ty),*) => {
        #[pymethods]
        impl PyEstimator {
            #[staticmethod]
            fn $prop<'py>(
                model: PyModel,
                parameter_names: Bound<'py, PyAny>,
                parameters: PyReadonlyArray2<f64>,
                input: PyReadonlyArray2<f64>,
            ) -> GradResult<'py> {
                match <$enum>::from(model) {
                    $(
                    <$enum>::$model => {
                        $prop::<$type>(parameter_names, parameters, input)
                    })*
                }
            }
        }
    };
}

macro_rules! impl_evaluate_gradients {
    (pure, [$($prop:ident),*], $models:tt) => {
        $(impl_evaluate_gradients!(1,PyModel,$prop,$models,1,2,3,4,5,max:6);)*
    };
    (binary, [$($prop:ident),*], $models:tt) => {
        $(impl_evaluate_gradients!(2,BinaryModels,$prop,$models,1,2,3,4,5,6,7,8,9,10,11,12,13,14,max:15);)*
    };
    ($n:literal, $enum:ty, $prop:ident, {$($model:ident: $type:ty),*}, $($p:literal,)* max: $max:literal) => {
        expand_models!($enum, $prop, $($model: $type),*);
        paste!(
        fn $prop<
            'py,
            R: ResidualHelmholtzEnergy<$n> + NamedParameters,
        >(
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
                R::[<$prop _parallel>](p, parameters.as_array(), input.as_array())
            } else)* if let Ok(p) = parameter_names.extract::<[String; $max]>() {
                R::[<$prop _parallel>](p, parameters.as_array(), input.as_array())
            } else {
                panic!("Gradients can only be evaluated for up to {} parameters!", $max)
            };
            (
                value.to_pyarray(parameter_names.py()),
                grad.to_pyarray(parameter_names.py()),
                status.to_pyarray(parameter_names.py()),
            )
        });
    };
}

impl_evaluate_gradients!(
    pure,
    [vapor_pressure, liquid_density, equilibrium_liquid_density],
    {PcSaftNonAssoc: PcSaftPure<4>, PcSaftFull: PcSaftPure<8>}
);

impl_evaluate_gradients!(
    binary,
    [bubble_point_pressure, dew_point_pressure],
    {PcSaftNonAssoc: PcSaftBinary<4>, PcSaftFull: PcSaftBinary<8>}
);
