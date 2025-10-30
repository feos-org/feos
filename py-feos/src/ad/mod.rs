use feos::pcsaft::{PcSaftBinary, PcSaftPure};
use feos_core::{ParametersAD, PropertiesAD};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use paste::paste;
use pyo3::prelude::*;

#[pyclass(name = "ModelAD", eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyModelAD {
    PcSaftNonAssoc,
    PcSaftFull,
}

enum BinaryModels {
    PcSaftNonAssoc,
    PcSaftFull,
}

impl From<PyModelAD> for BinaryModels {
    fn from(value: PyModelAD) -> Self {
        match value {
            PyModelAD::PcSaftNonAssoc => Self::PcSaftNonAssoc,
            PyModelAD::PcSaftFull => Self::PcSaftFull,
        }
    }
}

#[pyclass(name = "PropertiesAD")]
pub struct PyPropertiesAD;

type GradResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<bool>>,
);

macro_rules! expand_models {
    ($enum:ty, $prop:ident, $($model:ident: $type:ty),*) => {
        #[pymethods]
        impl PyPropertiesAD {
            #[staticmethod]
            fn $prop<'py>(
                model: PyModelAD,
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
        $(impl_evaluate_gradients!(1,PyModelAD,$prop,$models,0,1,2,3,4,5,max:6);)*
    };
    (binary, [$($prop:ident),*], $models:tt) => {
        $(impl_evaluate_gradients!(2,BinaryModels,$prop,$models,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,max:15);)*
    };
    ($n:literal, $enum:ty, $prop:ident, {$($model:ident: $type:ty),*}, $($p:literal,)* max: $max:literal) => {
        expand_models!($enum, $prop, $($model: $type),*);
        paste!(
        fn $prop<'py, R: ParametersAD<$n>>(
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
    {PcSaftNonAssoc: PcSaftPure<f64, 4>, PcSaftFull: PcSaftPure<f64, 8>}
);

impl_evaluate_gradients!(
    binary,
    [bubble_point_pressure, dew_point_pressure],
    {PcSaftNonAssoc: PcSaftBinary<f64, 4>, PcSaftFull: PcSaftBinary<f64, 8>}
);
