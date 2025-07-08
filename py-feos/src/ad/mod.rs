#[cfg(feature = "pcsaft")]
use feos_ad::eos::{PcSaftBinary, PcSaftPure};
use feos_ad::{BinaryProperty, NamedParameters, Property, PureProperty, ResidualHelmholtzEnergy};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

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
                property.evaluate_parallel::<Pure, Binary, $p>(p, parameters.as_array(), input.as_array())
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
