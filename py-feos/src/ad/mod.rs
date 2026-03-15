use feos::pcsaft::{PcSaftBinary, PcSaftPure};
use feos_core::{ParametersAD, PropertiesAD};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use paste::paste;
use pyo3::prelude::*;

#[pyclass(name = "EquationOfStateAD", eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyEquationOfStateAD {
    PcSaftNonAssoc,
    PcSaftFull,
}

enum BinaryModels {
    PcSaftNonAssoc,
    PcSaftFull,
}

impl From<PyEquationOfStateAD> for BinaryModels {
    fn from(value: PyEquationOfStateAD) -> Self {
        match value {
            PyEquationOfStateAD::PcSaftNonAssoc => Self::PcSaftNonAssoc,
            PyEquationOfStateAD::PcSaftFull => Self::PcSaftFull,
        }
    }
}

type GradResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<bool>>,
);

/// Calculate vapor pressures and derivatives w.r.t. model parameters.
///
/// Parameters
/// ----------
/// model: EquationOfStateAD
///     The equation of state to use.
/// parameter_names: List[string]
///     The name of the parameters for which derivatives are calculated.
/// parameters: np.ndarray[float]
///     The parameters for every data point.
/// input: np.ndarray[float]
///     The temperature (in K) for every data point.
///
/// Returns
/// -------
/// (np.ndarray[float], np.ndarray[float], np.ndarray[bool]): The vapor pressures (in Pa), gradients, and convergence status.
#[pyfunction]
pub fn vapor_pressure_derivatives<'py>(
    model: PyEquationOfStateAD,
    parameter_names: &Bound<'py, PyAny>,
    parameters: PyReadonlyArray2<f64>,
    input: PyReadonlyArray2<f64>,
) -> GradResult<'py> {
    _vapor_pressure_derivatives(model, parameter_names, parameters, input)
}

/// Calculate boiling temperatures and derivatives w.r.t. model parameters.
///
/// Parameters
/// ----------
/// model: EquationOfStateAD
///     The equation of state to use.
/// parameter_names: List[string]
///     The name of the parameters for which derivatives are calculated.
/// parameters: np.ndarray[float]
///     The parameters for every data point.
/// input: np.ndarray[float]
///     The pressure (in Pa) for every data point.
///
/// Returns
/// -------
/// (np.ndarray[float], np.ndarray[float], np.ndarray[bool]): The boiling temperature (in K), gradients, and convergence status.
#[pyfunction]
pub fn boiling_temperature_derivatives<'py>(
    model: PyEquationOfStateAD,
    parameter_names: &Bound<'py, PyAny>,
    parameters: PyReadonlyArray2<f64>,
    input: PyReadonlyArray2<f64>,
) -> GradResult<'py> {
    _boiling_temperature_derivatives(model, parameter_names, parameters, input)
}

/// Calculate liquid densities and derivatives w.r.t. model parameters.
///
/// Parameters
/// ----------
/// model: EquationOfStateAD
///     The equation of state to use.
/// parameter_names: List[string]
///     The name of the parameters for which derivatives are calculated.
/// parameters: np.ndarray[float]
///     The parameters for every data point.
/// input: np.ndarray[float]
///     The temperature (in K) and pressure (in Pa) for every data point.
///
/// Returns
/// -------
/// (np.ndarray[float], np.ndarray[float], np.ndarray[bool]): The liquid densities (in kmol/m³), gradients, and convergence status.
#[pyfunction]
pub fn liquid_density_derivatives<'py>(
    model: PyEquationOfStateAD,
    parameter_names: &Bound<'py, PyAny>,
    parameters: PyReadonlyArray2<f64>,
    input: PyReadonlyArray2<f64>,
) -> GradResult<'py> {
    _liquid_density_derivatives(model, parameter_names, parameters, input)
}

/// Calculate liquid densities at saturation and derivatives w.r.t. model parameters.
///
/// Parameters
/// ----------
/// model: EquationOfStateAD
///     The equation of state to use.
/// parameter_names: List[string]
///     The name of the parameters for which derivatives are calculated.
/// parameters: np.ndarray[float]
///     The parameters for every data point.
/// input: np.ndarray[float]
///     The temperature (in K) for every data point.
///
/// Returns
/// -------
/// (np.ndarray[float], np.ndarray[float], np.ndarray[bool]): The liquid densities (in kmol/m³), gradients, and convergence status.
#[pyfunction]
pub fn equilibrium_liquid_density_derivatives<'py>(
    model: PyEquationOfStateAD,
    parameter_names: &Bound<'py, PyAny>,
    parameters: PyReadonlyArray2<f64>,
    input: PyReadonlyArray2<f64>,
) -> GradResult<'py> {
    _equilibrium_liquid_density_derivatives(model, parameter_names, parameters, input)
}

/// Calculate bubble point pressures of binary mixtures and derivatives w.r.t. model parameters.
///
/// Parameters
/// ----------
/// model: EquationOfStateAD
///     The equation of state to use.
/// parameter_names: List[string]
///     The name of the parameters for which derivatives are calculated.
/// parameters: np.ndarray[float]
///     The parameters for every data point.
/// input: np.ndarray[float]
///     The temperature (in K), composition of the first component, and an initial guess for the
///     pressure (in Pa) for every data point.
///
/// Returns
/// -------
/// (np.ndarray[float], np.ndarray[float], np.ndarray[bool]): The bubble point pressures (in Pa), gradients, and convergence status.
#[pyfunction]
pub fn bubble_point_pressure_derivatives<'py>(
    model: PyEquationOfStateAD,
    parameter_names: &Bound<'py, PyAny>,
    parameters: PyReadonlyArray2<f64>,
    input: PyReadonlyArray2<f64>,
) -> GradResult<'py> {
    _bubble_point_pressure_derivatives(model, parameter_names, parameters, input)
}

/// Calculate dew point pressures of binary mixtures and derivatives w.r.t. model parameters.
///
/// Parameters
/// ----------
/// model: EquationOfStateAD
///     The equation of state to use.
/// parameter_names: List[string]
///     The name of the parameters for which derivatives are calculated.
/// parameters: np.ndarray[float]
///     The parameters for every data point.
/// input: np.ndarray[float]
///     The temperature (in K), composition of the first component, and an initial guess for the
///     pressure (in Pa) for every data point.
///
/// Returns
/// -------
/// (np.ndarray[float], np.ndarray[float], np.ndarray[bool]): The dew point pressures (in Pa), gradients, and convergence status.
#[pyfunction]
pub fn dew_point_pressure_derivatives<'py>(
    model: PyEquationOfStateAD,
    parameter_names: &Bound<'py, PyAny>,
    parameters: PyReadonlyArray2<f64>,
    input: PyReadonlyArray2<f64>,
) -> GradResult<'py> {
    _dew_point_pressure_derivatives(model, parameter_names, parameters, input)
}

macro_rules! expand_models {
    ($enum:ty, $prop:ident, $($model:ident: $type:ty),*) => {
        paste!(
        #[pyfunction]
        fn [<_ $prop _derivatives>]<'py>(
            model: PyEquationOfStateAD,
            parameter_names: &Bound<'py, PyAny>,
            parameters: PyReadonlyArray2<f64>,
            input: PyReadonlyArray2<f64>,
        ) -> GradResult<'py> {
            match <$enum>::from(model) {
                $(
                <$enum>::$model => {
                    $prop::<$type>(parameter_names, parameters, input)
                })*
            }
        });
    };
}

macro_rules! impl_evaluate_gradients {
    (pure, [$($prop:ident),*], $models:tt) => {
        $(impl_evaluate_gradients!(1,PyEquationOfStateAD,$prop,$models,0,1,2,3,4,5,max:6);)*
    };
    (binary, [$($prop:ident),*], $models:tt) => {
        $(impl_evaluate_gradients!(2,BinaryModels,$prop,$models,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,max:15);)*
    };
    ($n:literal, $enum:ty, $prop:ident, {$($model:ident: $type:ty),*}, $($p:literal,)* max: $max:literal) => {
        expand_models!($enum, $prop, $($model: $type),*);
        paste!(
        fn $prop<'py, R: ParametersAD<$n>>(
            parameter_names: &Bound<'py, PyAny>,
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
    [vapor_pressure, boiling_temperature, liquid_density, equilibrium_liquid_density],
    {PcSaftNonAssoc: PcSaftPure<f64, 4>, PcSaftFull: PcSaftPure<f64, 8>}
);

impl_evaluate_gradients!(
    binary,
    [bubble_point_pressure, dew_point_pressure],
    {PcSaftNonAssoc: PcSaftBinary<f64, 4>, PcSaftFull: PcSaftBinary<f64, 8>}
);
