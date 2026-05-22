use crate::eos::PyEquationOfState;
use feos::pcsaft::{PcSaftBinary, PcSaftPure};
use feos_core::ad::{
    BoilingTemperature, BubblePointPressure, DewPointPressure, EnthalpyOfVaporization,
    EquilibriumLiquidDensity, LiquidDensity, ParametersAD, PropertyAD,
    ResidualIsobaricHeatCapacity, VaporPressure,
};
use nalgebra::{U1, U2};
use num_dual::DualSVec;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use paste::paste;
use pyo3::prelude::*;

pub mod dataset;
pub use dataset::{PyBinaryDataset, PyPureDataset};

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

type EvalResult<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<bool>>);

type GradResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<bool>>,
);

#[pyclass(name = "Property")]
pub struct PyPropertyAD;

#[pymethods]
impl PyPropertyAD {
    /// Calculate vapor pressures in parallel.
    ///
    /// Parameters
    /// ----------
    /// eos: EquationOfState
    ///     The equation of state to use.
    /// input: np.ndarray[float]
    ///     The temperature (in K) for every data point.
    ///
    /// Returns
    /// -------
    /// (np.ndarray[float], np.ndarray[bool]): The vapor pressures (in Pa), and convergence status.
    #[staticmethod]
    pub fn vapor_pressure<'py>(
        py: Python<'py>,
        eos: &PyEquationOfState,
        input: PyReadonlyArray2<f64>,
    ) -> EvalResult<'py> {
        let (value, status) = VaporPressure::evaluate_parallel(&eos.0, input.as_array());
        (value.to_pyarray(py), status.to_pyarray(py))
    }

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
    #[staticmethod]
    pub fn vapor_pressure_derivatives<'py>(
        model: PyEquationOfStateAD,
        parameter_names: &Bound<'py, PyAny>,
        parameters: &Bound<'py, PyAny>,
        input: PyReadonlyArray2<f64>,
    ) -> GradResult<'py> {
        _vapor_pressure_derivatives(model, parameter_names, parameters, input)
    }

    /// Calculate boiling temperatures in parallel.
    ///
    /// Parameters
    /// ----------
    /// eos: EquationOfState
    ///     The equation of state to use.
    /// input: np.ndarray[float]
    ///     The pressure (in Pa) for every data point.
    ///
    /// Returns
    /// -------
    /// (np.ndarray[float], np.ndarray[bool]): The boiling temperature (in K), and convergence status.
    #[staticmethod]
    pub fn boiling_temperature<'py>(
        py: Python<'py>,
        eos: &PyEquationOfState,
        input: PyReadonlyArray2<f64>,
    ) -> EvalResult<'py> {
        let (value, status) = BoilingTemperature::evaluate_parallel(&eos.0, input.as_array());
        (value.to_pyarray(py), status.to_pyarray(py))
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
    #[staticmethod]
    pub fn boiling_temperature_derivatives<'py>(
        model: PyEquationOfStateAD,
        parameter_names: &Bound<'py, PyAny>,
        parameters: &Bound<'py, PyAny>,
        input: PyReadonlyArray2<f64>,
    ) -> GradResult<'py> {
        _boiling_temperature_derivatives(model, parameter_names, parameters, input)
    }

    /// Calculate liquid densities in parallel.
    ///
    /// Parameters
    /// ----------
    /// eos: EquationOfState
    ///     The equation of state to use.
    /// input: np.ndarray[float]
    ///     The temperature (in K) and pressure (in Pa) for every data point.
    ///
    /// Returns
    /// -------
    /// (np.ndarray[float], np.ndarray[bool]): The liquid densities (in kmol/m³), and convergence status.
    #[staticmethod]
    pub fn liquid_density<'py>(
        py: Python<'py>,
        eos: &PyEquationOfState,
        input: PyReadonlyArray2<f64>,
    ) -> EvalResult<'py> {
        let (value, status) = LiquidDensity::evaluate_parallel(&eos.0, input.as_array());
        (value.to_pyarray(py), status.to_pyarray(py))
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
    #[staticmethod]
    pub fn liquid_density_derivatives<'py>(
        model: PyEquationOfStateAD,
        parameter_names: &Bound<'py, PyAny>,
        parameters: &Bound<'py, PyAny>,
        input: PyReadonlyArray2<f64>,
    ) -> GradResult<'py> {
        _liquid_density_derivatives(model, parameter_names, parameters, input)
    }

    /// Calculate liquid densities at saturation in parallel.
    ///
    /// Parameters
    /// ----------
    /// eos: EquationOfState
    ///     The equation of state to use.
    /// input: np.ndarray[float]
    ///     The temperature (in K) for every data point.
    ///
    /// Returns
    /// -------
    /// (np.ndarray[float], np.ndarray[bool]): The liquid densities (in kmol/m³), and convergence status.
    #[staticmethod]
    pub fn equilibrium_liquid_density<'py>(
        py: Python<'py>,
        eos: &PyEquationOfState,
        input: PyReadonlyArray2<f64>,
    ) -> EvalResult<'py> {
        let (value, status) = EquilibriumLiquidDensity::evaluate_parallel(&eos.0, input.as_array());
        (value.to_pyarray(py), status.to_pyarray(py))
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
    #[staticmethod]
    pub fn equilibrium_liquid_density_derivatives<'py>(
        model: PyEquationOfStateAD,
        parameter_names: &Bound<'py, PyAny>,
        parameters: &Bound<'py, PyAny>,
        input: PyReadonlyArray2<f64>,
    ) -> GradResult<'py> {
        _equilibrium_liquid_density_derivatives(model, parameter_names, parameters, input)
    }

    /// Calculate enthalpy of vaporization in parallel.
    ///
    /// Parameters
    /// ----------
    /// eos: EquationOfState
    ///     The equation of state to use.
    /// input: np.ndarray[float]
    ///     The temperature (in K) for every data point.
    ///
    /// Returns
    /// -------
    /// (np.ndarray[float], np.ndarray[bool]):
    ///     The enthalpies of vaporization (in J/mol), and convergence status.
    #[staticmethod]
    pub fn enthalpy_of_vaporization<'py>(
        py: Python<'py>,
        eos: &PyEquationOfState,
        input: PyReadonlyArray2<f64>,
    ) -> EvalResult<'py> {
        let (value, status) = EnthalpyOfVaporization::evaluate_parallel(&eos.0, input.as_array());
        (value.to_pyarray(py), status.to_pyarray(py))
    }

    /// Calculate enthalpy of vaporization and derivatives w.r.t. model parameters.
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
    /// (np.ndarray[float], np.ndarray[float], np.ndarray[bool]):
    ///     The enthalpies of vaporization (in J/mol), gradients, and convergence status.
    #[staticmethod]
    pub fn enthalpy_of_vaporization_derivatives<'py>(
        model: PyEquationOfStateAD,
        parameter_names: &Bound<'py, PyAny>,
        parameters: &Bound<'py, PyAny>,
        input: PyReadonlyArray2<f64>,
    ) -> GradResult<'py> {
        _enthalpy_of_vaporization_derivatives(model, parameter_names, parameters, input)
    }

    /// Calculate residual isobaric molar heat capacities (liquid phase) in parallel.
    ///
    /// Parameters
    /// ----------
    /// eos: EquationOfState
    ///     The equation of state to use.
    /// input: np.ndarray[float]
    ///     The temperature (in K) and pressure (in Pa) for every data point.
    ///
    /// Returns
    /// -------
    /// (np.ndarray[float], np.ndarray[bool]):
    ///     The residual isobaric heat capacities (in J/(mol·K)), and convergence status.
    #[staticmethod]
    pub fn residual_isobaric_heat_capacity<'py>(
        py: Python<'py>,
        eos: &PyEquationOfState,
        input: PyReadonlyArray2<f64>,
    ) -> EvalResult<'py> {
        let (value, status) =
            ResidualIsobaricHeatCapacity::evaluate_parallel(&eos.0, input.as_array());
        (value.to_pyarray(py), status.to_pyarray(py))
    }

    /// Calculate residual isobaric molar heat capacities (liquid phase) and derivatives w.r.t. model parameters.
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
    /// (np.ndarray[float], np.ndarray[float], np.ndarray[bool]):
    ///     The residual isobaric heat capacities (in J/(mol·K)), gradients, and convergence status.
    #[staticmethod]
    pub fn residual_isobaric_heat_capacity_derivatives<'py>(
        model: PyEquationOfStateAD,
        parameter_names: &Bound<'py, PyAny>,
        parameters: &Bound<'py, PyAny>,
        input: PyReadonlyArray2<f64>,
    ) -> GradResult<'py> {
        _residual_isobaric_heat_capacity_derivatives(model, parameter_names, parameters, input)
    }

    /// Calculate bubble point pressures of binary mixtures in parallel.
    ///
    /// Parameters
    /// ----------
    /// eos: EquationOfState
    ///     The equation of state to use.
    /// input: np.ndarray[float]
    ///     The temperature (in K), composition of the first component, and an initial guess for the
    ///     pressure (in Pa) for every data point.
    ///
    /// Returns
    /// -------
    /// (np.ndarray[float], np.ndarray[bool]): The bubble point pressures (in Pa), and convergence status.
    #[staticmethod]
    pub fn bubble_point_pressure<'py>(
        py: Python<'py>,
        eos: &PyEquationOfState,
        input: PyReadonlyArray2<f64>,
    ) -> EvalResult<'py> {
        let (value, status) = BubblePointPressure::evaluate_parallel(&eos.0, input.as_array());
        (value.to_pyarray(py), status.to_pyarray(py))
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
    #[staticmethod]
    pub fn bubble_point_pressure_derivatives<'py>(
        model: PyEquationOfStateAD,
        parameter_names: &Bound<'py, PyAny>,
        parameters: &Bound<'py, PyAny>,
        input: PyReadonlyArray2<f64>,
    ) -> GradResult<'py> {
        _bubble_point_pressure_derivatives(model, parameter_names, parameters, input)
    }

    /// Calculate dew point pressures of binary mixtures in parallel.
    ///
    /// Parameters
    /// ----------
    /// eos: EquationOfState
    ///     The equation of state to use.
    /// input: np.ndarray[float]
    ///     The temperature (in K), composition of the first component, and an initial guess for the
    ///     pressure (in Pa) for every data point.
    ///
    /// Returns
    /// -------
    /// (np.ndarray[float], np.ndarray[bool]): The dew point pressures (in Pa), and convergence status.
    #[staticmethod]
    pub fn dew_point_pressure<'py>(
        py: Python<'py>,
        eos: &PyEquationOfState,
        input: PyReadonlyArray2<f64>,
    ) -> EvalResult<'py> {
        let (value, status) = DewPointPressure::evaluate_parallel(&eos.0, input.as_array());
        (value.to_pyarray(py), status.to_pyarray(py))
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
    #[staticmethod]
    pub fn dew_point_pressure_derivatives<'py>(
        model: PyEquationOfStateAD,
        parameter_names: &Bound<'py, PyAny>,
        parameters: &Bound<'py, PyAny>,
        input: PyReadonlyArray2<f64>,
    ) -> GradResult<'py> {
        _dew_point_pressure_derivatives(model, parameter_names, parameters, input)
    }
}

macro_rules! expand_models {
    ($enum:ty, $prop:ident, $($model:ident: $type:ty),*) => {
        paste!(
        #[pyfunction]
        fn [<_ $prop _derivatives>]<'py>(
            model: PyEquationOfStateAD,
            parameter_names: &Bound<'py, PyAny>,
            parameters: &Bound<'py, PyAny>,
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
    (pure, [$($prop:ident: $prop_type:ty),*], $models:tt) => {
        $(impl_evaluate_gradients!(U1,PyEquationOfStateAD,$prop,$prop_type,$models,0,1,2,3,4,5,max:6);)*
    };
    (binary, [$($prop:ident: $prop_type:ty),*], $models:tt) => {
        $(impl_evaluate_gradients!(U2,BinaryModels,$prop,$prop_type,$models,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,max:15);)*
    };
    ($n:ty, $enum:ty, $prop:ident, $prop_type:ty, {$($model:ident: $type:ty),*}, $($p:literal,)* max: $max:literal) => {
        expand_models!($enum, $prop, $($model: $type),*);
        fn $prop<'py, R: ParametersAD<$n>>(
            parameter_names: &Bound<'py, PyAny>,
            parameters: &Bound<'py, PyAny>,
            input: PyReadonlyArray2<f64>,
        ) -> (
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray2<f64>>,
            Bound<'py, PyArray1<bool>>,
        )
        where
            $(R::Lifted<DualSVec<f64, f64, $p>>: Sync,)*
            R::Lifted<DualSVec<f64, f64, $max>>: Sync
        {
            let (value, grad, status) =
            if let Ok(pars) = parameters.extract::<PyReadonlyArray1<f64>>() {
                let pars = pars.as_slice().expect("Parameter array is not contiguous!");
                $(
                if let Ok(p) = parameter_names.extract::<[String; $p]>() {
                    <$prop_type>::evaluate_parallel_derivatives::<R, $p>(p, pars, input.as_array())
                } else)* if let Ok(p) = parameter_names.extract::<[String; $max]>() {
                    <$prop_type>::evaluate_parallel_derivatives::<R, $max>(p, pars, input.as_array())
                } else {
                    panic!("Gradients can only be evaluated for up to {} parameters!", $max)
                }
            } else if let Ok(pars) = parameters.extract::<PyReadonlyArray2<f64>>() {
                $(
                if let Ok(p) = parameter_names.extract::<[String; $p]>() {
                    <$prop_type>::evaluate_parallel_derivatives_params::<R, $p>(p, pars.as_array(), input.as_array())
                } else)* if let Ok(p) = parameter_names.extract::<[String; $max]>() {
                    <$prop_type>::evaluate_parallel_derivatives_params::<R, $max>(p, pars.as_array(), input.as_array())
                } else {
                    panic!("Gradients can only be evaluated for up to {} parameters!", $max)
                }
            } else {
                panic!("Argument `parameters` needs to be a 1D or 2D array!")
            };
            (
                value.to_pyarray(parameter_names.py()),
                grad.to_pyarray(parameter_names.py()),
                status.to_pyarray(parameter_names.py()),
            )
        }
    };
}

impl_evaluate_gradients!(
    pure,
    [vapor_pressure: VaporPressure, boiling_temperature: BoilingTemperature, liquid_density: LiquidDensity, equilibrium_liquid_density: EquilibriumLiquidDensity, enthalpy_of_vaporization: EnthalpyOfVaporization, residual_isobaric_heat_capacity: ResidualIsobaricHeatCapacity],
    {PcSaftNonAssoc: PcSaftPure<f64, 4>, PcSaftFull: PcSaftPure<f64, 8>}
);

impl_evaluate_gradients!(
    binary,
    [bubble_point_pressure: BubblePointPressure, dew_point_pressure: DewPointPressure],
    {PcSaftNonAssoc: PcSaftBinary<f64, 4>, PcSaftFull: PcSaftBinary<f64, 8>}
);
