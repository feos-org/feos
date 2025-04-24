use crate::eos::PyEquationOfState;
use crate::error::PyFeosError;
use crate::ideal_gas::IdealGasModel;
use crate::residual::ResidualModel;
use crate::PyVerbosity;
use feos::estimator::{
    BinaryPhaseDiagram, BinaryVleChemicalPotential, BinaryVlePressure, DataSet, Diffusion,
    EquilibriumLiquidDensity, Estimator, LiquidDensity, Loss, Phase, VaporPressure,
};
use feos_core::EquationOfState;
use ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyErr;
use quantity::*;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(name = "Loss", eq)]
pub(crate) enum PyLoss {
    Linear(),
    SoftL1(f64),
    Huber(f64),
    Cauchy(f64),
    Arctan(f64),
}

impl From<Loss> for PyLoss {
    fn from(value: Loss) -> Self {
        use Loss::*;
        match value {
            Linear => Self::Linear(),
            SoftL1(s) => Self::SoftL1(s),
            Huber(s) => Self::Huber(s),
            Cauchy(s) => Self::Cauchy(s),
            Arctan(s) => Self::Arctan(s),
        }
    }
}

impl From<PyLoss> for Loss {
    fn from(value: PyLoss) -> Self {
        use PyLoss::*;
        match value {
            Linear() => Self::Linear,
            SoftL1(s) => Self::SoftL1(s),
            Huber(s) => Self::Huber(s),
            Cauchy(s) => Self::Cauchy(s),
            Arctan(s) => Self::Arctan(s),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(name = "Phase", eq, eq_int)]
pub(crate) enum PyPhase {
    Vapor,
    Liquid,
}

impl From<Phase> for PyPhase {
    fn from(value: Phase) -> Self {
        use Phase::*;
        match value {
            Vapor => Self::Vapor,
            Liquid => Self::Liquid,
        }
    }
}

impl From<PyPhase> for Phase {
    fn from(value: PyPhase) -> Self {
        use PyPhase::*;
        match value {
            Vapor => Self::Vapor,
            Liquid => Self::Liquid,
        }
    }
}

/// A collection of experimental data that can be used to compute
/// cost functions and make predictions using an equation of state.
#[pyclass(name = "DataSet")]
#[derive(Clone)]
pub struct PyDataSet(Arc<dyn DataSet<EquationOfState<IdealGasModel, ResidualModel>>>);

#[pymethods]
impl PyDataSet {
    /// Compute the cost function for each input value.
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state that is used.
    /// loss : Loss
    ///     The loss function that is applied to residuals
    ///     to handle outliers.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray[Float]
    ///     The cost function evaluated for each experimental data point.
    ///
    /// Note
    /// ----
    /// The cost function that is used depends on the
    /// property. For most properties it is the absolute relative difference.
    /// See the constructors of the respective properties
    /// to learn about the cost functions that are used.
    #[pyo3(text_signature = "($self, eos, loss)")]
    fn cost<'py>(
        &self,
        eos: &PyEquationOfState,
        loss: PyLoss,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .0
            .cost(&eos.0, loss.into())
            .map_err(PyFeosError::from)?
            .view()
            .to_pyarray(py))
    }

    /// Return the property of interest for each data point
    /// of the input as computed by the equation of state.
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state that is used.
    ///
    /// Returns
    /// -------
    /// SIArray1
    #[pyo3(text_signature = "($self, eos)")]
    fn predict<'py>(
        &self,
        eos: &PyEquationOfState,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .0
            .predict(&eos.0)
            .map_err(PyFeosError::from)?
            .view()
            .to_pyarray(py))
    }

    /// Return the relative difference between experimental data
    /// and prediction of the equation of state.
    ///
    /// The relative difference is computed as:
    ///
    /// .. math:: \text{Relative Difference} = \frac{x_i^\text{prediction} - x_i^\text{experiment}}{x_i^\text{experiment}}
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state that is used.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray[Float]
    #[pyo3(text_signature = "($self, eos)")]
    fn relative_difference<'py>(
        &self,
        eos: &PyEquationOfState,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .0
            .relative_difference(&eos.0)
            .map_err(PyFeosError::from)?
            .view()
            .to_pyarray(py))
    }

    /// Return the mean absolute relative difference.
    ///
    /// The mean absolute relative difference is computed as:
    ///
    /// .. math:: \text{MARD} = \frac{1}{N}\sum_{i=1}^{N} \left|\frac{x_i^\text{prediction} - x_i^\text{experiment}}{x_i^\text{experiment}} \right|
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state that is used.
    ///
    /// Returns
    /// -------
    /// Float
    #[pyo3(text_signature = "($self, eos)")]
    fn mean_absolute_relative_difference(&self, eos: &PyEquationOfState) -> PyResult<f64> {
        Ok(self
            .0
            .mean_absolute_relative_difference(&eos.0)
            .map_err(PyFeosError::from)?)
    }

    /// Create a DataSet with experimental data for vapor pressure.
    ///
    /// Parameters
    /// ----------
    /// target : SIArray1
    ///     Experimental data for vapor pressure.
    /// temperature : SIArray1
    ///     Temperature for experimental data points.
    /// extrapolate : bool, optional
    ///     Use Antoine type equation to extrapolate vapor
    ///     pressure if experimental data is above critial
    ///     point of model. Defaults to False.
    /// critical_temperature : SINumber, optional
    ///     Estimate of the critical temperature used as initial
    ///     value for critical point calculation. Defaults to None.
    ///     For additional information, see note.
    /// max_iter : int, optional
    ///     The maximum number of iterations for critical point
    ///     and VLE algorithms.
    /// tol: float, optional
    ///     Solution tolerance for critical point
    ///     and VLE algorithms.
    /// verbosity : Verbosity, optional
    ///     Verbosity for critical point
    ///     and VLE algorithms.
    ///
    /// Returns
    /// -------
    /// ``DataSet``
    ///
    /// Note
    /// ----
    /// If no critical temperature is provided, the maximum of the `temperature` input
    /// is used. If that fails, the default temperatures of the critical point routine
    /// are used.
    #[staticmethod]
    #[pyo3(
        text_signature = "(target, temperature, extrapolate, critical_temperature=None, max_iter=None, tol=None, verbosity=None)"
    )]
    #[pyo3(signature = (target, temperature, extrapolate, critical_temperature=None, max_iter=None, tol=None, verbosity=None))]
    fn vapor_pressure(
        target: Pressure<Array1<f64>>,
        temperature: Temperature<Array1<f64>>,
        extrapolate: Option<bool>,
        critical_temperature: Option<Temperature>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        verbosity: Option<PyVerbosity>,
    ) -> Self {
        Self(Arc::new(VaporPressure::new(
            target,
            temperature,
            extrapolate.unwrap_or(false),
            critical_temperature,
            Some((max_iter, tol, verbosity.map(|v| v.into())).into()),
        )))
    }

    /// Create a DataSet with experimental data for liquid density.
    ///
    /// Parameters
    /// ----------
    /// target : SIArray1
    ///     Experimental data for liquid density.
    /// temperature : SIArray1
    ///     Temperature for experimental data points.
    /// pressure : SIArray1
    ///     Pressure for experimental data points.
    ///
    /// Returns
    /// -------
    /// DataSet
    #[staticmethod]
    #[pyo3(text_signature = "(target, temperature, pressure)")]
    fn liquid_density(
        target: MassDensity<Array1<f64>>,
        temperature: Temperature<Array1<f64>>,
        pressure: Pressure<Array1<f64>>,
    ) -> Self {
        Self(Arc::new(LiquidDensity::new(target, temperature, pressure)))
    }

    /// Create a DataSet with experimental data for liquid density
    /// for a vapor liquid equilibrium.
    ///
    /// Parameters
    /// ----------
    /// target : SIArray1
    ///     Experimental data for liquid density.
    /// temperature : SIArray1
    ///     Temperature for experimental data points.
    /// max_iter : int, optional
    ///     The maximum number of iterations for critical point
    ///     and VLE algorithms.
    /// tol: float, optional
    ///     Solution tolerance for critical point
    ///     and VLE algorithms.
    /// verbosity : Verbosity, optional
    ///     Verbosity for critical point
    ///     and VLE algorithms.
    ///
    /// Returns
    /// -------
    /// DataSet
    #[staticmethod]
    #[pyo3(text_signature = "(target, temperature, max_iter=None, tol=None, verbosity=None)")]
    #[pyo3(signature = (target, temperature, max_iter=None, tol=None, verbosity=None))]
    fn equilibrium_liquid_density(
        target: MassDensity<Array1<f64>>,
        temperature: Temperature<Array1<f64>>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        verbosity: Option<PyVerbosity>,
    ) -> Self {
        Self(Arc::new(EquilibriumLiquidDensity::new(
            target,
            temperature,
            Some((max_iter, tol, verbosity.map(|v| v.into())).into()),
        )))
    }

    /// Create a DataSet with experimental data for binary
    /// phase equilibria using the chemical potential residual.
    ///
    /// Parameters
    /// ----------
    /// temperature : SIArray1
    ///     Temperature of the experimental data points.
    /// pressure : SIArray1
    ///     Pressure of the experimental data points.
    /// liquid_molefracs : np.array[float]
    ///     Molar composition of component 1 in the liquid phase.
    /// vapor_molefracs : np.array[float]
    ///     Molar composition of component 1 in the vapor phase.
    ///
    /// Returns
    /// -------
    /// DataSet
    #[staticmethod]
    #[pyo3(text_signature = "(temperature, pressure, liquid_molefracs, vapor_molefracs)")]
    fn binary_vle_chemical_potential(
        temperature: Temperature<Array1<f64>>,
        pressure: Pressure<Array1<f64>>,
        liquid_molefracs: &Bound<'_, PyArray1<f64>>,
        vapor_molefracs: &Bound<'_, PyArray1<f64>>,
    ) -> Self {
        Self(Arc::new(BinaryVleChemicalPotential::new(
            temperature,
            pressure,
            liquid_molefracs.to_owned_array(),
            vapor_molefracs.to_owned_array(),
        )))
    }

    /// Create a DataSet with experimental data for binary
    /// phase equilibria using the pressure residual.
    ///
    /// Parameters
    /// ----------
    /// temperature : SIArray1
    ///     Temperature of the experimental data points.
    /// pressure : SIArray1
    ///     Pressure of the experimental data points.
    /// molefracs : np.array[float]
    ///     Molar composition of component 1 in the considered phase.
    /// phase : Phase
    ///     The phase of the experimental data points.
    ///
    /// Returns
    /// -------
    /// DataSet
    #[staticmethod]
    #[pyo3(text_signature = "(temperature, pressure, molefracs, phase)")]
    fn binary_vle_pressure(
        temperature: Temperature<Array1<f64>>,
        pressure: Pressure<Array1<f64>>,
        molefracs: &Bound<'_, PyArray1<f64>>,
        phase: PyPhase,
    ) -> Self {
        Self(Arc::new(BinaryVlePressure::new(
            temperature,
            pressure,
            molefracs.to_owned_array(),
            phase.into(),
        )))
    }

    /// Create a DataSet with experimental data for binary
    /// phase diagrams using the distance residual.
    ///
    /// Parameters
    /// ----------
    /// specification : SINumber
    ///     The constant temperature/pressure of the isotherm/isobar.
    /// temperature_or_pressure : SIArray1
    ///     The temperature (isobar) or pressure (isotherm) of the
    ///     experimental data points.
    /// liquid_molefracs : np.array[float], optional
    ///     Molar composition of component 1 in the liquid phase.
    /// vapor_molefracs : np.array[float], optional
    ///     Molar composition of component 1 in the vapor phase.
    /// npoints : int, optional
    ///     The resolution of the phase diagram used to calculate
    ///     the distance residual.
    ///
    /// Returns
    /// -------
    /// DataSet
    #[staticmethod]
    #[pyo3(
        text_signature = "(specification, temperature_or_pressure, liquid_molefracs=None, vapor_molefracs=None, npoints=None)"
    )]
    #[pyo3(signature = (specification, temperature_or_pressure, liquid_molefracs=None, vapor_molefracs=None, npoints=None))]
    fn binary_phase_diagram(
        specification: Bound<'_, PyAny>,
        temperature_or_pressure: Bound<'_, PyAny>,
        liquid_molefracs: Option<&Bound<'_, PyArray1<f64>>>,
        vapor_molefracs: Option<&Bound<'_, PyArray1<f64>>>,
        npoints: Option<usize>,
    ) -> PyResult<Self> {
        if let Ok(t) = specification.extract::<Temperature>() {
            Ok(Self(Arc::new(BinaryPhaseDiagram::new(
                t,
                temperature_or_pressure.extract()?,
                liquid_molefracs.map(|x| x.to_owned_array()),
                vapor_molefracs.map(|x| x.to_owned_array()),
                npoints,
            ))))
        } else if let Ok(p) = specification.extract::<Pressure>() {
            Ok(Self(Arc::new(BinaryPhaseDiagram::new(
                p,
                temperature_or_pressure.extract()?,
                liquid_molefracs.map(|x| x.to_owned_array()),
                vapor_molefracs.map(|x| x.to_owned_array()),
                npoints,
            ))))
        } else {
            Err(PyErr::new::<PyValueError, _>(format!(
                "Wrong units! Expected K or Pa, got {}.",
                temperature_or_pressure.call_method0("__repr__")?
            )))
        }
    }

    /// Return `target` as ``SIArray1``.
    #[getter]
    fn get_target<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.0.target().view().to_pyarray(py)
    }

    /// Return number of stored data points.
    #[getter]
    fn get_datapoints(&self) -> usize {
        self.0.datapoints()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

/// A collection of `DataSet`s that can be used to compute metrics for experimental data.
///
/// Parameters
/// ----------
/// data : List[DataSet]
///     The properties and experimental data points to add to
///     the estimator.
/// weights : List[float]
///     The weight of each property. When computing the cost function,
///     the weights are normalized (sum of weights equals unity).
/// losses : List[Loss]
///     The loss functions for each property.
///
/// Returns
/// -------
/// Estimator
#[pyclass(name = "Estimator")]
pub struct PyEstimator(Estimator<EquationOfState<IdealGasModel, ResidualModel>>);

#[pymethods]
impl PyEstimator {
    #[new]
    #[pyo3(text_signature = "(data, weights, losses)")]
    fn new(data: Vec<PyDataSet>, weights: Vec<f64>, losses: Vec<PyLoss>) -> Self {
        Self(Estimator::new(
            data.iter().map(|d| d.0.clone()).collect(),
            weights,
            losses.iter().map(|&l| l.into()).collect(),
        ))
    }

    /// Compute the cost function for each ``DataSet``.
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state that is used.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray[Float]
    ///     The cost function evaluated for each experimental data point
    ///     of each ``DataSet``.
    ///
    /// Note
    /// ----
    /// The cost function is:
    ///
    /// - The relative difference between prediction and target value,
    /// - to which a loss function is applied,
    /// - and which is weighted according to the number of datapoints,
    /// - and the relative weights as defined in the Estimator object.
    #[pyo3(text_signature = "($self, eos)")]
    fn cost<'py>(
        &self,
        eos: &PyEquationOfState,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .0
            .cost(&eos.0)
            .map_err(PyFeosError::from)?
            .view()
            .to_pyarray(py))
    }

    /// Return the properties as computed by the
    /// equation of state for each `DataSet`.
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state that is used.
    ///
    /// Returns
    /// -------
    /// List[SIArray1]
    #[pyo3(text_signature = "($self, eos)")]
    fn predict<'py>(
        &self,
        eos: &PyEquationOfState,
        py: Python<'py>,
    ) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
        Ok(self
            .0
            .predict(&eos.0)
            .map_err(PyFeosError::from)?
            .iter()
            .map(|d| d.view().to_pyarray(py))
            .collect())
    }

    /// Return the relative difference between experimental data
    /// and prediction of the equation of state for each ``DataSet``.
    ///
    /// The relative difference is computed as:
    ///
    /// .. math:: \text{Relative Difference} = \frac{x_i^\text{prediction} - x_i^\text{experiment}}{x_i^\text{experiment}}
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state that is used.
    ///
    /// Returns
    /// -------
    /// List[numpy.ndarray[Float]]
    #[pyo3(text_signature = "($self, eos)")]
    fn relative_difference<'py>(
        &self,
        eos: &PyEquationOfState,
        py: Python<'py>,
    ) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
        Ok(self
            .0
            .relative_difference(&eos.0)
            .map_err(PyFeosError::from)?
            .iter()
            .map(|d| d.view().to_pyarray(py))
            .collect())
    }

    /// Return the mean absolute relative difference for each ``DataSet``.
    ///
    /// The mean absolute relative difference is computed as:
    ///
    /// .. math:: \text{MARD} = \frac{1}{N}\sum_{i=1}^{N} \left|\frac{x_i^\text{prediction} - x_i^\text{experiment}}{x_i^\text{experiment}} \right|
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state that is used.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray[Float]
    #[pyo3(text_signature = "($self, eos)")]
    fn mean_absolute_relative_difference<'py>(
        &self,
        eos: &PyEquationOfState,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .0
            .mean_absolute_relative_difference(&eos.0)
            .map_err(PyFeosError::from)?
            .view()
            .to_pyarray(py))
    }

    /// Return the stored ``DataSet``s.
    ///
    /// Returns
    /// -------
    /// List[DataSet]
    #[getter]
    fn get_datasets(&self) -> Vec<PyDataSet> {
        self.0
            .datasets()
            .iter()
            .cloned()
            .map(|ds| PyDataSet(ds))
            .collect()
    }

    fn _repr_markdown_(&self) -> String {
        self.0._repr_markdownn_()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

#[pymethods]
impl PyDataSet {
    /// Create a DataSet with experimental data for viscosity.
    ///
    /// Parameters
    /// ----------
    /// target : SIArray1
    ///     Experimental data for viscosity.
    /// temperature : SIArray1
    ///     Temperature for experimental data points.
    /// pressure : SIArray1
    ///     Pressure for experimental data points.
    /// phase : List[Phase], optional
    ///     Phase of data. Used to determine the starting
    ///     density for the density iteration. If provided,
    ///     resulting states may not be stable.
    ///
    /// Returns
    /// -------
    /// DataSet
    #[staticmethod]
    #[pyo3(text_signature = "(target, temperature, pressure, phase=None)")]
    #[pyo3(signature = (target, temperature, pressure, phase=None))]
    fn viscosity(
        target: quantity::Viscosity<Array1<f64>>,
        temperature: Temperature<Array1<f64>>,
        pressure: Pressure<Array1<f64>>,
        phase: Option<Vec<PyPhase>>,
    ) -> Self {
        let p = phase.map(|p| (p.into_iter().map(|p| p.into()).collect()));
        Self(Arc::new(feos::estimator::Viscosity::new(
            target,
            temperature,
            pressure,
            p.as_ref(),
        )))
    }

    /// Create a DataSet with experimental data for thermal conductivity.
    ///
    /// Parameters
    /// ----------
    /// target : SIArray1
    ///     Experimental data for thermal conductivity.
    /// temperature : SIArray1
    ///     Temperature for experimental data points.
    /// pressure : SIArray1
    ///     Pressure for experimental data points.
    /// phase : List[Phase], optional
    ///     Phase of data. Used to determine the starting
    ///     density for the density iteration. If provided,
    ///     resulting states may not be stable.
    ///
    /// Returns
    /// -------
    /// DataSet
    #[staticmethod]
    #[pyo3(text_signature = "(target, temperature, pressure, phase=None)")]
    #[pyo3(signature = (target, temperature, pressure, phase=None))]
    fn thermal_conductivity(
        target: quantity::ThermalConductivity<Array1<f64>>,
        temperature: Temperature<Array1<f64>>,
        pressure: Pressure<Array1<f64>>,
        phase: Option<Vec<PyPhase>>,
    ) -> Self {
        let p = phase.map(|p| (p.into_iter().map(|p| p.into()).collect()));
        Self(Arc::new(feos::estimator::ThermalConductivity::new(
            target,
            temperature,
            pressure,
            p.as_ref(),
        )))
    }

    /// Create a DataSet with experimental data for diffusion coefficient.
    ///
    /// Parameters
    /// ----------
    /// target : SIArray1
    ///     Experimental data for diffusion coefficient.
    /// temperature : SIArray1
    ///     Temperature for experimental data points.
    /// pressure : SIArray1
    ///     Pressure for experimental data points.
    /// phase : List[Phase], optional
    ///     Phase of data. Used to determine the starting
    ///     density for the density iteration. If provided,
    ///     resulting states may not be stable.
    ///
    /// Returns
    /// -------
    /// DataSet
    #[staticmethod]
    #[pyo3(text_signature = "(target, temperature, pressure, phase=None)")]
    #[pyo3(signature = (target, temperature, pressure, phase=None))]
    fn diffusion(
        target: quantity::Diffusivity<Array1<f64>>,
        temperature: Temperature<Array1<f64>>,
        pressure: Pressure<Array1<f64>>,
        phase: Option<Vec<PyPhase>>,
    ) -> Self {
        let p = phase.map(|p| (p.into_iter().map(|p| p.into()).collect()));
        Self(Arc::new(Diffusion::new(
            target,
            temperature,
            pressure,
            p.as_ref(),
        )))
    }
}
