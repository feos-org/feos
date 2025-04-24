use crate::{
    eos::PyEquationOfState,
    error::PyFeosError,
    ideal_gas::IdealGasModel,
    residual::ResidualModel,
    state::{PyContributions, PyState, PyStateVec},
    PyVerbosity,
};
use feos_core::{
    Components, Contributions, EquationOfState, PhaseDiagram, PhaseDiagramHetero, PhaseEquilibrium,
};
use ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use quantity::*;
use std::collections::HashMap;
use typenum::P3;

/// A thermodynamic two phase equilibrium state.
#[pyclass(name = "PhaseEquilibrium")]
#[derive(Clone)]
pub struct PyPhaseEquilibrium(
    pub PhaseEquilibrium<EquationOfState<IdealGasModel, ResidualModel>, 2>,
);

#[pymethods]
impl PyPhaseEquilibrium {
    /// Create a liquid and vapor state in equilibrium
    /// for a pure substance.
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state.
    /// temperature_or_pressure : SINumber
    ///     The system temperature or pressure.
    /// initial_state : PhaseEquilibrium, optional
    ///     A phase equilibrium used as initial guess.
    ///     Can speed up convergence.
    /// max_iter : int, optional
    ///     The maximum number of iterations.
    /// tol: float, optional
    ///     The solution tolerance.
    /// verbosity : Verbosity, optional
    ///     The verbosity.
    ///
    /// Returns
    /// -------
    /// PhaseEquilibrium
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     When pressure iteration fails or no phase equilibrium is found.
    #[staticmethod]
    #[pyo3(
        text_signature = "(eos, temperature_or_pressure, initial_state=None, max_iter=None, tol=None, verbosity=None)"
    )]
    #[pyo3(signature = (eos, temperature_or_pressure, initial_state=None, max_iter=None, tol=None, verbosity=None))]
    pub(crate) fn pure(
        eos: PyEquationOfState,
        temperature_or_pressure: Bound<'_, PyAny>,
        initial_state: Option<&PyPhaseEquilibrium>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        verbosity: Option<PyVerbosity>,
    ) -> PyResult<Self> {
        if let Ok(t) = temperature_or_pressure.extract::<Temperature>() {
            Ok(Self(
                PhaseEquilibrium::pure(
                    &eos.0,
                    t,
                    initial_state.map(|s| &s.0),
                    (max_iter, tol, verbosity.map(|v| v.into())).into(),
                )
                .map_err(PyFeosError::from)?,
            ))
        } else if let Ok(p) = temperature_or_pressure.extract::<Pressure>() {
            Ok(Self(
                PhaseEquilibrium::pure(
                    &eos.0,
                    p,
                    initial_state.map(|s| &s.0),
                    (max_iter, tol, verbosity.map(|v| v.into())).into(),
                )
                .map_err(PyFeosError::from)?,
            ))
        } else {
            Err(PyErr::new::<PyValueError, _>(format!(
                "Wrong units! Expected K or Pa, got {}.",
                temperature_or_pressure.call_method0("__repr__")?
            )))
        }
    }

    /// Create a liquid and vapor state in equilibrium
    /// for given temperature, pressure and feed composition.
    ///
    /// Can also be used to calculate liquid liquid phase separation.
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state.
    /// temperature : SINumber
    ///     The system temperature.
    /// pressure : SINumber
    ///     The system pressure.
    /// feed : SIArray1
    ///     Feed composition (units of amount of substance).
    /// initial_state : PhaseEquilibrium, optional
    ///     A phase equilibrium used as initial guess.
    ///     Can speed up convergence.
    /// max_iter : int, optional
    ///     The maximum number of iterations.
    /// tol: float, optional
    ///     The solution tolerance.
    /// verbosity : Verbosity, optional
    ///     The verbosity.
    ///
    /// Returns
    /// -------
    /// PhaseEquilibrium
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     When pressure iteration fails or no phase equilibrium is found.
    #[staticmethod]
    #[pyo3(
        text_signature = "(eos, temperature, pressure, feed, initial_state=None, max_iter=None, tol=None, verbosity=None, non_volatile_components=None)"
    )]
    #[pyo3(signature = (eos, temperature, pressure, feed, initial_state=None, max_iter=None, tol=None, verbosity=None, non_volatile_components=None))]
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn tp_flash(
        eos: PyEquationOfState,
        temperature: Temperature,
        pressure: Pressure,
        feed: Moles<Array1<f64>>,
        initial_state: Option<&PyPhaseEquilibrium>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        verbosity: Option<PyVerbosity>,
        non_volatile_components: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        Ok(Self(
            PhaseEquilibrium::tp_flash(
                &eos.0,
                temperature,
                pressure,
                &feed,
                initial_state.map(|s| &s.0),
                (max_iter, tol, verbosity.map(|v| v.into())).into(),
                non_volatile_components,
            )
            .map_err(PyFeosError::from)?,
        ))
    }

    /// Compute a phase equilibrium for given temperature
    /// or pressure and liquid mole fractions.
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state.
    /// temperature_or_pressure : SINumber
    ///     The system temperature_or_pressure.
    /// liquid_molefracs : numpy.ndarray
    ///     The mole fraction of the liquid phase.
    /// tp_init : SINumber, optional
    ///     The system pressure/temperature used as starting
    ///     condition for the iteration.
    /// vapor_molefracs : numpy.ndarray, optional
    ///     The mole fraction of the vapor phase used as
    ///     starting condition for iteration.
    /// max_iter_inner : int, optional
    ///     The maximum number of inner iterations.
    /// max_iter_outer : int, optional
    ///     The maximum number of outer iterations.
    /// tol_inner : float, optional
    ///     The solution tolerance in the inner loop.
    /// tol_outer : float, optional
    ///     The solution tolerance in the outer loop.
    /// verbosity : Verbosity, optional
    ///     The verbosity.
    ///
    /// Returns
    /// -------
    /// PhaseEquilibrium
    #[staticmethod]
    #[pyo3(
        text_signature = "(eos, temperature_or_pressure, liquid_molefracs, tp_init=None, vapor_molefracs=None, max_iter_inner=None, max_iter_outer=None, tol_inner=None, tol_outer=None, verbosity=None)"
    )]
    #[pyo3(signature = (eos, temperature_or_pressure, liquid_molefracs, tp_init=None, vapor_molefracs=None, max_iter_inner=None, max_iter_outer=None, tol_inner=None, tol_outer=None, verbosity=None))]
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn bubble_point<'py>(
        eos: PyEquationOfState,
        temperature_or_pressure: Bound<'_, PyAny>,
        liquid_molefracs: &Bound<'py, PyArray1<f64>>,
        tp_init: Option<Bound<'_, PyAny>>,
        vapor_molefracs: Option<&Bound<'py, PyArray1<f64>>>,
        max_iter_inner: Option<usize>,
        max_iter_outer: Option<usize>,
        tol_inner: Option<f64>,
        tol_outer: Option<f64>,
        verbosity: Option<PyVerbosity>,
    ) -> PyResult<Self> {
        let x = vapor_molefracs.map(|m| m.to_owned_array());
        if let Ok(t) = temperature_or_pressure.extract::<Temperature>() {
            Ok(Self(
                PhaseEquilibrium::bubble_point(
                    &eos.0,
                    t,
                    &liquid_molefracs.to_owned_array(),
                    tp_init.map(|p| p.extract()).transpose()?,
                    x.as_ref(),
                    (
                        (max_iter_inner, tol_inner, verbosity.map(|v| v.into())).into(),
                        (max_iter_outer, tol_outer, verbosity.map(|v| v.into())).into(),
                    ),
                )
                .map_err(PyFeosError::from)?,
            ))
        } else if let Ok(p) = temperature_or_pressure.extract::<Pressure>() {
            Ok(Self(
                PhaseEquilibrium::bubble_point(
                    &eos.0,
                    p,
                    &liquid_molefracs.to_owned_array(),
                    tp_init.map(|p| p.extract()).transpose()?,
                    x.as_ref(),
                    (
                        (max_iter_inner, tol_inner, verbosity.map(|v| v.into())).into(),
                        (max_iter_outer, tol_outer, verbosity.map(|v| v.into())).into(),
                    ),
                )
                .map_err(PyFeosError::from)?,
            ))
        } else {
            Err(PyErr::new::<PyValueError, _>(format!(
                "Wrong units! Expected K or Pa, got {}.",
                temperature_or_pressure.call_method0("__repr__")?
            )))
        }
    }

    /// Compute a phase equilibrium for given temperature
    /// or pressure and vapor mole fractions.
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state.
    /// temperature_or_pressure : SINumber
    ///     The system temperature or pressure.
    /// vapor_molefracs : numpy.ndarray
    ///     The mole fraction of the vapor phase.
    /// tp_init : SINumber, optional
    ///     The system pressure/temperature used as starting
    ///     condition for the iteration.
    /// liquid_molefracs : numpy.ndarray, optional
    ///     The mole fraction of the liquid phase used as
    ///     starting condition for iteration.
    /// max_iter_inner : int, optional
    ///     The maximum number of inner iterations.
    /// max_iter_outer : int, optional
    ///     The maximum number of outer iterations.
    /// tol_inner : float, optional
    ///     The solution tolerance in the inner loop.
    /// tol_outer : float, optional
    ///     The solution tolerance in the outer loop.
    /// verbosity : Verbosity, optional
    ///     The verbosity.
    ///
    /// Returns
    /// -------
    /// PhaseEquilibrium
    #[staticmethod]
    #[pyo3(
        text_signature = "(eos, temperature_or_pressure, vapor_molefracs, tp_init=None, liquid_molefracs=None, max_iter_inner=None, max_iter_outer=None, tol_inner=None, tol_outer=None, verbosity=None)"
    )]
    #[pyo3(signature = (eos, temperature_or_pressure, vapor_molefracs, tp_init=None, liquid_molefracs=None, max_iter_inner=None, max_iter_outer=None, tol_inner=None, tol_outer=None, verbosity=None))]
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn dew_point<'py>(
        eos: PyEquationOfState,
        temperature_or_pressure: Bound<'_, PyAny>,
        vapor_molefracs: &Bound<'py, PyArray1<f64>>,
        tp_init: Option<Bound<'_, PyAny>>,
        liquid_molefracs: Option<&Bound<'py, PyArray1<f64>>>,
        max_iter_inner: Option<usize>,
        max_iter_outer: Option<usize>,
        tol_inner: Option<f64>,
        tol_outer: Option<f64>,
        verbosity: Option<PyVerbosity>,
    ) -> PyResult<Self> {
        let x = liquid_molefracs.map(|m| m.to_owned_array());
        if let Ok(t) = temperature_or_pressure.extract::<Temperature>() {
            Ok(Self(
                PhaseEquilibrium::dew_point(
                    &eos.0,
                    t,
                    &vapor_molefracs.to_owned_array(),
                    tp_init.map(|p| p.extract()).transpose()?,
                    x.as_ref(),
                    (
                        (max_iter_inner, tol_inner, verbosity.map(|v| v.into())).into(),
                        (max_iter_outer, tol_outer, verbosity.map(|v| v.into())).into(),
                    ),
                )
                .map_err(PyFeosError::from)?,
            ))
        } else if let Ok(p) = temperature_or_pressure.extract::<Pressure>() {
            Ok(Self(
                PhaseEquilibrium::dew_point(
                    &eos.0,
                    p,
                    &vapor_molefracs.to_owned_array(),
                    tp_init.map(|p| p.extract()).transpose()?,
                    x.as_ref(),
                    (
                        (max_iter_inner, tol_inner, verbosity.map(|v| v.into())).into(),
                        (max_iter_outer, tol_outer, verbosity.map(|v| v.into())).into(),
                    ),
                )
                .map_err(PyFeosError::from)?,
            ))
        } else {
            Err(PyErr::new::<PyValueError, _>(format!(
                "Wrong units! Expected K or Pa, got {}.",
                temperature_or_pressure.call_method0("__repr__")?
            )))
        }
    }

    /// Creates a new PhaseEquilibrium that contains two states at the
    /// specified temperature, pressure and moles.
    ///
    /// The constructor can be used in custom phase equilibrium solvers or,
    /// e.g., to generate initial guesses for an actual VLE solver.
    /// In general, the two states generated are NOT in an equilibrium.
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state.
    /// temperature : SINumber
    ///     The system temperature.
    /// pressure : SINumber
    ///     The system pressure.
    /// vapor_moles : SIArray1
    ///     Amount of substance of the vapor phase.
    /// liquid_moles : SIArray1
    ///     Amount of substance of the liquid phase.
    ///
    /// Returns
    /// -------
    /// PhaseEquilibrium
    #[staticmethod]
    pub(crate) fn new_npt(
        eos: PyEquationOfState,
        temperature: Temperature,
        pressure: Pressure,
        vapor_moles: Moles<Array1<f64>>,
        liquid_moles: Moles<Array1<f64>>,
    ) -> PyResult<Self> {
        Ok(Self(
            PhaseEquilibrium::new_npt(&eos.0, temperature, pressure, &vapor_moles, &liquid_moles)
                .map_err(PyFeosError::from)?,
        ))
    }

    #[getter]
    fn get_vapor(&self) -> PyState {
        PyState(self.0.vapor().clone())
    }

    #[getter]
    fn get_liquid(&self) -> PyState {
        PyState(self.0.liquid().clone())
    }

    /// Calculate the pure component vapor-liquid equilibria for all
    /// components in the system.
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state.
    /// temperature_or_pressure : SINumber
    ///     The system temperature or pressure.
    ///
    /// Returns
    /// -------
    /// list[PhaseEquilibrium]
    #[staticmethod]
    fn vle_pure_comps(
        eos: PyEquationOfState,
        temperature_or_pressure: Bound<'_, PyAny>,
    ) -> PyResult<Vec<Option<Self>>> {
        if let Ok(t) = temperature_or_pressure.extract::<Temperature>() {
            Ok(PhaseEquilibrium::vle_pure_comps(&eos.0, t)
                .into_iter()
                .map(|o| o.map(Self))
                .collect())
        } else if let Ok(p) = temperature_or_pressure.extract::<Pressure>() {
            Ok(PhaseEquilibrium::vle_pure_comps(&eos.0, p)
                .into_iter()
                .map(|o| o.map(Self))
                .collect())
        } else {
            Err(PyErr::new::<PyValueError, _>(format!(
                "Wrong units! Expected K or Pa, got {}.",
                temperature_or_pressure.call_method0("__repr__")?
            )))
        }
    }

    /// Calculate the pure component vapor pressures for all the
    /// components in the system.
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state.
    /// temperature : SINumber
    ///     The system temperature.
    ///
    /// Returns
    /// -------
    /// list[SINumber]
    #[staticmethod]
    fn vapor_pressure(eos: PyEquationOfState, temperature: Temperature) -> Vec<Option<Pressure>> {
        PhaseEquilibrium::vapor_pressure(&eos.0, temperature)
    }

    /// Calculate the pure component boiling temperatures for all the
    /// components in the system.
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state.
    /// pressure : SINumber
    ///     The system pressure.
    ///
    /// Returns
    /// -------
    /// list[SINumber]
    #[staticmethod]
    fn boiling_temperature(eos: PyEquationOfState, pressure: Pressure) -> Vec<Option<Temperature>> {
        PhaseEquilibrium::boiling_temperature(&eos.0, pressure)
    }

    fn _repr_markdown_(&self) -> String {
        self.0._repr_markdown_()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

/// A thermodynamic three phase equilibrium state.
#[pyclass(name = "ThreePhaseEquilibrium")]
#[derive(Clone)]
struct PyThreePhaseEquilibrium(PhaseEquilibrium<EquationOfState<IdealGasModel, ResidualModel>, 3>);

#[pymethods]
impl PyPhaseEquilibrium {
    /// Calculate a heteroazeotrope in a binary mixture for a given temperature
    /// or pressure.
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state.
    /// temperature_or_pressure : SINumber
    ///     The system temperature or pressure.
    /// x_init : list[float]
    ///     Initial guesses for the liquid molefracs of component 1
    ///     at the heteroazeotropic point.
    /// tp_init : SINumber, optional
    ///     Initial guess for the temperature/pressure at the
    ///     heteroszeotropic point.
    /// max_iter : int, optional
    ///     The maximum number of iterations.
    /// tol: float, optional
    ///     The solution tolerance.
    /// verbosity : Verbosity, optional
    ///     The verbosity.
    /// max_iter_bd_inner : int, optional
    ///     The maximum number of inner iterations in the bubble/dew point iteration.
    /// max_iter_bd_outer : int, optional
    ///     The maximum number of outer iterations in the bubble/dew point iteration.
    /// tol_bd_inner : float, optional
    ///     The solution tolerance in the inner loop of the bubble/dew point iteration.
    /// tol_bd_outer : float, optional
    ///     The solution tolerance in the outer loop of the bubble/dew point iteration.
    /// verbosity_bd : Verbosity, optional
    ///     The verbosity of the bubble/dew point iteration.
    #[staticmethod]
    #[pyo3(
        text_signature = "(eos, temperature_or_pressure, x_init, tp_init=None, max_iter=None, tol=None, verbosity=None, max_iter_bd_inner=None, max_iter_bd_outer=None, tol_bd_inner=None, tol_bd_outer=None, verbosity_bd=None)"
    )]
    #[pyo3(signature = (eos, temperature_or_pressure, x_init, tp_init=None, max_iter=None, tol=None, verbosity=None, max_iter_bd_inner=None, max_iter_bd_outer=None, tol_bd_inner=None, tol_bd_outer=None, verbosity_bd=None))]
    #[expect(clippy::too_many_arguments)]
    fn heteroazeotrope(
        eos: PyEquationOfState,
        temperature_or_pressure: Bound<'_, PyAny>,
        x_init: (f64, f64),
        tp_init: Option<Bound<'_, PyAny>>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        verbosity: Option<PyVerbosity>,
        max_iter_bd_inner: Option<usize>,
        max_iter_bd_outer: Option<usize>,
        tol_bd_inner: Option<f64>,
        tol_bd_outer: Option<f64>,
        verbosity_bd: Option<PyVerbosity>,
    ) -> PyResult<PyThreePhaseEquilibrium> {
        if let Ok(t) = temperature_or_pressure.extract::<Temperature>() {
            Ok(PyThreePhaseEquilibrium(
                PhaseEquilibrium::heteroazeotrope(
                    &eos.0,
                    t,
                    x_init,
                    tp_init.map(|t| t.extract()).transpose()?,
                    (max_iter, tol, verbosity.map(|v| v.into())).into(),
                    (
                        (
                            max_iter_bd_inner,
                            tol_bd_inner,
                            verbosity_bd.map(|v| v.into()),
                        )
                            .into(),
                        (
                            max_iter_bd_outer,
                            tol_bd_outer,
                            verbosity_bd.map(|v| v.into()),
                        )
                            .into(),
                    ),
                )
                .map_err(PyFeosError::from)?,
            ))
        } else if let Ok(p) = temperature_or_pressure.extract::<Pressure>() {
            Ok(PyThreePhaseEquilibrium(
                PhaseEquilibrium::heteroazeotrope(
                    &eos.0,
                    p,
                    x_init,
                    tp_init.map(|t| t.extract()).transpose()?,
                    (max_iter, tol, verbosity.map(|v| v.into())).into(),
                    (
                        (
                            max_iter_bd_inner,
                            tol_bd_inner,
                            verbosity_bd.map(|v| v.into()),
                        )
                            .into(),
                        (
                            max_iter_bd_outer,
                            tol_bd_outer,
                            verbosity_bd.map(|v| v.into()),
                        )
                            .into(),
                    ),
                )
                .map_err(PyFeosError::from)?,
            ))
        } else {
            Err(PyErr::new::<PyValueError, _>(format!(
                "Wrong units! Expected K or Pa, got {}.",
                temperature_or_pressure.call_method0("__repr__")?
            )))
        }
    }
}

#[pymethods]
impl PyThreePhaseEquilibrium {
    #[getter]
    fn get_vapor(&self) -> PyState {
        PyState(self.0.vapor().clone())
    }

    #[getter]
    fn get_liquid1(&self) -> PyState {
        PyState(self.0.liquid1().clone())
    }

    #[getter]
    fn get_liquid2(&self) -> PyState {
        PyState(self.0.liquid2().clone())
    }

    fn _repr_markdown_(&self) -> String {
        self.0._repr_markdown_()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

#[pymethods]
impl PyState {
    /// Calculates a two phase Tp-flash with the state as feed.
    ///
    /// Parameters
    /// ----------
    /// initial_state : PhaseEquilibrium, optional
    ///     A phase equilibrium used as initial guess.
    ///     Can speed up convergence.
    /// max_iter : int, optional
    ///     The maximum number of iterations.
    /// tol: float, optional
    ///     The solution tolerance.
    /// verbosity : Verbosity, optional
    ///     The verbosity.
    ///
    /// Returns
    /// -------
    /// PhaseEquilibrium
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     When pressure iteration fails or no phase equilibrium is found.
    #[pyo3(
        text_signature = "($self, initial_state=None, max_iter=None, tol=None, verbosity=None, non_volatile_components=None)"
    )]
    #[pyo3(signature = (initial_state=None, max_iter=None, tol=None, verbosity=None, non_volatile_components=None))]
    pub(crate) fn tp_flash(
        &self,
        initial_state: Option<&PyPhaseEquilibrium>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        verbosity: Option<PyVerbosity>,
        non_volatile_components: Option<Vec<usize>>,
    ) -> PyResult<PyPhaseEquilibrium> {
        Ok(PyPhaseEquilibrium(
            self.0
                .tp_flash(
                    initial_state.map(|s| &s.0),
                    (max_iter, tol, verbosity.map(|v| v.into())).into(),
                    non_volatile_components,
                )
                .map_err(PyFeosError::from)?,
        ))
    }
}

/// Phase diagram for a pure component or a binary mixture.
///
/// Parameters
/// ----------
/// phase_equilibria : [PhaseEquilibrium]
///     A list of individual phase equilibria.
///
/// Returns
/// -------
/// PhaseDiagram : the resulting phase diagram
#[pyclass(name = "PhaseDiagram")]
pub struct PyPhaseDiagram(PhaseDiagram<EquationOfState<IdealGasModel, ResidualModel>, 2>);

#[pymethods]
impl PyPhaseDiagram {
    #[new]
    fn new(phase_equilibria: Vec<PyPhaseEquilibrium>) -> Self {
        Self(PhaseDiagram::new(
            phase_equilibria.into_iter().map(|p| p.0).collect(),
        ))
    }

    /// Calculate a pure component phase diagram.
    ///
    /// Parameters
    /// ----------
    /// eos: Eos
    ///     The equation of state.
    /// min_temperature: SINumber
    ///     The lower limit for the temperature.
    /// npoints: int
    ///     The number of points.
    /// critical_temperature: SINumber, optional
    ///     An estimate for the critical temperature to initialize
    ///     the calculation if necessary. For most components not necessary.
    ///     Defaults to `None`.
    /// max_iter : int, optional
    ///     The maximum number of iterations.
    /// tol: float, optional
    ///     The solution tolerance.
    /// verbosity : Verbosity, optional
    ///     The verbosity.
    ///
    /// Returns
    /// -------
    /// PhaseDiagram
    #[staticmethod]
    #[pyo3(
        text_signature = "(eos, min_temperature, npoints, critical_temperature=None, max_iter=None, tol=None, verbosity=None)"
    )]
    #[pyo3(signature = (eos, min_temperature, npoints, critical_temperature=None, max_iter=None, tol=None, verbosity=None))]
    pub(crate) fn pure(
        eos: &PyEquationOfState,
        min_temperature: Temperature,
        npoints: usize,
        critical_temperature: Option<Temperature>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        verbosity: Option<PyVerbosity>,
    ) -> PyResult<Self> {
        let dia = PhaseDiagram::pure(
            &eos.0,
            min_temperature,
            npoints,
            critical_temperature,
            (max_iter, tol, verbosity.map(|v| v.into())).into(),
        )
        .map_err(PyFeosError::from)?;
        Ok(Self(dia))
    }

    /// Calculate a pure component phase diagram in parallel.
    ///
    /// Parameters
    /// ----------
    /// eos : Eos
    ///     The equation of state.
    /// min_temperature: SINumber
    ///     The lower limit for the temperature.
    /// npoints : int
    ///     The number of points.
    /// chunksize : int
    ///     The number of points that are calculated in sequence
    ///     within a thread.
    /// nthreads : int
    ///     Number of threads.
    /// critical_temperature: SINumber, optional
    ///     An estimate for the critical temperature to initialize
    ///     the calculation if necessary. For most components not necessary.
    ///     Defaults to `None`.
    /// max_iter : int, optional
    ///     The maximum number of iterations.
    /// tol: float, optional
    ///     The solution tolerance.
    /// verbosity : Verbosity, optional
    ///     The verbosity.
    ///
    /// Returns
    /// -------
    /// PhaseDiagram
    #[cfg(feature = "rayon")]
    #[staticmethod]
    #[pyo3(
        text_signature = "(eos, min_temperature, npoints, chunksize, nthreads, critical_temperature=None, max_iter=None, tol=None, verbosity=None)"
    )]
    #[pyo3(signature = (eos, min_temperature, npoints, chunksize, nthreads, critical_temperature=None, max_iter=None, tol=None, verbosity=None))]
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn par_pure(
        eos: &PyEquationOfState,
        min_temperature: Temperature,
        npoints: usize,
        chunksize: usize,
        nthreads: usize,
        critical_temperature: Option<Temperature>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        verbosity: Option<PyVerbosity>,
    ) -> PyResult<Self> {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(nthreads)
            .build()
            .map_err(PyFeosError::from)?;
        let dia = PhaseDiagram::par_pure(
            &eos.0,
            min_temperature,
            npoints,
            chunksize,
            thread_pool,
            critical_temperature,
            (max_iter, tol, verbosity.map(|v| v.into())).into(),
        )
        .map_err(PyFeosError::from)?;
        Ok(Self(dia))
    }

    /// Calculate the bubble point line of a mixture with given composition.
    ///
    /// In the resulting phase diagram, the liquid states correspond to the
    /// bubble point line while the vapor states contain the corresponding
    /// equilibrium states at different compositions.
    ///
    /// Parameters
    /// ----------
    /// eos: Eos
    ///     The equation of state.
    /// moles: SIArray1
    ///     The moles of the individual components
    /// min_temperature: SINumber
    ///     The lower limit for the temperature.
    /// npoints: int
    ///     The number of points.
    /// critical_temperature: SINumber, optional
    ///     An estimate for the critical temperature to initialize
    ///     the calculation if necessary. For most components not necessary.
    ///     Defaults to `None`.
    /// max_iter_inner : int, optional
    ///     The maximum number of inner iterations in the bubble/dew point iteration.
    /// max_iter_outer : int, optional
    ///     The maximum number of outer iterations in the bubble/dew point iteration.
    /// tol_inner : float, optional
    ///     The solution tolerance in the inner loop of the bubble/dew point iteration.
    /// tol_outer : float, optional
    ///     The solution tolerance in the outer loop of the bubble/dew point iteration.
    /// verbosity : Verbosity, optional
    ///     The verbosity of the bubble/dew point iteration.
    ///
    /// Returns
    /// -------
    /// PhaseDiagram
    #[staticmethod]
    #[pyo3(
        text_signature = "(eos, moles, min_temperature, npoints, critical_temperature=None, max_iter_inner=None, max_iter_outer=None, tol_inner=None, tol_outer=None, verbosity=None)"
    )]
    #[pyo3(signature = (eos, moles, min_temperature, npoints, critical_temperature=None, max_iter_inner=None, max_iter_outer=None, tol_inner=None, tol_outer=None, verbosity=None))]
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn bubble_point_line(
        eos: &PyEquationOfState,
        moles: Moles<Array1<f64>>,
        min_temperature: Temperature,
        npoints: usize,
        critical_temperature: Option<Temperature>,
        max_iter_inner: Option<usize>,
        max_iter_outer: Option<usize>,
        tol_inner: Option<f64>,
        tol_outer: Option<f64>,
        verbosity: Option<PyVerbosity>,
    ) -> PyResult<Self> {
        let dia = PhaseDiagram::bubble_point_line(
            &eos.0,
            &moles,
            min_temperature,
            npoints,
            critical_temperature,
            (
                (max_iter_inner, tol_inner, verbosity.map(|v| v.into())).into(),
                (max_iter_outer, tol_outer, verbosity.map(|v| v.into())).into(),
            ),
        )
        .map_err(PyFeosError::from)?;
        Ok(Self(dia))
    }

    /// Calculate the dew point line of a mixture with given composition.
    ///
    /// In the resulting phase diagram, the vapor states correspond to the
    /// dew point line while the liquid states contain the corresponding
    /// equilibrium states at different compositions.
    ///
    /// Parameters
    /// ----------
    /// eos: Eos
    ///     The equation of state.
    /// moles: SIArray1
    ///     The moles of the individual components
    /// min_temperature: SINumber
    ///     The lower limit for the temperature.
    /// npoints: int
    ///     The number of points.
    /// critical_temperature: SINumber, optional
    ///     An estimate for the critical temperature to initialize
    ///     the calculation if necessary. For most components not necessary.
    ///     Defaults to `None`.
    /// max_iter_inner : int, optional
    ///     The maximum number of inner iterations in the bubble/dew point iteration.
    /// max_iter_outer : int, optional
    ///     The maximum number of outer iterations in the bubble/dew point iteration.
    /// tol_inner : float, optional
    ///     The solution tolerance in the inner loop of the bubble/dew point iteration.
    /// tol_outer : float, optional
    ///     The solution tolerance in the outer loop of the bubble/dew point iteration.
    /// verbosity : Verbosity, optional
    ///     The verbosity of the bubble/dew point iteration.
    ///
    /// Returns
    /// -------
    /// PhaseDiagram
    #[staticmethod]
    #[pyo3(
        text_signature = "(eos, moles, min_temperature, npoints, critical_temperature=None, max_iter_inner=None, max_iter_outer=None, tol_inner=None, tol_outer=None, verbosity=None)"
    )]
    #[pyo3(signature = (eos, moles, min_temperature, npoints, critical_temperature=None, max_iter_inner=None, max_iter_outer=None, tol_inner=None, tol_outer=None, verbosity=None))]
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn dew_point_line(
        eos: &PyEquationOfState,
        moles: Moles<Array1<f64>>,
        min_temperature: Temperature,
        npoints: usize,
        critical_temperature: Option<Temperature>,
        max_iter_inner: Option<usize>,
        max_iter_outer: Option<usize>,
        tol_inner: Option<f64>,
        tol_outer: Option<f64>,
        verbosity: Option<PyVerbosity>,
    ) -> PyResult<Self> {
        let dia = PhaseDiagram::dew_point_line(
            &eos.0,
            &moles,
            min_temperature,
            npoints,
            critical_temperature,
            (
                (max_iter_inner, tol_inner, verbosity.map(|v| v.into())).into(),
                (max_iter_outer, tol_outer, verbosity.map(|v| v.into())).into(),
            ),
        )
        .map_err(PyFeosError::from)?;
        Ok(Self(dia))
    }

    /// Calculate the spinodal lines for a mixture with fixed composition.
    ///
    /// Parameters
    /// ----------
    /// eos: Eos
    ///     The equation of state.
    /// moles: SIArray1
    ///     The moles of the individual components
    /// min_temperature: SINumber
    ///     The lower limit for the temperature.
    /// npoints: int
    ///     The number of points.
    /// critical_temperature: SINumber, optional
    ///     An estimate for the critical temperature to initialize
    ///     the calculation if necessary. For most components not necessary.
    ///     Defaults to `None`.
    /// max_iter : int, optional
    ///     The maximum number of iterations.
    /// tol: float, optional
    ///     The solution tolerance.
    /// verbosity : Verbosity, optional
    ///     The verbosity.
    ///
    /// Returns
    /// -------
    /// PhaseDiagram
    #[staticmethod]
    #[pyo3(
        text_signature = "(eos, moles, min_temperature, npoints, critical_temperature=None, max_iter=None, tol=None, verbosity=None)"
    )]
    #[pyo3(signature = (eos, moles, min_temperature, npoints, critical_temperature=None, max_iter=None, tol=None, verbosity=None))]
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn spinodal(
        eos: &PyEquationOfState,
        moles: Moles<Array1<f64>>,
        min_temperature: Temperature,
        npoints: usize,
        critical_temperature: Option<Temperature>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        verbosity: Option<PyVerbosity>,
    ) -> PyResult<Self> {
        let dia = PhaseDiagram::spinodal(
            &eos.0,
            &moles,
            min_temperature,
            npoints,
            critical_temperature,
            (max_iter, tol, verbosity.map(|v| v.into())).into(),
        )
        .map_err(PyFeosError::from)?;
        Ok(Self(dia))
    }

    #[getter]
    pub(crate) fn get_states(&self) -> Vec<PyPhaseEquilibrium> {
        self.0
            .states
            .iter()
            .map(|vle| PyPhaseEquilibrium(vle.clone()))
            .collect()
    }

    #[getter]
    pub(crate) fn get_vapor(&self) -> PyStateVec {
        self.0.vapor().into()
    }

    #[getter]
    pub(crate) fn get_liquid(&self) -> PyStateVec {
        self.0.liquid().into()
    }

    /// Returns the phase diagram as dictionary.
    ///
    /// Parameters
    /// ----------
    /// contributions : Contributions, optional
    ///     The contributions to consider when calculating properties.
    ///     Defaults to Contributions.Total.
    ///
    /// Returns
    /// -------
    /// Dict[str, List[float]]
    ///     Keys: property names. Values: property for each state.
    ///
    /// Notes
    /// -----
    /// - temperature : K
    /// - pressure : Pa
    /// - densities : mol / m³
    /// - mass densities : kg / m³
    /// - molar enthalpies : kJ / mol
    /// - molar entropies : kJ / mol / K
    /// - specific enthalpies : kJ / kg
    /// - specific entropies : kJ / kg / K
    /// - xi: liquid molefraction of component i
    /// - yi: vapor molefraction of component i
    /// - component index `i` matches to order of components in parameters.
    #[pyo3(signature = (contributions=PyContributions::Total), text_signature = "($self, contributions)")]
    pub(crate) fn to_dict(&self, contributions: PyContributions) -> HashMap<String, Vec<f64>> {
        let n = self.0.states[0].liquid().eos.components();
        let c = Contributions::from(contributions);
        let mut dict = HashMap::with_capacity(16 + 2 * n);
        if n != 1 {
            let xs = self.0.liquid().molefracs();
            let ys = self.0.vapor().molefracs();
            for i in 0..n {
                dict.insert(format!("x{}", i), xs.column(i).to_vec());
                dict.insert(format!("y{}", i), ys.column(i).to_vec());
            }
        }
        dict.insert(
            String::from("temperature"),
            self.0
                .vapor()
                .temperature()
                .convert_to(KELVIN)
                .into_raw_vec_and_offset()
                .0,
        );

        // Check if liquid and vapor pressures are different (e.g. if )
        let p_v = self
            .0
            .vapor()
            .pressure()
            .convert_to(PASCAL)
            .into_raw_vec_and_offset()
            .0;
        let p_l = self
            .0
            .liquid()
            .pressure()
            .convert_to(PASCAL)
            .into_raw_vec_and_offset()
            .0;
        let different_pressures = p_v
            .iter()
            .zip(p_l.iter())
            .any(|(pv, pl)| (pv - pl).abs() / pv > 1e-3);

        if different_pressures {
            dict.insert(String::from("pressure vapor"), p_v);
            dict.insert(String::from("pressure liquid"), p_l);
        } else {
            dict.insert(String::from("pressure"), p_v);
        }
        dict.insert(
            String::from("density liquid"),
            self.0
                .liquid()
                .density()
                .convert_to(MOL / METER.powi::<P3>())
                .into_raw_vec_and_offset()
                .0,
        );
        dict.insert(
            String::from("density vapor"),
            self.0
                .vapor()
                .density()
                .convert_to(MOL / METER.powi::<P3>())
                .into_raw_vec_and_offset()
                .0,
        );
        dict.insert(
            String::from("molar enthalpy liquid"),
            self.0
                .liquid()
                .molar_enthalpy(c)
                .convert_to(KILO * JOULE / MOL)
                .into_raw_vec_and_offset()
                .0,
        );
        dict.insert(
            String::from("molar enthalpy vapor"),
            self.0
                .vapor()
                .molar_enthalpy(c)
                .convert_to(KILO * JOULE / MOL)
                .into_raw_vec_and_offset()
                .0,
        );
        dict.insert(
            String::from("molar entropy liquid"),
            self.0
                .liquid()
                .molar_entropy(c)
                .convert_to(KILO * JOULE / KELVIN / MOL)
                .into_raw_vec_and_offset()
                .0,
        );
        dict.insert(
            String::from("molar entropy vapor"),
            self.0
                .vapor()
                .molar_entropy(c)
                .convert_to(KILO * JOULE / KELVIN / MOL)
                .into_raw_vec_and_offset()
                .0,
        );
        if self.0.states[0].liquid().eos.residual.has_molar_weight() {
            dict.insert(
                String::from("mass density liquid"),
                self.0
                    .liquid()
                    .mass_density()
                    .convert_to(KILOGRAM / METER.powi::<P3>())
                    .into_raw_vec_and_offset()
                    .0,
            );
            dict.insert(
                String::from("mass density vapor"),
                self.0
                    .vapor()
                    .mass_density()
                    .convert_to(KILOGRAM / METER.powi::<P3>())
                    .into_raw_vec_and_offset()
                    .0,
            );
            dict.insert(
                String::from("specific enthalpy liquid"),
                self.0
                    .liquid()
                    .specific_enthalpy(c)
                    .convert_to(KILO * JOULE / KILOGRAM)
                    .into_raw_vec_and_offset()
                    .0,
            );
            dict.insert(
                String::from("specific enthalpy vapor"),
                self.0
                    .vapor()
                    .specific_enthalpy(c)
                    .convert_to(KILO * JOULE / KILOGRAM)
                    .into_raw_vec_and_offset()
                    .0,
            );
            dict.insert(
                String::from("specific entropy liquid"),
                self.0
                    .liquid()
                    .specific_entropy(c)
                    .convert_to(KILO * JOULE / KELVIN / KILOGRAM)
                    .into_raw_vec_and_offset()
                    .0,
            );
            dict.insert(
                String::from("specific entropy vapor"),
                self.0
                    .vapor()
                    .specific_entropy(c)
                    .convert_to(KILO * JOULE / KELVIN / KILOGRAM)
                    .into_raw_vec_and_offset()
                    .0,
            );
        }
        dict
    }

    /// Binary phase diagram calculated using bubble/dew point iterations.
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state.
    /// temperature_or_pressure: SINumber
    ///     The constant temperature or pressure.
    /// npoints: int, optional
    ///     The number of points (default 51).
    /// x_lle: (float, float), optional
    ///     An estimate for the molefractions of component 1
    ///     at the heteroazeotrop
    /// max_iter_inner : int, optional
    ///     The maximum number of inner iterations in the bubble/dew point iteration.
    /// max_iter_outer : int, optional
    ///     The maximum number of outer iterations in the bubble/dew point iteration.
    /// tol_inner : float, optional
    ///     The solution tolerance in the inner loop of the bubble/dew point iteration.
    /// tol_outer : float, optional
    ///     The solution tolerance in the outer loop of the bubble/dew point iteration.
    /// verbosity : Verbosity, optional
    ///     The verbosity of the bubble/dew point iteration.
    ///
    /// Returns
    /// -------
    /// PhaseDiagram
    #[staticmethod]
    #[pyo3(
        text_signature = "(eos, temperature_or_pressure, npoints=None, x_lle=None, max_iter_inner=None, max_iter_outer=None, tol_inner=None, tol_outer=None, verbosity=None)"
    )]
    #[pyo3(signature = (eos, temperature_or_pressure, npoints=None, x_lle=None, max_iter_inner=None, max_iter_outer=None, tol_inner=None, tol_outer=None, verbosity=None))]
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn binary_vle(
        eos: PyEquationOfState,
        temperature_or_pressure: Bound<'_, PyAny>,
        npoints: Option<usize>,
        x_lle: Option<(f64, f64)>,
        max_iter_inner: Option<usize>,
        max_iter_outer: Option<usize>,
        tol_inner: Option<f64>,
        tol_outer: Option<f64>,
        verbosity: Option<PyVerbosity>,
    ) -> PyResult<Self> {
        if let Ok(t) = temperature_or_pressure.extract::<Temperature>() {
            Ok(Self(
                PhaseDiagram::binary_vle(
                    &eos.0,
                    t,
                    npoints,
                    x_lle,
                    (
                        (max_iter_inner, tol_inner, verbosity.map(|v| v.into())).into(),
                        (max_iter_outer, tol_outer, verbosity.map(|v| v.into())).into(),
                    ),
                )
                .map_err(PyFeosError::from)?,
            ))
        } else if let Ok(p) = temperature_or_pressure.extract::<Pressure>() {
            Ok(Self(
                PhaseDiagram::binary_vle(
                    &eos.0,
                    p,
                    npoints,
                    x_lle,
                    (
                        (max_iter_inner, tol_inner, verbosity.map(|v| v.into())).into(),
                        (max_iter_outer, tol_outer, verbosity.map(|v| v.into())).into(),
                    ),
                )
                .map_err(PyFeosError::from)?,
            ))
        } else {
            Err(PyErr::new::<PyValueError, _>(format!(
                "Wrong units! Expected K or Pa, got {}.",
                temperature_or_pressure.call_method0("__repr__")?
            )))
        }
    }

    /// Create a new phase diagram using Tp flash calculations.
    ///
    /// The usual use case for this function is the calculation of
    /// liquid-liquid phase diagrams, but it can be used for vapor-
    /// liquid diagrams as well, as long as the feed composition is
    /// in a two phase region.
    ///
    /// Parameters
    /// ----------
    /// eos : EquationOfState
    ///     The equation of state.
    /// temperature_or_pressure: SINumber
    ///     The consant temperature or pressure.
    /// feed: SIArray1
    ///     Mole numbers in the (unstable) feed state.
    /// min_tp:
    ///     The lower limit of the temperature/pressure range.
    /// max_tp:
    ///     The upper limit of the temperature/pressure range.
    /// npoints: int, optional
    ///     The number of points (default 51).
    ///
    /// Returns
    /// -------
    /// PhaseDiagram
    #[staticmethod]
    #[pyo3(text_signature = "(eos, temperature_or_pressure, feed, min_tp, max_tp, npoints=None)")]
    #[pyo3(signature = (eos, temperature_or_pressure, feed, min_tp, max_tp, npoints=None))]
    pub(crate) fn lle(
        eos: PyEquationOfState,
        temperature_or_pressure: Bound<'_, PyAny>,
        feed: Moles<Array1<f64>>,
        min_tp: Bound<'_, PyAny>,
        max_tp: Bound<'_, PyAny>,
        npoints: Option<usize>,
    ) -> PyResult<Self> {
        if let Ok(t) = temperature_or_pressure.extract::<Temperature>() {
            Ok(Self(
                PhaseDiagram::lle(
                    &eos.0,
                    t,
                    &feed,
                    min_tp.extract()?,
                    max_tp.extract()?,
                    npoints,
                )
                .map_err(PyFeosError::from)?,
            ))
        } else if let Ok(p) = temperature_or_pressure.extract::<Pressure>() {
            Ok(Self(
                PhaseDiagram::lle(
                    &eos.0,
                    p,
                    &feed,
                    min_tp.extract()?,
                    max_tp.extract()?,
                    npoints,
                )
                .map_err(PyFeosError::from)?,
            ))
        } else {
            Err(PyErr::new::<PyValueError, _>(format!(
                "Wrong units! Expected K or Pa, got {}.",
                temperature_or_pressure.call_method0("__repr__")?
            )))
        }
    }
}

/// Phase diagram for a binary mixture exhibiting a heteroazeotrope.
#[pyclass(name = "PhaseDiagramHetero")]
pub struct PyPhaseDiagramHetero(PhaseDiagramHetero<EquationOfState<IdealGasModel, ResidualModel>>);

#[pymethods]
impl PyPhaseDiagram {
    /// Phase diagram for a binary mixture exhibiting a heteroazeotrope.
    ///
    /// Parameters
    /// ----------
    /// eos: SaftFunctional
    ///     The SAFT Helmholtz energy functional.
    /// temperature_or_pressure: SINumber
    ///     The temperature_or_pressure.
    /// x_lle: SINumber
    ///     Initial values for the molefractions of component 1
    ///     at the heteroazeotrop.
    /// tp_lim_lle: SINumber, optional
    ///     The minimum temperature up to which the LLE is calculated.
    ///     If it is not provided, no LLE is calcualted.
    /// tp_init_vlle: SINumber, optional
    ///     Initial value for the calculation of the VLLE.
    /// npoints_vle: int, optional
    ///     The number of points for the VLE (default 51).
    /// npoints_lle: int, optional
    ///     The number of points for the LLE (default 51).
    /// max_iter_inner : int, optional
    ///     The maximum number of inner iterations in the bubble/dew point iteration.
    /// max_iter_outer : int, optional
    ///     The maximum number of outer iterations in the bubble/dew point iteration.
    /// tol_inner : float, optional
    ///     The solution tolerance in the inner loop of the bubble/dew point iteration.
    /// tol_outer : float, optional
    ///     The solution tolerance in the outer loop of the bubble/dew point iteration.
    /// verbosity : Verbosity, optional
    ///     The verbosity of the bubble/dew point iteration.
    ///
    /// Returns
    /// -------
    /// PhaseDiagramHetero
    #[staticmethod]
    #[pyo3(
        text_signature = "(eos, temperature_or_pressure, x_lle, tp_lim_lle=None, tp_init_vlle=None, npoints_vle=None, npoints_lle=None, max_iter_inner=None, max_iter_outer=None, tol_inner=None, tol_outer=None, verbosity=None)"
    )]
    #[pyo3(signature = (eos, temperature_or_pressure, x_lle, tp_lim_lle=None, tp_init_vlle=None, npoints_vle=None, npoints_lle=None, max_iter_inner=None, max_iter_outer=None, tol_inner=None, tol_outer=None, verbosity=None))]
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn binary_vlle(
        eos: PyEquationOfState,
        temperature_or_pressure: Bound<'_, PyAny>,
        x_lle: (f64, f64),
        tp_lim_lle: Option<Bound<'_, PyAny>>,
        tp_init_vlle: Option<Bound<'_, PyAny>>,
        npoints_vle: Option<usize>,
        npoints_lle: Option<usize>,
        max_iter_inner: Option<usize>,
        max_iter_outer: Option<usize>,
        tol_inner: Option<f64>,
        tol_outer: Option<f64>,
        verbosity: Option<PyVerbosity>,
    ) -> PyResult<PyPhaseDiagramHetero> {
        if let Ok(t) = temperature_or_pressure.extract::<Temperature>() {
            Ok(PyPhaseDiagramHetero(
                PhaseDiagram::binary_vlle(
                    &eos.0,
                    t,
                    x_lle,
                    tp_lim_lle.map(|t| t.extract()).transpose()?,
                    tp_init_vlle.map(|t| t.extract()).transpose()?,
                    npoints_vle,
                    npoints_lle,
                    (
                        (max_iter_inner, tol_inner, verbosity.map(|v| v.into())).into(),
                        (max_iter_outer, tol_outer, verbosity.map(|v| v.into())).into(),
                    ),
                )
                .map_err(PyFeosError::from)?,
            ))
        } else if let Ok(p) = temperature_or_pressure.extract::<Pressure>() {
            Ok(PyPhaseDiagramHetero(
                PhaseDiagram::binary_vlle(
                    &eos.0,
                    p,
                    x_lle,
                    tp_lim_lle.map(|t| t.extract()).transpose()?,
                    tp_init_vlle.map(|t| t.extract()).transpose()?,
                    npoints_vle,
                    npoints_lle,
                    (
                        (max_iter_inner, tol_inner, verbosity.map(|v| v.into())).into(),
                        (max_iter_outer, tol_outer, verbosity.map(|v| v.into())).into(),
                    ),
                )
                .map_err(PyFeosError::from)?,
            ))
        } else {
            Err(PyErr::new::<PyValueError, _>(format!(
                "Wrong units! Expected K or Pa, got {}.",
                temperature_or_pressure.call_method0("__repr__")?
            )))
        }
    }
}

#[pymethods]
impl PyPhaseDiagramHetero {
    #[getter]
    pub(crate) fn get_vle(&self) -> PyPhaseDiagram {
        PyPhaseDiagram(self.0.vle().clone())
    }

    #[getter]
    pub(crate) fn get_vle1(&self) -> PyPhaseDiagram {
        PyPhaseDiagram(self.0.vle1.clone())
    }

    #[getter]
    pub(crate) fn get_vle2(&self) -> PyPhaseDiagram {
        PyPhaseDiagram(self.0.vle2.clone())
    }

    #[getter]
    pub(crate) fn get_lle(&self) -> Option<PyPhaseDiagram> {
        self.0.lle.as_ref().map(|d| PyPhaseDiagram(d.clone()))
    }
}
