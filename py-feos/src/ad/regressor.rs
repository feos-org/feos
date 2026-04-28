use std::collections::HashMap;

use feos::pcsaft::{PcSaftBinary, PcSaftPure};
use feos_core::parameter_optimization::{
    BinaryDataset, BubblePointDataset, BubblePointRecord, Dataset, DewPointDataset, DewPointRecord,
    DynRegressor, EnthalpyOfVaporizationDataset, EnthalpyOfVaporizationRecord,
    EquilibriumLiquidDensityDataset, EquilibriumLiquidDensityRecord, ParameterOptimizationError,
    LiquidDensityDataset, LiquidDensityRecord, LossFunction, NonConvergenceStrategy, PureDataset,
    Regressor, RegressorConfig, RegressorResult, ResidualFunction,
    ResidualIsobaricHeatCapacityDataset, ResidualIsobaricHeatCapacityRecord, VaporPressureDataset,
    VaporPressureRecord,
};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::PyEquationOfStateAD;

macro_rules! py_dataset {
    (
        $py_name:ident,
        $rust_type:ty,
        $display:expr,
        $csv_doc:expr
    ) => {
        #[doc = concat!("CSV columns: ``", $csv_doc, "``")]
        #[pyclass(name = $display)]
        pub struct $py_name {
            pub(crate) inner: $rust_type,
        }

        #[pymethods]
        impl $py_name {
            /// Load from a CSV file.
            ///
            /// Args:
            ///     path (str): Path to the CSV file.
            ///     name (str, optional): Dataset name used in regressor diagnostics
            ///         and results. Must be unique within a regressor; defaults to
            ///         the property name (e.g. `"vapor pressure"`).
            #[staticmethod]
            #[pyo3(signature = (path, name=None))]
            pub fn from_csv(path: &str, name: Option<&str>) -> PyResult<Self> {
                <$rust_type>::from_csv(std::path::Path::new(path))
                    .map(|mut inner| {
                        if let Some(n) = name {
                            inner = inner.with_name(n);
                        }
                        Self { inner }
                    })
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            }

            /// Property name.
            #[getter]
            pub fn name(&self) -> &str {
                Dataset::name(&self.inner)
            }

            /// Number of data points.
            pub fn __len__(&self) -> usize {
                Dataset::target(&self.inner).len()
            }

            /// Target values.
            pub fn target<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
                Dataset::target(&self.inner).to_owned().to_pyarray(py)
            }

            pub fn __repr__(&self) -> String {
                format!("{}(n={})", $display, Dataset::target(&self.inner).len())
            }
        }
    };
}

py_dataset!(
    PyVaporPressureDataset,
    VaporPressureDataset,
    "VaporPressureDataset",
    "temperature_k, vapor_pressure_pa"
);

#[pymethods]
impl PyVaporPressureDataset {
    /// Construct from numpy arrays.
    ///
    /// Args:
    ///     temperature_k (np.ndarray): Temperatures in K.
    ///     vapor_pressure_pa (np.ndarray): Vapor pressures in Pa.
    ///     name (str, optional): Dataset name (must be unique within a regressor).
    #[new]
    #[pyo3(signature = (temperature_k, vapor_pressure_pa, name=None))]
    pub fn new(
        temperature_k: PyReadonlyArray1<f64>,
        vapor_pressure_pa: PyReadonlyArray1<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let temperature_k = temperature_k.as_array();
        let vapor_pressure_pa = vapor_pressure_pa.as_array();
        if temperature_k.len() != vapor_pressure_pa.len() {
            return Err(PyValueError::new_err(
                "temperature_k and vapor_pressure_pa must have the same length",
            ));
        }
        let records = temperature_k
            .iter()
            .zip(vapor_pressure_pa.iter())
            .map(|(&t, &p)| VaporPressureRecord {
                temperature_k: t,
                vapor_pressure_pa: p,
            })
            .collect();
        let mut inner = VaporPressureDataset::from_records(records);
        if let Some(n) = name {
            inner = inner.with_name(n);
        }
        Ok(Self { inner })
    }
}

py_dataset!(
    PyLiquidDensityDataset,
    LiquidDensityDataset,
    "LiquidDensityDataset",
    "temperature_k, pressure_pa, liquid_density_kmol_m3"
);

#[pymethods]
impl PyLiquidDensityDataset {
    /// Construct from numpy arrays.
    ///
    /// Args:
    ///     temperature_k (np.ndarray): Temperatures in K.
    ///     pressure_pa (np.ndarray): Pressures in Pa.
    ///     liquid_density_kmol_m3 (np.ndarray): Liquid molar densities in kmol/m³.
    ///     name (str, optional): Dataset name (must be unique within a regressor).
    #[new]
    #[pyo3(signature = (temperature_k, pressure_pa, liquid_density_kmol_m3, name=None))]
    pub fn new(
        temperature_k: PyReadonlyArray1<f64>,
        pressure_pa: PyReadonlyArray1<f64>,
        liquid_density_kmol_m3: PyReadonlyArray1<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let temperature_k = temperature_k.as_array();
        let pressure_pa = pressure_pa.as_array();
        let liquid_density_kmol_m3 = liquid_density_kmol_m3.as_array();
        let n = temperature_k.len();
        if pressure_pa.len() != n || liquid_density_kmol_m3.len() != n {
            return Err(PyValueError::new_err(
                "all arrays must have the same length",
            ));
        }
        let records = temperature_k
            .iter()
            .zip(pressure_pa.iter())
            .zip(liquid_density_kmol_m3.iter())
            .map(|((&t, &p), &rho)| LiquidDensityRecord {
                temperature_k: t,
                pressure_pa: p,
                liquid_density_kmol_m3: rho,
            })
            .collect();
        let mut inner = LiquidDensityDataset::from_records(records);
        if let Some(n) = name {
            inner = inner.with_name(n);
        }
        Ok(Self { inner })
    }
}

py_dataset!(
    PyEquilibriumLiquidDensityDataset,
    EquilibriumLiquidDensityDataset,
    "EquilibriumLiquidDensityDataset",
    "temperature_k, liquid_density_kmol_m3"
);

#[pymethods]
impl PyEquilibriumLiquidDensityDataset {
    /// Construct from numpy arrays.
    ///
    /// Args:
    ///     temperature_k (np.ndarray): Temperatures in K.
    ///     liquid_density_kmol_m3 (np.ndarray): Saturated liquid molar densities in kmol/m³.
    ///     name (str, optional): Dataset name (must be unique within a regressor).
    #[new]
    #[pyo3(signature = (temperature_k, liquid_density_kmol_m3, name=None))]
    pub fn new(
        temperature_k: PyReadonlyArray1<f64>,
        liquid_density_kmol_m3: PyReadonlyArray1<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let temperature_k = temperature_k.as_array();
        let liquid_density_kmol_m3 = liquid_density_kmol_m3.as_array();
        if temperature_k.len() != liquid_density_kmol_m3.len() {
            return Err(PyValueError::new_err(
                "temperature_k and liquid_density_kmol_m3 must have the same length",
            ));
        }
        let records = temperature_k
            .iter()
            .zip(liquid_density_kmol_m3.iter())
            .map(|(&t, &rho)| EquilibriumLiquidDensityRecord {
                temperature_k: t,
                liquid_density_kmol_m3: rho,
            })
            .collect();
        let mut inner = EquilibriumLiquidDensityDataset::from_records(records);
        if let Some(n) = name {
            inner = inner.with_name(n);
        }
        Ok(Self { inner })
    }
}

py_dataset!(
    PyEnthalpyOfVaporizationDataset,
    EnthalpyOfVaporizationDataset,
    "EnthalpyOfVaporizationDataset",
    "temperature_k, dh_vap_j_mol"
);

#[pymethods]
impl PyEnthalpyOfVaporizationDataset {
    /// Construct from numpy arrays.
    ///
    /// Args:
    ///     temperature_k (np.ndarray): Temperatures in K.
    ///     dh_vap_j_mol (np.ndarray): Enthalpy of vaporization in J/mol.
    ///     name (str, optional): Dataset name (must be unique within a regressor).
    #[new]
    #[pyo3(signature = (temperature_k, dh_vap_j_mol, name=None))]
    pub fn new(
        temperature_k: PyReadonlyArray1<f64>,
        dh_vap_j_mol: PyReadonlyArray1<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let temperature_k = temperature_k.as_array();
        let dh_vap_j_mol = dh_vap_j_mol.as_array();
        if temperature_k.len() != dh_vap_j_mol.len() {
            return Err(PyValueError::new_err(
                "temperature_k and dh_vap_j_mol must have the same length",
            ));
        }
        let records = temperature_k
            .iter()
            .zip(dh_vap_j_mol.iter())
            .map(|(&t, &p)| EnthalpyOfVaporizationRecord {
                temperature_k: t,
                dh_vap_j_mol: p,
            })
            .collect();
        let mut inner = EnthalpyOfVaporizationDataset::from_records(records);
        if let Some(n) = name {
            inner = inner.with_name(n);
        }
        Ok(Self { inner })
    }
}

py_dataset!(
    PyResidualIsobaricHeatCapacityDataset,
    ResidualIsobaricHeatCapacityDataset,
    "ResidualIsobaricHeatCapacityDataset",
    "temperature_k, pressure_pa, cp_res_j_molk"
);

#[pymethods]
impl PyResidualIsobaricHeatCapacityDataset {
    /// Construct from numpy arrays.
    ///
    /// Args:
    ///     temperature_k (np.ndarray): Temperatures in K.
    ///     pressure_pa (np.ndarray): Pressures in Pa.
    ///     cp_res_j_molk (np.ndarray): Residual isobaric molar heat capacities in J/(mol·K).
    ///     name (str, optional): Dataset name (must be unique within a regressor).
    #[new]
    #[pyo3(signature = (temperature_k, pressure_pa, cp_res_j_molk, name=None))]
    pub fn new(
        temperature_k: PyReadonlyArray1<f64>,
        pressure_pa: PyReadonlyArray1<f64>,
        cp_res_j_molk: PyReadonlyArray1<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let temperature_k = temperature_k.as_array();
        let pressure_pa = pressure_pa.as_array();
        let cp_res_j_molk = cp_res_j_molk.as_array();
        let n = temperature_k.len();
        if pressure_pa.len() != n || cp_res_j_molk.len() != n {
            return Err(PyValueError::new_err(
                "all arrays must have the same length",
            ));
        }
        let records = temperature_k
            .iter()
            .zip(pressure_pa.iter())
            .zip(cp_res_j_molk.iter())
            .map(|((&t, &p), &cp)| ResidualIsobaricHeatCapacityRecord {
                temperature_k: t,
                pressure_pa: p,
                cp_res_j_molk: cp,
            })
            .collect();
        let mut inner = ResidualIsobaricHeatCapacityDataset::from_records(records);
        if let Some(n) = name {
            inner = inner.with_name(n);
        }
        Ok(Self { inner })
    }
}

py_dataset!(
    PyBubblePointDataset,
    BubblePointDataset,
    "BubblePointDataset",
    "temperature_k, liquid_molefrac_1, bubble_pressure_pa"
);

#[pymethods]
impl PyBubblePointDataset {
    /// Construct from numpy arrays.
    ///
    /// Args:
    ///     temperature_k (np.ndarray): Temperatures in K.
    ///     liquid_molefrac_1 (np.ndarray): Liquid-phase mole fractions of component 1.
    ///     bubble_pressure_pa (np.ndarray): Bubble point pressures in Pa.
    ///         Also used as the initial VLE solver guess.
    ///     name (str, optional): Dataset name (must be unique within a regressor).
    #[new]
    #[pyo3(signature = (temperature_k, liquid_molefrac_1, bubble_pressure_pa, name=None))]
    pub fn new(
        temperature_k: PyReadonlyArray1<f64>,
        liquid_molefrac_1: PyReadonlyArray1<f64>,
        bubble_pressure_pa: PyReadonlyArray1<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let temperature_k = temperature_k.as_array();
        let liquid_molefrac_1 = liquid_molefrac_1.as_array();
        let bubble_pressure_pa = bubble_pressure_pa.as_array();
        let n = temperature_k.len();
        if liquid_molefrac_1.len() != n || bubble_pressure_pa.len() != n {
            return Err(PyValueError::new_err(
                "all arrays must have the same length",
            ));
        }
        let records = temperature_k
            .iter()
            .zip(liquid_molefrac_1.iter())
            .zip(bubble_pressure_pa.iter())
            .map(|((&t, &x), &p)| BubblePointRecord {
                temperature_k: t,
                liquid_molefrac_1: x,
                bubble_pressure_pa: p,
            })
            .collect();
        let mut inner = BubblePointDataset::from_records(records);
        if let Some(n) = name {
            inner = inner.with_name(n);
        }
        Ok(Self { inner })
    }
}

py_dataset!(
    PyDewPointDataset,
    DewPointDataset,
    "DewPointDataset",
    "temperature_k, vapor_molefrac_1, dew_pressure_pa"
);

#[pymethods]
impl PyDewPointDataset {
    /// Construct from numpy arrays.
    ///
    /// Args:
    ///     temperature_k (np.ndarray): Temperatures in K.
    ///     vapor_molefrac_1 (np.ndarray): Vapor-phase mole fractions of component 1.
    ///     dew_pressure_pa (np.ndarray): Dew point pressures in Pa.
    ///         Also used as the initial VLE solver guess.
    ///     name (str, optional): Dataset name (must be unique within a regressor).
    #[new]
    #[pyo3(signature = (temperature_k, vapor_molefrac_1, dew_pressure_pa, name=None))]
    pub fn new(
        temperature_k: PyReadonlyArray1<f64>,
        vapor_molefrac_1: PyReadonlyArray1<f64>,
        dew_pressure_pa: PyReadonlyArray1<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let temperature_k = temperature_k.as_array();
        let vapor_molefrac_1 = vapor_molefrac_1.as_array();
        let dew_pressure_pa = dew_pressure_pa.as_array();
        let n = temperature_k.len();
        if vapor_molefrac_1.len() != n || dew_pressure_pa.len() != n {
            return Err(PyValueError::new_err(
                "all arrays must have the same length",
            ));
        }
        let records = temperature_k
            .iter()
            .zip(vapor_molefrac_1.iter())
            .zip(dew_pressure_pa.iter())
            .map(|((&t, &y), &p)| DewPointRecord {
                temperature_k: t,
                vapor_molefrac_1: y,
                dew_pressure_pa: p,
            })
            .collect();
        let mut inner = DewPointDataset::from_records(records);
        if let Some(n) = name {
            inner = inner.with_name(n);
        }
        Ok(Self { inner })
    }
}

/// Loss function applied to the dimensionless relative residual during fitting.
///
/// Examples:
///     >>> LossFunction.l2()
///     LossFunction.L2
///     >>> LossFunction.huber(0.1)
///     LossFunction.Huber(delta=0.1)
#[pyclass(name = "LossFunction")]
pub struct PyLossFunction {
    pub(crate) inner: LossFunction,
}

#[pymethods]
impl PyLossFunction {
    /// L2 loss: all residuals weighted equally.
    #[staticmethod]
    pub fn l2() -> Self {
        Self {
            inner: LossFunction::L2,
        }
    }

    /// Huber loss: down-weights points with relative error > `delta`.
    ///
    /// Args:
    ///     delta (float): Dimensionless threshold on the relative residual
    ///         ``r = (calc − exp) / exp``.  E.g. ``0.1`` means 10 %.
    #[staticmethod]
    pub fn huber(delta: f64) -> PyResult<Self> {
        if delta <= 0.0 {
            return Err(PyValueError::new_err("Huber delta must be > 0"));
        }
        Ok(Self {
            inner: LossFunction::Huber { delta },
        })
    }

    pub fn __repr__(&self) -> String {
        match &self.inner {
            LossFunction::L2 => "LossFunction.L2".to_string(),
            LossFunction::Huber { delta } => format!("LossFunction.Huber(delta={delta})"),
        }
    }
}

/// Strategy for handling data points where EoS evaluation did not converge.
///
/// Examples:
///     >>> NonConvergenceStrategy.ignore()
///     NonConvergenceStrategy.Ignore
///     >>> NonConvergenceStrategy.penalty(10.0)
///     NonConvergenceStrategy.Penalty(10.0)
#[pyclass(name = "NonConvergenceStrategy")]
pub struct PyNonConvergenceStrategy {
    pub(crate) inner: NonConvergenceStrategy,
}

#[pymethods]
impl PyNonConvergenceStrategy {
    /// Exclude non-converged points entirely (zero residual and Jacobian).
    #[staticmethod]
    pub fn ignore() -> Self {
        Self {
            inner: NonConvergenceStrategy::Ignore,
        }
    }

    /// Apply a fixed dimensionless relative residual penalty.
    ///
    /// Args:
    ///     value (float): Penalty value, e.g. `10.0` = 1000 % relative error.
    #[staticmethod]
    pub fn penalty(value: f64) -> PyResult<Self> {
        if value <= 0.0 {
            return Err(PyValueError::new_err("value must be > 0"));
        }
        Ok(Self {
            inner: NonConvergenceStrategy::Penalty(value),
        })
    }

    pub fn __repr__(&self) -> String {
        match self.inner {
            NonConvergenceStrategy::Ignore => "NonConvergenceStrategy.Ignore".to_string(),
            NonConvergenceStrategy::Penalty(v) => {
                format!("NonConvergenceStrategy.Penalty({v})")
            }
        }
    }
}

/// Levenberg-Marquardt hyperparameters.
///
/// Args:
///     ftol (float): Terminate when the relative reduction in the objective
///         function is ≤ ``ftol``. Default: 1e-8.
///     xtol (float): Terminate when the relative change in parameters between
///         iterations is ≤ ``xtol``. Default: 1e-8.
///     gtol (float): Terminate when the residual and all Jacobian columns are
///         nearly orthogonal (scale-invariant first-order optimality check).
///         Set to ``0.0`` to disable. Default: 1e-8.
///     stepbound (float): Factor for the initial trust-region step bound, in
///         ``[0.1, 100]``. Smaller values are more conservative. Default: 0.1.
///     patience (int): Maximum number of residual evaluations =
///         ``patience × (n_params + 1)``. Default: 100.
///     scale_diag (bool): Rescale parameters internally using Jacobian column
///         norms. Default: True.
///     strategy (NonConvergenceStrategy, optional): How to treat non-converged
///         EoS evaluations. Default: ``NonConvergenceStrategy.penalty(10.0)``.
#[pyclass(name = "RegressorConfig")]
#[derive(Clone)]
pub struct PyRegressorConfig {
    pub(crate) inner: RegressorConfig,
}

#[pymethods]
impl PyRegressorConfig {
    #[new]
    #[pyo3(signature = (ftol=1e-8, xtol=1e-8, gtol=1e-8, stepbound=0.1, patience=100, scale_diag=true, strategy=None))]
    pub fn new(
        ftol: f64,
        xtol: f64,
        gtol: f64,
        stepbound: f64,
        patience: usize,
        scale_diag: bool,
        strategy: Option<Bound<'_, PyNonConvergenceStrategy>>,
    ) -> Self {
        Self {
            inner: RegressorConfig {
                ftol,
                xtol,
                gtol,
                stepbound,
                patience,
                scale_diag,
                strategy: strategy
                    .map(|s| s.borrow().inner.clone())
                    .unwrap_or_default(),
            },
        }
    }

    #[getter]
    pub fn ftol(&self) -> f64 {
        self.inner.ftol
    }
    #[getter]
    pub fn xtol(&self) -> f64 {
        self.inner.xtol
    }
    #[getter]
    pub fn gtol(&self) -> f64 {
        self.inner.gtol
    }
    #[getter]
    pub fn stepbound(&self) -> f64 {
        self.inner.stepbound
    }
    #[getter]
    pub fn patience(&self) -> usize {
        self.inner.patience
    }
    #[getter]
    pub fn scale_diag(&self) -> bool {
        self.inner.scale_diag
    }
    #[getter]
    pub fn strategy(&self) -> PyNonConvergenceStrategy {
        PyNonConvergenceStrategy {
            inner: self.inner.strategy.clone(),
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "RegressorConfig(ftol={}, xtol={}, gtol={}, stepbound={}, patience={}, \
             scale_diag={}, strategy={})",
            self.inner.ftol,
            self.inner.xtol,
            self.inner.gtol,
            self.inner.stepbound,
            self.inner.patience,
            self.inner.scale_diag,
            PyNonConvergenceStrategy {
                inner: self.inner.strategy.clone()
            }
            .__repr__(),
        )
    }
}

/// Results from a completed parameter fit.
///
/// Attributes:
///     optimal_params (list[float]): Optimal physical parameters in canonical
///         order.
///     fitted_param_names (list[str]): Names of the parameters that were
///         optimised.
///     all_param_names (list[str]): All canonical parameter names in order.
///     aad_per_dataset (list[float | None]): Per-dataset average absolute
///         relative deviation in percent.
///     dataset_names (list[str]): Dataset names in order.
///     convergence_stats (list[tuple[int, int]]): Per-dataset
///         `(n_converged, n_total)`.
///     converged (bool): Whether LM reported successful convergence.
///     n_evaluations (int): Number of residual evaluations performed by LM.
///     objective_function (float): Final value of the LM objective.
#[pyclass(name = "RegressorResult")]
pub struct PyRegressorResult {
    pub(crate) inner: RegressorResult,
}

#[pymethods]
impl PyRegressorResult {
    #[getter]
    pub fn optimal_params(&self) -> Vec<f64> {
        self.inner.optimal_params.clone()
    }
    #[getter]
    pub fn fitted_param_names(&self) -> Vec<String> {
        self.inner.fitted_param_names.clone()
    }
    #[getter]
    pub fn all_param_names(&self) -> Vec<String> {
        self.inner.all_param_names.clone()
    }
    #[getter]
    pub fn aad_per_dataset(&self) -> Vec<Option<f64>> {
        self.inner.aad_per_dataset.clone()
    }
    #[getter]
    pub fn dataset_names(&self) -> Vec<String> {
        self.inner.dataset_names.clone()
    }
    #[getter]
    pub fn convergence_stats(&self) -> Vec<(usize, usize)> {
        self.inner.convergence_stats.clone()
    }
    #[getter]
    pub fn converged(&self) -> bool {
        self.inner.converged
    }
    /// Description of why the optimiser stopped.
    #[getter]
    pub fn termination_reason(&self) -> &str {
        &self.inner.termination_reason
    }
    #[getter]
    pub fn n_evaluations(&self) -> usize {
        self.inner.n_evaluations
    }
    #[getter]
    pub fn objective_function(&self) -> f64 {
        self.inner.objective_function
    }
    /// Wall-clock time spent inside the LM optimiser, in milliseconds.
    #[getter]
    pub fn elapsed_ms(&self) -> f64 {
        self.inner.elapsed.as_secs_f64() * 1000.0
    }

    /// All parameters as `{name: value}`.
    ///
    /// Fitted parameters at their optimal values, all others at their initial values.
    ///
    /// Note:
    ///
    /// The dict is returned in canonical parameter order.
    /// This means `list(result.all_parameters().values())`
    /// is a valid argument to :meth:`PureRegressor.predict`.
    ///
    /// Pass this directly to :meth:`PureRegressor.evaluate_datasets` or
    /// :meth:`BinaryRegressor.evaluate_datasets` to compare model predictions
    /// against the experimental data at the fitted optimum.
    pub fn all_parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, &val) in self
            .inner
            .all_param_names
            .iter()
            .zip(self.inner.optimal_params.iter())
        {
            d.set_item(name, val)?;
        }
        Ok(d)
    }

    /// Fitted parameters as `{name: value}`, in canonical order.
    ///
    /// Only the parameters that were optimised are included.
    /// For all  parameters (including fixed ones), use :meth:`all_parameters`.
    pub fn optimal_params_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for name in &self.inner.fitted_param_names {
            if let Some(pos) = self.inner.all_param_names.iter().position(|n| n == name) {
                d.set_item(name, self.inner.optimal_params[pos])?;
            }
        }
        Ok(d)
    }

    pub fn __repr__(&self) -> String {
        let aad_strs: Vec<String> = self
            .inner
            .aad_per_dataset
            .iter()
            .zip(self.inner.dataset_names.iter())
            .map(|(aad, name)| match aad {
                Some(v) => format!("{name}: {v:.2}%"),
                None => format!("{name}: no convergence"),
            })
            .collect();
        format!(
            "RegressorResult(converged={}, n_eval={}, elapsed={:.1}ms, aad=[{}])",
            self.inner.converged,
            self.inner.n_evaluations,
            self.inner.elapsed.as_secs_f64() * 1000.0,
            aad_strs.join(", ")
        )
    }
}

macro_rules! regressor_methods {
    ($py_type:ty, $display:expr) => {
        #[pymethods]
        impl $py_type {
            /// Evaluate the model at the given physical parameters.
            ///
            /// Args:
            ///     params (list[float]): Full parameter vector in canonical
            ///         order (`all_param_names`).
            ///
            /// Returns:
            ///     tuple[np.ndarray, np.ndarray, np.ndarray]:
            ///         `(predicted, gradients, converged)` where
            ///         `gradients` has shape `[n_points, n_fitted_params]`.
            pub fn predict<'py>(
                &self,
                py: Python<'py>,
                params: Vec<f64>,
            ) -> (
                Bound<'py, PyArray1<f64>>,
                Bound<'py, PyArray2<f64>>,
                Bound<'py, PyArray1<bool>>,
            ) {
                let (pred, grad, conv) = self.inner.predict(&params);
                (
                    pred.to_pyarray(py),
                    grad.to_pyarray(py),
                    conv.to_pyarray(py),
                )
            }

            /// Target values across all datasets (concatenated).
            pub fn target<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
                self.inner.target().to_pyarray(py)
            }

            /// Evaluate each dataset at the given parameters.
            ///
            /// Returns a list of dicts where each contains:
            ///
            /// - Input columns keyed by their property name
            /// - The experimental target column, keyed by its property name
            /// - The model predictions, keyed as `"<target_name>_predicted"`
            /// - `"relative_deviation"` — `(predicted − target) / target`
            /// - `"converged"` — boolean mask
            ///
            /// Args:
            ///     params (dict[str, float], optional): All EOS parameters keyed by
            ///         name, as returned by :attr:`RegressorResult.all_parameters`.
            ///         If omitted, the regressor's current parameters are used — i.e.
            ///         the optimal values after :meth:`fit`, or the initial values
            ///         before fitting.
            ///
            /// Returns:
            ///     list[dict[str, np.ndarray]]: One dict per dataset, in the
            ///         same order as the datasets passed to the constructor.
            ///
            /// Examples:
            ///     >>> result = regressor.fit()
            ///     >>> ds = regressor.evaluate_datasets()           # uses fitted params
            ///     >>> ds = regressor.evaluate_datasets(result.all_parameters)  # explicit
            ///     >>> dfs = [pd.DataFrame(d) for d in ds]
            #[pyo3(signature = (params=None))]
            pub fn evaluate_datasets<'py>(
                &self,
                py: Python<'py>,
                params: Option<HashMap<String, f64>>,
            ) -> PyResult<Vec<Bound<'py, PyDict>>> {
                let param_vec = match params {
                    None => self.inner.optimal_params(),
                    Some(map) => {
                        let names = self.inner.all_param_names();
                        names
                            .iter()
                            .map(|n| {
                                map.get(n).copied().ok_or_else(|| {
                                    PyValueError::new_err(format!("missing parameter '{n}'"))
                                })
                            })
                            .collect::<PyResult<Vec<f64>>>()?
                    }
                };

                let results = self.inner.evaluate_datasets(&param_vec);

                // Create Python dictionaries
                results
                    .into_iter()
                    .map(|r| {
                        let d = PyDict::new(py);
                        for (name, col) in r.inputs {
                            d.set_item(name, col.to_pyarray(py))?;
                        }
                        d.set_item(r.target_name, r.target.to_pyarray(py))?;
                        d.set_item(
                            format!("{}_predicted", r.target_name),
                            r.predicted.to_pyarray(py),
                        )?;
                        d.set_item("relative_deviation", r.relative_deviation.to_pyarray(py))?;
                        d.set_item("converged", r.converged.to_pyarray(py))?;
                        Ok(d)
                    })
                    .collect()
            }

            /// Run Levenberg-Marquardt optimisation.
            ///
            /// Args:
            ///     config (RegressorConfig, optional): Regressor hyperparameters
            ///         (including non-convergence strategy).
            ///     loss (LossFunction, optional): Loss function applied to
            ///         relative residuals.
            ///
            /// Returns:
            ///     RegressorResult
            #[pyo3(signature = (config=None, loss=None))]
            pub fn fit(
                &mut self,
                config: Option<PyRegressorConfig>,
                loss: Option<Bound<'_, PyLossFunction>>,
            ) -> PyRegressorResult {
                let config = config.map(|c| c.inner).unwrap_or_default();
                let loss = loss.map(|l| l.borrow().inner.clone());
                PyRegressorResult {
                    inner: self.inner.fit(config, loss),
                }
            }

            /// All canonical parameter names in order.
            #[getter]
            pub fn all_param_names(&self) -> Vec<String> {
                self.inner.all_param_names()
            }

            /// Names of the parameters being optimised.
            #[getter]
            pub fn fitted_param_names(&self) -> Vec<String> {
                self.inner.fitted_param_names()
            }

            /// Dataset names in order.
            #[getter]
            pub fn dataset_names(&self) -> Vec<String> {
                self.inner.dataset_names()
            }

            pub fn __repr__(&self) -> String {
                format!(
                    "{}(params={:?}, fitting={:?}, datasets=[{}])",
                    $display,
                    self.inner.all_param_names(),
                    self.inner.fitted_param_names(),
                    self.inner
                        .dataset_names()
                        .iter()
                        .map(|s| format!("\"{s}\""))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
    };
}

fn validate_weights(weights: &[f64], n_datasets: usize) -> PyResult<()> {
    if weights.len() != n_datasets {
        return Err(PyValueError::new_err(format!(
            "weights has length {} but there are {} datasets",
            weights.len(),
            n_datasets
        )));
    }
    if weights.iter().any(|&w| w <= 0.0) {
        return Err(PyValueError::new_err(
            "all dataset weights must be positive",
        ));
    }
    Ok(())
}

fn parse_residual_fn(s: &str) -> PyResult<ResidualFunction> {
    match s {
        "difference" => Ok(ResidualFunction::Difference),
        "log_difference" => Ok(ResidualFunction::LogDifference),
        "relative_difference" => Ok(ResidualFunction::RelativeDifference),
        _ => Err(PyValueError::new_err(format!(
            "unknown residual function '{s}'; valid: 'difference', 'log_difference', 'relative_difference'"
        ))),
    }
}

/// Levenberg-Marquardt regressor for pure-component parameter fitting.
///
/// Args:
///     model (EquationOfStateAD): Equation of state to use.
///     datasets (list): One or more datasets for pure substance properties.
///     params (dict[str, float]): Initial values for all parameters keyed by name.
///     fit (list[str]): Names of the parameters to optimise.
///     residual (str, optional): Residual function. Possible values are
///         ``"relative_difference"`` (default), ``"log_difference"``, or ``"difference"``.
///     weights (list[float], optional): Per-dataset weights, one per dataset.
///         All weights must be positive. Default: 1.0 for every dataset.
///
/// Examples:
///     >>> vp  = VaporPressureDataset.from_csv("vp.csv")
///     >>> rho = LiquidDensityDataset.from_csv("rho.csv")
///     >>> reg = PureRegressor(
///     ...     model=EquationOfStateAD.PcSaftNonAssoc,
///     ...     datasets=[vp, rho],
///     ...     params={"m": 2.5, "sigma": 3.4, "epsilon_k": 280.0, "mu": 0.0},
///     ...     fit=["sigma", "epsilon_k"],
///     ...     weights=[1.0, 2.0],
///     ... )
///     >>> result = reg.fit()
#[pyclass(name = "PureRegressor")]
pub struct PyPureRegressor {
    inner: Box<dyn DynRegressor>,
}

#[pymethods]
impl PyPureRegressor {
    #[new]
    #[pyo3(signature = (model, datasets, params, fit, residual=None, weights=None))]
    pub fn new(
        model: PyEquationOfStateAD,
        datasets: Vec<Bound<'_, PyAny>>,
        params: HashMap<String, f64>,
        fit: Vec<String>,
        residual: Option<&str>,
        weights: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let ds = extract_pure_datasets(&datasets)?;
        if let Some(ref w) = weights {
            validate_weights(w, ds.len())?;
        }
        let fit_strs: Vec<&str> = fit.iter().map(|s| s.as_str()).collect();
        let residual_fn = residual.map(parse_residual_fn).transpose()?;
        let inner: Box<dyn DynRegressor> = match model {
            PyEquationOfStateAD::PcSaftNonAssoc => {
                let r = Regressor::<PcSaftPure<f64, 4>, _>::new(ds, params, &fit_strs)
                    .map_err(|e: ParameterOptimizationError| PyValueError::new_err(e.to_string()))?;
                let r = match residual_fn {
                    Some(rf) => r.with_residual_fn(rf),
                    None => r,
                };
                let r = match weights {
                    Some(w) => r
                        .with_weights(w)
                        .map_err(|e: ParameterOptimizationError| PyValueError::new_err(e.to_string()))?,
                    None => r,
                };
                Box::new(r)
            }
            PyEquationOfStateAD::PcSaftFull => {
                let r = Regressor::<PcSaftPure<f64, 8>, _>::new(ds, params, &fit_strs)
                    .map_err(|e: ParameterOptimizationError| PyValueError::new_err(e.to_string()))?;
                let r = match residual_fn {
                    Some(rf) => r.with_residual_fn(rf),
                    None => r,
                };
                let r = match weights {
                    Some(w) => r
                        .with_weights(w)
                        .map_err(|e: ParameterOptimizationError| PyValueError::new_err(e.to_string()))?,
                    None => r,
                };
                Box::new(r)
            }
        };
        Ok(Self { inner })
    }
}

regressor_methods!(PyPureRegressor, "PureRegressor");

/// Levenberg-Marquardt regressor for binary-mixture parameter fitting.
///
/// Args:
///     model (EquationOfStateAD): Equation of state to use.
///     datasets (list): One or more datasets for mixture properties.
///     params (dict[str, float]): Initial values for all parameters keyed by name.
///     fit (list[str]): Names of the parameters to optimise.
///     residual (str, optional): Residual function — ``"relative_difference"``
///         (default), ``"log_difference"``, or ``"difference"``.
///     weights (list[float], optional): Per-dataset weights, one per dataset.
///         All weights must be positive. Default: 1.0 for every dataset.
///
/// Examples:
///     >>> bp = BubblePointDataset.from_csv("bubble.csv")
///     >>> reg = BinaryRegressor(
///     ...     model=EquationOfStateAD.PcSaftNonAssoc,
///     ...     datasets=[bp],
///     ...     params={
///     ...         "m1": 3.0,
///     ...         # more parameters for substance 1 and 2
///     ...         "k_ij": 0.01
///     ...     },
///     ...     fit=["k_ij"],
///     ... )
///     >>> result = reg.fit()
#[pyclass(name = "BinaryRegressor")]
pub struct PyBinaryRegressor {
    inner: Box<dyn DynRegressor>,
}

#[pymethods]
impl PyBinaryRegressor {
    #[new]
    #[pyo3(signature = (model, datasets, params, fit, residual=None, weights=None))]
    pub fn new(
        model: PyEquationOfStateAD,
        datasets: Vec<Bound<'_, PyAny>>,
        params: HashMap<String, f64>,
        fit: Vec<String>,
        residual: Option<&str>,
        weights: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let ds = extract_binary_datasets(&datasets)?;
        if let Some(ref w) = weights {
            validate_weights(w, ds.len())?;
        }
        let fit_strs: Vec<&str> = fit.iter().map(|s| s.as_str()).collect();
        let residual_fn = residual.map(parse_residual_fn).transpose()?;
        let inner: Box<dyn DynRegressor> = match model {
            PyEquationOfStateAD::PcSaftNonAssoc => {
                let r = Regressor::<PcSaftBinary<f64, 4>, _>::new(ds, params, &fit_strs)
                    .map_err(|e: ParameterOptimizationError| PyValueError::new_err(e.to_string()))?;
                let r = match residual_fn {
                    Some(rf) => r.with_residual_fn(rf),
                    None => r,
                };
                let r = match weights {
                    Some(w) => r
                        .with_weights(w)
                        .map_err(|e: ParameterOptimizationError| PyValueError::new_err(e.to_string()))?,
                    None => r,
                };
                Box::new(r)
            }
            PyEquationOfStateAD::PcSaftFull => {
                let r = Regressor::<PcSaftBinary<f64, 8>, _>::new(ds, params, &fit_strs)
                    .map_err(|e: ParameterOptimizationError| PyValueError::new_err(e.to_string()))?;
                let r = match residual_fn {
                    Some(rf) => r.with_residual_fn(rf),
                    None => r,
                };
                let r = match weights {
                    Some(w) => r
                        .with_weights(w)
                        .map_err(|e: ParameterOptimizationError| PyValueError::new_err(e.to_string()))?,
                    None => r,
                };
                Box::new(r)
            }
        };
        Ok(Self { inner })
    }
}

regressor_methods!(PyBinaryRegressor, "BinaryRegressor");

/// Downcast a Python list to `Vec<PureDataset>`.
fn extract_pure_datasets(list: &[Bound<'_, PyAny>]) -> PyResult<Vec<PureDataset>> {
    if list.is_empty() {
        return Err(PyValueError::new_err("datasets list must not be empty"));
    }
    list.iter()
        .map(|d| {
            if let Ok(vp) = d.extract::<PyRef<PyVaporPressureDataset>>() {
                Ok(PureDataset::VaporPressure(vp.inner.clone()))
            } else if let Ok(rho) = d.extract::<PyRef<PyLiquidDensityDataset>>() {
                Ok(PureDataset::LiquidDensity(rho.inner.clone()))
            } else if let Ok(eq) = d.extract::<PyRef<PyEquilibriumLiquidDensityDataset>>() {
                Ok(PureDataset::EquilibriumLiquidDensity(eq.inner.clone()))
            } else if let Ok(eq) = d.extract::<PyRef<PyEnthalpyOfVaporizationDataset>>() {
                Ok(PureDataset::EnthalpyOfVaporization(eq.inner.clone()))
            } else if let Ok(cp) = d.extract::<PyRef<PyResidualIsobaricHeatCapacityDataset>>() {
                Ok(PureDataset::ResidualIsobaricHeatCapacity(cp.inner.clone()))
            } else {
                Err(PyTypeError::new_err(format!(
                    "expected VaporPressureDataset, LiquidDensityDataset, \
                     EquilibriumLiquidDensityDataset, EnthalpyOfVaporizationDataset, or \
                     ResidualIsobaricHeatCapacityDataset; got {}",
                    d.get_type().name()?
                )))
            }
        })
        .collect()
}

/// Downcast a Python list to `Vec<BinaryDataset>`.
fn extract_binary_datasets(list: &[Bound<'_, PyAny>]) -> PyResult<Vec<BinaryDataset>> {
    if list.is_empty() {
        return Err(PyValueError::new_err("datasets list must not be empty"));
    }
    list.iter()
        .map(|d| {
            if let Ok(bp) = d.extract::<PyRef<PyBubblePointDataset>>() {
                Ok(BinaryDataset::BubblePoint(bp.inner.clone()))
            } else if let Ok(dp) = d.extract::<PyRef<PyDewPointDataset>>() {
                Ok(BinaryDataset::DewPoint(dp.inner.clone()))
            } else {
                Err(PyTypeError::new_err(format!(
                    "expected BubblePointDataset or DewPointDataset; got {}",
                    d.get_type().name()?
                )))
            }
        })
        .collect()
}
