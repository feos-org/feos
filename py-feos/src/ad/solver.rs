use std::collections::HashMap;

use feos::pcsaft::{PcSaftBinary, PcSaftPure};
use feos_core::parameter_optimization::{
    BinaryDataset, BubblePointDataset, BubblePointRecord, Dataset, DewPointDataset, DewPointRecord,
    DynSolver, EquilibriumLiquidDensityDataset, EquilibriumLiquidDensityRecord, FitConfig,
    FitResult, FittingError, LiquidDensityDataset, LiquidDensityRecord, LossFunction,
    NonConvergenceStrategy, PureDataset, Regressor, VaporPressureDataset, VaporPressureRecord,
};
use ndarray::Array2;
use numpy::{PyArray1, PyArray2, ToPyArray};
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
            ///     name (str, optional): Dataset name used in solver diagnostics
            ///         and results. Must be unique within a solver; defaults to
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
    ///     name (str, optional): Dataset name (must be unique within a solver).
    #[new]
    #[pyo3(signature = (temperature_k, vapor_pressure_pa, name=None))]
    pub fn new(
        temperature_k: Vec<f64>,
        vapor_pressure_pa: Vec<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        if temperature_k.len() != vapor_pressure_pa.len() {
            return Err(PyValueError::new_err(
                "temperature_k and vapor_pressure_pa must have the same length",
            ));
        }
        let records = temperature_k
            .into_iter()
            .zip(vapor_pressure_pa)
            .map(|(t, p)| VaporPressureRecord {
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
    "temperature_k, pressure_pa, liquid_density_molm3"
);

#[pymethods]
impl PyLiquidDensityDataset {
    /// Construct from numpy arrays.
    ///
    /// Args:
    ///     temperature_k (np.ndarray): Temperatures in K.
    ///     pressure_pa (np.ndarray): Pressures in Pa.
    ///     liquid_density_molm3 (np.ndarray): Liquid molar densities in mol/m³.
    ///     name (str, optional): Dataset name (must be unique within a solver).
    #[new]
    #[pyo3(signature = (temperature_k, pressure_pa, liquid_density_molm3, name=None))]
    pub fn new(
        temperature_k: Vec<f64>,
        pressure_pa: Vec<f64>,
        liquid_density_molm3: Vec<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let n = temperature_k.len();
        if pressure_pa.len() != n || liquid_density_molm3.len() != n {
            return Err(PyValueError::new_err(
                "all arrays must have the same length",
            ));
        }
        let records = (0..n)
            .map(|i| LiquidDensityRecord {
                temperature_k: temperature_k[i],
                pressure_pa: pressure_pa[i],
                liquid_density_molm3: liquid_density_molm3[i],
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
    "temperature_k, liquid_density_molm3"
);

#[pymethods]
impl PyEquilibriumLiquidDensityDataset {
    /// Construct from numpy arrays.
    ///
    /// Args:
    ///     temperature_k (np.ndarray): Temperatures in K.
    ///     liquid_density_molm3 (np.ndarray): Saturated liquid molar densities in mol/m³.
    ///     name (str, optional): Dataset name (must be unique within a solver).
    #[new]
    #[pyo3(signature = (temperature_k, liquid_density_molm3, name=None))]
    pub fn new(
        temperature_k: Vec<f64>,
        liquid_density_molm3: Vec<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        if temperature_k.len() != liquid_density_molm3.len() {
            return Err(PyValueError::new_err(
                "temperature_k and liquid_density_molm3 must have the same length",
            ));
        }
        let records = temperature_k
            .into_iter()
            .zip(liquid_density_molm3)
            .map(|(t, rho)| EquilibriumLiquidDensityRecord {
                temperature_k: t,
                liquid_density_molm3: rho,
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
    ///     name (str, optional): Dataset name (must be unique within a solver).
    #[new]
    #[pyo3(signature = (temperature_k, liquid_molefrac_1, bubble_pressure_pa, name=None))]
    pub fn new(
        temperature_k: Vec<f64>,
        liquid_molefrac_1: Vec<f64>,
        bubble_pressure_pa: Vec<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let n = temperature_k.len();
        if liquid_molefrac_1.len() != n || bubble_pressure_pa.len() != n {
            return Err(PyValueError::new_err(
                "all arrays must have the same length",
            ));
        }
        let records = (0..n)
            .map(|i| BubblePointRecord {
                temperature_k: temperature_k[i],
                liquid_molefrac_1: liquid_molefrac_1[i],
                bubble_pressure_pa: bubble_pressure_pa[i],
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
    ///     name (str, optional): Dataset name (must be unique within a solver).
    #[new]
    #[pyo3(signature = (temperature_k, vapor_molefrac_1, dew_pressure_pa, name=None))]
    pub fn new(
        temperature_k: Vec<f64>,
        vapor_molefrac_1: Vec<f64>,
        dew_pressure_pa: Vec<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let n = temperature_k.len();
        if vapor_molefrac_1.len() != n || dew_pressure_pa.len() != n {
            return Err(PyValueError::new_err(
                "all arrays must have the same length",
            ));
        }
        let records = (0..n)
            .map(|i| DewPointRecord {
                temperature_k: temperature_k[i],
                vapor_molefrac_1: vapor_molefrac_1[i],
                dew_pressure_pa: dew_pressure_pa[i],
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
///     >>> NonConvergenceStrategy.adaptive_penalty(5.0)
///     NonConvergenceStrategy.AdaptivePenalty(factor=5.0)
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

    /// Penalty scaled by `factor × max |r_rel|` over converged points.
    ///
    /// Falls back to `factor` when no points converged.
    ///
    /// Args:
    ///     factor (float): Scaling factor, e.g. `5.0`.
    #[staticmethod]
    pub fn adaptive_penalty(factor: f64) -> PyResult<Self> {
        if factor <= 0.0 {
            return Err(PyValueError::new_err("factor must be > 0"));
        }
        Ok(Self {
            inner: NonConvergenceStrategy::AdaptivePenalty(factor),
        })
    }

    pub fn __repr__(&self) -> String {
        match &self.inner {
            NonConvergenceStrategy::Ignore => "NonConvergenceStrategy.Ignore".to_string(),
            NonConvergenceStrategy::Penalty(v) => {
                format!("NonConvergenceStrategy.Penalty({v})")
            }
            NonConvergenceStrategy::AdaptivePenalty(f) => {
                format!("NonConvergenceStrategy.AdaptivePenalty(factor={f})")
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
#[pyclass(name = "FitConfig")]
#[derive(Clone)]
pub struct PyFitConfig {
    pub(crate) inner: FitConfig,
}

#[pymethods]
impl PyFitConfig {
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
            inner: FitConfig {
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
            "FitConfig(ftol={}, xtol={}, gtol={}, stepbound={}, patience={}, \
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
#[pyclass(name = "FitResult")]
pub struct PyFitResult {
    pub(crate) inner: FitResult,
}

#[pymethods]
impl PyFitResult {
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
    /// Fitted parameters at thei optimal values, all others at their initial values.
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
            "FitResult(converged={}, n_eval={}, elapsed={:.1}ms, aad=[{}])",
            self.inner.converged,
            self.inner.n_evaluations,
            self.inner.elapsed.as_secs_f64() * 1000.0,
            aad_strs.join(", ")
        )
    }
}

macro_rules! solver_methods {
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
            ///     params (dict[str, float]): All EOS parameters as returned by
            ///         :attr:`FitResult.all_parameters`.
            ///
            /// Returns:
            ///     list[dict[str, np.ndarray]]: One dict per dataset, in the
            ///         same order as the datasets passed to the constructor.
            ///
            /// Examples:
            ///     >>> result = regressor.fit()
            ///     >>> ds = regressor.evaluate_datasets(result.all_parameters)
            ///     >>> dfs = [pd.DataFrame(d) for d in ds] # one pandas.DataFrame per Dataset
            pub fn evaluate_datasets<'py>(
                &self,
                py: Python<'py>,
                params: HashMap<String, f64>,
            ) -> PyResult<Vec<Bound<'py, PyDict>>> {
                // Build the full param vector in canonical order.
                let names = self.inner.all_param_names();
                let param_vec: Result<Vec<f64>, _> = names
                    .iter()
                    .map(|n| {
                        params.get(n).copied().ok_or_else(|| {
                            PyValueError::new_err(format!("missing parameter '{n}'"))
                        })
                    })
                    .collect();
                let param_vec = param_vec?;

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
            ///     config (FitConfig, optional): Regressor hyperparameters
            ///         (including non-convergence strategy).
            ///     loss (LossFunction, optional): Loss function applied to
            ///         relative residuals.
            ///
            /// Returns:
            ///     FitResult
            #[pyo3(signature = (config=None, loss=None))]
            pub fn fit(
                &mut self,
                config: Option<PyFitConfig>,
                loss: Option<Bound<'_, PyLossFunction>>,
            ) -> PyFitResult {
                let config = config.map(|c| c.inner).unwrap_or_default();
                let loss = loss.map(|l| l.borrow().inner.clone());
                PyFitResult {
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

/// Levenberg-Marquardt solver for pure-component parameter fitting.
///
/// Args:
///     model (EquationOfStateAD): Equation of state to use.
///     datasets (list): One or more datasets for pure substance properties.
///     params (dict[str, float]): Initial values for all parameters keyed by name.
///     fit (list[str]): Names of the parameters to optimise.
///
/// Examples:
///     >>> vp  = VaporPressureDataset.from_csv("vp.csv")
///     >>> rho = LiquidDensityDataset.from_csv("rho.csv")
///     >>> solver = PureRegressor(
///     ...     model=EquationOfStateAD.PcSaftNonAssoc,
///     ...     datasets=[vp, rho],
///     ...     params={"m": 2.5, "sigma": 3.4, "epsilon_k": 280.0, "mu": 0.0},
///     ...     fit=["sigma", "epsilon_k"],
///     ... )
///     >>> result = solver.fit()
#[pyclass(name = "PureRegressor")]
pub struct PyPureRegressor {
    inner: Box<dyn DynSolver>,
}

#[pymethods]
impl PyPureRegressor {
    #[new]
    #[pyo3(signature = (model, datasets, params, fit))]
    pub fn new(
        model: PyEquationOfStateAD,
        datasets: Vec<Bound<'_, PyAny>>,
        params: HashMap<String, f64>,
        fit: Vec<String>,
    ) -> PyResult<Self> {
        let ds = extract_pure_datasets(&datasets)?;
        let fit_strs: Vec<&str> = fit.iter().map(|s| s.as_str()).collect();
        let inner: Box<dyn DynSolver> = match model {
            PyEquationOfStateAD::PcSaftNonAssoc => Box::new(
                Regressor::<PcSaftPure<f64, 4>, _>::new(ds, params, &fit_strs)
                    .map_err(|e: FittingError| PyValueError::new_err(e.to_string()))?,
            ),
            PyEquationOfStateAD::PcSaftFull => Box::new(
                Regressor::<PcSaftPure<f64, 8>, _>::new(ds, params, &fit_strs)
                    .map_err(|e: FittingError| PyValueError::new_err(e.to_string()))?,
            ),
        };
        Ok(Self { inner })
    }
}

solver_methods!(PyPureRegressor, "PureRegressor");

/// Levenberg-Marquardt solver for binary-mixture parameter fitting.
///
/// Args:
///     model (EquationOfStateAD): Equation of state to use.
///     datasets (list): One or more datasets for mixture properties.
///     params (dict[str, float]): Initial values for all parameters keyed by name.
///     fit (list[str]): Names of the parameters to optimise.
///
/// Examples:
///     >>> bp = BubblePointDataset.from_csv("bubble.csv")
///     >>> solver = BinaryRegressor(
///     ...     model=EquationOfStateAD.PcSaftNonAssoc,
///     ...     datasets=[bp],
///     ...     params={
///     ...         "m1": 3.0,
///     ...         # more parameters for substance 1 and 2
///     ...         "k_ij": 0.01
///     ...     },
///     ...     fit=["k_ij"],
///     ... )
///     >>> result = solver.fit()
#[pyclass(name = "BinaryRegressor")]
pub struct PyBinaryRegressor {
    inner: Box<dyn DynSolver>,
}

#[pymethods]
impl PyBinaryRegressor {
    #[new]
    #[pyo3(signature = (model, datasets, params, fit))]
    pub fn new(
        model: PyEquationOfStateAD,
        datasets: Vec<Bound<'_, PyAny>>,
        params: HashMap<String, f64>,
        fit: Vec<String>,
    ) -> PyResult<Self> {
        let ds = extract_binary_datasets(&datasets)?;
        let fit_strs: Vec<&str> = fit.iter().map(|s| s.as_str()).collect();
        let inner: Box<dyn DynSolver> = match model {
            PyEquationOfStateAD::PcSaftNonAssoc => Box::new(
                Regressor::<PcSaftBinary<f64, 4>, _>::new(ds, params, &fit_strs)
                    .map_err(|e: FittingError| PyValueError::new_err(e.to_string()))?,
            ),
            PyEquationOfStateAD::PcSaftFull => Box::new(
                Regressor::<PcSaftBinary<f64, 8>, _>::new(ds, params, &fit_strs)
                    .map_err(|e: FittingError| PyValueError::new_err(e.to_string()))?,
            ),
        };
        Ok(Self { inner })
    }
}

solver_methods!(PyBinaryRegressor, "BinaryRegressor");

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
            } else {
                Err(PyTypeError::new_err(format!(
                    "expected VaporPressureDataset, LiquidDensityDataset, or \
                     EquilibriumLiquidDensityDataset; got {}",
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
