use feos_core::parameter_optimization::{
    BinaryDataset, BinaryProperty, BubblePointRecord, Dataset, DewPointRecord,
    EnthalpyOfVaporizationRecord, EquilibriumLiquidDensityRecord, LiquidDensityRecord, PureDataset,
    PureProperty, ResidualIsobaricHeatCapacityRecord, VaporPressureRecord,
};
use ndarray::{Array2, ArrayView1};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::eos::PyEquationOfState;

/// Run a `Dataset::evaluate` against a single model or a list of models.
///
/// If `models` extracts as a single `PyEquationOfState`, returns the
/// `(predicted, converged)` arrays as 1D. If it extracts as a sequence of
/// models, returns them stacked as 2D arrays with shape `[n_points, n_models]`.
fn evaluate_models<'py, D: Dataset>(
    py: Python<'py>,
    dataset: &D,
    models: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    if let Ok(model) = models.extract::<PyRef<PyEquationOfState>>() {
        let (pred, ok) = dataset.evaluate(&model.0);
        return Ok((pred.to_pyarray(py).into_any(), ok.to_pyarray(py).into_any()));
    }

    let model_refs: Vec<PyRef<PyEquationOfState>> = models.extract().map_err(|_| {
        PyTypeError::new_err(
            "expected an EquationOfState or a sequence of EquationOfState instances",
        )
    })?;

    let n_points = dataset.target().len();
    let n_models = model_refs.len();
    let mut pred = Array2::<f64>::from_elem((n_points, n_models), f64::NAN);
    let mut ok = Array2::<bool>::from_elem((n_points, n_models), false);
    for (j, model) in model_refs.iter().enumerate() {
        let (p, c) = dataset.evaluate(&model.0);
        pred.column_mut(j).assign(&p);
        ok.column_mut(j).assign(&c);
    }
    Ok((pred.to_pyarray(py).into_any(), ok.to_pyarray(py).into_any()))
}

fn ensure_same_len(arrays: &[(&str, usize)]) -> PyResult<usize> {
    let Some((_, n)) = arrays.first() else {
        return Ok(0);
    };
    if arrays.iter().any(|(_, len)| len != n) {
        let names = arrays
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(PyValueError::new_err(format!(
            "all arrays must have the same length: {names}"
        )));
    }
    Ok(*n)
}

fn collect_records_2<R>(
    a: (&str, ArrayView1<'_, f64>),
    b: (&str, ArrayView1<'_, f64>),
    f: impl Fn(f64, f64) -> R,
) -> PyResult<Vec<R>> {
    ensure_same_len(&[(a.0, a.1.len()), (b.0, b.1.len())])?;
    Ok(a.1.iter().zip(b.1.iter()).map(|(&a, &b)| f(a, b)).collect())
}

fn collect_records_3<R>(
    a: (&str, ArrayView1<'_, f64>),
    b: (&str, ArrayView1<'_, f64>),
    c: (&str, ArrayView1<'_, f64>),
    f: impl Fn(f64, f64, f64) -> R,
) -> PyResult<Vec<R>> {
    ensure_same_len(&[(a.0, a.1.len()), (b.0, b.1.len()), (c.0, c.1.len())])?;
    Ok(a.1
        .iter()
        .zip(b.1.iter())
        .zip(c.1.iter())
        .map(|((&a, &b), &c)| f(a, b, c))
        .collect())
}

fn parse_pure_property(property: &str) -> PyResult<PureProperty> {
    match property {
        "vapor_pressure" => Ok(PureProperty::VaporPressure),
        "liquid_density" => Ok(PureProperty::LiquidDensity),
        "equilibrium_liquid_density" => Ok(PureProperty::EquilibriumLiquidDensity),
        "enthalpy_of_vaporization" => Ok(PureProperty::EnthalpyOfVaporization),
        "residual_isobaric_heat_capacity" => Ok(PureProperty::ResidualIsobaricHeatCapacity),
        _ => Err(PyValueError::new_err(format!(
            "unknown pure property '{property}'; valid: \
             'vapor_pressure', 'liquid_density', 'equilibrium_liquid_density', \
             'enthalpy_of_vaporization', 'residual_isobaric_heat_capacity'"
        ))),
    }
}

fn parse_binary_property(property: &str) -> PyResult<BinaryProperty> {
    match property {
        "bubble_point_pressure" | "bubble_point" => Ok(BinaryProperty::BubblePointPressure),
        "dew_point_pressure" | "dew_point" => Ok(BinaryProperty::DewPointPressure),
        _ => Err(PyValueError::new_err(format!(
            "unknown binary property '{property}'; valid: \
             'bubble_point_pressure', 'dew_point_pressure'"
        ))),
    }
}

#[pyclass(name = "PureDataset")]
pub struct PyPureDataset {
    pub(crate) inner: PureDataset,
}

#[pymethods]
impl PyPureDataset {
    /// Load pure-component data from CSV.
    ///
    /// Args:
    ///     path (str): Path to the CSV file.
    ///     property (str): Property identifier. Valid values are
    ///         ``"vapor_pressure"``, ``"liquid_density"``,
    ///         ``"equilibrium_liquid_density"``, ``"enthalpy_of_vaporization"``,
    ///         and ``"residual_isobaric_heat_capacity"``.
    ///     name (str, optional): Dataset name used in regressor diagnostics.
    #[staticmethod]
    #[pyo3(signature = (path, property, name=None))]
    pub fn from_csv(path: &str, property: &str, name: Option<&str>) -> PyResult<Self> {
        Self::from_csv_for_property(parse_pure_property(property)?, path, name)
    }

    /// Load vapor pressure data from CSV.
    ///
    /// CSV columns: ``temperature_k, vapor_pressure_pa``.
    #[staticmethod]
    #[pyo3(signature = (path, name=None))]
    pub fn vapor_pressure_from_csv(path: &str, name: Option<&str>) -> PyResult<Self> {
        Self::from_csv_for_property(PureProperty::VaporPressure, path, name)
    }

    /// Construct vapor pressure data from numpy arrays.
    #[staticmethod]
    #[pyo3(signature = (temperature_k, vapor_pressure_pa, name=None))]
    pub fn vapor_pressure(
        temperature_k: PyReadonlyArray1<f64>,
        vapor_pressure_pa: PyReadonlyArray1<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let temperature_k = temperature_k.as_array();
        let vapor_pressure_pa = vapor_pressure_pa.as_array();
        let records = collect_records_2(
            ("temperature_k", temperature_k),
            ("vapor_pressure_pa", vapor_pressure_pa),
            |temperature_k, vapor_pressure_pa| VaporPressureRecord {
                temperature_k,
                vapor_pressure_pa,
            },
        )?;
        Ok(Self::with_optional_name(
            PureDataset::vapor_pressure(records),
            name,
        ))
    }

    /// Load liquid density data from CSV.
    ///
    /// CSV columns: ``temperature_k, pressure_pa, liquid_density_kmol_m3``.
    #[staticmethod]
    #[pyo3(signature = (path, name=None))]
    pub fn liquid_density_from_csv(path: &str, name: Option<&str>) -> PyResult<Self> {
        Self::from_csv_for_property(PureProperty::LiquidDensity, path, name)
    }

    /// Construct liquid density data from numpy arrays.
    #[staticmethod]
    #[pyo3(signature = (temperature_k, pressure_pa, liquid_density_kmol_m3, name=None))]
    pub fn liquid_density(
        temperature_k: PyReadonlyArray1<f64>,
        pressure_pa: PyReadonlyArray1<f64>,
        liquid_density_kmol_m3: PyReadonlyArray1<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let temperature_k = temperature_k.as_array();
        let pressure_pa = pressure_pa.as_array();
        let liquid_density_kmol_m3 = liquid_density_kmol_m3.as_array();
        let records = collect_records_3(
            ("temperature_k", temperature_k),
            ("pressure_pa", pressure_pa),
            ("liquid_density_kmol_m3", liquid_density_kmol_m3),
            |temperature_k, pressure_pa, liquid_density_kmol_m3| LiquidDensityRecord {
                temperature_k,
                pressure_pa,
                liquid_density_kmol_m3,
            },
        )?;
        Ok(Self::with_optional_name(
            PureDataset::liquid_density(records),
            name,
        ))
    }

    /// Load saturated liquid density data from CSV.
    ///
    /// CSV columns: ``temperature_k, liquid_density_kmol_m3``.
    #[staticmethod]
    #[pyo3(signature = (path, name=None))]
    pub fn equilibrium_liquid_density_from_csv(path: &str, name: Option<&str>) -> PyResult<Self> {
        Self::from_csv_for_property(PureProperty::EquilibriumLiquidDensity, path, name)
    }

    /// Construct saturated liquid density data from numpy arrays.
    #[staticmethod]
    #[pyo3(signature = (temperature_k, liquid_density_kmol_m3, name=None))]
    pub fn equilibrium_liquid_density(
        temperature_k: PyReadonlyArray1<f64>,
        liquid_density_kmol_m3: PyReadonlyArray1<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let temperature_k = temperature_k.as_array();
        let liquid_density_kmol_m3 = liquid_density_kmol_m3.as_array();
        let records = collect_records_2(
            ("temperature_k", temperature_k),
            ("liquid_density_kmol_m3", liquid_density_kmol_m3),
            |temperature_k, liquid_density_kmol_m3| EquilibriumLiquidDensityRecord {
                temperature_k,
                liquid_density_kmol_m3,
            },
        )?;
        Ok(Self::with_optional_name(
            PureDataset::equilibrium_liquid_density(records),
            name,
        ))
    }

    /// Load enthalpy of vaporization data from CSV.
    ///
    /// CSV columns: ``temperature_k, dh_vap_j_mol``.
    #[staticmethod]
    #[pyo3(signature = (path, name=None))]
    pub fn enthalpy_of_vaporization_from_csv(path: &str, name: Option<&str>) -> PyResult<Self> {
        Self::from_csv_for_property(PureProperty::EnthalpyOfVaporization, path, name)
    }

    /// Construct enthalpy of vaporization data from numpy arrays.
    #[staticmethod]
    #[pyo3(signature = (temperature_k, dh_vap_j_mol, name=None))]
    pub fn enthalpy_of_vaporization(
        temperature_k: PyReadonlyArray1<f64>,
        dh_vap_j_mol: PyReadonlyArray1<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let temperature_k = temperature_k.as_array();
        let dh_vap_j_mol = dh_vap_j_mol.as_array();
        let records = collect_records_2(
            ("temperature_k", temperature_k),
            ("dh_vap_j_mol", dh_vap_j_mol),
            |temperature_k, dh_vap_j_mol| EnthalpyOfVaporizationRecord {
                temperature_k,
                dh_vap_j_mol,
            },
        )?;
        Ok(Self::with_optional_name(
            PureDataset::enthalpy_of_vaporization(records),
            name,
        ))
    }

    /// Load residual isobaric heat capacity data from CSV.
    ///
    /// CSV columns: ``temperature_k, pressure_pa, cp_res_j_molk``.
    #[staticmethod]
    #[pyo3(signature = (path, name=None))]
    pub fn residual_isobaric_heat_capacity_from_csv(
        path: &str,
        name: Option<&str>,
    ) -> PyResult<Self> {
        Self::from_csv_for_property(PureProperty::ResidualIsobaricHeatCapacity, path, name)
    }

    /// Construct residual isobaric heat capacity data from numpy arrays.
    #[staticmethod]
    #[pyo3(signature = (temperature_k, pressure_pa, cp_res_j_molk, name=None))]
    pub fn residual_isobaric_heat_capacity(
        temperature_k: PyReadonlyArray1<f64>,
        pressure_pa: PyReadonlyArray1<f64>,
        cp_res_j_molk: PyReadonlyArray1<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let temperature_k = temperature_k.as_array();
        let pressure_pa = pressure_pa.as_array();
        let cp_res_j_molk = cp_res_j_molk.as_array();
        let records = collect_records_3(
            ("temperature_k", temperature_k),
            ("pressure_pa", pressure_pa),
            ("cp_res_j_molk", cp_res_j_molk),
            |temperature_k, pressure_pa, cp_res_j_molk| ResidualIsobaricHeatCapacityRecord {
                temperature_k,
                pressure_pa,
                cp_res_j_molk,
            },
        )?;
        Ok(Self::with_optional_name(
            PureDataset::residual_isobaric_heat_capacity(records),
            name,
        ))
    }

    /// Property name.
    #[getter]
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// Number of data points.
    pub fn __len__(&self) -> usize {
        self.inner.target().len()
    }

    /// Target values.
    pub fn target<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.target().to_owned().to_pyarray(py)
    }

    /// Input values.
    pub fn inputs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.inputs().to_owned().to_pyarray(py)
    }

    /// Evaluate the dataset's property for one or more models (no gradients).
    ///
    /// Args:
    ///     models: A single ``EquationOfState`` or a list of them. Each
    ///         model must describe a single component (``components() == 1``).
    ///
    /// Returns:
    ///     ``(predicted, converged)``. For a single model, both are 1D arrays
    ///     of length ``n_points``. For a list of ``n_models`` models, both are
    ///     2D arrays of shape ``[n_points, n_models]``; column ``k``
    ///     corresponds to ``models[k]``. Non-converged points are reported as
    ///     ``NaN`` in ``predicted`` and ``False`` in ``converged``.
    pub fn evaluate<'py>(
        &self,
        py: Python<'py>,
        models: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        evaluate_models(py, &self.inner, models)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PureDataset(property={}, n={})",
            self.inner.name(),
            self.inner.target().len()
        )
    }
}

impl PyPureDataset {
    fn from_csv_for_property(
        property: PureProperty,
        path: &str,
        name: Option<&str>,
    ) -> PyResult<Self> {
        PureDataset::from_csv(property, std::path::Path::new(path))
            .map(|inner| Self::with_optional_name(inner, name))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn with_optional_name(mut inner: PureDataset, name: Option<&str>) -> Self {
        if let Some(n) = name {
            inner = inner.with_name(n);
        }
        Self { inner }
    }
}

#[pyclass(name = "BinaryDataset")]
pub struct PyBinaryDataset {
    pub(crate) inner: BinaryDataset,
}

#[pymethods]
impl PyBinaryDataset {
    /// Load binary-mixture data from CSV.
    ///
    /// Args:
    ///     path (str): Path to the CSV file.
    ///     property (str): Property identifier. Valid values are
    ///         ``"bubble_point_pressure"`` and ``"dew_point_pressure"``.
    ///         Short aliases ``"bubble_point"`` and ``"dew_point"`` are also accepted.
    ///     name (str, optional): Dataset name used in regressor diagnostics.
    #[staticmethod]
    #[pyo3(signature = (path, property, name=None))]
    pub fn from_csv(path: &str, property: &str, name: Option<&str>) -> PyResult<Self> {
        Self::from_csv_for_property(parse_binary_property(property)?, path, name)
    }

    /// Load bubble point pressure data from CSV.
    ///
    /// CSV columns: ``temperature_k, liquid_molefrac_1, bubble_pressure_pa``.
    #[staticmethod]
    #[pyo3(signature = (path, name=None))]
    pub fn bubble_point_pressure_from_csv(path: &str, name: Option<&str>) -> PyResult<Self> {
        Self::from_csv_for_property(BinaryProperty::BubblePointPressure, path, name)
    }

    /// Construct bubble point pressure data from numpy arrays.
    #[staticmethod]
    #[pyo3(signature = (temperature_k, liquid_molefrac_1, bubble_pressure_pa, name=None))]
    pub fn bubble_point_pressure(
        temperature_k: PyReadonlyArray1<f64>,
        liquid_molefrac_1: PyReadonlyArray1<f64>,
        bubble_pressure_pa: PyReadonlyArray1<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let temperature_k = temperature_k.as_array();
        let liquid_molefrac_1 = liquid_molefrac_1.as_array();
        let bubble_pressure_pa = bubble_pressure_pa.as_array();
        let records = collect_records_3(
            ("temperature_k", temperature_k),
            ("liquid_molefrac_1", liquid_molefrac_1),
            ("bubble_pressure_pa", bubble_pressure_pa),
            |temperature_k, liquid_molefrac_1, bubble_pressure_pa| BubblePointRecord {
                temperature_k,
                liquid_molefrac_1,
                bubble_pressure_pa,
            },
        )?;
        Ok(Self::with_optional_name(
            BinaryDataset::bubble_point_pressure(records),
            name,
        ))
    }

    /// Load dew point pressure data from CSV.
    ///
    /// CSV columns: ``temperature_k, vapor_molefrac_1, dew_pressure_pa``.
    #[staticmethod]
    #[pyo3(signature = (path, name=None))]
    pub fn dew_point_pressure_from_csv(path: &str, name: Option<&str>) -> PyResult<Self> {
        Self::from_csv_for_property(BinaryProperty::DewPointPressure, path, name)
    }

    /// Construct dew point pressure data from numpy arrays.
    #[staticmethod]
    #[pyo3(signature = (temperature_k, vapor_molefrac_1, dew_pressure_pa, name=None))]
    pub fn dew_point_pressure(
        temperature_k: PyReadonlyArray1<f64>,
        vapor_molefrac_1: PyReadonlyArray1<f64>,
        dew_pressure_pa: PyReadonlyArray1<f64>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let temperature_k = temperature_k.as_array();
        let vapor_molefrac_1 = vapor_molefrac_1.as_array();
        let dew_pressure_pa = dew_pressure_pa.as_array();
        let records = collect_records_3(
            ("temperature_k", temperature_k),
            ("vapor_molefrac_1", vapor_molefrac_1),
            ("dew_pressure_pa", dew_pressure_pa),
            |temperature_k, vapor_molefrac_1, dew_pressure_pa| DewPointRecord {
                temperature_k,
                vapor_molefrac_1,
                dew_pressure_pa,
            },
        )?;
        Ok(Self::with_optional_name(
            BinaryDataset::dew_point_pressure(records),
            name,
        ))
    }

    /// Property name.
    #[getter]
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// Number of data points.
    pub fn __len__(&self) -> usize {
        self.inner.target().len()
    }

    /// Target values.
    pub fn target<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.target().to_owned().to_pyarray(py)
    }

    /// Evaluate the dataset's property for one or more models (no gradients).
    ///
    /// Args:
    ///     models: A single ``EquationOfState`` or a list of them. Each
    ///         model must describe a binary system (``components() == 2``).
    ///
    /// Returns:
    ///     ``(predicted, converged)``. For a single model, both are 1D arrays
    ///     of length ``n_points``. For a list of ``n_models`` models, both are
    ///     2D arrays of shape ``[n_points, n_models]``; column ``k``
    ///     corresponds to ``models[k]``. Non-converged points are reported as
    ///     ``NaN`` in ``predicted`` and ``False`` in ``converged``.
    pub fn evaluate<'py>(
        &self,
        py: Python<'py>,
        models: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        evaluate_models(py, &self.inner, models)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "BinaryDataset(property={}, n={})",
            self.inner.name(),
            self.inner.target().len()
        )
    }
}

impl PyBinaryDataset {
    fn from_csv_for_property(
        property: BinaryProperty,
        path: &str,
        name: Option<&str>,
    ) -> PyResult<Self> {
        BinaryDataset::from_csv(property, std::path::Path::new(path))
            .map(|inner| Self::with_optional_name(inner, name))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn with_optional_name(mut inner: BinaryDataset, name: Option<&str>) -> Self {
        if let Some(n) = name {
            inner = inner.with_name(n);
        }
        Self { inner }
    }
}
