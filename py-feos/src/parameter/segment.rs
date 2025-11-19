use crate::error::PyFeosError;
use feos_core::parameter::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pythonize::{depythonize, pythonize};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Parameters describing individual segments.
#[derive(Serialize, Deserialize, Clone)]
#[serde(from = "SegmentRecord<Value, Value>")]
#[serde(into = "SegmentRecord<Value, Value>")]
#[pyclass(module = "feos.feos", name = "SegmentRecord")]
pub struct PySegmentRecord {
    #[pyo3(get)]
    identifier: String,
    #[pyo3(get)]
    molarweight: f64,
    model_record: Value,
    association_sites: Vec<AssociationRecord<Value>>,
}

impl From<PySegmentRecord> for SegmentRecord<Value, Value> {
    fn from(value: PySegmentRecord) -> Self {
        Self {
            identifier: value.identifier,
            molarweight: value.molarweight,
            model_record: value.model_record,
            association_sites: value.association_sites,
        }
    }
}

impl From<SegmentRecord<Value, Value>> for PySegmentRecord {
    fn from(value: SegmentRecord<Value, Value>) -> Self {
        Self {
            identifier: value.identifier,
            molarweight: value.molarweight,
            model_record: value.model_record,
            association_sites: value.association_sites,
        }
    }
}

#[pymethods]
impl PySegmentRecord {
    #[new]
    #[pyo3(signature = (identifier, molarweight, **parameters))]
    fn new(
        identifier: String,
        molarweight: f64,
        parameters: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let Some(parameters) = parameters else {
            return Err(PyFeosError::Error(
                "No model parameters provided for SegmentRecord".to_string(),
            )
            .into());
        };
        parameters.set_item("identifier", identifier)?;
        parameters.set_item("molarweight", molarweight)?;
        Ok(depythonize(parameters)?)
    }

    #[getter]
    fn get_model_record<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize(py, &self.model_record).map_err(PyErr::from)
    }

    #[getter]
    fn get_association_sites<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize(py, &self.association_sites).map_err(PyErr::from)
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize(py, &self).map_err(PyErr::from)
    }

    #[staticmethod]
    pub fn from_json_str(s: &str) -> PyResult<Self> {
        Ok(serde_json::from_str(s).map_err(PyFeosError::from)?)
    }

    pub fn to_json_str(&self) -> PyResult<String> {
        Ok(serde_json::to_string(&self).map_err(PyFeosError::from)?)
    }

    /// Read a list of [SegmentRecord]s from a JSON file.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to file containing the segment records.
    ///
    /// Returns
    /// -------
    /// [SegmentRecord]
    #[staticmethod]
    pub fn from_json(path: &str) -> PyResult<Vec<Self>> {
        Ok(SegmentRecord::from_json(path)
            .map_err(PyFeosError::from)?
            .into_iter()
            .map(|r| r.into())
            .collect())
    }

    fn __repr__(&self) -> String {
        SegmentRecord::from(self.clone()).to_string()
    }
}

/// Binary segment/segment interaction parameters.
#[derive(Serialize, Deserialize, Clone)]
#[serde(from = "BinaryRecord<String, Value, Value>")]
#[serde(into = "BinaryRecord<String, Value, Value>")]
#[pyclass(module = "feos.feos", name = "BinarySegmentRecord")]
pub struct PyBinarySegmentRecord {
    #[pyo3(get)]
    pub id1: String,
    #[pyo3(get)]
    pub id2: String,
    pub model_record: Option<Value>,
    pub association_sites: Vec<BinaryAssociationRecord<Value>>,
}

impl From<PyBinarySegmentRecord> for BinaryRecord<String, Value, Value> {
    fn from(value: PyBinarySegmentRecord) -> Self {
        Self {
            id1: value.id1,
            id2: value.id2,
            model_record: value.model_record,
            association_sites: value.association_sites,
        }
    }
}

impl From<BinaryRecord<String, Value, Value>> for PyBinarySegmentRecord {
    fn from(value: BinaryRecord<String, Value, Value>) -> Self {
        Self {
            id1: value.id1,
            id2: value.id2,
            model_record: value.model_record,
            association_sites: value.association_sites,
        }
    }
}

#[pymethods]
impl PyBinarySegmentRecord {
    #[new]
    #[pyo3(signature = (id1, id2, **parameters))]
    fn new(id1: String, id2: String, parameters: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let Some(parameters) = parameters else {
            return Err(PyFeosError::Error(
                "No model parameters provided for BinaryRecord".to_string(),
            )
            .into());
        };
        parameters.set_item("id1", id1)?;
        parameters.set_item("id2", id2)?;
        Ok(depythonize(parameters)?)
    }

    #[getter]
    fn get_model_record<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize(py, &self.model_record).map_err(PyErr::from)
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize(py, &self).map_err(PyErr::from)
    }

    #[staticmethod]
    pub fn from_json_str(s: &str) -> PyResult<Self> {
        Ok(serde_json::from_str(s).map_err(PyFeosError::from)?)
    }

    pub fn to_json_str(&self) -> PyResult<String> {
        Ok(serde_json::to_string(&self).map_err(PyFeosError::from)?)
    }

    fn __repr__(&self) -> String {
        BinaryRecord::from(self.clone()).to_string()
    }
}
