//! Structs for parameter objects.
//!
//! - PyPureRecord
//! - PyBinaryRecord
use super::identifier::{PyIdentifier, PyIdentifierOption};
use crate::error::PyFeosError;
use feos_core::parameter::*;
use pyo3::types::PyDict;
use pyo3::{exceptions::PyValueError, prelude::*};
use pythonize::{depythonize, pythonize};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Parameters that describe a pure component.
#[derive(Serialize, Deserialize, Clone)]
#[serde(from = "PureRecord<Value, Value>")]
#[serde(into = "PureRecord<Value, Value>")]
#[pyclass(name = "PureRecord")]
pub struct PyPureRecord {
    #[pyo3(get)]
    pub identifier: PyIdentifier,
    #[pyo3(get)]
    pub molarweight: f64,
    pub model_record: Value,
    pub association_sites: Vec<AssociationRecord<Value>>,
}

impl From<PyPureRecord> for PureRecord<Value, Value> {
    fn from(value: PyPureRecord) -> Self {
        Self {
            identifier: value.identifier.0,
            molarweight: value.molarweight,
            model_record: value.model_record,
            association_sites: value.association_sites,
        }
    }
}

impl From<PureRecord<Value, Value>> for PyPureRecord {
    fn from(value: PureRecord<Value, Value>) -> Self {
        Self {
            identifier: PyIdentifier(value.identifier),
            molarweight: value.molarweight,
            model_record: value.model_record,
            association_sites: value.association_sites,
        }
    }
}

#[pymethods]
impl PyPureRecord {
    #[new]
    #[pyo3(signature = (identifier, molarweight, **parameters))]
    fn new(
        py: Python,
        identifier: PyIdentifier,
        molarweight: f64,
        parameters: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let Some(parameters) = parameters else {
            return Err(PyErr::new::<PyValueError, _>(
                "No model parameters provided for PureRecord",
            ));
        };
        parameters.set_item("identifier", pythonize(py, &identifier)?)?;
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

    /// Read a list of `PureRecord`s from a JSON file.
    ///
    /// Parameters
    /// ----------
    /// substances : list[str]
    ///     List of component identifiers.
    /// path : str
    ///     Path to file containing the segment records.
    /// identifier_option : IdentifierOption
    ///     The type of identifier used in the substance list.
    ///
    /// Returns
    /// -------
    /// [SegmentRecord]
    #[staticmethod]
    pub fn from_json(
        substances: Vec<String>,
        file: &str,
        identifier_option: PyIdentifierOption,
    ) -> PyResult<Vec<Self>> {
        Ok(
            PureRecord::from_json(&substances, file, identifier_option.into())
                .map_err(PyFeosError::from)?
                .into_iter()
                .map(|r| r.into())
                .collect(),
        )
    }

    fn __repr__(&self) -> String {
        PureRecord::from(self.clone()).to_string()
    }
}

/// Binary interaction parameters.
#[derive(Serialize, Deserialize, Clone)]
#[serde(from = "BinaryRecord<Identifier, Value, Value>")]
#[serde(into = "BinaryRecord<Identifier, Value, Value>")]
#[pyclass(name = "BinaryRecord")]
pub struct PyBinaryRecord {
    #[pyo3(get)]
    pub id1: PyIdentifier,
    #[pyo3(get)]
    pub id2: PyIdentifier,
    pub model_record: Option<Value>,
    pub association_sites: Vec<BinaryAssociationRecord<Value>>,
}

impl From<PyBinaryRecord> for BinaryRecord<Identifier, Value, Value> {
    fn from(value: PyBinaryRecord) -> Self {
        Self {
            id1: value.id1.0,
            id2: value.id2.0,
            model_record: value.model_record,
            association_sites: value.association_sites,
        }
    }
}

impl From<BinaryRecord<Identifier, Value, Value>> for PyBinaryRecord {
    fn from(value: BinaryRecord<Identifier, Value, Value>) -> Self {
        Self {
            id1: PyIdentifier(value.id1),
            id2: PyIdentifier(value.id2),
            model_record: value.model_record,
            association_sites: value.association_sites,
        }
    }
}

#[pymethods]
impl PyBinaryRecord {
    #[new]
    #[pyo3(signature = (id1, id2, **parameters))]
    fn new(
        py: Python,
        id1: PyIdentifier,
        id2: PyIdentifier,
        parameters: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let Some(parameters) = parameters else {
            return Err(PyFeosError::Error(
                "No model parameters provided for BinaryRecord".to_string(),
            )
            .into());
        };
        parameters.set_item("id1", pythonize(py, &id1)?)?;
        parameters.set_item("id2", pythonize(py, &id2)?)?;
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
