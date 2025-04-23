//! Structs for parameter objects.
//!
//! - PyPureRecord
//! - PyBinaryRecord
use feos_core::{parameter::*, FeosError, FeosResult};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyDict;
use pythonize::{depythonize, pythonize};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;

use crate::error::PyFeosError;

use super::identifier::{PyIdentifier, PyIdentifierOption};

#[pyclass(name = "PureRecord")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyPureRecord {
    #[pyo3(get)]
    pub identifier: PyIdentifier,
    #[pyo3(get)]
    #[serde(default)]
    pub molarweight: f64,
    pub model_record: Value,
}

impl<M> TryFrom<&PureRecord<M>> for PyPureRecord
where
    M: Serialize + Clone,
{
    type Error = FeosError;
    fn try_from(value: &PureRecord<M>) -> Result<Self, Self::Error> {
        Ok(serde_json::from_value(serde_json::to_value(
            value.clone(),
        )?)?)
    }
}

impl<M> TryInto<PureRecord<M>> for PyPureRecord
where
    for<'de> M: Deserialize<'de>,
{
    type Error = FeosError;
    fn try_into(self) -> FeosResult<PureRecord<M>> {
        Ok(serde_json::from_value(serde_json::to_value(self)?)?)
    }
}

impl PyPureRecord {
    pub(crate) fn from_json(
        substances: &[PyBackedStr],
        file: &PyBackedStr,
        identifier_option: PyIdentifierOption,
    ) -> FeosResult<Vec<Self>> {
        // create list of substances
        let mut queried: HashSet<_> = substances.iter().map(|s| s.to_string()).collect();
        // raise error on duplicate detection
        if queried.len() != substances.len() {
            return Err(FeosError::IncompatibleParameters(
                "A substance was defined more than once.".to_string(),
            ));
        }

        let reader = BufReader::new(File::open::<&str>(file.as_ref())?);
        let file_records: Vec<Self> = serde_json::from_reader(reader)?;
        let mut records: HashMap<_, _> = HashMap::with_capacity(substances.len());

        // build map, draining list of queried substances in the process
        for record in file_records {
            if let Some(id) = record.identifier.as_str(identifier_option.into()) {
                queried.take(id).map(|id| records.insert(id, record));
            }
            // all parameters parsed
            if queried.is_empty() {
                break;
            }
        }

        // report missing parameters
        if !queried.is_empty() {
            return Err(FeosError::ComponentsNotFound(format!("{:?}", queried)));
        };

        // collect into vec in correct order
        Ok(substances
            .iter()
            .map(|s| records.get(&s.to_string()).unwrap().clone())
            .collect())
    }
}

#[pymethods]
impl PyPureRecord {
    #[new]
    #[pyo3(signature = (identifier, molarweight, **parameters))]
    fn new(
        identifier: PyIdentifier,
        molarweight: f64,
        parameters: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if parameters.is_none() {
            return Err(PyFeosError::Error(
                "No model parameters provided for PureRecord".to_string(),
            )
            .into());
        }
        let model_record = depythonize(parameters.unwrap()).map_err(PyErr::from)?;
        Ok(Self {
            identifier,
            molarweight,
            model_record,
        })
    }

    #[getter]
    fn get_model_record<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pythonize(py, &self.model_record)
            .map_err(PyErr::from)
            .and_then(|d| d.downcast_into::<PyDict>().map_err(PyErr::from))
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pythonize(py, &self)
            .map_err(PyErr::from)
            .and_then(|d| d.downcast_into::<PyDict>().map_err(PyErr::from))
    }

    #[staticmethod]
    pub fn from_json_str(s: &str) -> PyResult<Self> {
        Ok(serde_json::from_str(s).map_err(PyFeosError::from)?)
    }

    pub fn to_json_str(&self) -> PyResult<String> {
        Ok(serde_json::to_string(&self).map_err(PyFeosError::from)?)
    }

    fn __repr__(&self) -> PyResult<String> {
        let params: PyResult<String> = Python::with_gil(|py| {
            Ok(self
                .get_model_record(py)?
                .iter()
                .map(|(p, v)| format!(", {p}={v}"))
                .collect::<Vec<_>>()
                .join(""))
        });
        Ok(format!(
            "PureRecord(identifier={}, molarweight={}{})",
            self.identifier.0, self.molarweight, params?
        ))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[pyclass(name = "BinaryRecord")]
pub struct PyBinaryRecord {
    #[pyo3(get)]
    pub id1: PyIdentifier,
    #[pyo3(get)]
    pub id2: PyIdentifier,
    pub model_record: Value,
}

impl<M> TryInto<BinaryRecord<M>> for PyBinaryRecord
where
    for<'de> M: Deserialize<'de>,
{
    type Error = FeosError;
    fn try_into(self) -> FeosResult<BinaryRecord<M>> {
        Ok(serde_json::from_value(serde_json::to_value(self)?)?)
    }
}

#[pymethods]
impl PyBinaryRecord {
    #[new]
    #[pyo3(signature = (id1, id2, **parameters))]
    fn new(
        id1: PyIdentifier,
        id2: PyIdentifier,
        parameters: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if parameters.is_none() {
            return Err(PyFeosError::Error(
                "No model parameters provided for BinaryRecord".to_string(),
            )
            .into());
        }
        let model_record = depythonize(parameters.unwrap()).map_err(PyErr::from)?;
        Ok(Self {
            id1,
            id2,
            model_record,
        })
    }

    #[getter]
    fn get_model_record<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pythonize(py, &self.model_record)
            .map_err(PyErr::from)
            .and_then(|d| d.downcast_into::<PyDict>().map_err(PyErr::from))
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pythonize(py, &self)
            .map_err(PyErr::from)
            .and_then(|d| d.downcast_into::<PyDict>().map_err(PyErr::from))
    }

    #[staticmethod]
    pub fn from_json_str(s: &str) -> PyResult<Self> {
        Ok(serde_json::from_str(s).map_err(PyFeosError::from)?)
    }

    pub fn to_json_str(&self) -> PyResult<String> {
        Ok(serde_json::to_string(&self).map_err(PyFeosError::from)?)
    }

    fn __repr__(&self) -> PyResult<String> {
        let params: PyResult<String> = Python::with_gil(|py| {
            Ok(self
                .get_model_record(py)?
                .iter()
                .map(|(p, v)| format!(", {p}={v}"))
                .collect::<Vec<_>>()
                .join(""))
        });
        Ok(format!(
            "BinaryRecord(id1={}, id2={}{})",
            self.id1.0, self.id2.0, params?
        ))
    }
}
