use std::{fs::File, io::BufReader};

use crate::error::{PyFeosError, PyFeosResult};
use feos_core::{
    parameter::{BinarySegmentRecord, SegmentRecord},
    FeosError, FeosResult,
};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyDict;
use pythonize::{depythonize, pythonize, PythonizeError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::model_record;

#[derive(Debug, Serialize, Deserialize, Clone)]
#[pyclass(name = "SegmentRecord")]
pub struct PySegmentRecord {
    #[pyo3(get)]
    identifier: String,
    #[pyo3(get)]
    molarweight: f64,
    model_record: Value,
}

impl<M> TryInto<SegmentRecord<M>> for PySegmentRecord
where
    for<'de> M: Deserialize<'de>,
{
    type Error = FeosError;
    fn try_into(self) -> FeosResult<SegmentRecord<M>> {
        Ok(serde_json::from_value(serde_json::to_value(self)?)?)
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
        if parameters.is_none() {
            return Err(PyFeosError::Error(
                "No model parameters provided for SegmentRecord".to_string(),
            )
            .into());
        }
        let model_record = depythonize(parameters.unwrap())?;
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
            "SegmentRecord(identifier={}, molarweight={}{})",
            self.identifier, self.molarweight, params?
        ))
    }

    /// Read a list of `SegmentRecord`s from a JSON file.
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
        Ok(
            serde_json::from_reader(BufReader::new(File::open(path)?))
                .map_err(PyFeosError::from)?,
        )
    }
}

/// A collection of parameters that model interactions between two segments.
#[pyclass(name = "BinarySegmentRecord")]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PyBinarySegmentRecord {
    /// Identifier of the first component
    #[pyo3(get)]
    pub id1: String,
    /// Identifier of the second component
    #[pyo3(get)]
    pub id2: String,
    /// Binary interaction parameter(s)
    #[pyo3(get)]
    pub model_record: f64,
}

// For future extennsion to generic model parameter M
// impl<M> TryInto<BinarySegmentRecord<M>> for PyBinarySegmentRecord
// where
//     for<'de> M: Deserialize<'de>,
// {
//     type Error = FeosError;
//     fn try_into(self) -> FeosResult<BinarySegmentRecord<M>> {
//         Ok(serde_json::from_value(serde_json::to_value(self)?)?)
//     }
// }

impl From<BinarySegmentRecord> for PyBinarySegmentRecord {
    fn from(value: BinarySegmentRecord) -> Self {
        Self {
            id1: value.id1,
            id2: value.id2,
            model_record: value.model_record,
        }
    }
}

impl Into<BinarySegmentRecord> for PyBinarySegmentRecord {
    fn into(self) -> BinarySegmentRecord {
        BinarySegmentRecord {
            id1: self.id1,
            id2: self.id2,
            model_record: self.model_record,
        }
    }
}

#[pymethods]
impl PyBinarySegmentRecord {
    #[new]
    fn py_new(id1: String, id2: String, model_record: f64) -> PyResult<Self> {
        Ok(Self {
            id1,
            id2,
            model_record,
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "BinarySegmentRecord(id1={}, id2={}, model_record={})",
            self.id1, self.id2, self.model_record
        ))
    }

    /// Creates record from json string.
    #[staticmethod]
    fn from_json_str(json: &str) -> PyResult<Self> {
        Ok(serde_json::from_str(json).map_err(PyFeosError::from)?)
    }

    /// Creates a json string from record.
    fn to_json_str(&self) -> PyResult<String> {
        Ok(serde_json::to_string(&self).map_err(PyFeosError::from)?)
    }
}
