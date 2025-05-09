//! Structs for parameter objects.
//!
//! - PyPureRecord
//! - PyBinaryRecord
use super::identifier::{PyIdentifier, PyIdentifierOption};
use crate::error::PyFeosError;
use feos_core::parameter::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pythonize::{depythonize, pythonize};
use serde::{Deserialize, Serialize};
use serde_json::Value;

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

// impl PyPureRecord {
//     pub(crate) fn from_json(
//         substances: &[String],
//         file: &String,
//         identifier_option: PyIdentifierOption,
//     ) -> FeosResult<Vec<Self>> {
//         let pure_records = PureRecord::from_json(substances, file, identifier_option.0)?;
//         // create list of substances
//         let mut queried: HashSet<_> = substances.iter().map(|s| s.to_string()).collect();
//         // raise error on duplicate detection
//         if queried.len() != substances.len() {
//             return Err(FeosError::IncompatibleParameters(
//                 "A substance was defined more than once.".to_string(),
//             ));
//         }

//         let reader = BufReader::new(File::open::<&str>(file.as_ref())?);
//         let file_records: Vec<Self> = serde_json::from_reader(reader)?;
//         let mut records: HashMap<_, _> = HashMap::with_capacity(substances.len());

//         // build map, draining list of queried substances in the process
//         for record in file_records {
//             if let Some(id) = record.identifier.as_str(identifier_option) {
//                 queried.take(id).map(|id| records.insert(id, record));
//             }
//             // all parameters parsed
//             if queried.is_empty() {
//                 break;
//             }
//         }

//         // report missing parameters
//         if !queried.is_empty() {
//             return Err(FeosError::ComponentsNotFound(format!("{:?}", queried)));
//         };

//         // collect into vec in correct order
//         Ok(substances
//             .iter()
//             .map(|s| records.get(&s.to_string()).unwrap().clone())
//             .collect())
//     }
// }

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
            return Err(PyFeosError::Error(
                "No model parameters provided for PureRecord".to_string(),
            )
            .into());
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
        // Ok(self
        //     .try_into()
        //     .map_err(PyFeosError::from)
        //     .map(|r: PureRecord<Value, Value>| r.to_string())?)

        // let params = self
        //     .get_model_record(py)?
        //     .iter()
        //     .map(|(p, v)| format!(", {p}={v}"))
        //     .collect::<Vec<_>>()
        //     .join("");
        // Ok(format!(
        //     "PureRecord(identifier={}, molarweight={}{})",
        //     self.identifier.0, self.molarweight, params
        // ))
    }
}

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
        // let params: PyResult<String> = Python::with_gil(|py| {
        //     Ok(self
        //         .get_model_record(py)?
        //         .iter()
        //         .map(|(p, v)| format!(", {p}={v}"))
        //         .collect::<Vec<_>>()
        //         .join(""))
        // });
        // Ok(format!(
        //     "BinaryRecord(id1={}, id2={}{})",
        //     self.id1.0, self.id2.0, params?
        // ))
    }
}
