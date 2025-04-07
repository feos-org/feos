use crate::parameter::*;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use ndarray::Array2;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyDict, PyList};
use pythonize::{depythonize, pythonize};
use serde::{Deserialize, Serialize};
mod fragmentation;
pub use fragmentation::PySmartsRecord;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;

impl From<ParameterError> for PyErr {
    fn from(e: ParameterError) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }
}

#[pymethods]
impl Identifier {
    #[new]
    #[pyo3(
        text_signature = "(cas=None, name=None, iupac_name=None, smiles=None, inchi=None, formula=None)",
        signature = (cas=None, name=None, iupac_name=None, smiles=None, inchi=None, formula=None)
    )]
    fn py_new(
        cas: Option<&str>,
        name: Option<&str>,
        iupac_name: Option<&str>,
        smiles: Option<&str>,
        inchi: Option<&str>,
        formula: Option<&str>,
    ) -> Self {
        Self::new(cas, name, iupac_name, smiles, inchi, formula)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.to_string())
    }

    /// Creates record from json string.
    #[staticmethod]
    fn from_json_str(json: &str) -> Result<Self, ParameterError> {
        Ok(serde_json::from_str(json)?)
    }

    /// Creates a json string from record.
    fn to_json_str(&self) -> Result<String, ParameterError> {
        Ok(serde_json::to_string(&self)?)
    }
}

#[pymethods]
impl ChemicalRecord {
    #[new]
    #[pyo3(text_signature = "(identifier, segments, bonds=None)", signature = (identifier, segments, bonds=None))]
    fn py_new(
        identifier: Identifier,
        segments: Vec<String>,
        bonds: Option<Vec<[usize; 2]>>,
    ) -> Self {
        Self::new(identifier, segments, bonds)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.to_string())
    }

    /// Creates record from json string.
    #[staticmethod]
    fn from_json_str(json: &str) -> Result<Self, ParameterError> {
        Ok(serde_json::from_str(json)?)
    }

    /// Creates a json string from record.
    fn to_json_str(&self) -> Result<String, ParameterError> {
        Ok(serde_json::to_string(&self)?)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[pyclass(name = "BinaryRecord")]
pub struct PyBinaryRecord {
    #[pyo3(get)]
    id1: Identifier,
    #[pyo3(get)]
    id2: Identifier,
    model_record: Value,
}

impl<M> TryInto<BinaryRecord<M>> for PyBinaryRecord
where
    for<'de> M: Deserialize<'de>,
{
    type Error = ParameterError;
    fn try_into(self) -> Result<BinaryRecord<M>, ParameterError> {
        Ok(serde_json::from_value(serde_json::to_value(self)?)?)
    }
}

#[pymethods]
impl PyBinaryRecord {
    #[new]
    #[pyo3(signature = (id1, id2, **parameters))]
    fn new(
        id1: Identifier,
        id2: Identifier,
        parameters: Option<&Bound<'_, PyDict>>,
    ) -> Result<Self, ParameterError> {
        if parameters.is_none() {
            return Err(ParameterError::Error(
                "No model parameters provided for BinaryRecord".to_string(),
            ));
        }
        let model_record =
            depythonize(parameters.unwrap()).map_err(|e| ParameterError::Error(e.to_string()))?;
        Ok(Self {
            id1,
            id2,
            model_record,
        })
    }

    #[getter]
    fn get_model_record<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyDict>, ParameterError> {
        pythonize(py, &self.model_record)
            .map_err(|e| ParameterError::Error(e.to_string()))
            .and_then(|d| {
                d.downcast_into::<PyDict>()
                    .map_err(|e| ParameterError::Error(e.to_string()))
            })
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyDict>, ParameterError> {
        pythonize(py, &self)
            .map_err(|e| ParameterError::Error(e.to_string()))
            .and_then(|d| {
                d.downcast_into::<PyDict>()
                    .map_err(|e| ParameterError::Error(e.to_string()))
            })
    }

    #[staticmethod]
    pub fn from_json_str(s: &str) -> Result<Self, ParameterError> {
        Ok(serde_json::from_str(s)?)
    }

    pub fn to_json_str(&self) -> Result<String, ParameterError> {
        Ok(serde_json::to_string(&self)?)
    }

    fn __repr__(&self) -> Result<String, ParameterError> {
        let params: Result<String, ParameterError> = Python::with_gil(|py| {
            Ok(self
                .get_model_record(py)?
                .iter()
                .map(|(p, v)| format!(", {p}={v}"))
                .collect::<Vec<_>>()
                .join(""))
        });
        Ok(format!(
            "BinaryRecord(id1={}, id2={}{})",
            self.id1, self.id2, params?
        ))
    }
}

#[pymethods]
impl BinarySegmentRecord {
    #[new]
    fn py_new(id1: String, id2: String, model_record: f64) -> PyResult<Self> {
        Ok(Self::new(id1, id2, model_record))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.to_string())
    }

    /// Creates record from json string.
    #[staticmethod]
    fn from_json_str(json: &str) -> Result<Self, ParameterError> {
        Ok(serde_json::from_str(json)?)
    }

    /// Creates a json string from record.
    fn to_json_str(&self) -> Result<String, ParameterError> {
        Ok(serde_json::to_string(&self)?)
    }
}

#[pyclass(name = "PureRecord")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyPureRecord {
    #[pyo3(get)]
    identifier: Identifier,
    #[pyo3(get)]
    molarweight: f64,
    model_record: Value,
}

impl<M> TryFrom<&PureRecord<M>> for PyPureRecord
where
    M: Serialize + Clone,
{
    type Error = ParameterError;
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
    type Error = ParameterError;
    fn try_into(self) -> Result<PureRecord<M>, ParameterError> {
        Ok(serde_json::from_value(serde_json::to_value(self)?)?)
    }
}

impl PyPureRecord {
    fn from_json(
        substances: &[PyBackedStr],
        file: &PyBackedStr,
        identifier_option: IdentifierOption,
    ) -> Result<Vec<Self>, ParameterError> {
        // create list of substances
        let mut queried: HashSet<_> = substances.iter().map(|s| s.to_string()).collect();
        // raise error on duplicate detection
        if queried.len() != substances.len() {
            return Err(ParameterError::IncompatibleParameters(
                "A substance was defined more than once.".to_string(),
            ));
        }

        let reader = BufReader::new(File::open::<&str>(file.as_ref())?);
        let file_records: Vec<Self> = serde_json::from_reader(reader)?;
        let mut records: HashMap<_, _> = HashMap::with_capacity(substances.len());

        // build map, draining list of queried substances in the process
        for record in file_records {
            if let Some(id) = record.identifier.as_string(identifier_option) {
                queried.take(&id).map(|id| records.insert(id, record));
            }
            // all parameters parsed
            if queried.is_empty() {
                break;
            }
        }

        // report missing parameters
        if !queried.is_empty() {
            return Err(ParameterError::ComponentsNotFound(format!("{:?}", queried)));
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
        identifier: Identifier,
        molarweight: f64,
        parameters: Option<&Bound<'_, PyDict>>,
    ) -> Result<Self, ParameterError> {
        if parameters.is_none() {
            return Err(ParameterError::Error(
                "No model parameters provided for PureRecord".to_string(),
            ));
        }
        let model_record =
            depythonize(parameters.unwrap()).map_err(|e| ParameterError::Error(e.to_string()))?;
        Ok(Self {
            identifier,
            molarweight,
            model_record,
        })
    }

    #[getter]
    fn get_model_record<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyDict>, ParameterError> {
        pythonize(py, &self.model_record)
            .map_err(|e| ParameterError::Error(e.to_string()))
            .and_then(|d| {
                d.downcast_into::<PyDict>()
                    .map_err(|e| ParameterError::Error(e.to_string()))
            })
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyDict>, ParameterError> {
        pythonize(py, &self)
            .map_err(|e| ParameterError::Error(e.to_string()))
            .and_then(|d| {
                d.downcast_into::<PyDict>()
                    .map_err(|e| ParameterError::Error(e.to_string()))
            })
    }

    #[staticmethod]
    pub fn from_json_str(s: &str) -> Result<Self, ParameterError> {
        Ok(serde_json::from_str(s)?)
    }

    pub fn to_json_str(&self) -> Result<String, ParameterError> {
        Ok(serde_json::to_string(&self)?)
    }

    fn __repr__(&self) -> Result<String, ParameterError> {
        let params: Result<String, ParameterError> = Python::with_gil(|py| {
            Ok(self
                .get_model_record(py)?
                .iter()
                .map(|(p, v)| format!(", {p}={v}"))
                .collect::<Vec<_>>()
                .join(""))
        });
        Ok(format!(
            "PureRecord(identifier={}, molarweight={}{})",
            self.identifier, self.molarweight, params?
        ))
    }
}

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
    type Error = ParameterError;
    fn try_into(self) -> Result<SegmentRecord<M>, ParameterError> {
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
            return Err(ParameterError::Error(
                "No model parameters provided for SegmentRecord".to_string(),
            )
            .into());
        }
        let model_record =
            depythonize(parameters.unwrap()).map_err(|e| ParameterError::Error(e.to_string()))?;
        Ok(Self {
            identifier,
            molarweight,
            model_record,
        })
    }

    #[getter]
    fn get_model_record<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyDict>, ParameterError> {
        pythonize(py, &self.model_record)
            .map_err(|e| ParameterError::Error(e.to_string()))
            .and_then(|d| {
                d.downcast_into::<PyDict>()
                    .map_err(|e| ParameterError::Error(e.to_string()))
            })
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyDict>, ParameterError> {
        pythonize(py, &self)
            .map_err(|e| ParameterError::Error(e.to_string()))
            .and_then(|d| {
                d.downcast_into::<PyDict>()
                    .map_err(|e| ParameterError::Error(e.to_string()))
            })
    }

    #[staticmethod]
    pub fn from_json_str(s: &str) -> Result<Self, ParameterError> {
        Ok(serde_json::from_str(s)?)
    }

    pub fn to_json_str(&self) -> Result<String, ParameterError> {
        Ok(serde_json::to_string(&self)?)
    }

    fn __repr__(&self) -> Result<String, ParameterError> {
        let params: Result<String, ParameterError> = Python::with_gil(|py| {
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
    pub fn from_json(path: &str) -> Result<Vec<Self>, ParameterError> {
        Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
    }
}

#[pyclass(name = "Parameters")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyParameters {
    #[pyo3(get)]
    pub pure_records: Vec<PyPureRecord>,
    pub binary_records: Vec<([usize; 2], Value)>,
}

impl PyParameters {
    pub fn try_convert<P: Parameter>(self) -> Result<P, ParameterError> {
        let n = self.pure_records.len();
        let pure_records = self
            .pure_records
            .into_iter()
            .map(|r| r.try_into())
            .collect::<Result<_, _>>()?;
        let binary_records = if self.binary_records.is_empty() {
            None
        } else {
            let mut br = Array2::default((n, n));
            for ([i, j], r) in self.binary_records {
                let r: P::Binary = serde_json::from_value(r)?;
                br[[i, j]] = r.clone();
                br[[j, i]] = r;
            }
            Some(br)
        };
        P::from_records(pure_records, binary_records)
    }
}

#[pymethods]
impl PyParameters {
    /// Creates parameters from records.
    ///
    /// Parameters
    /// ----------
    /// pure_records : List[PureRecord]
    ///     A list of pure component parameters.
    /// binary_records : List[BinaryRecord], optional, defaults to []
    ///     A list containing records for binary interactions.
    /// identifier_option : IdentifierOption, optional, defaults to IdentifierOption.Name
    ///     Identifier that is used to search binary records.
    #[staticmethod]
    #[pyo3(
                signature = (pure_records, binary_records=vec![], identifier_option=IdentifierOption::Name),
                text_signature = "(pure_records, binary_records=[], identifier_option=IdentifierOption.Name)"
            )]
    fn from_records(
        pure_records: Vec<PyPureRecord>,
        binary_records: Vec<PyBinaryRecord>,
        identifier_option: IdentifierOption,
    ) -> Result<Self, ParameterError> {
        // Build Hashmap (id, id) -> BinaryRecord
        let binary_map: HashMap<_, _> = {
            binary_records
                .iter()
                .filter_map(|br| {
                    let id1 = br.id1.as_string(identifier_option);
                    let id2 = br.id2.as_string(identifier_option);
                    id1.and_then(|id1| id2.map(|id2| ((id1, id2), br.model_record.clone())))
                })
                .collect()
        };

        // look up pure records in Hashmap
        let binary_records = pure_records
            .iter()
            .enumerate()
            .array_combinations()
            .filter_map(|[(i1, p1), (i2, p2)]| {
                let id1 = p1
                    .identifier
                    .as_string(identifier_option)
                    .unwrap_or_else(|| {
                        panic!(
                            "No {} for pure record {} ({}).",
                            identifier_option, i1, p1.identifier
                        )
                    });
                let id2 = p2
                    .identifier
                    .as_string(identifier_option)
                    .unwrap_or_else(|| {
                        panic!(
                            "No {} for pure record {} ({}).",
                            identifier_option, i2, p2.identifier
                        )
                    });
                binary_map
                    .get(&(id1.clone(), id2.clone()))
                    .or_else(|| binary_map.get(&(id2, id1)))
                    .map(|br| ([i1, i2], br.clone()))
            })
            .collect();

        Ok(Self {
            pure_records,
            binary_records,
        })
    }

    /// Creates parameters for a pure component from a pure record.
    ///
    /// Parameters
    /// ----------
    /// pure_record : PureRecord
    ///     The pure component parameters.
    #[staticmethod]
    fn new_pure(pure_record: PyPureRecord) -> Self {
        Self {
            pure_records: vec![pure_record],
            binary_records: vec![],
        }
    }

    /// Creates parameters for a binary system from pure records and an optional
    /// binary interaction parameter or binary interaction parameter record.
    ///
    /// Parameters
    /// ----------
    /// pure_records : [PureRecord]
    ///     A list of pure component parameters.
    /// binary_record : float or BinaryRecord, optional
    ///     The binary interaction parameter or binary interaction record.
    #[staticmethod]
    #[pyo3(text_signature = "(pure_records, binary_record=None)", signature = (pure_records, binary_record=None))]
    fn new_binary(pure_records: Vec<PyPureRecord>, binary_record: Option<PyBinaryRecord>) -> Self {
        let binary_records = binary_record
            .into_iter()
            .map(|r| ([0, 1], r.model_record))
            .collect();
        Self {
            pure_records,
            binary_records,
        }
    }

    /// Creates parameters from json files.
    ///
    /// Parameters
    /// ----------
    /// substances : List[str]
    ///     The substances to search.
    /// pure_path : str
    ///     Path to file containing pure substance parameters.
    /// binary_path : str, optional
    ///     Path to file containing binary substance parameters.
    /// identifier_option : IdentifierOption, optional, defaults to IdentifierOption.Name
    ///     Identifier that is used to search substance.
    #[staticmethod]
    #[pyo3(
        signature = (substances, pure_path, binary_path=None, identifier_option=IdentifierOption::Name),
        text_signature = "(substances, pure_path, binary_path=None, identifier_option)"
    )]
    fn from_json(
        substances: Vec<PyBackedStr>,
        pure_path: PyBackedStr,
        binary_path: Option<PyBackedStr>,
        identifier_option: IdentifierOption,
    ) -> Result<Self, ParameterError> {
        Self::from_multiple_json(
            vec![(substances, pure_path)],
            binary_path,
            identifier_option,
        )
    }

    /// Creates parameters from json files.
    ///
    /// Parameters
    /// ----------
    /// input : List[Tuple[List[str], str]]
    ///     The substances to search and their respective parameter files.
    ///     E.g. [(["methane", "propane"], "parameters/alkanes.json"), (["methanol"], "parameters/alcohols.json")]
    /// binary_path : str, optional
    ///     Path to file containing binary substance parameters.
    /// identifier_option : IdentifierOption, optional, defaults to IdentifierOption.Name
    ///     Identifier that is used to search substance.
    #[staticmethod]
    #[pyo3(
        signature = (input, binary_path=None, identifier_option=IdentifierOption::Name),
        text_signature = "(input, binary_path=None, identifier_option)"
    )]
    fn from_multiple_json(
        input: Vec<(Vec<PyBackedStr>, PyBackedStr)>,
        binary_path: Option<PyBackedStr>,
        identifier_option: IdentifierOption,
    ) -> Result<Self, ParameterError> {
        // total number of substances queried
        let nsubstances = input.iter().map(|(substances, _)| substances.len()).sum();

        // queried substances with removed duplicates
        let queried: HashSet<_> = input
            .iter()
            .flat_map(|(substances, _)| substances)
            .collect();

        // check if there are duplicates
        if queried.len() != nsubstances {
            return Err(ParameterError::IncompatibleParameters(
                "A substance was defined more than once.".to_string(),
            ));
        }

        let mut pure_records = Vec::with_capacity(nsubstances);

        // collect parameters from files into single map
        for (substances, file) in input {
            pure_records.extend(PyPureRecord::from_json(
                &substances,
                &file,
                identifier_option,
            )?);
        }

        let binary_records = if let Some(path) = binary_path {
            let reader = BufReader::new(File::open::<&str>(path.as_ref())?);
            serde_json::from_reader(reader)?
        } else {
            Vec::new()
        };

        Self::from_records(pure_records, binary_records, identifier_option)
    }

    // /// Generates JSON-formatted string for pure and binary records (if initialized).
    // ///
    // /// Parameters
    // /// ----------
    // /// pretty : bool
    // ///     Whether to use pretty (True) or dense (False) formatting. Defaults to True.
    // ///
    // /// Returns
    // /// -------
    // /// str : The JSON-formatted string.
    // #[pyo3(
    //     signature = (pretty=true),
    //     text_signature = "(pretty=True)"
    // )]
    // fn to_json_str(&self, pretty: bool) -> Result<(String, Option<String>), ParameterError> {
    //     Self::to_json_str(&self.0, pretty)
    // }

    #[getter]
    fn get_binary_records<'py>(
        &self,
        py: Python<'py>,
    ) -> Result<Bound<'py, PyList>, ParameterError> {
        pythonize(py, &self.binary_records)
            .map_err(|e| ParameterError::Error(e.to_string()))
            .and_then(|d| {
                d.downcast_into::<PyList>()
                    .map_err(|e| ParameterError::Error(e.to_string()))
            })
    }
}

#[pyclass(name = "GcParameters", get_all)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyGcParameters {
    chemical_records: Vec<ChemicalRecord>,
    segment_records: Vec<PySegmentRecord>,
    binary_segment_records: Option<Vec<BinarySegmentRecord>>,
}

impl PyGcParameters {
    pub fn try_convert_homosegmented<P: Parameter>(self) -> Result<P, ParameterError>
    where
        P::Pure: FromSegments<usize>,
        P::Binary: FromSegmentsBinary<usize>,
    {
        let segment_records = self
            .segment_records
            .into_iter()
            .map(|r| r.try_into())
            .collect::<Result<_, _>>()?;
        P::from_segments(
            self.chemical_records,
            segment_records,
            self.binary_segment_records,
        )
    }

    pub fn try_convert_heterosegmented<P: ParameterHetero>(self) -> Result<P, ParameterError>
    where
        P::Chemical: From<ChemicalRecord>,
    {
        let segment_records = self
            .segment_records
            .into_iter()
            .map(|r| r.try_into())
            .collect::<Result<_, _>>()?;
        P::from_segments(
            self.chemical_records,
            segment_records,
            self.binary_segment_records,
        )
    }
}

#[pymethods]
impl PyGcParameters {
    /// Creates parameters from segment records.
    ///
    /// Parameters
    /// ----------
    /// chemical_records : [ChemicalRecord]
    ///     A list of pure component chemical records.
    /// segment_records : [SegmentRecord]
    ///     A list of records containing the parameters of
    ///     all individual segments.
    /// binary_segment_records : [BinarySegmentRecord], optional
    ///     A list of binary segment-segment parameters.
    #[staticmethod]
    #[pyo3(text_signature = "(chemical_records, segment_records, binary_segment_records=None)",
    signature = (chemical_records, segment_records, binary_segment_records=None))]
    fn from_segments(
        chemical_records: Vec<ChemicalRecord>,
        segment_records: Vec<PySegmentRecord>,
        binary_segment_records: Option<Vec<BinarySegmentRecord>>,
    ) -> Self {
        Self {
            chemical_records,
            segment_records,
            binary_segment_records,
        }
    }

    /// Creates parameters using segments from json file.
    ///
    /// Parameters
    /// ----------
    /// substances : List[str]
    ///     The substances to search.
    /// pure_path : str
    ///     Path to file containing pure substance parameters.
    /// segments_path : str
    ///     Path to file containing segment parameters.
    /// binary_path : str, optional
    ///     Path to file containing binary segment-segment parameters.
    /// identifier_option : IdentifierOption, optional, defaults to IdentifierOption.Name
    ///     Identifier that is used to search substance.
    #[staticmethod]
    #[pyo3(
        signature = (substances, pure_path, segments_path, binary_path=None, identifier_option=IdentifierOption::Name),
        text_signature = "(substances, pure_path, segments_path, binary_path=None, identifier_option=IdentiferOption.Name)"
    )]
    fn from_json_segments(
        substances: Vec<PyBackedStr>,
        pure_path: PyBackedStr,
        segments_path: PyBackedStr,
        binary_path: Option<PyBackedStr>,
        identifier_option: IdentifierOption,
    ) -> Result<Self, ParameterError> {
        let queried: IndexSet<String> = substances
            .iter()
            .map(|identifier| identifier.to_string())
            .collect();

        let reader = BufReader::new(File::open(&pure_path as &str)?);
        let chemical_records: Vec<ChemicalRecord> = serde_json::from_reader(reader)?;
        let mut record_map: IndexMap<_, _> = chemical_records
            .into_iter()
            .filter_map(|record| {
                record
                    .identifier
                    .as_string(identifier_option)
                    .map(|i| (i, record))
            })
            .collect();

        // Compare queried components and available components
        let available: IndexSet<String> = record_map
            .keys()
            .map(|identifier| identifier.to_string())
            .collect();
        if !queried.is_subset(&available) {
            let missing: Vec<String> = queried.difference(&available).cloned().collect();
            return Err(ParameterError::ComponentsNotFound(format!("{:?}", missing)));
        };

        // Collect all pure records that were queried
        let chemical_records: Vec<_> = queried
            .iter()
            .filter_map(|identifier| record_map.shift_remove(&identifier.clone()))
            .collect();

        // Read segment records
        let segment_records: Vec<PySegmentRecord> =
            PySegmentRecord::from_json(&segments_path as &str)?;

        // Read binary records
        let binary_records = binary_path
            .as_ref()
            .map(|file_binary| {
                let reader = BufReader::new(File::open(file_binary as &str)?);
                let binary_records: Result<Vec<BinarySegmentRecord>, ParameterError> =
                    Ok(serde_json::from_reader(reader)?);
                binary_records
            })
            .transpose()?;

        Ok(Self::from_segments(
            chemical_records,
            segment_records,
            binary_records,
        ))
    }

    /// Creates parameters from SMILES and segment records.
    ///
    /// Requires an installation of rdkit.
    ///
    /// Parameters
    /// ----------
    /// identifier : [str | Identifier]
    ///     A list of SMILES codes or [Identifier] objects.
    /// smarts_records : [SmartsRecord]
    ///     A list of records containing the SMARTS codes used
    ///     to fragment the molecule.
    /// segment_records : [SegmentRecord]
    ///     A list of records containing the parameters of
    ///     all individual segments.
    /// binary_segment_records : [BinarySegmentRecord], optional
    ///     A list of binary segment-segment parameters.
    #[staticmethod]
    #[pyo3(
        text_signature = "(identifier, smarts_records, segment_records, binary_segment_records=None)"
    )]
    #[pyo3(signature = (identifier, smarts_records, segment_records, binary_segment_records=None))]
    fn from_smiles(
        identifier: Vec<Bound<'_, PyAny>>,
        smarts_records: Vec<PySmartsRecord>,
        segment_records: Vec<PySegmentRecord>,
        binary_segment_records: Option<Vec<BinarySegmentRecord>>,
    ) -> PyResult<Self> {
        let chemical_records: Vec<_> = identifier
            .into_iter()
            .map(|i| ChemicalRecord::from_smiles(&i, smarts_records.clone()))
            .collect::<PyResult<_>>()?;
        Ok(Self::from_segments(
            chemical_records,
            segment_records,
            binary_segment_records,
        ))
    }

    /// Creates parameters from SMILES using segments from json file.
    ///
    /// Requires an installation of rdkit.
    ///
    /// Parameters
    /// ----------
    /// identifier : list[str | Identifier]
    ///     A list of SMILES codes or [Identifier] objects.
    /// smarts_path : str
    ///     Path to file containing SMARTS records.
    /// segments_path : str
    ///     Path to file containing segment parameters.
    /// binary_path : str, optional
    ///     Path to file containing binary segment-segment parameters.
    #[staticmethod]
    #[pyo3(
        signature = (identifier, smarts_path, segments_path, binary_path=None),
        text_signature = "(identifier, smarts_path, segments_path, binary_path=None)"
    )]
    fn from_json_smiles(
        identifier: Vec<Bound<'_, PyAny>>,
        smarts_path: PyBackedStr,
        segments_path: PyBackedStr,
        binary_path: Option<PyBackedStr>,
    ) -> PyResult<Self> {
        let smarts_records = PySmartsRecord::from_json(&smarts_path)?;
        let segment_records = PySegmentRecord::from_json(&segments_path)?;
        let binary_segment_records = binary_path
            .as_ref()
            .map(|p| BinarySegmentRecord::from_json(p as &str))
            .transpose()?;
        Self::from_smiles(
            identifier,
            smarts_records,
            segment_records,
            binary_segment_records,
        )
    }
}
