use crate::parameter::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pythonize::{depythonize, pythonize};
use serde::{Deserialize, Serialize};
mod fragmentation;
pub use fragmentation::PySmartsRecord;
use serde_json::Value;
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

#[macro_export]
macro_rules! impl_parameter {
        ($parameter:ty, $py_parameter:ty) => {
        use pyo3::pybacked::*;

        #[pymethods]
        impl $py_parameter {
            /// Creates parameters from records.
            ///
            /// Parameters
            /// ----------
            /// pure_records : [PureRecord]
            ///     A list of pure component parameters.
            /// binary_records : numpy.ndarray[float] or List[BinaryRecord], optional
            ///     A matrix of binary interaction parameters or a list
            ///     containing records for binary interactions.
            /// identifier_option : IdentifierOption, optional, defaults to IdentifierOption.Name
            ///     Identifier that is used to search binary records.
            #[staticmethod]
            #[pyo3(
                signature = (pure_records, binary_records=None, identifier_option=IdentifierOption::Name),
                text_signature = "(pure_records, binary_records=None, identifier_option=None)"
            )]
            fn from_records(
                pure_records: Vec<PyPureRecord>,
                binary_records: Option<&Bound<'_, PyAny>>,
                identifier_option: IdentifierOption,
            ) -> PyResult<Self> {
                let prs = pure_records.into_iter().map(|pr| pr.try_into()).collect::<Result<Vec<_>, _>>()?;
                let binary_records = binary_records
                    .map(|binary_records| {
                        if let Ok(br) = binary_records.extract::<PyReadonlyArray2<f64>>() {
                            Ok(Some(br.as_array().mapv(|r| r.try_into().unwrap())))
                        } else if let Ok(br) = binary_records.extract::<Vec<PyBinaryRecord>>() {
                            let brs: Vec<_> = br.into_iter().map(|br| br.try_into()).collect::<Result<_,_>>()?;
                            Ok(<$parameter>::binary_matrix_from_records(
                                &prs,
                                &brs,
                                identifier_option,
                            ))
                        } else {
                            Err(PyErr::new::<PyTypeError, _>(format!(
                                "Could not parse binary input!"
                            )))
                        }
                    })
                    .transpose()?
                    .flatten();
                Ok(Self(Arc::new(Parameter::from_records(prs, binary_records)?)))
            }

            /// Creates parameters for a pure component from a pure record.
            ///
            /// Parameters
            /// ----------
            /// pure_record : PureRecord
            ///     The pure component parameters.
            #[staticmethod]
            fn new_pure(pure_record: PyPureRecord) -> PyResult<Self> {
                Ok(Self(Arc::new(<$parameter>::new_pure(pure_record.try_into()?)?)))
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
            fn new_binary(
                pure_records: Vec<PyPureRecord>,
                binary_record: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<Self> {
                let prs = pure_records.into_iter().map(|pr| pr.try_into()).collect::<Result<_,_>>()?;
                let br = binary_record
                    .map(|br| {
                        if let Ok(r) = br.extract::<f64>() {
                            Ok(r.try_into()?)
                        } else if let Ok(r) = br.extract::<PyBinaryRecord>() {
                            let r: BinaryRecord<<$parameter as Parameter>::Binary> = r.try_into()?;
                            Ok(r.model_record)
                        } else {
                            Err(PyErr::new::<PyTypeError, _>(format!(
                                "Could not parse binary input!"
                            )))
                        }
                    })
                    .transpose()?;
                Ok(Self(Arc::new(<$parameter>::new_binary(prs, br)?)))
            }

            // /// Creates parameters from model records with default values for the molar weight,
            // /// identifiers, and binary interaction parameters.
            // ///
            // /// Parameters
            // /// ----------
            // /// model_records : [ModelRecord]
            // ///     A list of model parameters.
            // #[staticmethod]
            // fn from_model_records(model_records: Vec<$py_model_record>) -> PyResult<Self> {
            //     let mrs = model_records.into_iter().map(|mr| mr.0).collect();
            //     Ok(Self(Arc::new(<$parameter>::from_model_records(mrs)?)))
            // }

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
                pure_path: String,
                binary_path: Option<String>,
                identifier_option: IdentifierOption,
            ) -> Result<Self, ParameterError> {
                let substances = substances.iter().map(|s| &**s).collect();
                Ok(Self(Arc::new(<$parameter>::from_json(
                    substances,
                    pure_path,
                    binary_path,
                    identifier_option,
                )?)))
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
                identifier_option: Option<IdentifierOption>,
            ) -> Result<Self, ParameterError> {
                let input: Vec<(Vec<&str>, &str)> = input.iter().map(|(c, f)| (c.iter().map(|c| &**c).collect(), &**f)).collect();
                Ok(Self(Arc::new(<$parameter>::from_multiple_json(
                    &input,
                    binary_path.as_deref(),
                    identifier_option.unwrap_or(IdentifierOption::Name),
                )?)))
            }

            /// Generates JSON-formatted string for pure and binary records (if initialized).
            ///
            /// Parameters
            /// ----------
            /// pretty : bool
            ///     Whether to use pretty (true) or dense (false) formatting. Defaults to true.
            ///
            /// Returns
            /// -------
            /// str : The JSON-formatted string.
            #[pyo3(
                signature = (pretty=true),
                text_signature = "(pretty=true)"
            )]
            fn to_json_str(&self, pretty: bool) -> Result<(String, Option<String>), ParameterError> {
                <$parameter>::to_json_str(&self.0, pretty)
            }

            #[getter]
            fn get_pure_records(&self) -> PyResult<Vec<PyPureRecord>> {
                let pure_records = self.0.records().0;
                Ok(pure_records
                    .iter()
                    .map(|r| PyPureRecord::try_from(r))
                    .collect::<Result<_,ParameterError>>()?)
            }

            // #[getter]
            // fn get_binary_records<'py>(&self, py: Python<'py>) -> Result<Option<Bound<'py, PyAny>>, ParameterError> {
            //     let binary_records = self.0.records().1;
            //     let Some(br) = binary_records else {
            //         return Ok(None)
            //     };

            //     if br.shape()[0] == 0 {
            //         return Ok(None)
            //     }
            //     // todo: map f64 or binary records
            // }
        }
    };
}

#[macro_export]
macro_rules! impl_parameter_from_segments {
    ($parameter:ty, $py_parameter:ty) => {
        use pyo3::pybacked::*;

        #[pymethods]
        impl $py_parameter {
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
            ) -> PyResult<Self> {
                Ok(Self(Arc::new(<$parameter>::from_segments(
                    chemical_records,
                    segment_records.into_iter().map(|sr| sr.try_into()).collect::<Result<_, ParameterError>>()?,
                    binary_segment_records,
                )?)))
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
                text_signature = "(substances, pure_path, segments_path, binary_path=None, identifier_option)"
            )]
            fn from_json_segments(
                substances: Vec<PyBackedStr>,
                pure_path: String,
                segments_path: String,
                binary_path: Option<String>,
                identifier_option: IdentifierOption,
            ) -> PyResult<Self> {
                let substances: Vec<_> = substances.iter().map(|s| &**s).collect();
                Ok(Self(Arc::new(<$parameter>::from_json_segments(
                    &substances,
                    pure_path,
                    segments_path,
                    binary_path,
                    identifier_option,
                )?)))
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
            #[pyo3(text_signature = "(identifier, smarts_records, segment_records, binary_segment_records=None)")]
            #[pyo3(signature = (identifier, smarts_records, segment_records, binary_segment_records=None))]
            fn from_smiles(
                identifier: Vec<Bound<'_,PyAny>>,
                smarts_records: Vec<PySmartsRecord>,
                segment_records: Vec<PySegmentRecord>,
                binary_segment_records: Option<Vec<BinarySegmentRecord>>,
            ) -> PyResult<Self> {
                let chemical_records: Vec<_> = identifier
                    .into_iter()
                    .map(|i| ChemicalRecord::from_smiles(&i, smarts_records.clone()))
                    .collect::<PyResult<_>>()?;
                Self::from_segments(chemical_records, segment_records, binary_segment_records)
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
                identifier: Vec<Bound<'_,PyAny>>,
                smarts_path: String,
                segments_path: String,
                binary_path: Option<String>,
            ) -> PyResult<Self> {

                let smarts_records = PySmartsRecord::from_json(&smarts_path)?;
                let segment_records = PySegmentRecord::from_json(&segments_path)?;
                let binary_segment_records = binary_path.map(|p| BinarySegmentRecord::from_json(&p)).transpose()?;
                Self::from_smiles(
                    identifier,
                    smarts_records,
                    segment_records,
                    binary_segment_records,
                )
            }
        }
    };
}
