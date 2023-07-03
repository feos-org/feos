use crate::impl_json_handling;
use crate::parameter::{BinaryRecord, ChemicalRecord, Identifier, ParameterError};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

impl From<ParameterError> for PyErr {
    fn from(e: ParameterError) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }
}

/// Create an identifier for a pure substance.
///
/// Parameters
/// ----------
/// cas : str, optional
///     CAS number.
/// name : str, optional
///     name
/// iupac_name : str, optional
///     Iupac name.
/// smiles : str, optional
///     Canonical SMILES
/// inchi : str, optional
///     Inchi number
/// formula : str, optional
///     Molecular formula.
///
/// Returns
/// -------
/// Identifier
#[pyclass(name = "Identifier")]
#[derive(Clone)]
#[pyo3(
    text_signature = "(cas=None, name=None, iupac_name=None, smiles=None, inchi=None, formula=None)"
)]
pub struct PyIdentifier(pub Identifier);

#[pymethods]
impl PyIdentifier {
    #[new]
    fn new(
        cas: Option<&str>,
        name: Option<&str>,
        iupac_name: Option<&str>,
        smiles: Option<&str>,
        inchi: Option<&str>,
        formula: Option<&str>,
    ) -> Self {
        Self(Identifier::new(
            cas, name, iupac_name, smiles, inchi, formula,
        ))
    }

    #[getter]
    fn get_cas(&self) -> Option<String> {
        self.0.cas.clone()
    }

    #[setter]
    fn set_cas(&mut self, cas: &str) {
        self.0.cas = Some(cas.to_string());
    }

    #[getter]
    fn get_name(&self) -> Option<String> {
        self.0.name.clone()
    }

    #[setter]
    fn set_name(&mut self, name: &str) {
        self.0.name = Some(name.to_string());
    }

    #[getter]
    fn get_iupac_name(&self) -> Option<String> {
        self.0.iupac_name.clone()
    }

    #[setter]
    fn set_iupac_name(&mut self, iupac_name: &str) {
        self.0.iupac_name = Some(iupac_name.to_string());
    }

    #[getter]
    fn get_smiles(&self) -> Option<String> {
        self.0.smiles.clone()
    }

    #[setter]
    fn set_smiles(&mut self, smiles: &str) {
        self.0.smiles = Some(smiles.to_string());
    }

    #[getter]
    fn get_inchi(&self) -> Option<String> {
        self.0.inchi.clone()
    }

    #[setter]
    fn set_inchi(&mut self, inchi: &str) {
        self.0.inchi = Some(inchi.to_string());
    }

    #[getter]
    fn get_formula(&self) -> Option<String> {
        self.0.formula.clone()
    }

    #[setter]
    fn set_formula(&mut self, formula: &str) {
        self.0.formula = Some(formula.to_string());
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PyIdentifier);

/// Create a chemical record for a pure substance.
///
/// Parameters
/// ----------
/// identifier : Identifier
///     The identifier of the pure component.
/// segments : [str]
///     List of segments, that the molecule consists of.
/// bonds : [[int, int]], optional
///     List of bonds with the indices starting at 0 and
///     referring to the list of segments passed as first
///     argument. If no bonds are specified, the molecule
///     is assumed to be linear.
///
/// Returns
/// -------
/// ChemicalRecord
#[pyclass(name = "ChemicalRecord")]
#[derive(Clone)]
#[pyo3(text_signature = "(identifier, segments, bonds=None)")]
pub struct PyChemicalRecord(pub ChemicalRecord);

#[pymethods]
impl PyChemicalRecord {
    #[new]
    fn new(
        identifier: PyIdentifier,
        segments: Vec<String>,
        bonds: Option<Vec<[usize; 2]>>,
    ) -> Self {
        Self(ChemicalRecord::new(identifier.0, segments, bonds))
    }

    #[getter]
    fn get_identifier(&self) -> PyIdentifier {
        PyIdentifier(self.0.identifier.clone())
    }

    #[getter]
    fn get_segments(&self) -> Vec<String> {
        self.0.segments.clone()
    }

    #[getter]
    fn get_bonds(&self) -> Vec<[usize; 2]> {
        self.0.bonds.clone()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PyChemicalRecord);

#[macro_export]
macro_rules! impl_binary_record {
    () => {
        #[pyclass(name = "BinaryModelRecord")]
        #[derive(Clone)]
        struct PyBinaryModelRecord(f64);
        impl_binary_record!(f64, PyBinaryModelRecord);
    };
    ($model_record:ident, $py_model_record:ident) => {
        /// Create a record for a binary interaction parameter.
        ///
        /// Parameters
        /// ----------
        /// id1 : Identifier
        ///     The identifier of the first component.
        /// id2 : Identifier
        ///     The identifier of the second component.
        /// model_record : float or BinaryModelRecord
        ///     The binary interaction parameter.
        ///
        /// Returns
        /// -------
        /// BinaryRecord
        #[pyclass(name = "BinaryRecord")]
        #[derive(Clone)]
        pub struct PyBinaryRecord(pub BinaryRecord<Identifier, $model_record>);

        #[pymethods]
        impl PyBinaryRecord {
            #[new]
            fn new(id1: PyIdentifier, id2: PyIdentifier, model_record: &PyAny) -> PyResult<Self> {
                if let Ok(mr) = model_record.extract::<f64>() {
                    Ok(Self(BinaryRecord::new(id1.0, id2.0, mr.try_into()?)))
                } else if let Ok(mr) = model_record.extract::<$py_model_record>() {
                    Ok(Self(BinaryRecord::new(id1.0, id2.0, mr.0)))
                } else {
                    Err(PyErr::new::<PyTypeError, _>(format!(
                        "Could not parse model_record input!"
                    )))
                }
            }

            /// Read a list of `BinaryRecord`s from a JSON file.
            ///
            /// Parameters
            /// ----------
            /// path : str
            ///     Path to file containing the binary records.
            ///
            /// Returns
            /// -------
            /// [BinaryRecord]
            #[staticmethod]
            #[pyo3(text_signature = "(path)")]
            fn from_json(path: &str) -> Result<Vec<Self>, ParameterError> {
                Ok(BinaryRecord::from_json(path)?
                    .into_iter()
                    .map(Self)
                    .collect())
            }

            #[getter]
            fn get_id1(&self) -> PyIdentifier {
                PyIdentifier(self.0.id1.clone())
            }

            #[setter]
            fn set_id1(&mut self, id1: PyIdentifier) {
                self.0.id1 = id1.0;
            }

            #[getter]
            fn get_id2(&self) -> PyIdentifier {
                PyIdentifier(self.0.id2.clone())
            }

            #[setter]
            fn set_id2(&mut self, id2: PyIdentifier) {
                self.0.id2 = id2.0;
            }

            #[getter]
            fn get_model_record(&self, py: Python) -> PyObject {
                if let Ok(mr) = f64::try_from(self.0.model_record.clone()) {
                    mr.to_object(py)
                } else {
                    $py_model_record(self.0.model_record.clone()).into_py(py)
                }
            }

            #[setter]
            fn set_model_record(&mut self, model_record: &PyAny) -> PyResult<()> {
                if let Ok(mr) = model_record.extract::<f64>() {
                    self.0.model_record = mr.try_into()?;
                } else if let Ok(mr) = model_record.extract::<$py_model_record>() {
                    self.0.model_record = mr.0;
                } else {
                    return Err(PyErr::new::<PyTypeError, _>(format!(
                        "Could not parse model_record input!"
                    )));
                }
                Ok(())
            }

            fn __repr__(&self) -> PyResult<String> {
                Ok(self.0.to_string())
            }
        }

        impl_json_handling!(PyBinaryRecord);
    };
}

/// Create a record for a binary segment interaction parameter.
///
/// Parameters
/// ----------
/// id1 : str
///     The identifier of the first segment.
/// id2 : str
///     The identifier of the second segment.
/// model_record : float
///     The binary segment interaction parameter.
///
/// Returns
/// -------
/// BinarySegmentRecord
#[pyclass(name = "BinarySegmentRecord")]
#[derive(Clone)]
pub struct PyBinarySegmentRecord(pub BinaryRecord<String, f64>);

#[pymethods]
impl PyBinarySegmentRecord {
    #[new]
    fn new(id1: String, id2: String, model_record: f64) -> PyResult<Self> {
        Ok(Self(BinaryRecord::new(id1, id2, model_record)))
    }

    /// Read a list of `BinarySegmentRecord`s from a JSON file.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to file containing the binary records.
    ///
    /// Returns
    /// -------
    /// [BinarySegmentRecord]
    #[staticmethod]
    #[pyo3(text_signature = "(path)")]
    fn from_json(path: &str) -> Result<Vec<Self>, ParameterError> {
        Ok(BinaryRecord::from_json(path)?
            .into_iter()
            .map(Self)
            .collect())
    }

    #[getter]
    fn get_id1(&self) -> String {
        self.0.id1.clone()
    }

    #[setter]
    fn set_id1(&mut self, id1: String) {
        self.0.id1 = id1;
    }

    #[getter]
    fn get_id2(&self) -> String {
        self.0.id2.clone()
    }

    #[setter]
    fn set_id2(&mut self, id2: String) {
        self.0.id2 = id2;
    }

    #[getter]
    fn get_model_record(&self) -> f64 {
        self.0.model_record
    }

    #[setter]
    fn set_model_record(&mut self, model_record: f64) {
        self.0.model_record = model_record;
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PyBinarySegmentRecord);

#[macro_export]
macro_rules! impl_pure_record {
    ($model_record:ident, $py_model_record:ident, $ideal_gas_record:ident, $py_ideal_gas_record:ident) => {
        /// All information required to characterize a pure component.
        ///
        /// Parameters
        /// ----------
        /// identifier : Identifier
        ///     The identifier of the pure component.
        /// molarweight : float
        ///     The molar weight (in g/mol) of the pure component.
        /// model_record : ModelRecord
        ///     The pure component model parameters.
        /// ideal_gas_record: IdealGasRecord, optional
        ///     The pure component parameters for the ideal gas model.
        ///
        /// Returns
        /// -------
        /// PureRecord
        #[pyclass(name = "PureRecord")]
        #[pyo3(text_signature = "(identifier, molarweight, model_record, ideal_gas_record=None)")]
        #[derive(Clone)]
        pub struct PyPureRecord(pub PureRecord<$model_record, $ideal_gas_record>);

        #[pymethods]
        impl PyPureRecord {
            #[new]
            fn new(
                identifier: PyIdentifier,
                molarweight: f64,
                model_record: $py_model_record,
                ideal_gas_record: Option<$py_ideal_gas_record>,
            ) -> PyResult<Self> {
                Ok(Self(PureRecord::new(
                    identifier.0,
                    molarweight,
                    model_record.0,
                    ideal_gas_record.map(|ig| ig.0),
                )))
            }

            #[getter]
            fn get_identifier(&self) -> PyIdentifier {
                PyIdentifier(self.0.identifier.clone())
            }

            #[setter]
            fn set_identifier(&mut self, identifier: PyIdentifier) {
                self.0.identifier = identifier.0;
            }

            #[getter]
            fn get_molarweight(&self) -> f64 {
                self.0.molarweight
            }

            #[setter]
            fn set_molarweight(&mut self, molarweight: f64) {
                self.0.molarweight = molarweight;
            }

            #[getter]
            fn get_model_record(&self) -> $py_model_record {
                $py_model_record(self.0.model_record.clone())
            }

            #[setter]
            fn set_model_record(&mut self, model_record: $py_model_record) {
                self.0.model_record = model_record.0;
            }

            #[getter]
            fn get_ideal_gas_record(&self) -> Option<$py_ideal_gas_record> {
                self.0.ideal_gas_record.clone().map($py_ideal_gas_record)
            }

            #[setter]
            fn set_ideal_gas_record(&mut self, ideal_gas_record: $py_ideal_gas_record) {
                self.0.ideal_gas_record = Some(ideal_gas_record.0);
            }

            fn __repr__(&self) -> PyResult<String> {
                Ok(self.0.to_string())
            }
        }

        impl_json_handling!(PyPureRecord);
    };
}

#[macro_export]
macro_rules! impl_segment_record {
    ($model_record:ident, $py_model_record:ident, $ideal_gas_record:ident, $py_ideal_gas_record:ident) => {
        /// All information required to characterize a single segment.
        ///
        /// Parameters
        /// ----------
        /// identifier : str
        ///     The identifier of the segment.
        /// molarweight : float
        ///     The molar weight (in g/mol) of the segment.
        /// model_record : ModelRecord
        ///     The segment model parameters.
        /// ideal_gas_record: IdealGasRecord, optional
        ///     The segment ideal gas parameters.
        ///
        /// Returns
        /// -------
        /// SegmentRecord
        #[pyclass(name = "SegmentRecord")]
        #[pyo3(text_signature = "(identifier, molarweight, model_record, ideal_gas_record=None)")]
        #[derive(Clone)]
        pub struct PySegmentRecord(SegmentRecord<$model_record, $ideal_gas_record>);

        #[pymethods]
        impl PySegmentRecord {
            #[new]
            fn new(
                identifier: String,
                molarweight: f64,
                model_record: $py_model_record,
                ideal_gas_record: Option<$py_ideal_gas_record>,
            ) -> PyResult<Self> {
                Ok(Self(SegmentRecord::new(
                    identifier,
                    molarweight,
                    model_record.0,
                    ideal_gas_record.map(|ig| ig.0),
                )))
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
            fn from_json(path: &str) -> Result<Vec<Self>, ParameterError> {
                Ok(SegmentRecord::from_json(path)?
                    .into_iter()
                    .map(Self)
                    .collect())
            }

            #[getter]
            fn get_identifier(&self) -> String {
                self.0.identifier.clone()
            }

            #[setter]
            fn set_identifier(&mut self, identifier: String) {
                self.0.identifier = identifier;
            }

            #[getter]
            fn get_molarweight(&self) -> f64 {
                self.0.molarweight
            }

            #[setter]
            fn set_molarweight(&mut self, molarweight: f64) {
                self.0.molarweight = molarweight;
            }

            #[getter]
            fn get_model_record(&self) -> $py_model_record {
                $py_model_record(self.0.model_record.clone())
            }

            #[setter]
            fn set_model_record(&mut self, model_record: $py_model_record) {
                self.0.model_record = model_record.0;
            }

            #[getter]
            fn get_ideal_gas_record(&self) -> Option<$py_ideal_gas_record> {
                self.0.ideal_gas_record.clone().map($py_ideal_gas_record)
            }

            #[setter]
            fn set_ideal_gas_record(&mut self, ideal_gas_record: $py_ideal_gas_record) {
                self.0.ideal_gas_record = Some(ideal_gas_record.0);
            }

            fn __repr__(&self) -> PyResult<String> {
                Ok(self.0.to_string())
            }
        }

        impl_json_handling!(PySegmentRecord);
    };
}

#[macro_export]
macro_rules! impl_parameter {
    ($parameter:ty, $py_parameter:ty) => {
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
            /// search_option : IdentifierOption, optional, defaults to IdentifierOption.Name
            ///     Identifier that is used to search binary records.
            #[staticmethod]
            #[pyo3(
                signature = (pure_records, binary_records=None, search_option=IdentifierOption::Name),
                text_signature = "(pure_records, binary_records=None, search_option=None)"
            )]
            fn from_records(
                pure_records: Vec<PyPureRecord>,
                binary_records: Option<&PyAny>,
                search_option: IdentifierOption,
            ) -> PyResult<Self> {
                let prs = pure_records.into_iter().map(|pr| pr.0).collect();
                if let Some(binary_records) = binary_records {
                    let brs = if let Ok(br) = binary_records.extract::<PyReadonlyArray2<f64>>() {
                        Ok(br.to_owned_array().mapv(|r| r.try_into().unwrap()))
                    } else if let Ok(br) = binary_records.extract::<Vec<PyBinaryRecord>>() {
                        let brs: Vec<_> = br.into_iter().map(|br| br.0).collect();
                        Ok(<$parameter>::binary_matrix_from_records(
                            &prs,
                            &brs,
                            search_option,
                        )?)
                    } else {
                        Err(PyErr::new::<PyTypeError, _>(format!(
                            "Could not parse binary input!"
                        )))
                    };
                    Ok(Self(Arc::new(<$parameter>::from_records(
                        prs,
                        brs.unwrap(),
                    )?)))
                } else {
                    let n = prs.len();
                    Ok(Self(Arc::new(<$parameter>::from_records(
                        prs,
                        Array2::from_elem([n, n], <$parameter as Parameter>::Binary::default()),
                    )?)))
                }
            }

            /// Creates parameters for a pure component from a pure record.
            ///
            /// Parameters
            /// ----------
            /// pure_record : PureRecord
            ///     The pure component parameters.
            #[staticmethod]
            fn new_pure(pure_record: PyPureRecord) -> PyResult<Self> {
                Ok(Self(Arc::new(<$parameter>::new_pure(pure_record.0)?)))
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
            #[pyo3(text_signature = "(pure_records, binary_record=None)")]
            fn new_binary(
                pure_records: Vec<PyPureRecord>,
                binary_record: Option<&PyAny>,
            ) -> PyResult<Self> {
                let prs = pure_records.into_iter().map(|pr| pr.0).collect();
                let br = binary_record
                    .map(|br| {
                        if let Ok(r) = br.extract::<f64>() {
                            Ok(r.try_into()?)
                        } else if let Ok(r) = br.extract::<PyBinaryRecord>() {
                            Ok(r.0.model_record)
                        } else {
                            Err(PyErr::new::<PyTypeError, _>(format!(
                                "Could not parse binary input!"
                            )))
                        }
                    })
                    .transpose()?;
                Ok(Self(Arc::new(<$parameter>::new_binary(prs, br)?)))
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
            /// search_option : IdentifierOption, optional, defaults to IdentifierOption.Name
            ///     Identifier that is used to search substance.
            #[staticmethod]
            #[pyo3(
                signature = (substances, pure_path, binary_path=None, search_option=IdentifierOption::Name),
                text_signature = "(substances, pure_path, binary_path=None, search_option)"
            )]
            fn from_json(
                substances: Vec<&str>,
                pure_path: String,
                binary_path: Option<String>,
                search_option: IdentifierOption,
            ) -> Result<Self, ParameterError> {
                Ok(Self(Arc::new(<$parameter>::from_json(
                    substances,
                    pure_path,
                    binary_path,
                    search_option,
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
            /// search_option : IdentifierOption, optional, defaults to IdentifierOption.Name
            ///     Identifier that is used to search substance.
            #[staticmethod]
            #[pyo3(
                signature = (input, binary_path=None, search_option=IdentifierOption::Name),
                text_signature = "(input, binary_path=None, search_option)"
            )]
            fn from_multiple_json(
                input: Vec<(Vec<&str>, &str)>,
                binary_path: Option<&str>,
                search_option: Option<IdentifierOption>,
            ) -> Result<Self, ParameterError> {
                Ok(Self(Arc::new(<$parameter>::from_multiple_json(
                    &input,
                    binary_path,
                    search_option.unwrap_or(IdentifierOption::Name),
                )?)))
            }

            #[getter]
            fn get_pure_records(&self) -> Vec<PyPureRecord> {
                self.0
                    .records()
                    .0
                    .iter()
                    .map(|r| PyPureRecord(r.clone()))
                    .collect()
            }

            #[getter]
            fn get_binary_records<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
                self.0
                    .records()
                    .1
                    .mapv(|r| f64::try_from(r).unwrap())
                    .view()
                    .to_pyarray(py)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_parameter_from_segments {
    ($parameter:ty, $py_parameter:ty) => {
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
            #[pyo3(text_signature = "(chemical_records, segment_records, binary_segment_records=None)")]
            fn from_segments(
                chemical_records: Vec<PyChemicalRecord>,
                segment_records: Vec<PySegmentRecord>,
                binary_segment_records: Option<Vec<PyBinarySegmentRecord>>,
            ) -> Result<Self, ParameterError> {
                Ok(Self(Arc::new(<$parameter>::from_segments(
                    chemical_records.into_iter().map(|cr| cr.0).collect(),
                    segment_records.into_iter().map(|sr| sr.0).collect(),
                    binary_segment_records.map(|r| r.into_iter().map(|r| BinaryRecord{id1:r.0.id1,id2:r.0.id2,model_record:r.0.model_record.into()}).collect()),
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
            /// search_option : IdentifierOption, optional, defaults to IdentifierOption.Name
            ///     Identifier that is used to search substance.
            #[staticmethod]
            #[pyo3(
                signature = (substances, pure_path, segments_path, binary_path=None, search_option=IdentifierOption::Name),
                text_signature = "(substances, pure_path, segments_path, binary_path=None, search_option)"
            )]
            fn from_json_segments(
                substances: Vec<&str>,
                pure_path: String,
                segments_path: String,
                binary_path: Option<String>,
                search_option: IdentifierOption,
            ) -> Result<Self, ParameterError> {
                Ok(Self(Arc::new(<$parameter>::from_json_segments(
                    &substances,
                    pure_path,
                    segments_path,
                    binary_path,
                    search_option,
                )?)))
            }
        }
    };
}

#[macro_export]
macro_rules! impl_json_handling {
    ($py_parameter:ty) => {
        #[pymethods]
        impl $py_parameter {
            /// Creates record from json string.
            #[staticmethod]
            fn from_json_str(json: &str) -> Result<Self, ParameterError> {
                Ok(Self(serde_json::from_str(json)?))
            }

            /// Creates a json string from record.
            fn to_json_str(&self) -> Result<String, ParameterError> {
                Ok(serde_json::to_string(&self.0)?)
            }
        }
    };
}
