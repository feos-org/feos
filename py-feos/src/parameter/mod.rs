use crate::error::PyFeosError;
use feos_core::{parameter::*, FeosError};
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use ndarray::Array2;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pythonize::{pythonize, PythonizeError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::fs::File;
use std::io::BufReader;

mod chemical_record;
mod fragmentation;
mod identifier;
mod model_record;
mod segment;

// Export for wheel.
pub(crate) use chemical_record::PyChemicalRecord;
pub(crate) use fragmentation::PySmartsRecord;
pub(crate) use identifier::{PyIdentifier, PyIdentifierOption};
pub(crate) use model_record::{PyBinaryRecord, PyPureRecord};
pub(crate) use segment::{PyBinarySegmentRecord, PySegmentRecord};

#[pyclass(name = "Parameters")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyParameters {
    #[pyo3(get)]
    pub pure_records: Vec<PyPureRecord>,
    pub binary_records: Vec<([usize; 2], Value)>,
}

impl PyParameters {
    pub fn try_convert<P: Parameter>(self) -> PyResult<P> {
        let n = self.pure_records.len();
        let pure_records = self
            .pure_records
            .into_iter()
            .map(|r| r.try_into())
            .collect::<Result<_, _>>()
            .map_err(PyFeosError::from)?;
        let binary_records = if self.binary_records.is_empty() {
            None
        } else {
            let mut br = Array2::default((n, n));
            for ([i, j], r) in self.binary_records {
                let r: P::Binary = serde_json::from_value(r).map_err(PyFeosError::from)?;
                br[[i, j]] = r.clone();
                br[[j, i]] = r;
            }
            Some(br)
        };
        Ok(P::from_records(pure_records, binary_records).map_err(PyFeosError::from)?)
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
        signature = (pure_records, binary_records=vec![], identifier_option=PyIdentifierOption::Name),
        text_signature = "(pure_records, binary_records=[], identifier_option=IdentifierOption.Name)"
    )]
    fn from_records(
        pure_records: Vec<PyPureRecord>,
        binary_records: Vec<PyBinaryRecord>,
        identifier_option: PyIdentifierOption,
    ) -> PyResult<Self> {
        // Build Hashmap (id, id) -> BinaryRecord
        let binary_map: HashMap<_, _> = {
            binary_records
                .iter()
                .filter_map(|br| {
                    let id1 = br.id1.as_str(identifier_option);
                    let id2 = br.id2.as_str(identifier_option);
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
                let id1 = p1.identifier.as_str(identifier_option).unwrap_or_else(|| {
                    panic!(
                        "No {} for pure record {} ({}).",
                        IdentifierOption::from(identifier_option),
                        i1,
                        p1.identifier.0
                    )
                });
                let id2 = p2.identifier.as_str(identifier_option).unwrap_or_else(|| {
                    panic!(
                        "No {} for pure record {} ({}).",
                        IdentifierOption::from(identifier_option),
                        i2,
                        p2.identifier.0
                    )
                });
                binary_map
                    .get(&(id1, id2))
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
        signature = (substances, pure_path, binary_path=None, identifier_option=PyIdentifierOption::Name),
        text_signature = "(substances, pure_path, binary_path=None, identifier_option)"
    )]
    fn from_json(
        substances: Vec<PyBackedStr>,
        pure_path: PyBackedStr,
        binary_path: Option<PyBackedStr>,
        identifier_option: PyIdentifierOption,
    ) -> PyResult<Self> {
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
        signature = (input, binary_path=None, identifier_option=PyIdentifierOption::Name),
        text_signature = "(input, binary_path=None, identifier_option)"
    )]
    fn from_multiple_json(
        input: Vec<(Vec<PyBackedStr>, PyBackedStr)>,
        binary_path: Option<PyBackedStr>,
        identifier_option: PyIdentifierOption,
    ) -> PyResult<Self> {
        // total number of substances queried
        let nsubstances = input.iter().map(|(substances, _)| substances.len()).sum();

        // queried substances with removed duplicates
        let queried: HashSet<_> = input
            .iter()
            .flat_map(|(substances, _)| substances)
            .collect();

        // check if there are duplicates
        if queried.len() != nsubstances {
            return Err(PyFeosError::from(FeosError::IncompatibleParameters(
                "A substance was defined more than once.".to_string(),
            ))
            .into());
        }

        let mut pure_records = Vec::with_capacity(nsubstances);

        // collect parameters from files into single map
        for (substances, file) in input {
            pure_records.extend(
                PyPureRecord::from_json(&substances, &file, identifier_option)
                    .map_err(PyFeosError::from)?,
            );
        }

        let binary_records = if let Some(path) = binary_path {
            let reader = BufReader::new(File::open::<&str>(path.as_ref())?);
            serde_json::from_reader(reader).map_err(PyFeosError::from)?
        } else {
            Vec::new()
        };

        Self::from_records(pure_records, binary_records, identifier_option)
    }

    /// Generates JSON-formatted string for pure and binary records (if initialized).
    ///
    /// Parameters
    /// ----------
    /// pretty : bool
    ///     Whether to use pretty (True) or dense (False) formatting. Defaults to True.
    ///
    /// Returns
    /// -------
    /// str : The JSON-formatted string.
    #[pyo3(
        signature = (pretty=true),
        text_signature = "(pretty=True)"
    )]
    fn to_json_str(&self, pretty: bool) -> PyResult<(String, Option<String>)> {
        let pr_json = if pretty {
            serde_json::to_string_pretty(&self.pure_records)
        } else {
            serde_json::to_string(&self.pure_records)
        }
        .map_err(PyFeosError::from)?;
        let br_json = (!self.binary_records.is_empty())
            .then(|| {
                if pretty {
                    serde_json::to_string_pretty(&self.binary_records)
                } else {
                    serde_json::to_string(&self.binary_records)
                }
            })
            .transpose()
            .map_err(PyFeosError::from)?;
        Ok((pr_json, br_json))
    }

    #[getter]
    fn get_binary_records<'py>(
        &self,
        py: Python<'py>,
    ) -> Result<Bound<'py, PyAny>, PythonizeError> {
        pythonize(py, &self.binary_records)
    }

    fn __repr__(&self) -> PyResult<String> {
        let (mut pr, br) = self.to_json_str(true)?;
        if let Some(br) = br {
            pr += "\n\n";
            pr += &br;
        }
        Ok(pr)
    }

    fn _repr_markdown_(&self) -> String {
        // crate consistent list of component names
        let component_names: Vec<_> = self
            .pure_records
            .iter()
            .enumerate()
            .map(|(i, r)| {
                r.identifier
                    .0
                    .as_readable_str()
                    .map_or_else(|| format!("Component {}", i + 1), |s| s.to_owned())
            })
            .collect();

        // collect all pure component parameters
        let params: IndexSet<_> = self
            .pure_records
            .iter()
            .flat_map(|r| {
                serde_json::from_value::<IndexMap<String, Value>>(r.model_record.clone())
                    .unwrap()
                    .into_keys()
            })
            .collect();

        let mut output = String::new();
        let o = &mut output;

        // print pure component parameters in a table
        write!(o, "|component|molarweight|").unwrap();
        for p in &params {
            write!(o, "{}|", p).unwrap();
        }
        write!(o, "\n|-|-|").unwrap();
        for _ in &params {
            write!(o, "-|").unwrap();
        }
        for (record, comp) in self.pure_records.iter().zip(&component_names) {
            write!(o, "\n|{}|{}|", comp, record.molarweight).unwrap();
            let model_record =
                serde_json::from_value::<IndexMap<String, Value>>(record.model_record.clone())
                    .unwrap();
            for p in &params {
                if let Some(val) = model_record.get(p) {
                    write!(o, "{}|", val)
                } else {
                    write!(o, "-|")
                }
                .unwrap();
            }
        }

        if !self.binary_records.is_empty() {
            // collect all binary interaction parameters
            let params: IndexSet<_> = self
                .binary_records
                .iter()
                .flat_map(|(_, r)| {
                    serde_json::from_value::<IndexMap<String, Value>>(r.clone())
                        .unwrap()
                        .into_keys()
                })
                .collect();

            // print binary interaction parameters
            write!(o, "\n\n|component 1|component 2|").unwrap();
            for p in &params {
                write!(o, "{}|", p).unwrap();
            }
            write!(o, "\n|-|-|").unwrap();
            for _ in &params {
                write!(o, "-|").unwrap();
            }
            for ([i, j], r) in &self.binary_records {
                write!(o, "\n|{}|{}|", component_names[*i], component_names[*j]).unwrap();
                let model_record =
                    serde_json::from_value::<IndexMap<String, Value>>(r.clone()).unwrap();
                for p in &params {
                    if let Some(val) = model_record.get(p) {
                        write!(o, "{}|", val)
                    } else {
                        write!(o, "-|")
                    }
                    .unwrap();
                }
            }
        }

        output
    }
}

#[pyclass(name = "GcParameters", get_all)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyGcParameters {
    chemical_records: Vec<PyChemicalRecord>,
    segment_records: Vec<PySegmentRecord>,
    binary_segment_records: Option<Vec<PyBinarySegmentRecord>>,
}

impl PyGcParameters {
    pub fn try_convert_homosegmented<P: Parameter>(self) -> PyResult<P>
    where
        P::Pure: FromSegments<usize>,
        P::Binary: FromSegmentsBinary<usize>,
    {
        let segment_records = self
            .segment_records
            .into_iter()
            .map(|r| r.try_into())
            .collect::<Result<_, _>>()
            .map_err(PyFeosError::from)?;
        let chemical_records: Vec<ChemicalRecord> = self
            .chemical_records
            .into_iter()
            .map(|r| r.into())
            .collect();
        let binary_segment_records = self
            .binary_segment_records
            .map(|bsr| bsr.into_iter().map(|r| r.into()).collect());
        Ok(
            P::from_segments(chemical_records, segment_records, binary_segment_records)
                .map_err(PyFeosError::from)?,
        )
    }

    pub fn try_convert_heterosegmented<P: ParameterHetero>(self) -> PyResult<P>
    where
        P::Chemical: From<ChemicalRecord>,
    {
        let segment_records = self
            .segment_records
            .into_iter()
            .map(|r| r.try_into())
            .collect::<Result<_, _>>()
            .map_err(PyFeosError::from)?;
        let chemical_records: Vec<ChemicalRecord> = self
            .chemical_records
            .into_iter()
            .map(|r| r.into())
            .collect();
        let binary_segment_records = self
            .binary_segment_records
            .map(|bsr| bsr.into_iter().map(|r| r.into()).collect());
        Ok(
            P::from_segments(chemical_records, segment_records, binary_segment_records)
                .map_err(PyFeosError::from)?,
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
        chemical_records: Vec<PyChemicalRecord>,
        segment_records: Vec<PySegmentRecord>,
        binary_segment_records: Option<Vec<PyBinarySegmentRecord>>,
    ) -> Self {
        Self {
            chemical_records,
            segment_records,
            binary_segment_records,
        }
    }

    // /// Creates parameters using segments from json file.
    // ///
    // /// Parameters
    // /// ----------
    // /// substances : List[str]
    // ///     The substances to search.
    // /// pure_path : str
    // ///     Path to file containing pure substance parameters.
    // /// segments_path : str
    // ///     Path to file containing segment parameters.
    // /// binary_path : str, optional
    // ///     Path to file containing binary segment-segment parameters.
    // /// identifier_option : IdentifierOption, optional, defaults to IdentifierOption.Name
    // ///     Identifier that is used to search substance.
    // #[staticmethod]
    // #[pyo3(
    //     signature = (substances, pure_path, segments_path, binary_path=None, identifier_option=PyIdentifierOption::Name),
    //     text_signature = "(substances, pure_path, segments_path, binary_path=None, identifier_option=IdentiferOption.Name)"
    // )]
    // fn from_json_segments(
    //     substances: Vec<PyBackedStr>,
    //     pure_path: PyBackedStr,
    //     segments_path: PyBackedStr,
    //     binary_path: Option<PyBackedStr>,
    //     identifier_option: PyIdentifierOption,
    // ) -> PyResult<Self> {
    //     let queried: IndexSet<_> = substances
    //         .iter()
    //         .map(|identifier| identifier as &str)
    //         .collect();

    //     let reader = BufReader::new(File::open(&pure_path as &str)?);
    //     let chemical_records: Vec<ChemicalRecord> = serde_json::from_reader(reader)?;
    //     let mut record_map: IndexMap<_, _> = chemical_records
    //         .into_iter()
    //         .filter_map(|record| {
    //             record
    //                 .identifier
    //                 .as_str(identifier_option.into())
    //                 .map(|i| i.to_owned())
    //                 .map(|i| (i, record))
    //         })
    //         .collect();

    //     // Compare queried components and available components
    //     let available: IndexSet<_> = record_map
    //         .keys()
    //         .map(|identifier| identifier as &str)
    //         .collect();
    //     if !queried.is_subset(&available) {
    //         let missing: Vec<_> = queried.difference(&available).cloned().collect();
    //         return Err(FeosError::ComponentsNotFound(format!("{:?}", missing)));
    //     };

    //     // Collect all pure records that were queried
    //     let chemical_records: Vec<_> = queried
    //         .into_iter()
    //         .filter_map(|identifier| record_map.shift_remove(identifier))
    //         .map(|r| r.into())
    //         .collect();

    //     // Read segment records
    //     let segment_records: Vec<PySegmentRecord> =
    //         PySegmentRecord::from_json(&segments_path as &str)?;

    //     // Read binary records
    //     let binary_records = binary_path
    //         .as_ref()
    //         .map(|file_binary| {
    //             let reader = BufReader::new(File::open(file_binary as &str)?);
    //             let binary_records: Result<Vec<BinarySegmentRecord>, ParameterError> =
    //                 Ok(serde_json::from_reader(reader)?);
    //             binary_records
    //         })
    //         .transpose()?;

    //     Ok(Self::from_segments(
    //         chemical_records,
    //         segment_records,
    //         binary_records,
    //     ))
    // }

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
        binary_segment_records: Option<Vec<PyBinarySegmentRecord>>,
    ) -> PyResult<Self> {
        let chemical_records: Vec<_> = identifier
            .into_iter()
            .map(|i| PyChemicalRecord::from_smiles(&i, smarts_records.clone()))
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
            .map(|p| {
                BinarySegmentRecord::from_json(p as &str)
                    .map(|brs| brs.into_iter().map(|r| r.into()).collect())
            })
            .transpose()
            .map_err(PyFeosError::from)?;
        Self::from_smiles(
            identifier,
            smarts_records,
            segment_records,
            binary_segment_records,
        )
    }
}
