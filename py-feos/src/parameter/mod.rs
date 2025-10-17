use crate::error::PyFeosError;
use feos_core::parameter::*;
use feos_core::{FeosError, FeosResult};
use indexmap::IndexSet;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pythonize::depythonize;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Write;

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

/// Set of parameters that fully characterizes a mixture.
#[pyclass(name = "Parameters")]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyParameters {
    pub pure_records: Vec<PureRecord<Value, Value>>,
    pub binary_records: Vec<BinaryRecord<usize, Value, Value>>,
}

impl TryFrom<PyParameters> for Parameters<Value, Value, Value> {
    type Error = FeosError;
    fn try_from(value: PyParameters) -> FeosResult<Self> {
        Self::new(value.pure_records, value.binary_records)
    }
}

impl PyParameters {
    pub fn try_convert<P, B, A>(self) -> PyResult<Parameters<P, B, A>>
    where
        for<'de> P: Deserialize<'de> + Clone,
        for<'de> B: Deserialize<'de> + Clone,
        for<'de> A: Deserialize<'de> + Clone,
    {
        let pure_records = self
            .pure_records
            .into_iter()
            .map(|r| Ok(serde_json::from_value(serde_json::to_value(r)?)?))
            .collect::<Result<_, PyFeosError>>()?;
        let binary_records = self
            .binary_records
            .into_iter()
            .map(|r| Ok(serde_json::from_value(serde_json::to_value(r)?)?))
            .collect::<Result<_, PyFeosError>>()?;
        Ok(Parameters::new(pure_records, binary_records).map_err(PyFeosError::from)?)
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
        let pure_records: Vec<_> = pure_records.into_iter().map(PureRecord::from).collect();
        let binary_records: Vec<_> = binary_records.into_iter().map(BinaryRecord::from).collect();
        let binary_records = Parameters::binary_matrix_from_records(
            &pure_records,
            &binary_records,
            identifier_option.into(),
        )
        .map_err(PyFeosError::from)?;
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
            pure_records: vec![pure_record.into()],
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
    /// binary_parameters : float or BinaryRecord, optional
    ///     The binary interaction parameter or binary interaction record.
    #[staticmethod]
    #[pyo3(signature = (pure_records, **binary_parameters))]
    fn new_binary(
        pure_records: [PyPureRecord; 2],
        binary_parameters: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let pure_records = pure_records.into_iter().map(|r| r.into()).collect();
        let binary_records = binary_parameters
            .iter()
            .map(|binary_parameters| {
                binary_parameters.set_item("id1", 0)?;
                binary_parameters.set_item("id2", 1)?;
                depythonize(binary_parameters)
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            pure_records,
            binary_records,
        })
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
        substances: Vec<String>,
        pure_path: String,
        binary_path: Option<String>,
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
        input: Vec<(Vec<String>, String)>,
        binary_path: Option<String>,
        identifier_option: PyIdentifierOption,
    ) -> PyResult<Self> {
        let pure_records = PureRecord::from_multiple_json(&input, identifier_option.into())
            .map_err(PyFeosError::from)?;
        let binary_records: Vec<_> = binary_path
            .map_or_else(|| Ok(Vec::new()), BinaryRecord::from_json)
            .map_err(PyFeosError::from)?;
        let binary_records = Parameters::binary_matrix_from_records(
            &pure_records,
            &binary_records,
            identifier_option.into(),
        )
        .map_err(PyFeosError::from)?;
        Ok(Self {
            pure_records,
            binary_records,
        })
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
    fn get_pure_records(&self) -> Vec<PyPureRecord> {
        self.pure_records
            .iter()
            .map(|pr| PyPureRecord::from(pr.clone()))
            .collect()
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
                    .as_readable_str()
                    .map_or_else(|| format!("Component {}", i + 1), |s| s.to_owned())
            })
            .collect();

        // collect all pure component parameters
        let params: IndexSet<_> = self
            .pure_records
            .iter()
            .flat_map(|r| r.model_record.as_object().unwrap().keys())
            .collect();

        // collect association parameters and count the association sites
        let [mut na, mut nb, mut nc] = [0.0; 3];
        let mut assoc_params = IndexSet::new();
        for r in &self.pure_records {
            for s in &r.association_sites {
                na += s.na;
                nb += s.nb;
                nc += s.nc;
                if let Some(p) = &s.parameters {
                    assoc_params.extend(p.as_object().unwrap().keys());
                }
            }
        }

        let mut output = String::new();
        let o = &mut output;

        // print pure component parameters in a table
        write!(o, "|component|molarweight|").unwrap();
        for p in &params {
            write!(o, "{p}|").unwrap();
        }
        if na + nb + nc > 0.0 {
            write!(o, "sites|").unwrap();
            if na > 0.0 {
                write!(o, "na|").unwrap();
            }
            if nb > 0.0 {
                write!(o, "nb|").unwrap();
            }
            if nc > 0.0 {
                write!(o, "nc|").unwrap();
            }
            for p in &assoc_params {
                write!(o, "{p}|").unwrap();
            }
        }
        write!(o, "\n|-|-|").unwrap();
        for _ in &params {
            write!(o, "-|").unwrap();
        }
        if na + nb + nc > 0.0 {
            write!(o, "-|").unwrap();
            if na > 0.0 {
                write!(o, "-|").unwrap();
            }
            if nb > 0.0 {
                write!(o, "-|").unwrap();
            }
            if nc > 0.0 {
                write!(o, "-|").unwrap();
            }
            for _ in &assoc_params {
                write!(o, "-|").unwrap();
            }
        }
        for (record, comp) in self.pure_records.iter().zip(&component_names) {
            write!(o, "\n|{}|{}|", comp, record.molarweight).unwrap();
            let model_record = record.model_record.as_object().unwrap();
            for &p in &params {
                if let Some(val) = model_record.get(p) {
                    write!(o, "{val}|")
                } else {
                    write!(o, "-|")
                }
                .unwrap();
            }
            if !record.association_sites.is_empty() {
                let s = &record.association_sites[0];
                if na + nb + nc > 0.0 {
                    write!(o, "{}|", s.id).unwrap();
                    if na > 0.0 {
                        write!(o, "{}|", s.na).unwrap();
                    }
                    if nb > 0.0 {
                        write!(o, "{}|", s.nb).unwrap();
                    }
                    if nc > 0.0 {
                        write!(o, "{}|", s.nc).unwrap();
                    }
                    for &p in &assoc_params {
                        if let Some(par) = &s.parameters {
                            let assoc_record = par.as_object().unwrap();
                            if let Some(val) = assoc_record.get(p) {
                                write!(o, "{val}|")
                            } else {
                                write!(o, "-|")
                            }
                            .unwrap();
                        }
                    }
                }
            }
            for s in record.association_sites.iter().skip(1) {
                write!(o, "\n|||").unwrap();
                for &_ in &params {
                    write!(o, "|").unwrap();
                }
                if na + nb + nc > 0.0 {
                    write!(o, "{}|", s.id).unwrap();
                    if na > 0.0 {
                        write!(o, "{}|", s.na).unwrap();
                    }
                    if nb > 0.0 {
                        write!(o, "{}|", s.nb).unwrap();
                    }
                    if nc > 0.0 {
                        write!(o, "{}|", s.nc).unwrap();
                    }
                    for &p in &assoc_params {
                        if let Some(par) = &s.parameters {
                            let assoc_record = par.as_object().unwrap();
                            if let Some(val) = assoc_record.get(p) {
                                write!(o, "{val}|")
                            } else {
                                write!(o, "-|")
                            }
                            .unwrap();
                        }
                    }
                }
            }
        }

        if !self.binary_records.is_empty() {
            // collect all binary interaction parameters
            let params: IndexSet<_> = self
                .binary_records
                .iter()
                .flat_map(|r| {
                    r.model_record
                        .iter()
                        .flat_map(|r| r.as_object().unwrap().keys())
                })
                .collect();
            // collect all binary association parameters
            let assoc_params: IndexSet<_> = self
                .binary_records
                .iter()
                .flat_map(|r| {
                    r.association_sites
                        .iter()
                        .flat_map(|s| s.parameters.as_object().unwrap().keys())
                })
                .collect();

            // print binary interaction parameters
            write!(o, "\n\n|component 1|component 2|").unwrap();
            for p in &params {
                write!(o, "{p}|").unwrap();
            }
            if !assoc_params.is_empty() {
                write!(o, "site 1| site 2|").unwrap();
                for p in &assoc_params {
                    write!(o, "{p}|").unwrap();
                }
            }
            write!(o, "\n|-|-|").unwrap();
            for _ in &params {
                write!(o, "-|").unwrap();
            }
            if !assoc_params.is_empty() {
                write!(o, "-|-|").unwrap();
                for _ in &assoc_params {
                    write!(o, "-|").unwrap();
                }
            }
            for r in &self.binary_records {
                write!(o, "\n|{}|", component_names[r.id1]).unwrap();
                write!(o, "{}|", component_names[r.id2]).unwrap();
                if let Some(m) = &r.model_record {
                    let model_record = m.as_object().unwrap();
                    for &p in &params {
                        if let Some(val) = model_record.get(p) {
                            write!(o, "{val}|")
                        } else {
                            write!(o, "-|")
                        }
                        .unwrap();
                    }
                }
                if !r.association_sites.is_empty() {
                    let s = &r.association_sites[0];
                    write!(o, "{}|{}|", s.id1, s.id2).unwrap();
                    for &p in &assoc_params {
                        let assoc_record = s.parameters.as_object().unwrap();
                        if let Some(val) = assoc_record.get(p) {
                            write!(o, "{val}|")
                        } else {
                            write!(o, "-|")
                        }
                        .unwrap();
                    }
                }
                for s in r.association_sites.iter().skip(1) {
                    write!(o, "\n|||").unwrap();
                    for &_ in &params {
                        write!(o, "|").unwrap();
                    }
                    write!(o, "{}|{}|", s.id1, s.id2).unwrap();
                    for &p in &assoc_params {
                        let assoc_record = s.parameters.as_object().unwrap();
                        if let Some(val) = assoc_record.get(p) {
                            write!(o, "{val}|")
                        } else {
                            write!(o, "-|")
                        }
                        .unwrap();
                    }
                }
            }
        }

        output
    }
}

/// Combination of chemical information and segment parameters that is used to
/// parametrize a group-contribution model.
#[pyclass(name = "GcParameters")]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyGcParameters {
    chemical_records: Vec<ChemicalRecord>,
    segment_records: Vec<SegmentRecord<Value, Value>>,
    binary_segment_records: Option<Vec<BinaryRecord<String, Value, Value>>>,
}

impl PyGcParameters {
    pub fn try_convert_homosegmented<P, B, A>(self) -> PyResult<Parameters<P, B, A>>
    where
        for<'de> P: Deserialize<'de> + Clone + FromSegments,
        for<'de> B: Deserialize<'de> + Clone + FromSegmentsBinary + Default,
        for<'de> A: Deserialize<'de> + Clone,
    {
        let segment_records: Vec<_> = self
            .segment_records
            .into_iter()
            .map(|r| Ok(serde_json::from_value(serde_json::to_value(r)?)?))
            .collect::<Result<_, PyFeosError>>()?;
        let binary_segment_records = self
            .binary_segment_records
            .map(|bsr| {
                bsr.into_iter()
                    .map(|r| Ok(serde_json::from_value(serde_json::to_value(r)?)?))
                    .collect::<Result<Vec<_>, PyFeosError>>()
            })
            .transpose()?;
        Ok(Parameters::from_segments(
            self.chemical_records,
            &segment_records,
            binary_segment_records.as_deref(),
        )
        .map_err(PyFeosError::from)?)
    }

    pub fn try_convert_heterosegmented<P, B, A, C: GroupCount + Default>(
        self,
    ) -> PyResult<GcParameters<P, B, A, (), C>>
    where
        for<'de> P: Deserialize<'de> + Clone,
        for<'de> B: Deserialize<'de> + Clone,
        for<'de> A: Deserialize<'de> + Clone,
    {
        let segment_records = self
            .segment_records
            .into_iter()
            .map(|r| Ok(serde_json::from_value(serde_json::to_value(r)?)?))
            .collect::<Result<Vec<_>, PyFeosError>>()?;
        let binary_segment_records = self
            .binary_segment_records
            .map(|bsr| {
                bsr.into_iter()
                    .map(|r| Ok(serde_json::from_value(serde_json::to_value(r)?)?))
                    .collect::<Result<Vec<_>, PyFeosError>>()
            })
            .transpose()?;
        Ok(GcParameters::<P, B, A, (), C>::from_segments_hetero(
            self.chemical_records,
            &segment_records,
            binary_segment_records.as_deref(),
        )
        .map_err(PyFeosError::from)?)
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
        let chemical_records = chemical_records.into_iter().map(|r| r.into()).collect();
        let segment_records = segment_records.into_iter().map(|r| r.into()).collect();
        let binary_segment_records =
            binary_segment_records.map(|bsr| bsr.into_iter().map(|r| r.into()).collect());
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
        signature = (substances, pure_path, segments_path, binary_path=None, identifier_option=PyIdentifierOption::Name),
        text_signature = "(substances, pure_path, segments_path, binary_path=None, identifier_option=IdentiferOption.Name)"
    )]
    fn from_json_segments(
        substances: Vec<String>,
        pure_path: &str,
        segments_path: &str,
        binary_path: Option<&str>,
        identifier_option: PyIdentifierOption,
    ) -> PyResult<Self> {
        let chemical_records =
            PyChemicalRecord::from_json(substances, pure_path, identifier_option)?;
        let segment_records = PySegmentRecord::from_json(segments_path)?;
        let binary_segment_records = binary_path
            .as_ref()
            .map(|p| {
                BinarySegmentRecord::from_json(p as &str)
                    .map(|brs| brs.into_iter().map(|r| r.into()).collect())
            })
            .transpose()
            .map_err(PyFeosError::from)?;
        Ok(Self::from_segments(
            chemical_records,
            segment_records,
            binary_segment_records,
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
        smarts_path: &str,
        segments_path: &str,
        binary_path: Option<&str>,
    ) -> PyResult<Self> {
        let smarts_records = PySmartsRecord::from_json(smarts_path)?;
        let segment_records = PySegmentRecord::from_json(segments_path)?;
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
