//! Structures and traits that can be used to build model parameters for equations of state.

use conv::ValueInto;
use indexmap::{IndexMap, IndexSet};
use ndarray::Array2;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io;
use std::io::BufReader;
use std::path::Path;
use thiserror::Error;

mod chemical_record;
mod identifier;
mod model_record;
mod segment;

pub use chemical_record::{ChemicalRecord, SegmentCount};
pub use identifier::{Identifier, IdentifierOption};
pub use model_record::{BinaryRecord, FromSegments, FromSegmentsBinary, PureRecord};
pub use segment::SegmentRecord;

/// Constructor methods for parameters.
///
/// By implementing `Parameter` for a type, you define how parameters
/// of an equation of state can be constructed from a sequence of
/// single substance records and possibly binary interaction parameters.
pub trait Parameter
where
    Self: Sized,
{
    type Pure: Clone + DeserializeOwned;
    type Binary: Clone + DeserializeOwned + Default;

    /// Creates parameters from records for pure substances and possibly binary parameters.
    fn from_records(
        pure_records: Vec<PureRecord<Self::Pure>>,
        binary_records: Option<Array2<Self::Binary>>,
    ) -> Result<Self, ParameterError>;

    /// Creates parameters for a pure component from a pure record.
    fn new_pure(pure_record: PureRecord<Self::Pure>) -> Result<Self, ParameterError> {
        Self::from_records(vec![pure_record], None)
    }

    /// Creates parameters for a binary system from pure records and an optional
    /// binary interaction parameter.
    fn new_binary(
        pure_records: Vec<PureRecord<Self::Pure>>,
        binary_record: Option<Self::Binary>,
    ) -> Result<Self, ParameterError> {
        let binary_record = binary_record.map(|br| {
            Array2::from_shape_fn([2, 2], |(i, j)| {
                if i == j {
                    Self::Binary::default()
                } else {
                    br.clone()
                }
            })
        });
        Self::from_records(pure_records, binary_record)
    }

    /// Creates parameters from model records with default values for the molar weight,
    /// identifiers, and binary interaction parameters.
    fn from_model_records(model_records: Vec<Self::Pure>) -> Result<Self, ParameterError> {
        let pure_records = model_records
            .into_iter()
            .map(|r| PureRecord::new(Default::default(), Default::default(), r))
            .collect();
        Self::from_records(pure_records, None)
    }

    /// Return the original pure and binary records that were used to construct the parameters.
    #[allow(clippy::type_complexity)]
    fn records(&self) -> (&[PureRecord<Self::Pure>], Option<&Array2<Self::Binary>>);

    /// Helper function to build matrix from list of records in correct order.
    ///
    /// If the identifiers in `binary_records` are not a subset of those in
    /// `pure_records`, the `Default` implementation of Self::Binary is used.
    #[allow(clippy::expect_fun_call)]
    fn binary_matrix_from_records(
        pure_records: &Vec<PureRecord<Self::Pure>>,
        binary_records: &[BinaryRecord<Identifier, Self::Binary>],
        identifier_option: IdentifierOption,
    ) -> Option<Array2<Self::Binary>> {
        if binary_records.is_empty() {
            return None;
        }

        // Build Hashmap (id, id) -> BinaryRecord
        let binary_map: HashMap<(String, String), Self::Binary> = {
            binary_records
                .iter()
                .filter_map(|br| {
                    let id1 = br.id1.as_string(identifier_option);
                    let id2 = br.id2.as_string(identifier_option);
                    id1.and_then(|id1| id2.map(|id2| ((id1, id2), br.model_record.clone())))
                })
                .collect()
        };
        let n = pure_records.len();
        Some(Array2::from_shape_fn([n, n], |(i, j)| {
            let id1 = pure_records[i]
                .identifier
                .as_string(identifier_option)
                .expect(&format!(
                    "No identifier for given identifier_option for pure record {}.",
                    i
                ));
            let id2 = pure_records[j]
                .identifier
                .as_string(identifier_option)
                .expect(&format!(
                    "No identifier for given identifier_option for pure record {}.",
                    j
                ));
            binary_map
                .get(&(id1.clone(), id2.clone()))
                .or_else(|| binary_map.get(&(id2, id1)))
                .cloned()
                .unwrap_or_default()
        }))
    }

    /// Creates parameters from substance information stored in json files.
    fn from_json<P>(
        substances: Vec<&str>,
        file_pure: P,
        file_binary: Option<P>,
        identifier_option: IdentifierOption,
    ) -> Result<Self, ParameterError>
    where
        P: AsRef<Path>,
    {
        Self::from_multiple_json(&[(substances, file_pure)], file_binary, identifier_option)
    }

    /// Creates parameters from substance information stored in multiple json files.
    fn from_multiple_json<P>(
        input: &[(Vec<&str>, P)],
        file_binary: Option<P>,
        identifier_option: IdentifierOption,
    ) -> Result<Self, ParameterError>
    where
        P: AsRef<Path>,
    {
        // total number of substances queried
        let nsubstances = input
            .iter()
            .fold(0, |acc, (substances, _)| acc + substances.len());

        // queried substances with removed duplicates
        let queried: IndexSet<String> = input
            .iter()
            .flat_map(|(substances, _)| substances)
            .map(|substance| substance.to_string())
            .collect();

        // check if there are duplicates
        if queried.len() != nsubstances {
            return Err(ParameterError::IncompatibleParameters(
                "A substance was defined more than once.".to_string(),
            ));
        }

        let mut records: Vec<PureRecord<Self::Pure>> = Vec::with_capacity(nsubstances);

        // collect parameters from files into single map
        for (substances, file) in input {
            records.extend(PureRecord::<Self::Pure>::from_json(
                substances,
                file,
                identifier_option,
            )?);
        }

        let binary_records = if let Some(path) = file_binary {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        } else {
            Vec::new()
        };
        let record_matrix =
            Self::binary_matrix_from_records(&records, &binary_records, identifier_option);
        Self::from_records(records, record_matrix)
    }

    /// Creates parameters from the molecular structure and segment information.
    ///
    /// The [FromSegments] trait needs to be implemented for both the model record
    /// and the ideal gas record.
    fn from_segments<C: SegmentCount>(
        chemical_records: Vec<C>,
        segment_records: Vec<SegmentRecord<Self::Pure>>,
        binary_segment_records: Option<Vec<BinaryRecord<String, f64>>>,
    ) -> Result<Self, ParameterError>
    where
        Self::Pure: FromSegments<C::Count>,
        Self::Binary: FromSegmentsBinary<C::Count>,
    {
        // update the pure records with model and ideal gas records
        // calculated from the gc method
        let pure_records = chemical_records
            .iter()
            .map(|cr| {
                cr.segment_map(&segment_records).and_then(|segments| {
                    PureRecord::from_segments(cr.identifier().into_owned(), segments)
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Map: (id1, id2) -> model_record
        // empty, if no binary segment records are provided
        let binary_map: HashMap<_, _> = binary_segment_records
            .into_iter()
            .flat_map(|seg| seg.into_iter())
            .map(|br| ((br.id1, br.id2), br.model_record))
            .collect();

        // For every component:  map: id -> count
        let segment_counts: Vec<_> = chemical_records
            .iter()
            .map(|cr| cr.segment_count())
            .collect();

        // full matrix of binary records from the gc method.
        // If a specific segment-segment interaction is not in the binary map,
        // the default value is used.
        let n = pure_records.len();
        let mut binary_records = Array2::default([n, n]);
        for i in 0..n {
            for j in i + 1..n {
                let mut vec = Vec::new();
                for (id1, &n1) in segment_counts[i].iter() {
                    for (id2, &n2) in segment_counts[j].iter() {
                        let binary = binary_map
                            .get(&(id1.clone(), id2.clone()))
                            .or_else(|| binary_map.get(&(id2.clone(), id1.clone())))
                            .cloned()
                            .unwrap_or_default();
                        vec.push((binary, n1, n2));
                    }
                }
                let kij = Self::Binary::from_segments_binary(&vec)?;
                binary_records[(i, j)] = kij.clone();
                binary_records[(j, i)] = kij;
            }
        }

        Self::from_records(pure_records, Some(binary_records))
    }

    /// Creates parameters from segment information stored in json files.
    ///
    /// The [FromSegments] trait needs to be implemented for both the model record
    /// and the ideal gas record.
    fn from_json_segments<P>(
        substances: &[&str],
        file_pure: P,
        file_segments: P,
        file_binary: Option<P>,
        identifier_option: IdentifierOption,
    ) -> Result<Self, ParameterError>
    where
        P: AsRef<Path>,
        Self::Pure: FromSegments<usize>,
        Self::Binary: FromSegmentsBinary<usize>,
    {
        let queried: IndexSet<String> = substances
            .iter()
            .map(|identifier| identifier.to_string())
            .collect();

        let file = File::open(file_pure)?;
        let reader = BufReader::new(file);
        let chemical_records: Vec<ChemicalRecord> = serde_json::from_reader(reader)?;
        let mut record_map: HashMap<_, _> = chemical_records
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
            let msg = format!("{:?}", missing);
            return Err(ParameterError::ComponentsNotFound(msg));
        };

        // collect all pure records that were queried
        let chemical_records: Vec<_> = queried
            .iter()
            .filter_map(|identifier| record_map.remove(&identifier.clone()))
            .collect();

        // Read segment records
        let segment_records: Vec<SegmentRecord<Self::Pure>> =
            SegmentRecord::from_json(file_segments)?;

        // Read binary records
        let binary_records = file_binary
            .map(|file_binary| {
                let reader = BufReader::new(File::open(file_binary)?);
                let binary_records: Result<Vec<BinaryRecord<String, f64>>, ParameterError> =
                    Ok(serde_json::from_reader(reader)?);
                binary_records
            })
            .transpose()?;

        Self::from_segments(chemical_records, segment_records, binary_records)
    }

    /// Return a parameter set containing the subset of components specified in `component_list`.
    ///
    /// # Panics
    ///
    /// Panics if index in `component_list` is out of bounds or if
    /// [Parameter::from_records] fails.
    fn subset(&self, component_list: &[usize]) -> Self {
        let (pure_records, binary_records) = self.records();
        let pure_records = component_list
            .iter()
            .map(|&i| pure_records[i].clone())
            .collect();
        let n = component_list.len();
        let binary_records = binary_records.map(|br| {
            Array2::from_shape_fn([n, n], |(i, j)| {
                br[(component_list[i], component_list[j])].clone()
            })
        });

        Self::from_records(pure_records, binary_records)
            .expect("failed to create subset from parameters.")
    }
}

/// Dummy struct used for models that do not use binary interaction parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct NoBinaryModelRecord;

impl From<f64> for NoBinaryModelRecord {
    fn from(_: f64) -> Self {
        Self
    }
}

impl From<NoBinaryModelRecord> for f64 {
    fn from(_: NoBinaryModelRecord) -> Self {
        0.0 // nasty hack - panic crashes Ipython kernel, actual value is never used
    }
}

impl<T: Copy + ValueInto<f64>> FromSegmentsBinary<T> for NoBinaryModelRecord {
    fn from_segments_binary(_segments: &[(f64, T, T)]) -> Result<Self, ParameterError> {
        Ok(Self)
    }
}

impl fmt::Display for NoBinaryModelRecord {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

/// Constructor methods for parameters for heterosegmented models.
pub trait ParameterHetero: Sized {
    type Chemical: Clone;
    type Pure: Clone + DeserializeOwned;
    type Binary: Clone + DeserializeOwned;

    /// Creates parameters from the molecular structure and segment information.
    fn from_segments<C: Clone + Into<Self::Chemical>>(
        chemical_records: Vec<C>,
        segment_records: Vec<SegmentRecord<Self::Pure>>,
        binary_segment_records: Option<Vec<BinaryRecord<String, Self::Binary>>>,
    ) -> Result<Self, ParameterError>;

    /// Return the original records that were used to construct the parameters.
    #[allow(clippy::type_complexity)]
    fn records(
        &self,
    ) -> (
        &[Self::Chemical],
        &[SegmentRecord<Self::Pure>],
        &Option<Vec<BinaryRecord<String, Self::Binary>>>,
    );

    /// Creates parameters from segment information stored in json files.
    fn from_json_segments<P>(
        substances: &[&str],
        file_pure: P,
        file_segments: P,
        file_binary: Option<P>,
        identifier_option: IdentifierOption,
    ) -> Result<Self, ParameterError>
    where
        P: AsRef<Path>,
        ChemicalRecord: Into<Self::Chemical>,
    {
        let queried: IndexSet<String> = substances
            .iter()
            .map(|identifier| identifier.to_string())
            .collect();

        let reader = BufReader::new(File::open(file_pure)?);
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
            .filter_map(|identifier| record_map.remove(&identifier.clone()))
            .collect();

        // Read segment records
        let segment_records: Vec<SegmentRecord<Self::Pure>> =
            SegmentRecord::from_json(file_segments)?;

        // Read binary records
        let binary_records = file_binary
            .map(|file_binary| {
                let reader = BufReader::new(File::open(file_binary)?);
                let binary_records: Result<
                    Vec<BinaryRecord<String, Self::Binary>>,
                    ParameterError,
                > = Ok(serde_json::from_reader(reader)?);
                binary_records
            })
            .transpose()?;

        Self::from_segments(chemical_records, segment_records, binary_records)
    }

    /// Return a parameter set containing the subset of components specified in `component_list`.
    fn subset(&self, component_list: &[usize]) -> Self {
        let (chemical_records, segment_records, binary_segment_records) = self.records();
        let chemical_records: Vec<_> = component_list
            .iter()
            .map(|&i| chemical_records[i].clone())
            .collect();
        Self::from_segments(
            chemical_records,
            segment_records.to_vec(),
            binary_segment_records.clone(),
        )
        .unwrap()
    }
}

/// Error type for incomplete parameter information and IO problems.
#[derive(Error, Debug)]
pub enum ParameterError {
    #[error(transparent)]
    FileIO(#[from] io::Error),
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
    #[error("The following component(s) were not found: {0}")]
    ComponentsNotFound(String),
    #[error("The identifier '{0}' is not known. ['cas', 'name', 'iupacname', 'smiles', inchi', 'formula']")]
    IdentifierNotFound(String),
    #[error("Information missing.")]
    InsufficientInformation,
    #[error("Incompatible parameters: {0}")]
    IncompatibleParameters(String),
}
