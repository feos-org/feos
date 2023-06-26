//! Structures and traits that can be used to build model parameters for equations of state.

use indexmap::{IndexMap, IndexSet};
use ndarray::Array2;
use serde::de::DeserializeOwned;
use std::collections::HashMap;
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
    type IdealGas: Clone + DeserializeOwned;
    type Binary: Clone + DeserializeOwned + Default;

    /// Creates parameters from records for pure substances and possibly binary parameters.
    fn from_records(
        pure_records: Vec<PureRecord<Self::Pure, Self::IdealGas>>,
        binary_records: Array2<Self::Binary>,
    ) -> Result<Self, ParameterError>;

    /// Creates parameters for a pure component from a pure record.
    fn new_pure(
        pure_record: PureRecord<Self::Pure, Self::IdealGas>,
    ) -> Result<Self, ParameterError> {
        let binary_record = Array2::from_elem([1, 1], Self::Binary::default());
        Self::from_records(vec![pure_record], binary_record)
    }

    /// Creates parameters for a binary system from pure records and an optional
    /// binary interaction parameter.
    fn new_binary(
        pure_records: Vec<PureRecord<Self::Pure, Self::IdealGas>>,
        binary_record: Option<Self::Binary>,
    ) -> Result<Self, ParameterError> {
        let binary_record = Array2::from_shape_fn([2, 2], |(i, j)| {
            if i == j {
                Self::Binary::default()
            } else {
                binary_record.clone().unwrap_or_default()
            }
        });
        Self::from_records(pure_records, binary_record)
    }

    /// Return the original pure and binary records that were used to construct the parameters.
    #[allow(clippy::type_complexity)]
    fn records(
        &self,
    ) -> (
        &[PureRecord<Self::Pure, Self::IdealGas>],
        &Array2<Self::Binary>,
    );

    /// Helper function to build matrix from list of records in correct order.
    ///
    /// If the identifiers in `binary_records` are not a subset of those in
    /// `pure_records`, the `Default` implementation of Self::Binary is used.
    #[allow(clippy::expect_fun_call)]
    fn binary_matrix_from_records(
        pure_records: &Vec<PureRecord<Self::Pure, Self::IdealGas>>,
        binary_records: &[BinaryRecord<Identifier, Self::Binary>],
        search_option: IdentifierOption,
    ) -> Result<Array2<Self::Binary>, ParameterError> {
        // Build Hashmap (id, id) -> BinaryRecord
        let binary_map: HashMap<(String, String), Self::Binary> = {
            binary_records
                .iter()
                .filter_map(|br| {
                    let id1 = br.id1.as_string(search_option);
                    let id2 = br.id2.as_string(search_option);
                    id1.and_then(|id1| id2.map(|id2| ((id1, id2), br.model_record.clone())))
                })
                .collect()
        };
        let n = pure_records.len();
        Ok(Array2::from_shape_fn([n, n], |(i, j)| {
            let id1 = pure_records[i]
                .identifier
                .as_string(search_option)
                .expect(&format!(
                    "No identifier for given search_option for pure record {}.",
                    i
                ));
            let id2 = pure_records[j]
                .identifier
                .as_string(search_option)
                .expect(&format!(
                    "No identifier for given search_option for pure record {}.",
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
        search_option: IdentifierOption,
    ) -> Result<Self, ParameterError>
    where
        P: AsRef<Path>,
    {
        Self::from_multiple_json(&[(substances, file_pure)], file_binary, search_option)
    }

    /// Creates parameters from substance information stored in multiple json files.
    fn from_multiple_json<P>(
        input: &[(Vec<&str>, P)],
        file_binary: Option<P>,
        search_option: IdentifierOption,
    ) -> Result<Self, ParameterError>
    where
        P: AsRef<Path>,
    {
        let mut queried: IndexSet<String> = IndexSet::new();
        let mut record_map: HashMap<String, PureRecord<Self::Pure, Self::IdealGas>> =
            HashMap::new();

        for (substances, file) in input {
            substances.iter().try_for_each(|identifier| {
                match queried.insert(identifier.to_string()) {
                    true => Ok(()),
                    false => Err(ParameterError::IncompatibleParameters(format!(
                        "tried to add substance '{}' to system but it is already present.",
                        identifier
                    ))),
                }
            })?;
            let f = File::open(file)?;
            let reader = BufReader::new(f);

            let pure_records: Vec<PureRecord<Self::Pure, Self::IdealGas>> =
                serde_json::from_reader(reader)?;

            pure_records
                .into_iter()
                .filter_map(|record| {
                    record
                        .identifier
                        .as_string(search_option)
                        .map(|i| (i, record))
                })
                .for_each(|(i, r)| {
                    let _ = record_map.insert(i, r);
                });
        }

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
        let p = queried
            .iter()
            .filter_map(|identifier| record_map.remove(&identifier.clone()))
            .collect();

        let binary_records = if let Some(path) = file_binary {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        } else {
            Vec::new()
        };
        let record_matrix = Self::binary_matrix_from_records(&p, &binary_records, search_option)?;
        Self::from_records(p, record_matrix)
    }

    /// Creates parameters from the molecular structure and segment information.
    ///
    /// The [FromSegments] trait needs to be implemented for both the model record
    /// and the ideal gas record.
    fn from_segments<C: SegmentCount>(
        chemical_records: Vec<C>,
        segment_records: Vec<SegmentRecord<Self::Pure, Self::IdealGas>>,
        binary_segment_records: Option<Vec<BinaryRecord<String, Self::Binary>>>,
    ) -> Result<Self, ParameterError>
    where
        Self::Pure: FromSegments<C::Count>,
        Self::IdealGas: FromSegments<C::Count>,
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

        Self::from_records(pure_records, binary_records)
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
        search_option: IdentifierOption,
    ) -> Result<Self, ParameterError>
    where
        P: AsRef<Path>,
        Self::Pure: FromSegments<usize>,
        Self::IdealGas: FromSegments<usize>,
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
                    .as_string(search_option)
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
        let segment_records: Vec<SegmentRecord<Self::Pure, Self::IdealGas>> =
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
        let binary_records = Array2::from_shape_fn([n, n], |(i, j)| {
            binary_records[(component_list[i], component_list[j])].clone()
        });

        Self::from_records(pure_records, binary_records)
            .expect("failed to create subset from parameters.")
    }
}

/// Constructor methods for parameters for heterosegmented models.
pub trait ParameterHetero: Sized {
    type Chemical: Clone;
    type Pure: Clone + DeserializeOwned;
    type IdealGas: Clone + DeserializeOwned;
    type Binary: Clone + DeserializeOwned;

    /// Creates parameters from the molecular structure and segment information.
    fn from_segments<C: Clone + Into<Self::Chemical>>(
        chemical_records: Vec<C>,
        segment_records: Vec<SegmentRecord<Self::Pure, Self::IdealGas>>,
        binary_segment_records: Option<Vec<BinaryRecord<String, Self::Binary>>>,
    ) -> Result<Self, ParameterError>;

    /// Return the original records that were used to construct the parameters.
    #[allow(clippy::type_complexity)]
    fn records(
        &self,
    ) -> (
        &[Self::Chemical],
        &[SegmentRecord<Self::Pure, Self::IdealGas>],
        &Option<Vec<BinaryRecord<String, Self::Binary>>>,
    );

    /// Creates parameters from segment information stored in json files.
    fn from_json_segments<P>(
        substances: &[&str],
        file_pure: P,
        file_segments: P,
        file_binary: Option<P>,
        search_option: IdentifierOption,
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
                    .as_string(search_option)
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
        let segment_records: Vec<SegmentRecord<Self::Pure, Self::IdealGas>> =
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::joback::JobackRecord;
    use serde::{Deserialize, Serialize};
    use std::convert::TryFrom;

    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    struct MyPureModel {
        a: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
    struct MyBinaryModel {
        b: f64,
    }

    impl TryFrom<f64> for MyBinaryModel {
        type Error = &'static str;
        fn try_from(f: f64) -> Result<Self, Self::Error> {
            Ok(Self { b: f })
        }
    }

    struct MyParameter {
        pure_records: Vec<PureRecord<MyPureModel, JobackRecord>>,
        binary_records: Array2<MyBinaryModel>,
    }

    impl Parameter for MyParameter {
        type Pure = MyPureModel;
        type IdealGas = JobackRecord;
        type Binary = MyBinaryModel;
        fn from_records(
            pure_records: Vec<PureRecord<MyPureModel, JobackRecord>>,
            binary_records: Array2<MyBinaryModel>,
        ) -> Result<Self, ParameterError> {
            Ok(Self {
                pure_records,
                binary_records,
            })
        }

        fn records(
            &self,
        ) -> (
            &[PureRecord<MyPureModel, JobackRecord>],
            &Array2<MyBinaryModel>,
        ) {
            (&self.pure_records, &self.binary_records)
        }
    }

    #[test]
    fn from_records() -> Result<(), ParameterError> {
        let pr_json = r#"
        [
            {
                "identifier": {
                    "cas": "123-4-5"
                },
                "molarweight": 16.0426,
                "model_record": {
                    "a": 0.1
                }
            },
            {
                "identifier": {
                    "cas": "678-9-1"
                },
                "molarweight": 32.08412,
                "model_record": {
                    "a": 0.2
                }
            }
        ]
        "#;
        let br_json = r#"
        [
            {
                "id1": {
                    "cas": "123-4-5"
                },
                "id2": {
                    "cas": "678-9-1"
                },
                "model_record": {
                    "b": 12.0
                }
            }
        ]
        "#;
        let pure_records = serde_json::from_str(pr_json).expect("Unable to parse json.");
        let binary_records: Vec<_> = serde_json::from_str(br_json).expect("Unable to parse json.");
        let binary_matrix = MyParameter::binary_matrix_from_records(
            &pure_records,
            &binary_records,
            IdentifierOption::Cas,
        )?;
        let p = MyParameter::from_records(pure_records, binary_matrix)?;

        assert_eq!(p.pure_records[0].identifier.cas, Some("123-4-5".into()));
        assert_eq!(p.pure_records[1].identifier.cas, Some("678-9-1".into()));
        assert_eq!(p.binary_records[[0, 1]].b, 12.0);
        Ok(())
    }

    #[test]
    fn from_records_missing_binary() -> Result<(), ParameterError> {
        let pr_json = r#"
        [
            {
                "identifier": {
                    "cas": "123-4-5"
                },
                "molarweight": 16.0426,
                "model_record": {
                    "a": 0.1
                }
            },
            {
                "identifier": {
                    "cas": "678-9-1"
                },
                "molarweight": 32.08412,
                "model_record": {
                    "a": 0.2
                }
            }
        ]
        "#;
        let br_json = r#"
        [
            {
                "id1": {
                    "cas": "123-4-5"
                },
                "id2": {
                    "cas": "000-00-0"
                },
                "model_record": {
                    "b": 12.0
                }
            }
        ]
        "#;
        let pure_records = serde_json::from_str(pr_json).expect("Unable to parse json.");
        let binary_records: Vec<_> = serde_json::from_str(br_json).expect("Unable to parse json.");
        let binary_matrix = MyParameter::binary_matrix_from_records(
            &pure_records,
            &binary_records,
            IdentifierOption::Cas,
        )?;
        let p = MyParameter::from_records(pure_records, binary_matrix)?;

        assert_eq!(p.pure_records[0].identifier.cas, Some("123-4-5".into()));
        assert_eq!(p.pure_records[1].identifier.cas, Some("678-9-1".into()));
        assert_eq!(p.binary_records[[0, 1]], MyBinaryModel::default());
        assert_eq!(p.binary_records[[0, 1]].b, 0.0);
        Ok(())
    }

    #[test]
    fn from_records_correct_binary_order() -> Result<(), ParameterError> {
        let pr_json = r#"
        [
            {
                "identifier": {
                    "cas": "000-0-0"
                },
                "molarweight": 32.08412,
                "model_record": {
                    "a": 0.2
                }
            },
            {
                "identifier": {
                    "cas": "123-4-5"
                },
                "molarweight": 16.0426,
                "model_record": {
                    "a": 0.1
                }
            },
            {
                "identifier": {
                    "cas": "678-9-1"
                },
                "molarweight": 32.08412,
                "model_record": {
                    "a": 0.2
                }
            }
        ]
        "#;
        let br_json = r#"
        [
            {
                "id1": {
                    "cas": "123-4-5"
                },
                "id2": {
                    "cas": "678-9-1"
                },
                "model_record": {
                    "b": 12.0
                }
            }
        ]
        "#;
        let pure_records = serde_json::from_str(pr_json).expect("Unable to parse json.");
        let binary_records: Vec<_> = serde_json::from_str(br_json).expect("Unable to parse json.");
        let binary_matrix = MyParameter::binary_matrix_from_records(
            &pure_records,
            &binary_records,
            IdentifierOption::Cas,
        )?;
        let p = MyParameter::from_records(pure_records, binary_matrix)?;

        assert_eq!(p.pure_records[0].identifier.cas, Some("000-0-0".into()));
        assert_eq!(p.pure_records[1].identifier.cas, Some("123-4-5".into()));
        assert_eq!(p.pure_records[2].identifier.cas, Some("678-9-1".into()));
        assert_eq!(p.binary_records[[0, 1]], MyBinaryModel::default());
        assert_eq!(p.binary_records[[1, 0]], MyBinaryModel::default());
        assert_eq!(p.binary_records[[0, 2]], MyBinaryModel::default());
        assert_eq!(p.binary_records[[2, 0]], MyBinaryModel::default());
        assert_eq!(p.binary_records[[2, 1]].b, 12.0);
        assert_eq!(p.binary_records[[1, 2]].b, 12.0);
        Ok(())
    }
}
