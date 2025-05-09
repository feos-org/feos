//! Structures and traits that can be used to build model parameters for equations of state.

use crate::errors::*;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::ops::Deref;
use std::path::Path;

mod association;
mod chemical_record;
mod identifier;
mod model_record;
// mod segment;

pub use association::{AssociationRecord, BinaryAssociationRecord};
pub use chemical_record::{ChemicalRecord, CountType, SegmentCount};
pub use identifier::{Identifier, IdentifierOption};
pub use model_record::{
    BinaryRecord, BinarySegmentRecord, FromSegments, FromSegmentsBinary, PureRecord, SegmentRecord,
};
// pub use segment::{BinarySegmentRecord, SegmentRecord};

/// Constructor methods for parameters.
///
/// By implementing `Parameter` for a type, you define how parameters
/// of an equation of state can be constructed from a sequence of
/// single substance records and possibly binary interaction parameters.
pub trait Parameter
where
    Self: Sized,
{
    type Pure: Clone + DeserializeOwned + Serialize;
    type Binary: Clone + DeserializeOwned + Serialize;
    type Association: Clone + DeserializeOwned + Serialize;

    /// Creates parameters from records for pure substances and possibly binary parameters.
    fn from_records(
        pure_records: Vec<PureRecord<Self::Pure, Self::Association>>,
        binary_records: Vec<BinaryRecord<usize, Self::Binary, Self::Association>>,
    ) -> FeosResult<Self>;

    /// Creates parameters from records for pure substances and possibly binary parameters.
    fn from_records2(
        pure_records: Vec<PureRecord<Self::Pure, Self::Association>>,
        binary_records: Vec<BinaryRecord<Identifier, Self::Binary, Self::Association>>,
        identifier_option: IdentifierOption,
    ) -> FeosResult<Self> {
        let binary_records =
            Self::binary_matrix_from_records(&pure_records, &binary_records, identifier_option);
        Self::from_records(pure_records, binary_records)
    }

    /// Creates parameters for a pure component from a pure record.
    fn new_pure(pure_record: PureRecord<Self::Pure, Self::Association>) -> FeosResult<Self> {
        Self::from_records(vec![pure_record], vec![])
    }

    /// Creates parameters for a binary system from pure records and an optional
    /// binary interaction parameter.
    fn new_binary(
        pure_records: [PureRecord<Self::Pure, Self::Association>; 2],
        binary_record: Option<Self::Binary>,
        binary_association_records: Vec<BinaryAssociationRecord<Self::Association>>,
    ) -> FeosResult<Self> {
        let binary_record = vec![BinaryRecord::new(
            0,
            1,
            binary_record,
            binary_association_records,
        )];
        Self::from_records(pure_records.to_vec(), binary_record)
    }

    /// Creates parameters from model records with default values for the molar weight,
    /// identifiers, and binary interaction parameters.
    fn from_model_records(model_records: Vec<Self::Pure>) -> FeosResult<Self> {
        let pure_records = model_records
            .into_iter()
            .map(|r| PureRecord::new(Default::default(), Default::default(), r))
            .collect();
        Self::from_records(pure_records, vec![])
    }

    /// Return the original pure and binary records that were used to construct the parameters.
    #[expect(clippy::type_complexity)]
    fn records(
        &self,
    ) -> (
        &[PureRecord<Self::Pure, Self::Association>],
        &[BinaryRecord<usize, Self::Binary, Self::Association>],
    );

    /// Helper function to build matrix from list of records in correct order.
    fn binary_matrix_from_records(
        pure_records: &[PureRecord<Self::Pure, Self::Association>],
        binary_records: &[BinaryRecord<Identifier, Self::Binary, Self::Association>],
        identifier_option: IdentifierOption,
    ) -> Vec<BinaryRecord<usize, Self::Binary, Self::Association>> {
        // Build Hashmap (id, id) -> BinaryRecord
        let binary_map: HashMap<_, _> = {
            binary_records
                .iter()
                .filter_map(|br| {
                    let id1 = br.id1.as_str(identifier_option);
                    let id2 = br.id2.as_str(identifier_option);
                    id1.and_then(|id1| {
                        id2.map(|id2| ((id1, id2), (&br.model_record, &br.association_sites)))
                    })
                })
                .collect()
        };

        // look up pure records in Hashmap
        pure_records
            .iter()
            .enumerate()
            .array_combinations()
            .filter_map(|[(i1, p1), (i2, p2)]| {
                let id1 = p1.identifier.as_str(identifier_option).unwrap_or_else(|| {
                    panic!(
                        "No {} for pure record {} ({}).",
                        identifier_option, i1, p1.identifier
                    )
                });
                let id2 = p2.identifier.as_str(identifier_option).unwrap_or_else(|| {
                    panic!(
                        "No {} for pure record {} ({}).",
                        identifier_option, i2, p2.identifier
                    )
                });

                let records = if let Some(&(b, a)) = binary_map.get(&(id1, id2)) {
                    Some((b, a.clone()))
                } else if let Some(&(b, a)) = binary_map.get(&(id2, id1)) {
                    let a = a
                        .iter()
                        .cloned()
                        .map(|a| BinaryAssociationRecord::with_id(a.id2, a.id1, a.parameters))
                        .collect();
                    Some((b, a))
                } else {
                    None
                };
                records.map(|(b, a)| BinaryRecord::new(i1, i2, b.clone(), a))
            })
            .collect()
    }

    /// Creates parameters from substance information stored in json files.
    fn from_json<P, S>(
        substances: Vec<S>,
        file_pure: P,
        file_binary: Option<P>,
        identifier_option: IdentifierOption,
    ) -> FeosResult<Self>
    where
        P: AsRef<Path>,
        S: Deref<Target = str>,
    {
        Self::from_multiple_json(&[(substances, file_pure)], file_binary, identifier_option)
    }

    /// Creates parameters from substance information stored in multiple json files.
    fn from_multiple_json<P, S>(
        input: &[(Vec<S>, P)],
        file_binary: Option<P>,
        identifier_option: IdentifierOption,
    ) -> FeosResult<Self>
    where
        P: AsRef<Path>,
        S: Deref<Target = str>,
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
            return Err(FeosError::IncompatibleParameters(
                "A substance was defined more than once.".to_string(),
            ));
        }

        let mut records: Vec<PureRecord<Self::Pure, Self::Association>> =
            Vec::with_capacity(nsubstances);

        // collect parameters from files into single map
        for (substances, file) in input {
            records.extend(PureRecord::<Self::Pure, Self::Association>::from_json(
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
        // let record_matrix =
        //     Self::binary_matrix_from_records(&records, &binary_records, identifier_option);
        Self::from_records2(records, binary_records, identifier_option)
    }

    /// Creates parameters from the molecular structure and segment information.
    ///
    /// The [FromSegments] trait needs to be implemented for both the model record
    /// and the ideal gas record.
    fn from_segments<C: SegmentCount>(
        chemical_records: Vec<C>,
        segment_records: Vec<SegmentRecord<Self::Pure, Self::Association>>,
        binary_segment_records: Option<Vec<BinarySegmentRecord<Self::Binary, Self::Association>>>,
    ) -> FeosResult<Self>
    where
        Self::Pure: FromSegments<C::Count>,
        Self::Binary: FromSegmentsBinary<C::Count> + Default,
    {
        // Calculate the pure records from the GC method.
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
            .filter_map(|br| br.model_record.map(|b| ((br.id1, br.id2), b)))
            .collect();

        // For every component:  map: id -> count
        let segment_counts: Vec<_> = chemical_records
            .iter()
            .map(|cr| cr.segment_count())
            .collect();

        // full matrix of binary records from the gc method.
        // If a specific segment-segment interaction is not in the binary map,
        // the default value is used.
        let binary_records = segment_counts
            .iter()
            .enumerate()
            .array_combinations()
            .map(|[(i, sc1), (j, sc2)]| {
                let mut vec = Vec::new();
                for (id1, &n1) in sc1.iter() {
                    for (id2, &n2) in sc2.iter() {
                        let binary = binary_map
                            .get(&(id1.clone(), id2.clone()))
                            .or_else(|| binary_map.get(&(id2.clone(), id1.clone())))
                            .cloned()
                            .unwrap_or_default();
                        vec.push((binary, n1, n2));
                    }
                }
                Self::Binary::from_segments_binary(&vec)
                    .map(|br| BinaryRecord::new(i, j, Some(br), vec![]))
            })
            .collect::<Result<_, _>>()?;

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
        identifier_option: IdentifierOption,
    ) -> FeosResult<Self>
    where
        P: AsRef<Path>,
        Self::Pure: FromSegments<usize>,
        Self::Binary: FromSegmentsBinary<usize> + Default,
    {
        let queried: IndexSet<_> = substances.iter().copied().collect();

        let file = File::open(file_pure)?;
        let reader = BufReader::new(file);
        let chemical_records: Vec<ChemicalRecord> = serde_json::from_reader(reader)?;
        let mut record_map: HashMap<_, _> = chemical_records
            .into_iter()
            .filter_map(|record| {
                record
                    .identifier
                    .as_str(identifier_option)
                    .map(|i| i.to_owned())
                    .map(|i| (i, record))
            })
            .collect();

        // Compare queried components and available components
        let available: IndexSet<_> = record_map
            .keys()
            .map(|identifier| identifier as &str)
            .collect();
        if !queried.is_subset(&available) {
            let missing: Vec<_> = queried.difference(&available).cloned().collect();
            let msg = format!("{:?}", missing);
            return Err(FeosError::ComponentsNotFound(msg));
        };

        // collect all pure records that were queried
        let chemical_records: Vec<_> = queried
            .into_iter()
            .filter_map(|identifier| record_map.remove(identifier))
            .collect();

        // Read segment records
        let segment_records: Vec<SegmentRecord<Self::Pure, Self::Association>> =
            SegmentRecord::from_json(file_segments)?;

        // Read binary records
        let binary_records = file_binary
            .map(|file_binary| {
                let reader = BufReader::new(File::open(file_binary)?);
                let binary_records: FeosResult<
                    Vec<BinarySegmentRecord<Self::Binary, Self::Association>>,
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
        let mut binary_records = binary_records.to_vec();
        binary_records
            .retain(|r| component_list.contains(&r.id1) && component_list.contains(&r.id2));

        Self::from_records(pure_records, binary_records)
            .expect("failed to create subset from parameters.")
    }
}

/// Dummy struct used for models that do not use binary interaction parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct NoBinaryModelRecord;

impl<T: Copy> FromSegmentsBinary<T> for NoBinaryModelRecord {
    fn from_segments_binary(_segments: &[(Self, T, T)]) -> FeosResult<Self> {
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
    type Association: Clone + DeserializeOwned;

    /// Creates parameters from the molecular structure and segment information.
    fn from_segments<C: Clone + Into<Self::Chemical>>(
        chemical_records: Vec<C>,
        segment_records: Vec<SegmentRecord<Self::Pure, Self::Association>>,
        binary_segment_records: Option<Vec<BinarySegmentRecord<Self::Binary, Self::Association>>>,
    ) -> FeosResult<Self>;

    /// Return the original records that were used to construct the parameters.
    #[expect(clippy::type_complexity)]
    fn records(
        &self,
    ) -> (
        &[Self::Chemical],
        &[SegmentRecord<Self::Pure, Self::Association>],
        &Option<Vec<BinarySegmentRecord<Self::Binary, Self::Association>>>,
    );

    /// Creates parameters from segment information stored in json files.
    fn from_json_segments<P>(
        substances: &[&str],
        file_pure: P,
        file_segments: P,
        file_binary: Option<P>,
        identifier_option: IdentifierOption,
    ) -> FeosResult<Self>
    where
        P: AsRef<Path>,
        ChemicalRecord: Into<Self::Chemical>,
    {
        let queried: IndexSet<_> = substances
            .iter()
            .map(|identifier| identifier as &str)
            .collect();

        let reader = BufReader::new(File::open(file_pure)?);
        let chemical_records: Vec<ChemicalRecord> = serde_json::from_reader(reader)?;
        let mut record_map: IndexMap<_, _> = chemical_records
            .into_iter()
            .filter_map(|record| {
                record
                    .identifier
                    .as_str(identifier_option)
                    .map(|i| i.to_owned())
                    .map(|i| (i, record))
            })
            .collect();

        // Compare queried components and available components
        let available: IndexSet<_> = record_map
            .keys()
            .map(|identifier| identifier as &str)
            .collect();
        if !queried.is_subset(&available) {
            let missing: Vec<_> = queried.difference(&available).cloned().collect();
            return Err(FeosError::ComponentsNotFound(format!("{:?}", missing)));
        };

        // Collect all pure records that were queried
        let chemical_records: Vec<_> = queried
            .into_iter()
            .filter_map(|identifier| record_map.shift_remove(identifier))
            .collect();

        // Read segment records
        let segment_records: Vec<SegmentRecord<Self::Pure, Self::Association>> =
            SegmentRecord::from_json(file_segments)?;

        // Read binary records
        let binary_records = file_binary
            .map(|file_binary| {
                let reader = BufReader::new(File::open(file_binary)?);
                let binary_records: FeosResult<
                    Vec<BinarySegmentRecord<Self::Binary, Self::Association>>,
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
