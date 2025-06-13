//! Structures and traits that can be used to build model parameters for equations of state.
use crate::errors::*;
use indexmap::IndexSet;
use itertools::Itertools;
use ndarray::{Array1, Array2};
use quantity::{GRAM, MOL, MolarWeight};
use serde::de::DeserializeOwned;
use std::array;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::ops::Deref;
use std::path::Path;

mod association;
mod chemical_record;
mod identifier;
mod model_record;

pub use association::{AssociationRecord, BinaryAssociationRecord};
pub use chemical_record::{ChemicalRecord, GroupCount};
pub use identifier::{Identifier, IdentifierOption};
pub use model_record::{
    BinaryRecord, BinarySegmentRecord, BondRecord, FromSegments, FromSegmentsBinary, ModelRecord,
    PureRecord, SegmentRecord,
};

pub struct ParametersBase<I, P, B, A, Bo, C> {
    pub pure_records: Vec<ModelRecord<I, P, A, C>>,
    pub binary_records: Vec<BinaryRecord<usize, B, A>>,
    pub bond_records: Vec<BondRecord<Bo, C>>,
    pub molar_weight: MolarWeight<Array1<f64>>,
}

pub type Parameters<P, B, A> = ParametersBase<Identifier, P, B, A, (), ()>;
pub type ParametersHetero<P, B, A, Bo, C> = ParametersBase<String, P, B, A, Bo, C>;
pub type IdealGasParameters<I> = Parameters<I, (), ()>;

impl<I: Clone, P: Clone, B: Clone, A: Clone, Bo: Clone, C: Clone>
    ParametersBase<I, P, B, A, Bo, C>
{
    /// Return a parameter set containing the subset of components specified in `component_list`.
    ///
    /// # Panics
    ///
    /// Panics if index in `component_list` is out of bounds
    pub fn subset(&self, component_list: &[usize]) -> Self {
        let segment_list: Vec<_> = (0..self.pure_records.len())
            .filter(|&i| component_list.contains(&self.pure_records[i].component_index))
            .collect();
        let pure_records = segment_list
            .iter()
            .map(|&i| self.pure_records[i].clone())
            .collect();
        let mut binary_records = self.binary_records.clone();
        binary_records.retain(|r| segment_list.contains(&r.id1) && segment_list.contains(&r.id2));
        let mut bond_records = self.bond_records.clone();
        bond_records.retain(|r| segment_list.contains(&r.id1));
        let molar_weight = component_list
            .iter()
            .map(|&i| self.molar_weight.get(i))
            .collect();

        Self {
            pure_records,
            binary_records,
            bond_records,
            molar_weight,
        }
    }

    pub fn collate<F, T: Default + Copy, const N: usize>(&self, f: F) -> [Array1<T>; N]
    where
        F: Fn(&P) -> [T; N],
    {
        array::from_fn(|i| {
            self.pure_records
                .iter()
                .map(|pr| f(&pr.model_record)[i])
                .collect()
        })
    }

    pub fn collate_binary<F, T: Default + Copy, const N: usize>(&self, f: F) -> [Array2<T>; N]
    where
        F: Fn(&Option<B>) -> [T; N],
    {
        array::from_fn(|i| {
            let mut b_mat = Array2::default([self.pure_records.len(); 2]);
            for br in &self.binary_records {
                let b = f(&br.model_record)[i];
                b_mat[[br.id1, br.id2]] = b;
                b_mat[[br.id2, br.id1]] = b;
            }
            b_mat
        })
    }
}

impl<P: Clone, B: Clone, A: Clone> Parameters<P, B, A> {
    pub fn new(
        mut pure_records: Vec<PureRecord<P, A>>,
        binary_records: Vec<BinaryRecord<usize, B, A>>,
    ) -> Self {
        pure_records.iter_mut().enumerate().for_each(|(i, p)| {
            p.component_index = i;
        });
        let molar_weight = pure_records
            .iter()
            .map(|pr| pr.molarweight)
            .collect::<Array1<f64>>()
            * (GRAM / MOL);
        Self {
            pure_records,
            binary_records,
            bond_records: vec![],
            molar_weight,
        }
    }

    /// Creates parameters for a pure component from a pure record.
    pub fn new_pure(pure_record: PureRecord<P, A>) -> Self {
        Self::new(vec![pure_record], vec![])
    }

    /// Creates parameters for a binary system from pure records and an optional
    /// binary interaction parameter.
    pub fn new_binary(
        pure_records: [PureRecord<P, A>; 2],
        binary_record: Option<B>,
        binary_association_records: Vec<BinaryAssociationRecord<A>>,
    ) -> Self {
        let binary_record = vec![BinaryRecord::with_association(
            0,
            1,
            binary_record,
            binary_association_records,
        )];
        Self::new(pure_records.to_vec(), binary_record)
    }

    /// Creates parameters from records for pure substances and possibly binary parameters.
    pub fn from_records(
        pure_records: Vec<PureRecord<P, A>>,
        binary_records: Vec<BinaryRecord<Identifier, B, A>>,
        identifier_option: IdentifierOption,
    ) -> Self {
        let binary_records =
            Self::binary_matrix_from_records(&pure_records, &binary_records, identifier_option);
        Self::new(pure_records, binary_records)
    }

    /// Creates parameters from model records with default values for the molar weight,
    /// identifiers, association sites, and binary interaction parameters.
    pub fn from_model_records(model_records: Vec<P>) -> Self {
        let pure_records = model_records
            .into_iter()
            .map(|r| PureRecord::new(Default::default(), Default::default(), r))
            .collect();
        Self::new(pure_records, vec![])
    }

    /// Helper function to build matrix from list of records in correct order.
    pub fn binary_matrix_from_records(
        pure_records: &[PureRecord<P, A>],
        binary_records: &[BinaryRecord<Identifier, B, A>],
        identifier_option: IdentifierOption,
    ) -> Vec<BinaryRecord<usize, B, A>> {
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
                records.map(|(b, a)| BinaryRecord::with_association(i1, i2, b.clone(), a))
            })
            .collect()
    }

    /// Creates parameters from substance information stored in json files.
    pub fn from_json<F, S>(
        substances: Vec<S>,
        file_pure: F,
        file_binary: Option<F>,
        identifier_option: IdentifierOption,
    ) -> FeosResult<Self>
    where
        F: AsRef<Path>,
        S: Deref<Target = str>,
        P: DeserializeOwned,
        B: DeserializeOwned,
        A: DeserializeOwned,
    {
        Self::from_multiple_json(&[(substances, file_pure)], file_binary, identifier_option)
    }

    /// Creates parameters from substance information stored in multiple json files.
    pub fn from_multiple_json<F, S>(
        input: &[(Vec<S>, F)],
        file_binary: Option<F>,
        identifier_option: IdentifierOption,
    ) -> FeosResult<Self>
    where
        F: AsRef<Path>,
        S: Deref<Target = str>,
        P: DeserializeOwned,
        B: DeserializeOwned,
        A: DeserializeOwned,
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

        let mut records: Vec<PureRecord<P, A>> = Vec::with_capacity(nsubstances);

        // collect parameters from files into single map
        for (substances, file) in input {
            records.extend(PureRecord::<P, A>::from_json(
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
        Ok(Self::from_records(
            records,
            binary_records,
            identifier_option,
        ))
    }

    /// Creates parameters from the molecular structure and segment information.
    ///
    /// The [FromSegments] trait needs to be implemented for the model record
    /// and the binary interaction parameters.
    pub fn from_segments(
        chemical_records: Vec<ChemicalRecord>,
        segment_records: &[SegmentRecord<P, A>],
        binary_segment_records: Option<&[BinarySegmentRecord<B, A>]>,
    ) -> FeosResult<Self>
    where
        P: FromSegments,
        B: FromSegmentsBinary + Default,
    {
        let segment_map: HashMap<_, _> =
            segment_records.iter().map(|s| (&s.identifier, s)).collect();

        // Calculate the pure records from the GC method.
        let (group_counts, pure_records): (Vec<_>, _) = chemical_records
            .into_iter()
            .map(|cr| {
                let (identifier, group_counts, _) = GroupCount::into_groups(cr);
                let groups = group_counts.iter().map(|(s, c)| {
                    let Some(&x) = segment_map.get(s) else {
                        panic!("No segment record found for {s}");
                    };
                    (x.clone(), *c)
                });
                let pure_record = PureRecord::from_segments(identifier, groups)?;
                Ok::<_, FeosError>((group_counts, pure_record))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();

        // Map: (id1, id2) -> model_record
        // empty, if no binary segment records are provided
        let binary_map: HashMap<_, _> = binary_segment_records
            .into_iter()
            .flat_map(|seg| seg.iter())
            .filter_map(|br| br.model_record.as_ref().map(|b| ((&br.id1, &br.id2), b)))
            .collect();

        // full matrix of binary records from the gc method.
        // If a specific segment-segment interaction is not in the binary map,
        // the default value is used.
        let binary_records = group_counts
            .iter()
            .enumerate()
            .array_combinations()
            .map(|[(i, sc1), (j, sc2)]| {
                let mut vec = Vec::new();
                for (id1, n1) in sc1.iter() {
                    for (id2, n2) in sc2.iter() {
                        let binary = binary_map
                            .get(&(id1, id2))
                            .or_else(|| binary_map.get(&(id2, id1)))
                            .copied()
                            .cloned()
                            .unwrap_or_default();
                        vec.push((binary, *n1, *n2));
                    }
                }
                B::from_segments_binary(&vec).map(|br| BinaryRecord::new(i, j, Some(br)))
            })
            .collect::<Result<_, _>>()?;

        Ok(Self::new(pure_records, binary_records))
    }

    /// Creates parameters from segment information stored in json files.
    ///
    /// The [FromSegments] trait needs to be implemented for both the model record
    /// and the ideal gas record.
    pub fn from_json_segments<F>(
        substances: &[&str],
        file_pure: F,
        file_segments: F,
        file_binary: Option<F>,
        identifier_option: IdentifierOption,
    ) -> FeosResult<Self>
    where
        F: AsRef<Path>,
        P: FromSegments + DeserializeOwned,
        B: FromSegmentsBinary + DeserializeOwned + Default,
        A: DeserializeOwned,
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
        let segment_records: Vec<SegmentRecord<P, A>> = SegmentRecord::from_json(file_segments)?;

        // Read binary records
        let binary_records = file_binary
            .map(|file_binary| {
                let reader = BufReader::new(File::open(file_binary)?);
                let binary_records: FeosResult<Vec<BinarySegmentRecord<B, A>>> =
                    Ok(serde_json::from_reader(reader)?);
                binary_records
            })
            .transpose()?;

        Self::from_segments(
            chemical_records,
            &segment_records,
            binary_records.as_deref(),
        )
    }
}

impl<P: Clone, B: Clone, A: Clone, Bo: Clone> ParametersHetero<P, B, A, Bo, f64> {
    pub fn segment_counts(&self) -> Array1<f64> {
        self.pure_records.iter().map(|pr| pr.count).collect()
    }
}

impl<P: Clone, B: Clone, A: Clone, Bo: Clone, C: GroupCount + Default>
    ParametersHetero<P, B, A, Bo, C>
{
    pub fn from_segments(
        chemical_records: Vec<ChemicalRecord>,
        segment_records: &[SegmentRecord<P, A>],
        binary_segment_records: Option<&[BinarySegmentRecord<B, A>]>,
    ) -> Self
    where
        Bo: Default,
    {
        let mut bond_records = Vec::new();
        for s1 in segment_records.iter() {
            for s2 in segment_records.iter() {
                bond_records.push(BinarySegmentRecord::new(
                    s1.identifier.clone(),
                    s2.identifier.clone(),
                    Some(Bo::default()),
                ));
            }
        }
        Self::from_segments_with_bonds(
            chemical_records,
            segment_records,
            binary_segment_records,
            &bond_records,
        )
    }

    pub fn from_segments_with_bonds(
        chemical_records: Vec<ChemicalRecord>,
        segment_records: &[SegmentRecord<P, A>],
        binary_segment_records: Option<&[BinarySegmentRecord<B, A>]>,
        bond_records: &[BinarySegmentRecord<Bo, ()>],
    ) -> Self {
        let segment_map: HashMap<_, _> =
            segment_records.iter().map(|s| (&s.identifier, s)).collect();

        let mut bond_records_map = HashMap::new();
        for bond_record in bond_records {
            bond_records_map.insert((&bond_record.id1, &bond_record.id2), bond_record);
            bond_records_map.insert((&bond_record.id2, &bond_record.id1), bond_record);
        }

        let mut groups = Vec::new();
        let mut bonds = Vec::new();
        let mut molar_weight: Array1<f64> = Array1::zeros(chemical_records.len());
        for (i, cr) in chemical_records.into_iter().enumerate() {
            let (identifier, group_counts, bond_counts) = C::into_groups(cr);
            let n = groups.len();
            for (s, c) in &group_counts {
                let Some(&segment) = segment_map.get(s) else {
                    panic!("No segment record found for {s}");
                };
                molar_weight[i] += segment.molarweight * c.into_f64();
                groups.push(segment.apply_count(*c, i))
            }
            for ([a, b], c) in bond_counts {
                let id1 = &group_counts[a].0;
                let id2 = &group_counts[b].0;
                let Some(&bond) = bond_records_map.get(&(id1, id2)) else {
                    panic!("No bond record found for {id1}-{id2}");
                };
                let Some(bond) = bond.model_record.as_ref() else {
                    panic!("No bond record found for {id1}-{id2}");
                };
                bonds.push(BondRecord::with_count(a + n, b + n, bond.clone(), c));
            }
        }

        let mut binary_records = Vec::new();
        if let Some(binary_segment_records) = binary_segment_records {
            let mut binary_segment_records_map = HashMap::new();
            for binary_record in binary_segment_records {
                binary_segment_records_map
                    .insert((&binary_record.id1, &binary_record.id2), binary_record);
                binary_segment_records_map
                    .insert((&binary_record.id2, &binary_record.id1), binary_record);
            }

            for [(i1, s1), (i2, s2)] in groups.iter().enumerate().array_combinations() {
                if s1.component_index != s2.component_index {
                    let id1 = &s1.identifier;
                    let id2 = &s2.identifier;
                    if let Some(br) = binary_segment_records_map.get(&(id1, id2)) {
                        binary_records.push(BinaryRecord::with_association(
                            i1,
                            i2,
                            br.model_record.clone(),
                            br.association_sites.clone(),
                        ));
                    }
                }
            }
        }

        Self {
            pure_records: groups,
            binary_records,
            bond_records: bonds,
            molar_weight: molar_weight * (GRAM / MOL),
        }
    }

    /// Creates parameters from segment information stored in json files.
    ///
    /// The [FromSegments] trait needs to be implemented for both the model record
    /// and the ideal gas record.
    pub fn from_json_segments<F>(
        substances: &[&str],
        file_pure: F,
        file_segments: F,
        file_binary: Option<F>,
        identifier_option: IdentifierOption,
    ) -> FeosResult<Self>
    where
        F: AsRef<Path>,
        P: DeserializeOwned,
        B: DeserializeOwned + Default,
        A: DeserializeOwned,
        Bo: Default,
    {
        let (chemical_records, segment_records, binary_records) = Self::read_json(
            substances,
            file_pure,
            file_segments,
            file_binary,
            identifier_option,
        )?;

        Ok(Self::from_segments(
            chemical_records,
            &segment_records,
            binary_records.as_deref(),
        ))
    }

    /// Creates parameters from segment information stored in json files.
    ///
    /// The [FromSegments] trait needs to be implemented for both the model record
    /// and the ideal gas record.
    pub fn from_json_segments_with_bonds<F>(
        substances: &[&str],
        file_pure: F,
        file_segments: F,
        file_binary: Option<F>,
        file_bonds: F,
        identifier_option: IdentifierOption,
    ) -> FeosResult<Self>
    where
        F: AsRef<Path>,
        P: DeserializeOwned,
        B: DeserializeOwned + Default,
        A: DeserializeOwned,
        Bo: DeserializeOwned,
    {
        let (chemical_records, segment_records, binary_records) = Self::read_json(
            substances,
            file_pure,
            file_segments,
            file_binary,
            identifier_option,
        )?;

        // Read bond records
        let bond_records: Vec<_> = BinaryRecord::from_json(file_bonds)?;

        Ok(Self::from_segments_with_bonds(
            chemical_records,
            &segment_records,
            binary_records.as_deref(),
            &bond_records,
        ))
    }

    #[expect(clippy::type_complexity)]
    fn read_json<F>(
        substances: &[&str],
        file_pure: F,
        file_segments: F,
        file_binary: Option<F>,
        identifier_option: IdentifierOption,
    ) -> FeosResult<(
        Vec<ChemicalRecord>,
        Vec<SegmentRecord<P, A>>,
        Option<Vec<BinarySegmentRecord<B, A>>>,
    )>
    where
        F: AsRef<Path>,
        P: DeserializeOwned,
        B: DeserializeOwned + Default,
        A: DeserializeOwned,
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
        let chemical_records = queried
            .into_iter()
            .filter_map(|identifier| record_map.remove(identifier))
            .collect();

        // Read segment records
        let segment_records = SegmentRecord::from_json(file_segments)?;

        // Read binary records
        let binary_records = file_binary
            .map(|file_binary| BinaryRecord::from_json(file_binary))
            .transpose()?;

        Ok((chemical_records, segment_records, binary_records))
    }

    pub fn component_index(&self) -> Array1<usize> {
        self.pure_records
            .iter()
            .map(|pr| pr.component_index)
            .collect()
    }
}

// /// Constructor methods for parameters.
// ///
// /// By implementing `Parameter` for a type, you define how parameters
// /// of an equation of state can be constructed from a sequence of
// /// single substance records and possibly binary interaction parameters.
// pub trait Parameter
// where
//     Self: Sized,
// {
//     type Pure: Clone + DeserializeOwned + Serialize;
//     type Binary: Clone + DeserializeOwned + Serialize;
//     type Association: Clone + DeserializeOwned + Serialize;

//     /// Creates parameters from records for pure substances and possibly binary parameters.
//     fn from_records(
//         pure_records: Vec<PureRecord<Self::Pure, Self::Association>>,
//         binary_records: Vec<BinaryRecord<usize, Self::Binary, Self::Association>>,
//     ) -> FeosResult<Self>;

//     /// Creates parameters from records for pure substances and possibly binary parameters.
//     fn from_records2(
//         pure_records: Vec<PureRecord<Self::Pure, Self::Association>>,
//         binary_records: Vec<BinaryRecord<Identifier, Self::Binary, Self::Association>>,
//         identifier_option: IdentifierOption,
//     ) -> FeosResult<Self> {
//         let binary_records =
//             Self::binary_matrix_from_records(&pure_records, &binary_records, identifier_option);
//         Self::from_records(pure_records, binary_records)
//     }

//     /// Creates parameters for a pure component from a pure record.
//     fn new_pure(pure_record: PureRecord<Self::Pure, Self::Association>) -> FeosResult<Self> {
//         Self::from_records(vec![pure_record], vec![])
//     }

//     /// Creates parameters for a binary system from pure records and an optional
//     /// binary interaction parameter.
//     fn new_binary(
//         pure_records: [PureRecord<Self::Pure, Self::Association>; 2],
//         binary_record: Option<Self::Binary>,
//         binary_association_records: Vec<BinaryAssociationRecord<Self::Association>>,
//     ) -> FeosResult<Self> {
//         let binary_record = vec![BinaryRecord::new(
//             0,
//             1,
//             binary_record,
//             binary_association_records,
//         )];
//         Self::from_records(pure_records.to_vec(), binary_record)
//     }

//     /// Creates parameters from model records with default values for the molar weight,
//     /// identifiers, and binary interaction parameters.
//     fn from_model_records(model_records: Vec<Self::Pure>) -> FeosResult<Self> {
//         let pure_records = model_records
//             .into_iter()
//             .map(|r| PureRecord::new(Default::default(), Default::default(), r))
//             .collect();
//         Self::from_records(pure_records, vec![])
//     }

//     /// Return the original pure and binary records that were used to construct the parameters.
//     #[expect(clippy::type_complexity)]
//     fn records(
//         &self,
//     ) -> (
//         &[PureRecord<Self::Pure, Self::Association>],
//         &[BinaryRecord<usize, Self::Binary, Self::Association>],
//     );

//     /// Helper function to build matrix from list of records in correct order.
//     fn binary_matrix_from_records(
//         pure_records: &[PureRecord<Self::Pure, Self::Association>],
//         binary_records: &[BinaryRecord<Identifier, Self::Binary, Self::Association>],
//         identifier_option: IdentifierOption,
//     ) -> Vec<BinaryRecord<usize, Self::Binary, Self::Association>> {
//         // Build Hashmap (id, id) -> BinaryRecord
//         let binary_map: HashMap<_, _> = {
//             binary_records
//                 .iter()
//                 .filter_map(|br| {
//                     let id1 = br.id1.as_str(identifier_option);
//                     let id2 = br.id2.as_str(identifier_option);
//                     id1.and_then(|id1| {
//                         id2.map(|id2| ((id1, id2), (&br.model_record, &br.association_sites)))
//                     })
//                 })
//                 .collect()
//         };

//         // look up pure records in Hashmap
//         pure_records
//             .iter()
//             .enumerate()
//             .array_combinations()
//             .filter_map(|[(i1, p1), (i2, p2)]| {
//                 let id1 = p1.identifier.as_str(identifier_option).unwrap_or_else(|| {
//                     panic!(
//                         "No {} for pure record {} ({}).",
//                         identifier_option, i1, p1.identifier
//                     )
//                 });
//                 let id2 = p2.identifier.as_str(identifier_option).unwrap_or_else(|| {
//                     panic!(
//                         "No {} for pure record {} ({}).",
//                         identifier_option, i2, p2.identifier
//                     )
//                 });

//                 let records = if let Some(&(b, a)) = binary_map.get(&(id1, id2)) {
//                     Some((b, a.clone()))
//                 } else if let Some(&(b, a)) = binary_map.get(&(id2, id1)) {
//                     let a = a
//                         .iter()
//                         .cloned()
//                         .map(|a| BinaryAssociationRecord::with_id(a.id2, a.id1, a.parameters))
//                         .collect();
//                     Some((b, a))
//                 } else {
//                     None
//                 };
//                 records.map(|(b, a)| BinaryRecord::new(i1, i2, b.clone(), a))
//             })
//             .collect()
//     }

//     /// Creates parameters from substance information stored in json files.
//     fn from_json<P, S>(
//         substances: Vec<S>,
//         file_pure: P,
//         file_binary: Option<P>,
//         identifier_option: IdentifierOption,
//     ) -> FeosResult<Self>
//     where
//         P: AsRef<Path>,
//         S: Deref<Target = str>,
//     {
//         Self::from_multiple_json(&[(substances, file_pure)], file_binary, identifier_option)
//     }

//     /// Creates parameters from substance information stored in multiple json files.
//     fn from_multiple_json<P, S>(
//         input: &[(Vec<S>, P)],
//         file_binary: Option<P>,
//         identifier_option: IdentifierOption,
//     ) -> FeosResult<Self>
//     where
//         P: AsRef<Path>,
//         S: Deref<Target = str>,
//     {
//         // total number of substances queried
//         let nsubstances = input
//             .iter()
//             .fold(0, |acc, (substances, _)| acc + substances.len());

//         // queried substances with removed duplicates
//         let queried: IndexSet<String> = input
//             .iter()
//             .flat_map(|(substances, _)| substances)
//             .map(|substance| substance.to_string())
//             .collect();

//         // check if there are duplicates
//         if queried.len() != nsubstances {
//             return Err(FeosError::IncompatibleParameters(
//                 "A substance was defined more than once.".to_string(),
//             ));
//         }

//         let mut records: Vec<PureRecord<Self::Pure, Self::Association>> =
//             Vec::with_capacity(nsubstances);

//         // collect parameters from files into single map
//         for (substances, file) in input {
//             records.extend(PureRecord::<Self::Pure, Self::Association>::from_json(
//                 substances,
//                 file,
//                 identifier_option,
//             )?);
//         }

//         let binary_records = if let Some(path) = file_binary {
//             let file = File::open(path)?;
//             let reader = BufReader::new(file);
//             serde_json::from_reader(reader)?
//         } else {
//             Vec::new()
//         };
//         // let record_matrix =
//         //     Self::binary_matrix_from_records(&records, &binary_records, identifier_option);
//         Self::from_records2(records, binary_records, identifier_option)
//     }

//     /// Creates parameters from the molecular structure and segment information.
//     ///
//     /// The [FromSegments] trait needs to be implemented for both the model record
//     /// and the ideal gas record.
//     fn from_segments<C: SegmentCount>(
//         chemical_records: Vec<C>,
//         segment_records: Vec<SegmentRecord<Self::Pure, Self::Association>>,
//         binary_segment_records: Option<Vec<BinarySegmentRecord<Self::Binary, Self::Association>>>,
//     ) -> FeosResult<Self>
//     where
//         Self::Pure: FromSegments<C::Count>,
//         Self::Binary: FromSegmentsBinary<C::Count> + Default,
//     {
//         // Calculate the pure records from the GC method.
//         let pure_records = chemical_records
//             .iter()
//             .map(|cr| {
//                 cr.segment_map(&segment_records).and_then(|segments| {
//                     PureRecord::from_segments(cr.identifier().into_owned(), segments)
//                 })
//             })
//             .collect::<Result<Vec<_>, _>>()?;

//         // Map: (id1, id2) -> model_record
//         // empty, if no binary segment records are provided
//         let binary_map: HashMap<_, _> = binary_segment_records
//             .into_iter()
//             .flat_map(|seg| seg.into_iter())
//             .filter_map(|br| br.model_record.map(|b| ((br.id1, br.id2), b)))
//             .collect();

//         // For every component:  map: id -> count
//         let segment_counts: Vec<_> = chemical_records
//             .iter()
//             .map(|cr| cr.segment_count())
//             .collect();

//         // full matrix of binary records from the gc method.
//         // If a specific segment-segment interaction is not in the binary map,
//         // the default value is used.
//         let binary_records = segment_counts
//             .iter()
//             .enumerate()
//             .array_combinations()
//             .map(|[(i, sc1), (j, sc2)]| {
//                 let mut vec = Vec::new();
//                 for (id1, &n1) in sc1.iter() {
//                     for (id2, &n2) in sc2.iter() {
//                         let binary = binary_map
//                             .get(&(id1.clone(), id2.clone()))
//                             .or_else(|| binary_map.get(&(id2.clone(), id1.clone())))
//                             .cloned()
//                             .unwrap_or_default();
//                         vec.push((binary, n1, n2));
//                     }
//                 }
//                 Self::Binary::from_segments_binary(&vec)
//                     .map(|br| BinaryRecord::new(i, j, Some(br), vec![]))
//             })
//             .collect::<Result<_, _>>()?;

//         Self::from_records(pure_records, binary_records)
//     }

//     /// Creates parameters from segment information stored in json files.
//     ///
//     /// The [FromSegments] trait needs to be implemented for both the model record
//     /// and the ideal gas record.
//     fn from_json_segments<P>(
//         substances: &[&str],
//         file_pure: P,
//         file_segments: P,
//         file_binary: Option<P>,
//         identifier_option: IdentifierOption,
//     ) -> FeosResult<Self>
//     where
//         P: AsRef<Path>,
//         Self::Pure: FromSegments<usize>,
//         Self::Binary: FromSegmentsBinary<usize> + Default,
//     {
//         let queried: IndexSet<_> = substances.iter().copied().collect();

//         let file = File::open(file_pure)?;
//         let reader = BufReader::new(file);
//         let chemical_records: Vec<ChemicalRecord> = serde_json::from_reader(reader)?;
//         let mut record_map: HashMap<_, _> = chemical_records
//             .into_iter()
//             .filter_map(|record| {
//                 record
//                     .identifier
//                     .as_str(identifier_option)
//                     .map(|i| i.to_owned())
//                     .map(|i| (i, record))
//             })
//             .collect();

//         // Compare queried components and available components
//         let available: IndexSet<_> = record_map
//             .keys()
//             .map(|identifier| identifier as &str)
//             .collect();
//         if !queried.is_subset(&available) {
//             let missing: Vec<_> = queried.difference(&available).cloned().collect();
//             let msg = format!("{:?}", missing);
//             return Err(FeosError::ComponentsNotFound(msg));
//         };

//         // collect all pure records that were queried
//         let chemical_records: Vec<_> = queried
//             .into_iter()
//             .filter_map(|identifier| record_map.remove(identifier))
//             .collect();

//         // Read segment records
//         let segment_records: Vec<SegmentRecord<Self::Pure, Self::Association>> =
//             SegmentRecord::from_json(file_segments)?;

//         // Read binary records
//         let binary_records = file_binary
//             .map(|file_binary| {
//                 let reader = BufReader::new(File::open(file_binary)?);
//                 let binary_records: FeosResult<
//                     Vec<BinarySegmentRecord<Self::Binary, Self::Association>>,
//                 > = Ok(serde_json::from_reader(reader)?);
//                 binary_records
//             })
//             .transpose()?;

//         Self::from_segments(chemical_records, segment_records, binary_records)
//     }

//     /// Return a parameter set containing the subset of components specified in `component_list`.
//     ///
//     /// # Panics
//     ///
//     /// Panics if index in `component_list` is out of bounds or if
//     /// [Parameter::from_records] fails.
//     fn subset(&self, component_list: &[usize]) -> Self {
//         let (pure_records, binary_records) = self.records();
//         let pure_records = component_list
//             .iter()
//             .map(|&i| pure_records[i].clone())
//             .collect();
//         let mut binary_records = binary_records.to_vec();
//         binary_records
//             .retain(|r| component_list.contains(&r.id1) && component_list.contains(&r.id2));

//         Self::from_records(pure_records, binary_records)
//             .expect("failed to create subset from parameters.")
//     }
// }

// /// Dummy struct used for models that do not use binary interaction parameters.
// #[derive(Serialize, Deserialize, Clone, Default)]
// pub struct NoBinaryModelRecord;

// impl<T: Copy> FromSegmentsBinary<T> for NoBinaryModelRecord {
//     fn from_segments_binary(_segments: &[(Self, T, T)]) -> FeosResult<Self> {
//         Ok(Self)
//     }
// }

// impl fmt::Display for NoBinaryModelRecord {
//     fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
//         Ok(())
//     }
// }

// pub struct ParametersHetero<P, Bo, B, A> {
//     pub graph: Vec<UnGraph<SegmentRecord<P, A>, Bo>>,
//     pub binary_records: Vec<BinaryRecord<(usize, NodeIndex), B, A>>,
// }

// type ParametersHetero3<P, Bo, B, A> =
//     Parameters<UnGraph<SegmentRecord<P, A>, Bo>, Vec<BinarySegmentRecord<B, A>>, ()>;

// type MyParametersHetero = ParametersHetero3<f64, (), (), ()>;
// fn tedst() {
//     let x = MyParametersHetero::from_segments(vec![], vec![], None);
// }

// pub struct ParametersHetero2<P, Bo, B, A> {
//     pub group_counts: IndexSet<SegmentRecord<P, A>, f64>,
//     pub bond_counts: IndexSet<BinaryRecord<usize, Bo, ()>, f64>,
//     pub binary_records: Vec<BinaryRecord<usize, B, A>>,
// }

// impl<P, Bo, B, A> ParametersHetero<P, Bo, B, A> {
//     pub fn new(
//         graph: UnGraph<SegmentRecord<P, A>, Bo>,
//         binary_records: Vec<BinaryRecord<NodeIndex, B, A>>,
//     ) -> Self {
//         Self {
//             graph,
//             binary_records,
//         }
//     }
// }

// impl<P: Clone, B: Clone, A: Clone> ParametersHetero<P, (), B, A> {
//     pub fn from_segments(
//         chemical_records: &[ChemicalRecord],
//         segment_records: &[SegmentRecord<P, A>],
//         binary_segment_records: Option<&[BinarySegmentRecord<B, A>]>,
//     ) -> Self {
//         let segment_map: HashMap<_, _> =
//             segment_records.iter().map(|s| (&s.identifier, s)).collect();
//         let graph: Vec<_> = chemical_records
//             .iter()
//             .map(|cr| {
//                 let mut graph = UnGraph::new_undirected();
//                 let nodes: Vec<_> = cr
//                     .segments
//                     .iter()
//                     .map(|s| graph.add_node((segment_map[&s].clone(), 1.0)))
//                     .collect();
//                 for &[a, b] in &cr.bonds {
//                     graph.add_edge(nodes[a], nodes[b], ((), 1.0));
//                 }
//                 graph
//             })
//             .collect();

//         let binary_records = Self::binary_records(&graph, binary_segment_records);

//         Self {
//             graph,
//             binary_records,
//         }
//     }

//     pub fn from_segments_counts(
//         chemical_records: &[ChemicalRecord],
//         segment_records: &[SegmentRecord<P, A>],
//         binary_segment_records: Option<&[BinarySegmentRecord<B, A>]>,
//     ) -> Self {
//         let segment_map: HashMap<_, _> =
//             segment_records.iter().map(|s| (&s.identifier, s)).collect();
//         let graph: Vec<_> = chemical_records
//             .iter()
//             .map(|cr| {
//                 let group_map = HashMap::new();
//                 let mut graph = UnGraph::new_undirected();
//                 let nodes: Vec<_> = cr
//                     .segments
//                     .iter()
//                     .map(|s| graph.add_node((segment_map[&s].clone(), 1.0)))
//                     .collect();
//                 for &[a, b] in &cr.bonds {
//                     graph.add_edge(nodes[a], nodes[b], ((), 1.0));
//                 }
//                 graph
//             })
//             .collect();

//         let binary_records = Self::binary_records(&graph, binary_segment_records);

//         Self {
//             graph,
//             binary_records,
//         }
//     }

//     /// Creates parameters from segment information stored in json files.
//     pub fn from_json_segments<F: AsRef<Path>>(
//         substances: &[&str],
//         file_pure: F,
//         file_segments: F,
//         file_binary: Option<F>,
//         identifier_option: IdentifierOption,
//     ) -> FeosResult<Self>
//     where
//         P: DeserializeOwned,
//         B: DeserializeOwned,
//         A: DeserializeOwned,
//     {
//         let queried: IndexSet<_> = substances
//             .iter()
//             .map(|identifier| identifier as &str)
//             .collect();

//         let reader = BufReader::new(File::open(file_pure)?);
//         let chemical_records: Vec<ChemicalRecord> = serde_json::from_reader(reader)?;
//         let mut record_map: IndexMap<_, _> = chemical_records
//             .into_iter()
//             .filter_map(|record| {
//                 record
//                     .identifier
//                     .as_str(identifier_option)
//                     .map(|i| i.to_owned())
//                     .map(|i| (i, record))
//             })
//             .collect();

//         // Compare queried components and available components
//         let available: IndexSet<_> = record_map
//             .keys()
//             .map(|identifier| identifier as &str)
//             .collect();
//         if !queried.is_subset(&available) {
//             let missing: Vec<_> = queried.difference(&available).cloned().collect();
//             return Err(FeosError::ComponentsNotFound(format!("{:?}", missing)));
//         };

//         // Collect all pure records that were queried
//         let chemical_records: Vec<_> = queried
//             .into_iter()
//             .filter_map(|identifier| record_map.shift_remove(identifier))
//             .collect();

//         // Read segment records
//         let segment_records: Vec<SegmentRecord<P, A>> = SegmentRecord::from_json(file_segments)?;

//         // Read binary records
//         let binary_records = file_binary
//             .map(|file_binary| {
//                 let reader = BufReader::new(File::open(file_binary)?);
//                 let binary_records: FeosResult<Vec<BinarySegmentRecord<B, A>>> =
//                     Ok(serde_json::from_reader(reader)?);
//                 binary_records
//             })
//             .transpose()?;

//         Ok(Self::from_segments(
//             &chemical_records,
//             &segment_records,
//             binary_records.as_deref(),
//         ))
//     }
// }

// impl<P: Clone, Bo: Clone, B: Clone, A: Clone> ParametersHetero<P, Bo, B, A> {
//     pub fn from_segments_with_bonds(
//         chemical_records: &[ChemicalRecord],
//         segment_records: &[SegmentRecord<P, A>],
//         bond_records: &[BinarySegmentRecord<Bo, ()>],
//         binary_segment_records: Option<&[BinarySegmentRecord<B, A>]>,
//     ) -> Self {
//         let segment_map: HashMap<_, _> =
//             segment_records.iter().map(|s| (&s.identifier, s)).collect();

//         let mut bond_records_map = HashMap::new();
//         for bond_record in bond_records {
//             bond_records_map.insert((&bond_record.id1, &bond_record.id2), bond_record);
//             bond_records_map.insert((&bond_record.id2, &bond_record.id1), bond_record);
//         }

//         let graph: Vec<_> = chemical_records
//             .iter()
//             .map(|cr| {
//                 let mut graph = UnGraph::new_undirected();
//                 let nodes: Vec<_> = cr
//                     .segments
//                     .iter()
//                     .map(|s| graph.add_node((segment_map[&s].clone(), 1.0)))
//                     .collect();
//                 for &[a, b] in &cr.bonds {
//                     let id1 = &graph[nodes[a]].0.identifier;
//                     let id2 = &graph[nodes[b]].0.identifier;
//                     let Some(&bond) = bond_records_map.get(&(id1, id2)) else {
//                         panic!("No bond record found for {id1}-{id2}");
//                     };
//                     let Some(bond) = bond.model_record.as_ref() else {
//                         panic!("No bond record found for {id1}-{id2}");
//                     };
//                     graph.add_edge(nodes[a], nodes[b], (bond.clone(), 1.0));
//                 }
//                 graph
//             })
//             .collect();

//         let binary_records = Self::binary_records(&graph, binary_segment_records);

//         Self {
//             graph,
//             binary_records,
//         }
//     }

//     fn binary_records(
//         graph: &[UnGraph<(SegmentRecord<P, A>, f64), (Bo, f64)>],
//         binary_segment_records: Option<&[BinarySegmentRecord<B, A>]>,
//     ) -> Vec<BinaryRecord<(usize, NodeIndex), B, A>> {
//         let mut binary_records = Vec::new();
//         if let Some(binary_segment_records) = binary_segment_records {
//             let mut binary_segment_records_map = HashMap::new();
//             for binary_record in binary_segment_records {
//                 binary_segment_records_map
//                     .insert((&binary_record.id1, &binary_record.id2), binary_record);
//                 binary_segment_records_map
//                     .insert((&binary_record.id2, &binary_record.id1), binary_record);
//             }

//             for [(c1, n1), (c2, n2)] in graph
//                 .iter()
//                 .enumerate()
//                 .flat_map(|(i, g)| iter::repeat(i).zip(g.node_indices()))
//                 .array_combinations()
//             {
//                 if c1 != c2 {
//                     let id1 = &graph[c1][n1].0.identifier;
//                     let id2 = &graph[c2][n2].0.identifier;
//                     if let Some(br) = binary_segment_records_map.get(&(id1, id2)) {
//                         binary_records.push(BinaryRecord::new(
//                             (c1, n1),
//                             (c2, n2),
//                             br.model_record.clone(),
//                             br.association_sites.clone(),
//                         ));
//                     }
//                 }
//             }
//         }
//         binary_records
//     }

//     /// Return a parameter set containing the subset of components specified in `component_list`.
//     pub fn subset(&self, component_list: &[usize]) -> Self {
//         let graph = component_list
//             .iter()
//             .map(|&i| self.graph[i].clone())
//             .collect();
//         let mut binary_records = self.binary_records.to_vec();
//         binary_records
//             .retain(|r| component_list.contains(&r.id1.0) && component_list.contains(&r.id2.0));

//         Self {
//             graph,
//             binary_records,
//         }
//     }

//     pub fn into_group_counts(self) -> Self {
//         let groups = self.graph;
//     }

//     fn segments_to_group_counts(
//         graph: UnGraph<(SegmentRecord<P, A>, ()), (Bo, f64)>,
//     ) -> UnGraph<(SegmentRecord<P, A>, f64), (Bo, f64)> {
//         let groups: HashSet<_> = graph.node_weights().map(|(s, _)| &s.identifier).collect();
//         let new_graph = UnGraph::new_undirected();
//         for g in groups {
//             new_graph.add_node(weight)
//         }
//     }

//     pub fn molar_weight(&self) -> Array1<f64> {
//         self.graph
//             .iter()
//             .map(|g| g.node_weights().map(|pr| pr.0.molarweight).sum())
//             .collect()
//     }

//     pub fn collate<F, T: Default + Copy, const N: usize>(&self, f: F) -> [Array1<T>; N]
//     where
//         F: Fn(&P) -> [T; N],
//     {
//         array::from_fn(|i| {
//             self.graph
//                 .iter()
//                 .flat_map(|g| g.node_weights().map(|pr| f(&pr.0.model_record)[i]))
//                 .collect()
//         })
//     }

//     pub fn collate_binary<F, T: Default + Copy, const N: usize>(&self, f: F) -> [Array2<T>; N]
//     where
//         F: Fn(&Option<B>) -> [T; N],
//     {
//         let map: HashMap<_, _> = self
//             .graph
//             .iter()
//             .enumerate()
//             .flat_map(|(i, g)| g.node_indices().map(move |id| (i, id)))
//             .enumerate()
//             .map(|(i, x)| (x, i))
//             .collect();
//         let n = self.graph.iter().map(|g| g.node_count()).sum();
//         array::from_fn(|i| {
//             let mut b_mat = Array2::default([n; 2]);
//             for br in &self.binary_records {
//                 let b = f(&br.model_record)[i];
//                 b_mat[[map[&br.id1], map[&br.id2]]] = b;
//                 b_mat[[map[&br.id2], map[&br.id1]]] = b;
//             }
//             b_mat
//         })
//     }
// }

// /// Constructor methods for parameters for heterosegmented models.
// pub trait ParameterHetero: Sized {
//     type Chemical: Clone;
//     type Pure: Clone + DeserializeOwned;
//     type Binary: Clone + DeserializeOwned;
//     type Association: Clone + DeserializeOwned;

//     /// Creates parameters from the molecular structure and segment information.
//     fn from_segments<C: Clone + Into<Self::Chemical>>(
//         chemical_records: Vec<C>,
//         segment_records: Vec<SegmentRecord<Self::Pure, Self::Association>>,
//         binary_segment_records: Option<Vec<BinarySegmentRecord<Self::Binary, Self::Association>>>,
//     ) -> FeosResult<Self>;

//     /// Return the original records that were used to construct the parameters.
//     #[expect(clippy::type_complexity)]
//     fn records(
//         &self,
//     ) -> (
//         &[Self::Chemical],
//         &[SegmentRecord<Self::Pure, Self::Association>],
//         &Option<Vec<BinarySegmentRecord<Self::Binary, Self::Association>>>,
//     );

//     /// Creates parameters from segment information stored in json files.
//     fn from_json_segments<P>(
//         substances: &[&str],
//         file_pure: P,
//         file_segments: P,
//         file_binary: Option<P>,
//         identifier_option: IdentifierOption,
//     ) -> FeosResult<Self>
//     where
//         P: AsRef<Path>,
//         ChemicalRecord: Into<Self::Chemical>,
//     {
//         let queried: IndexSet<_> = substances
//             .iter()
//             .map(|identifier| identifier as &str)
//             .collect();

//         let reader = BufReader::new(File::open(file_pure)?);
//         let chemical_records: Vec<ChemicalRecord> = serde_json::from_reader(reader)?;
//         let mut record_map: IndexMap<_, _> = chemical_records
//             .into_iter()
//             .filter_map(|record| {
//                 record
//                     .identifier
//                     .as_str(identifier_option)
//                     .map(|i| i.to_owned())
//                     .map(|i| (i, record))
//             })
//             .collect();

//         // Compare queried components and available components
//         let available: IndexSet<_> = record_map
//             .keys()
//             .map(|identifier| identifier as &str)
//             .collect();
//         if !queried.is_subset(&available) {
//             let missing: Vec<_> = queried.difference(&available).cloned().collect();
//             return Err(FeosError::ComponentsNotFound(format!("{:?}", missing)));
//         };

//         // Collect all pure records that were queried
//         let chemical_records: Vec<_> = queried
//             .into_iter()
//             .filter_map(|identifier| record_map.shift_remove(identifier))
//             .collect();

//         // Read segment records
//         let segment_records: Vec<SegmentRecord<Self::Pure, Self::Association>> =
//             SegmentRecord::from_json(file_segments)?;

//         // Read binary records
//         let binary_records = file_binary
//             .map(|file_binary| {
//                 let reader = BufReader::new(File::open(file_binary)?);
//                 let binary_records: FeosResult<
//                     Vec<BinarySegmentRecord<Self::Binary, Self::Association>>,
//                 > = Ok(serde_json::from_reader(reader)?);
//                 binary_records
//             })
//             .transpose()?;

//         Self::from_segments(chemical_records, segment_records, binary_records)
//     }

//     /// Return a parameter set containing the subset of components specified in `component_list`.
//     fn subset(&self, component_list: &[usize]) -> Self {
//         let (chemical_records, segment_records, binary_segment_records) = self.records();
//         let chemical_records: Vec<_> = component_list
//             .iter()
//             .map(|&i| chemical_records[i].clone())
//             .collect();
//         Self::from_segments(
//             chemical_records,
//             segment_records.to_vec(),
//             binary_segment_records.clone(),
//         )
//         .unwrap()
//     }
// }
