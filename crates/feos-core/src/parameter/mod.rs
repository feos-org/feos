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

pub use association::{AssociationParameters, AssociationRecord, BinaryAssociationRecord};
pub use chemical_record::{ChemicalRecord, GroupCount};
pub use identifier::{Identifier, IdentifierOption};
pub use model_record::{
    BinaryRecord, BinarySegmentRecord, FromSegments, FromSegmentsBinary, ModelRecord, PureRecord,
    SegmentRecord,
};

#[derive(Clone)]
pub struct Pure<M, C> {
    pub identifier: String,
    pub model_record: M,
    pub count: C,
    pub component_index: usize,
}

#[derive(Clone, Copy)]
pub struct Binary<B, C> {
    pub id1: usize,
    pub id2: usize,
    pub model_record: B,
    pub count: C,
}

impl<B, C> Binary<B, C> {
    pub fn new(id1: usize, id2: usize, model_record: B, count: C) -> Self {
        Self {
            id1,
            id2,
            model_record,
            count,
        }
    }
}

impl<M: Clone> Pure<M, ()> {
    fn from_pure_record(model_record: M, component_index: usize) -> Self {
        Self {
            identifier: "".into(),
            model_record,
            count: (),
            component_index,
        }
    }
}

impl<M: Clone, C> Pure<M, C> {
    fn from_segment_record<A>(
        segment: &SegmentRecord<M, A>,
        count: C,
        component_index: usize,
    ) -> Self {
        Self {
            identifier: segment.identifier.clone(),
            model_record: segment.model_record.clone(),
            count,
            component_index,
        }
    }
}

pub struct ParametersBase<P, B, A, Bo, C> {
    pub pure: Vec<Pure<P, C>>,
    pub binary: Vec<Binary<B, ()>>,
    pub bonds: Vec<Binary<Bo, C>>,
    pub association: AssociationParameters<A>,
    pub identifiers: Vec<Identifier>,
    pub molar_weight: MolarWeight<Array1<f64>>,
}

pub type Parameters<P, B, A> = ParametersBase<P, B, A, (), ()>;
pub type ParametersHetero<P, B, A, Bo, C> = ParametersBase<P, B, A, Bo, C>;
pub type IdealGasParameters<I> = Parameters<I, (), ()>;

impl<P: Clone, B: Clone, A: Clone, Bo: Clone, C: Clone> ParametersBase<P, B, A, Bo, C> {
    /// Return a parameter set containing the subset of components specified in `component_list`.
    ///
    /// # Panics
    ///
    /// Panics if index in `component_list` is out of bounds
    pub fn subset(&self, component_list: &[usize]) -> Self {
        let segment_list: Vec<_> = (0..self.pure.len())
            .filter(|&i| component_list.contains(&self.pure[i].component_index))
            .collect();
        let pure_records = segment_list.iter().map(|&i| self.pure[i].clone()).collect();
        let mut binary_records = self.binary.clone();
        binary_records.retain(|r| segment_list.contains(&r.id1) && segment_list.contains(&r.id2));

        let mut bond_records = self.bonds.clone();
        bond_records.retain(|r| segment_list.contains(&r.id1));

        let association_parameters = self.association.subset(component_list);
        let identifiers = component_list
            .iter()
            .map(|&i| self.identifiers[i].clone())
            .collect();
        let molar_weight = component_list
            .iter()
            .map(|&i| self.molar_weight.get(i))
            .collect();

        Self {
            pure: pure_records,
            binary: binary_records,
            bonds: bond_records,
            association: association_parameters,
            identifiers,
            molar_weight,
        }
    }

    pub fn collate<F, T: Default + Copy, const N: usize>(&self, f: F) -> [Array1<T>; N]
    where
        F: Fn(&P) -> [T; N],
    {
        array::from_fn(|i| self.pure.iter().map(|pr| f(&pr.model_record)[i]).collect())
    }

    pub fn collate_binary<F, T: Default + Copy, const N: usize>(&self, f: F) -> [Array2<T>; N]
    where
        F: Fn(&B) -> [T; N],
    {
        array::from_fn(|i| {
            let mut b_mat = Array2::default([self.pure.len(); 2]);
            for br in &self.binary {
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
        pure_records: Vec<PureRecord<P, A>>,
        binary_records: Vec<BinaryRecord<usize, B, A>>,
    ) -> FeosResult<Self> {
        let association_parameters = AssociationParameters::new(&pure_records, &binary_records)?;
        let (identifiers, pure_records): (Vec<_>, _) = pure_records
            .into_iter()
            .enumerate()
            .map(|(i, pr)| {
                (
                    (pr.identifier, pr.molarweight),
                    Pure::from_pure_record(pr.model_record, i),
                )
            })
            .unzip();
        let (identifiers, molar_weight): (_, Vec<_>) = identifiers.into_iter().unzip();
        let binary_records = binary_records
            .into_iter()
            .filter_map(|br| br.model_record.map(|m| Binary::new(br.id1, br.id2, m, ())))
            .collect();

        Ok(Self {
            pure: pure_records,
            binary: binary_records,
            bonds: vec![],
            association: association_parameters,
            identifiers,
            molar_weight: Array1::from_vec(molar_weight) * (GRAM / MOL),
        })
    }

    /// Creates parameters for a pure component from a pure record.
    pub fn new_pure(pure_record: PureRecord<P, A>) -> FeosResult<Self> {
        Self::new(vec![pure_record], vec![])
    }

    /// Creates parameters for a binary system from pure records and an optional
    /// binary interaction parameter.
    pub fn new_binary(
        pure_records: [PureRecord<P, A>; 2],
        binary_record: Option<B>,
        binary_association_records: Vec<BinaryAssociationRecord<A>>,
    ) -> FeosResult<Self> {
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
    ) -> FeosResult<Self> {
        let binary_records =
            Self::binary_matrix_from_records(&pure_records, &binary_records, identifier_option)?;
        Self::new(pure_records, binary_records)
    }

    /// Creates parameters from model records with default values for the molar weight,
    /// identifiers, association sites, and binary interaction parameters.
    pub fn from_model_records(model_records: Vec<P>) -> FeosResult<Self> {
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
    ) -> FeosResult<Vec<BinaryRecord<usize, B, A>>> {
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
            .map(|[(i1, p1), (i2, p2)]| {
                let Some(id1) = p1.identifier.as_str(identifier_option) else {
                    return Err(FeosError::MissingParameters(format!(
                        "No {} for pure record {} ({}).",
                        identifier_option, i1, p1.identifier
                    )));
                };
                let Some(id2) = p2.identifier.as_str(identifier_option) else {
                    return Err(FeosError::MissingParameters(format!(
                        "No {} for pure record {} ({}).",
                        identifier_option, i2, p2.identifier
                    )));
                };
                Ok([(i1, id1), (i2, id2)])
            })
            .filter_map(|x| {
                x.map(|[(i1, id1), (i2, id2)]| {
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
                .transpose()
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
        let records = PureRecord::from_multiple_json(input, identifier_option)?;

        let binary_records = if let Some(path) = file_binary {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        } else {
            Vec::new()
        };

        Self::from_records(records, binary_records, identifier_option)
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
                let groups = group_counts
                    .iter()
                    .map(|(s, c)| {
                        segment_map.get(s).map(|&x| (x.clone(), *c)).ok_or_else(|| {
                            FeosError::MissingParameters(format!("No segment record found for {s}"))
                        })
                    })
                    .collect::<FeosResult<Vec<_>>>()?;
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

        Self::new(pure_records, binary_records)
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

impl<P, B, A, Bo> ParametersHetero<P, B, A, Bo, f64> {
    pub fn segment_counts(&self) -> Array1<f64> {
        self.pure.iter().map(|pr| pr.count).collect()
    }
}

impl<P: Clone, B: Clone, A: Clone, Bo: Clone, C: GroupCount + Default>
    ParametersHetero<P, B, A, Bo, C>
{
    pub fn from_segments_hetero(
        chemical_records: Vec<ChemicalRecord>,
        segment_records: &[SegmentRecord<P, A>],
        binary_segment_records: Option<&[BinarySegmentRecord<B, A>]>,
    ) -> FeosResult<Self>
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
    ) -> FeosResult<Self> {
        let segment_map: HashMap<_, _> =
            segment_records.iter().map(|s| (&s.identifier, s)).collect();

        let mut bond_records_map = HashMap::new();
        for bond_record in bond_records {
            bond_records_map.insert((&bond_record.id1, &bond_record.id2), bond_record);
            bond_records_map.insert((&bond_record.id2, &bond_record.id1), bond_record);
        }

        let mut groups = Vec::new();
        let mut association_sites = Vec::new();
        let mut bonds = Vec::new();
        let mut identifiers = Vec::new();
        let mut molar_weight: Array1<f64> = Array1::zeros(chemical_records.len());
        for (i, cr) in chemical_records.into_iter().enumerate() {
            let (identifier, group_counts, bond_counts) = C::into_groups(cr);
            let n = groups.len();
            identifiers.push(identifier);
            for (s, c) in &group_counts {
                let Some(&segment) = segment_map.get(s) else {
                    return Err(FeosError::MissingParameters(format!(
                        "No segment record found for {s}"
                    )));
                };
                molar_weight[i] += segment.molarweight * c.into_f64();
                groups.push(Pure::from_segment_record(segment, *c, i));
                association_sites.push(segment.association_sites.clone());
            }
            for ([a, b], c) in bond_counts {
                let id1 = &group_counts[a].0;
                let id2 = &group_counts[b].0;
                let Some(&bond) = bond_records_map.get(&(id1, id2)) else {
                    return Err(FeosError::MissingParameters(format!(
                        "No bond record found for {id1}-{id2}"
                    )));
                };
                let Some(bond) = bond.model_record.as_ref() else {
                    return Err(FeosError::MissingParameters(format!(
                        "No bond record found for {id1}-{id2}"
                    )));
                };
                bonds.push(Binary::new(a + n, b + n, bond.clone(), c));
            }
        }

        let mut binary_records = Vec::new();
        let mut binary_association_records = Vec::new();
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
                    if let Some(&br) = binary_segment_records_map.get(&(id1, id2)) {
                        if let Some(br) = &br.model_record {
                            binary_records.push(Binary::new(i1, i2, br.clone(), ()));
                        }
                        if !br.association_sites.is_empty() {
                            binary_association_records.push(Binary::new(
                                i1,
                                i2,
                                br.association_sites.clone(),
                                (),
                            ))
                        }
                    }
                }
            }
        }

        let association_parameters = AssociationParameters::new_hetero(
            &groups,
            &association_sites,
            &binary_association_records,
        )?;

        Ok(Self {
            pure: groups,
            binary: binary_records,
            bonds,
            association: association_parameters,
            identifiers,
            molar_weight: molar_weight * (GRAM / MOL),
        })
    }

    /// Creates parameters from segment information stored in json files.
    ///
    /// The [FromSegments] trait needs to be implemented for both the model record
    /// and the ideal gas record.
    pub fn from_json_segments_hetero<F>(
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

        Self::from_segments_hetero(
            chemical_records,
            &segment_records,
            binary_records.as_deref(),
        )
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

        Self::from_segments_with_bonds(
            chemical_records,
            &segment_records,
            binary_records.as_deref(),
            &bond_records,
        )
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
        self.pure.iter().map(|pr| pr.component_index).collect()
    }
}
