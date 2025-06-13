use super::{Identifier, IdentifierOption};
use crate::{FeosError, FeosResult};
use indexmap::IndexMap;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::ops::Deref;
use std::path::Path;

// Auxiliary structure used to deserialize chemical records without explicit bond information.
#[derive(Serialize, Deserialize)]
struct ChemicalRecordJSON {
    identifier: Identifier,
    segments: Vec<String>,
    bonds: Option<Vec<[usize; 2]>>,
}

/// Chemical information of a substance.
#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(from = "ChemicalRecordJSON")]
#[serde(into = "ChemicalRecordJSON")]
pub struct ChemicalRecord {
    pub identifier: Identifier,
    pub segments: Vec<String>,
    pub bonds: Vec<[usize; 2]>,
}

impl From<ChemicalRecordJSON> for ChemicalRecord {
    fn from(record: ChemicalRecordJSON) -> Self {
        Self::new(record.identifier, record.segments, record.bonds)
    }
}

impl From<ChemicalRecord> for ChemicalRecordJSON {
    fn from(record: ChemicalRecord) -> Self {
        Self {
            identifier: record.identifier,
            segments: record.segments,
            bonds: Some(record.bonds),
        }
    }
}

impl ChemicalRecord {
    /// Create a new `ChemicalRecord`.
    ///
    /// If no bonds are given, the molecule is assumed to be linear.
    pub fn new(
        identifier: Identifier,
        segments: Vec<String>,
        bonds: Option<Vec<[usize; 2]>>,
    ) -> ChemicalRecord {
        let bonds = bonds.unwrap_or_else(|| {
            (0..segments.len() - 1)
                .zip(1..segments.len())
                .map(|x| [x.0, x.1])
                .collect()
        });
        Self {
            identifier,
            segments,
            bonds,
        }
    }

    /// Create chemical records from a json file.
    pub fn from_json<P, S>(
        substances: &[S],
        file: P,
        identifier_option: IdentifierOption,
    ) -> FeosResult<Vec<Self>>
    where
        P: AsRef<Path>,
        S: Deref<Target = str>,
    {
        // create list of substances
        let mut queried: HashSet<&str> = substances.iter().map(|s| s.deref()).collect();
        // raise error on duplicate detection
        if queried.len() != substances.len() {
            return Err(FeosError::IncompatibleParameters(
                "A substance was defined more than once.".to_string(),
            ));
        }

        let f = File::open(file)?;
        let reader = BufReader::new(f);
        // use stream in the future
        let file_records: Vec<Self> = serde_json::from_reader(reader)?;
        let mut records: HashMap<&str, Self> = HashMap::with_capacity(substances.len());

        // build map, draining list of queried substances in the process
        for record in file_records {
            if let Some(id) = record.identifier.as_str(identifier_option) {
                queried.take(id).map(|id| records.insert(id, record));
            }
            // all parameters parsed
            if queried.is_empty() {
                break;
            }
        }

        // report missing parameters
        if !queried.is_empty() {
            return Err(FeosError::ComponentsNotFound(format!("{:?}", queried)));
        };

        // collect into vec in correct order
        Ok(substances
            .iter()
            .map(|s| records.remove(s.deref()).unwrap())
            .collect())
    }
}

impl std::fmt::Display for ChemicalRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ChemicalRecord(")?;
        write!(f, "\n\tidentifier={},", self.identifier)?;
        write!(f, "\n\tsegments={:?},", self.segments)?;
        write!(f, "\n\tbonds={:?}\n)", self.bonds)
    }
}
pub trait GroupCount: Copy {
    #[expect(clippy::type_complexity)]
    fn into_groups(
        chemical_record: ChemicalRecord,
    ) -> (Identifier, Vec<(String, Self)>, Vec<([usize; 2], Self)>);

    fn into_f64(self) -> f64;
}

impl GroupCount for f64 {
    fn into_groups(
        chemical_record: ChemicalRecord,
    ) -> (Identifier, Vec<(String, f64)>, Vec<([usize; 2], f64)>) {
        let mut group_counts = IndexMap::with_capacity(chemical_record.segments.len());
        let segment_to_group: Vec<_> = chemical_record
            .segments
            .into_iter()
            .map(|si| {
                let entry = group_counts.entry(si);
                let index = entry.index();
                *entry.or_insert(0.0) += 1.0;
                index
            })
            .collect();

        let mut bond_counts: IndexMap<_, _> = (0..group_counts.len())
            .array_combinations()
            .chain((0..group_counts.len()).map(|i| [i, i]))
            .map(|g| (g, 0.0))
            .collect();
        for [i, j] in chemical_record.bonds {
            let [s1, s2] = [segment_to_group[i], segment_to_group[j]];
            bond_counts.entry([s1, s2]).and_modify(|x| *x += 1.0);
            if s1 != s2 {
                bond_counts.entry([s2, s1]).and_modify(|x| *x += 1.0);
            }
        }
        let group_counts = group_counts.into_iter().collect();
        let bond_counts = bond_counts.into_iter().filter(|(_, c)| *c > 0.0).collect();

        (chemical_record.identifier, group_counts, bond_counts)
    }

    fn into_f64(self) -> f64 {
        self
    }
}

impl GroupCount for () {
    fn into_groups(
        chemical_record: ChemicalRecord,
    ) -> (Identifier, Vec<(String, ())>, Vec<([usize; 2], ())>) {
        let segments = chemical_record
            .segments
            .into_iter()
            .map(|s| (s, ()))
            .collect();
        let bonds = chemical_record.bonds.into_iter().map(|b| (b, ())).collect();
        (chemical_record.identifier, segments, bonds)
    }

    fn into_f64(self) -> f64 {
        1.0
    }
}

// pub trait CountType: Copy {
//     fn apply_count(self, x: f64) -> f64;
// }

// impl CountType for usize {
//     fn apply_count(self, x: f64) -> f64 {
//         self as f64 * x
//     }
// }

// impl CountType for f64 {
//     fn apply_count(self, x: f64) -> f64 {
//         self * x
//     }
// }

// /// Trait that enables parameter generation from generic molecular representations.
// pub trait SegmentCount {
//     type Count: CountType;

//     fn identifier(&self) -> Cow<Identifier>;

//     /// Count the number of occurences of each individual segment identifier in the
//     /// molecule.
//     ///
//     /// The map contains the segment identifier as key and the count as value.
//     fn segment_count(&self) -> Cow<HashMap<String, Self::Count>>;

//     /// Count the number of occurences of each individual segment in the
//     /// molecule.
//     ///
//     /// The map contains the segment record as key and the count as value.
//     #[expect(clippy::type_complexity)]
//     fn segment_map<M: Clone, A: Clone>(
//         &self,
//         segment_records: &[SegmentRecord<M, A>],
//     ) -> FeosResult<Vec<(SegmentRecord<M, A>, Self::Count)>> {
//         let count = self.segment_count();
//         let queried: HashSet<_> = count.keys().collect();
//         let mut segments: HashMap<_, SegmentRecord<M, A>> = segment_records
//             .iter()
//             .map(|r| (&r.identifier, r.clone()))
//             .collect();
//         let available = segments.keys().copied().collect();
//         if !queried.is_subset(&available) {
//             let missing: Vec<_> = queried.difference(&available).collect();
//             let msg = format!("{:?}", missing);
//             return Err(FeosError::ComponentsNotFound(msg));
//         };
//         Ok(count
//             .iter()
//             .map(|(s, c)| (segments.remove(s).unwrap(), *c))
//             .collect())
//     }
// }

// impl SegmentCount for ChemicalRecord {
//     type Count = usize;

//     fn identifier(&self) -> Cow<Identifier> {
//         Cow::Borrowed(&self.identifier)
//     }

//     fn segment_count(&self) -> Cow<HashMap<String, usize>> {
//         Cow::Owned(self.segment_count())
//     }
// }
