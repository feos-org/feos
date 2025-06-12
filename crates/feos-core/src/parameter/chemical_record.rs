use super::Identifier;
use indexmap::IndexMap;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

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

    /// Count the number of occurences of each individual segment identifier in the
    /// chemical record.
    ///
    /// The map contains the segment identifier as key and the count as value.
    #[expect(clippy::type_complexity)]
    pub fn into_groups(self) -> (Identifier, Vec<(String, f64)>, Vec<([usize; 2], f64)>) {
        let mut group_counts = IndexMap::with_capacity(self.segments.len());
        let segment_to_group: Vec<_> = self
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
        for [i, j] in self.bonds {
            let [s1, s2] = [segment_to_group[i], segment_to_group[j]];
            bond_counts.entry([s1, s2]).and_modify(|x| *x += 1.0);
            if s1 != s2 {
                bond_counts.entry([s2, s1]).and_modify(|x| *x += 1.0);
            }
        }
        let group_counts = group_counts.into_iter().collect();
        let bond_counts = bond_counts.into_iter().filter(|(_, c)| *c > 0.0).collect();

        (self.identifier, group_counts, bond_counts)
    }

    // /// Count the number of occurences of each individual segment in the
    // /// molecule.
    // ///
    // /// The map contains the segment record as key and the count as value.
    // #[expect(clippy::type_complexity)]
    // pub fn segment_map<M: Clone, A: Clone>(
    //     &self,
    //     segment_records: &[SegmentRecord<M, A>],
    // ) -> FeosResult<Vec<(SegmentRecord<M, A>, f64)>> {
    //     let count = self.segment_count();
    //     let queried: HashSet<_> = count.keys().collect();
    //     let mut segments: HashMap<_, SegmentRecord<M, A>> = segment_records
    //         .iter()
    //         .map(|r| (&r.identifier, r.clone()))
    //         .collect();
    //     let available = segments.keys().copied().collect();
    //     if !queried.is_subset(&available) {
    //         let missing: Vec<_> = queried.difference(&available).collect();
    //         let msg = format!("{:?}", missing);
    //         return Err(FeosError::ComponentsNotFound(msg));
    //     };
    //     Ok(count
    //         .iter()
    //         .map(|(s, c)| (segments.remove(s).unwrap(), *c))
    //         .collect())
    // }

    // /// Count the number of occurences of bonds between each pair of segment identifiers
    // /// in the chemical record.
    // ///
    // /// The map contains the segment identifiers as key and the count as value.
    // pub fn bond_count<T: NumAssign>(&self) -> HashMap<[String; 2], T> {
    //     let mut bond_counts = HashMap::new();
    //     for b in &self.bonds {
    //         let s1 = self.segments[b[0]].clone();
    //         let s2 = self.segments[b[1]].clone();
    //         let indices = if s1 > s2 { [s2, s1] } else { [s1, s2] };
    //         let entry = bond_counts.entry(indices).or_insert_with(|| T::zero());
    //         *entry += T::one();
    //     }
    //     bond_counts
    // }
}

impl std::fmt::Display for ChemicalRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ChemicalRecord(")?;
        write!(f, "\n\tidentifier={},", self.identifier)?;
        write!(f, "\n\tsegments={:?},", self.segments)?;
        write!(f, "\n\tbonds={:?}\n)", self.bonds)
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
