use super::identifier::Identifier;
use super::segment::SegmentRecord;
use super::ParameterError;
use conv::ValueInto;
use num_traits::NumAssign;
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

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
    pub fn segment_count<T: NumAssign>(&self) -> HashMap<String, T> {
        let mut counts = HashMap::with_capacity(self.segments.len());
        for si in &self.segments {
            let entry = counts.entry(si.clone()).or_insert_with(|| T::zero());
            *entry += T::one();
        }
        counts
    }

    /// Count the number of occurences of bonds between each pair of segment identifiers
    /// in the chemical record.
    ///
    /// The map contains the segment identifiers as key and the count as value.
    pub fn bond_count<T: NumAssign>(&self) -> HashMap<[String; 2], T> {
        let mut bond_counts = HashMap::new();
        for b in &self.bonds {
            let s1 = self.segments[b[0]].clone();
            let s2 = self.segments[b[1]].clone();
            let indices = if s1 > s2 { [s2, s1] } else { [s1, s2] };
            let entry = bond_counts.entry(indices).or_insert_with(|| T::zero());
            *entry += T::one();
        }
        bond_counts
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

/// Trait that enables parameter generation from generic molecular representations.
pub trait SegmentCount {
    type Count: Copy + ValueInto<f64>;

    fn identifier(&self) -> Cow<Identifier>;

    /// Count the number of occurences of each individual segment identifier in the
    /// molecule.
    ///
    /// The map contains the segment identifier as key and the count as value.
    fn segment_count(&self) -> Cow<HashMap<String, Self::Count>>;

    /// Count the number of occurences of each individual segment in the
    /// molecule.
    ///
    /// The map contains the segment record as key and the count as value.
    fn segment_map<M: Clone>(
        &self,
        segment_records: &[SegmentRecord<M>],
    ) -> Result<HashMap<SegmentRecord<M>, Self::Count>, ParameterError> {
        let count = self.segment_count();
        let queried: HashSet<_> = count.keys().cloned().collect();
        let mut segments: HashMap<String, SegmentRecord<M>> = segment_records
            .iter()
            .map(|r| (r.identifier.clone(), r.clone()))
            .collect();
        let available = segments.keys().cloned().collect();
        if !queried.is_subset(&available) {
            let missing: Vec<String> = queried.difference(&available).cloned().collect();
            let msg = format!("{:?}", missing);
            return Err(ParameterError::ComponentsNotFound(msg));
        };
        Ok(count
            .iter()
            .map(|(s, c)| (segments.remove(s).unwrap(), *c))
            .collect())
    }
}

impl SegmentCount for ChemicalRecord {
    type Count = usize;

    fn identifier(&self) -> Cow<Identifier> {
        Cow::Borrowed(&self.identifier)
    }

    fn segment_count(&self) -> Cow<HashMap<String, usize>> {
        Cow::Owned(self.segment_count())
    }
}
