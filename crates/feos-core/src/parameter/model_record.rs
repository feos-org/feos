use super::{AssociationRecord, BinaryAssociationRecord, CountType, Identifier, IdentifierOption};
use crate::FeosResult;
use crate::errors::FeosError;
use num_traits::Zero;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::ops::Deref;
use std::path::Path;

/// A collection of parameters with an arbitrary identifier.
#[derive(Serialize, Deserialize, Clone)]
pub struct Record<I, M, A> {
    pub identifier: I,
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub molarweight: f64,
    #[serde(flatten)]
    pub model_record: M,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default = "Vec::new")]
    pub association_sites: Vec<AssociationRecord<A>>,
}

/// A collection of parameters of a pure substance.
pub type PureRecord<M, A> = Record<Identifier, M, A>;

/// Parameters describing an individual segment of a molecule.
pub type SegmentRecord<M, A> = Record<String, M, A>;

impl<I, M, A> Record<I, M, A> {
    /// Create a new `PureRecord`.
    pub fn new(identifier: I, molarweight: f64, model_record: M) -> Self {
        Self {
            identifier,
            molarweight,
            model_record,
            association_sites: Vec::new(),
        }
    }

    /// Create a new `PureRecord` including association information.
    pub fn with_association(
        identifier: I,
        molarweight: f64,
        model_record: M,
        association_sites: Vec<AssociationRecord<A>>,
    ) -> Self {
        Self {
            identifier,
            molarweight,
            model_record,
            association_sites,
        }
    }

    /// Update the `PureRecord` from segment counts.
    ///
    /// The [FromSegments] trait needs to be implemented for both the model record
    /// and the ideal gas record.
    pub fn from_segments<S, T>(identifier: I, segments: S) -> FeosResult<Self>
    where
        T: CountType,
        M: FromSegments<T>,
        S: IntoIterator<Item = (SegmentRecord<M, A>, T)>,
    {
        let mut molarweight = 0.0;
        let mut model_segments = Vec::new();
        let association_sites = segments
            .into_iter()
            .flat_map(|(s, n)| {
                molarweight += n.apply_count(s.molarweight);
                model_segments.push((s.model_record, n));
                s.association_sites.into_iter().map(move |record| {
                    AssociationRecord::with_id(
                        record.id,
                        record.parameters,
                        n.apply_count(record.na),
                        n.apply_count(record.nb),
                        n.apply_count(record.nc),
                    )
                })
            })
            .collect();
        let model_record = M::from_segments(&model_segments)?;

        Ok(Self::with_association(
            identifier,
            molarweight,
            model_record,
            association_sites,
        ))
    }
}

impl<M, A> PureRecord<M, A> {
    /// Create pure substance parameters from a json file.
    pub fn from_json<P, S>(
        substances: &[S],
        file: P,
        identifier_option: IdentifierOption,
    ) -> FeosResult<Vec<Self>>
    where
        P: AsRef<Path>,
        S: Deref<Target = str>,
        M: DeserializeOwned,
        A: DeserializeOwned,
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

impl<M, A> SegmentRecord<M, A> {
    /// Read a list of `SegmentRecord`s from a JSON file.
    pub fn from_json<P: AsRef<Path>>(file: P) -> FeosResult<Vec<Self>>
    where
        M: DeserializeOwned,
        A: DeserializeOwned,
    {
        Ok(serde_json::from_reader(BufReader::new(File::open(file)?))?)
    }
}

impl<I: Serialize, M: Serialize, A: Serialize> std::fmt::Display for Record<I, M, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = serde_json::to_string(self).unwrap().replace("\"", "");
        let s = s.replace(",", ", ").replace(":", ": ");
        write!(f, "PureRecord({})", &s[1..s.len() - 1])
    }
}

/// Trait for models that implement a homosegmented group contribution
/// method
pub trait FromSegments<T>: Clone {
    /// Constructs the record from a list of segment records with their
    /// number of occurences.
    fn from_segments(segments: &[(Self, T)]) -> FeosResult<Self>;
}

/// Trait for models that implement a homosegmented group contribution
/// method and have a combining rule for binary interaction parameters.
pub trait FromSegmentsBinary<T>: Clone {
    /// Constructs the binary record from a list of segment records with
    /// their number of occurences.
    fn from_segments_binary(segments: &[(Self, T, T)]) -> FeosResult<Self>;
}

/// A collection of parameters that model interactions between two substances.
#[derive(Serialize, Deserialize, Clone)]
pub struct BinaryRecord<I, B, A> {
    /// Identifier of the first component
    pub id1: I,
    /// Identifier of the second component
    pub id2: I,
    /// Binary interaction parameter(s)
    #[serde(flatten)]
    pub model_record: Option<B>,
    /// Binary association records
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default = "Vec::new")]
    pub association_sites: Vec<BinaryAssociationRecord<A>>,
}

/// A collection of parameters that model interactions between two segments.
pub type BinarySegmentRecord<M, A> = BinaryRecord<String, M, A>;

impl<I, B, A> BinaryRecord<I, B, A> {
    /// Crates a new `BinaryRecord`.
    pub fn new(
        id1: I,
        id2: I,
        model_record: Option<B>,
        association_sites: Vec<BinaryAssociationRecord<A>>,
    ) -> Self {
        Self {
            id1,
            id2,
            model_record,
            association_sites,
        }
    }

    /// Read a list of `BinaryRecord`s from a JSON file.
    pub fn from_json<P: AsRef<Path>>(file: P) -> FeosResult<Vec<Self>>
    where
        I: DeserializeOwned,
        B: DeserializeOwned,
        A: DeserializeOwned,
    {
        Ok(serde_json::from_reader(BufReader::new(File::open(file)?))?)
    }
}

impl<I: Serialize + Clone, B: Serialize + Clone, A: Serialize + Clone> std::fmt::Display
    for BinaryRecord<I, B, A>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = serde_json::to_string(self).unwrap().replace("\"", "");
        let s = s.replace(",", ", ").replace(":", ": ");
        write!(f, "BinaryRecord({})", &s[1..s.len() - 1])
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Serialize, Deserialize, Debug, Default, Clone)]
    struct TestModelRecordSegments {
        a: f64,
    }

    #[test]
    fn deserialize() {
        let r = r#"
        {
            "identifier": {
                "cas": "123-4-5"
            },
            "molarweight": 16.0426,
            "a": 0.1
        }
        "#;
        let record: PureRecord<TestModelRecordSegments, ()> =
            serde_json::from_str(r).expect("Unable to parse json.");
        assert_eq!(record.identifier.cas, Some("123-4-5".into()))
    }

    #[test]
    fn deserialize_list() {
        let r = r#"
        [
            {
                "identifier": {
                    "cas": "1"
                },
                "molarweight": 1.0,
                "a": 1.0
            },
            {
                "identifier": {
                    "cas": "2"
                },
                "molarweight": 2.0,
                "a": 2.0
            }
        ]"#;
        let records: Vec<PureRecord<TestModelRecordSegments, ()>> =
            serde_json::from_str(r).expect("Unable to parse json.");
        assert_eq!(records[0].identifier.cas, Some("1".into()));
        assert_eq!(records[1].identifier.cas, Some("2".into()))
    }
}
