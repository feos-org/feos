use super::identifier::Identifier;
use super::segment::SegmentRecord;
use super::{IdentifierOption, ParameterError};
use conv::ValueInto;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// A collection of parameters of a pure substance.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PureRecord<M> {
    pub identifier: Identifier,
    #[serde(default)]
    pub molarweight: f64,
    pub model_record: M,
}

impl<M> PureRecord<M> {
    /// Create a new `PureRecord`.
    pub fn new(identifier: Identifier, molarweight: f64, model_record: M) -> Self {
        Self {
            identifier,
            molarweight,
            model_record,
        }
    }

    /// Update the `PureRecord` from segment counts.
    ///
    /// The [FromSegments] trait needs to be implemented for both the model record
    /// and the ideal gas record.
    pub fn from_segments<S, T>(identifier: Identifier, segments: S) -> Result<Self, ParameterError>
    where
        T: Copy + ValueInto<f64>,
        M: FromSegments<T>,
        S: IntoIterator<Item = (SegmentRecord<M>, T)>,
    {
        let mut molarweight = 0.0;
        let mut model_segments = Vec::new();
        for (s, n) in segments {
            molarweight += s.molarweight * n.value_into().unwrap();
            model_segments.push((s.model_record, n));
        }
        let model_record = M::from_segments(&model_segments)?;

        Ok(Self::new(identifier, molarweight, model_record))
    }

    /// Create pure substance parameters from a json file.
    pub fn from_json<P>(
        substances: &[&str],
        file: P,
        identifier_option: IdentifierOption,
    ) -> Result<Vec<Self>, ParameterError>
    where
        P: AsRef<Path>,
        M: Clone + DeserializeOwned,
    {
        // create list of substances
        let mut queried: HashSet<String> = substances.iter().map(|s| s.to_string()).collect();
        // raise error on duplicate detection
        if queried.len() != substances.len() {
            return Err(ParameterError::IncompatibleParameters(
                "A substance was defined more than once.".to_string(),
            ));
        }

        let f = File::open(file)?;
        let reader = BufReader::new(f);
        // use stream in the future
        let file_records: Vec<Self> = serde_json::from_reader(reader)?;
        let mut records: HashMap<String, Self> = HashMap::with_capacity(substances.len());

        // build map, draining list of queried substances in the process
        for record in file_records {
            if let Some(id) = record.identifier.as_string(identifier_option) {
                queried.take(&id).map(|id| records.insert(id, record));
            }
            // all parameters parsed
            if queried.is_empty() {
                break;
            }
        }

        // report missing parameters
        if !queried.is_empty() {
            return Err(ParameterError::ComponentsNotFound(format!("{:?}", queried)));
        };

        // collect into vec in correct order
        Ok(substances
            .iter()
            .map(|s| records.get(&s.to_string()).unwrap().clone())
            .collect())
    }
}

impl<M> std::fmt::Display for PureRecord<M>
where
    M: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PureRecord(")?;
        write!(f, "\n\tidentifier={},", self.identifier)?;
        write!(f, "\n\tmolarweight={},", self.molarweight)?;
        write!(f, "\n\tmodel_record={}", self.model_record)?;
        write!(f, "\n)")
    }
}

/// Trait for models that implement a homosegmented group contribution
/// method
pub trait FromSegments<T>: Clone {
    /// Constructs the record from a list of segment records with their
    /// number of occurences.
    fn from_segments(segments: &[(Self, T)]) -> Result<Self, ParameterError>;
}

/// Trait for models that implement a homosegmented group contribution
/// method and have a combining rule for binary interaction parameters.
pub trait FromSegmentsBinary<T>: Clone {
    /// Constructs the binary record from a list of segment records with
    /// their number of occurences.
    fn from_segments_binary(segments: &[(f64, T, T)]) -> Result<Self, ParameterError>;
}

/// A collection of parameters that model interactions between two
/// substances or segments.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BinaryRecord<I, B> {
    /// Identifier of the first component
    pub id1: I,
    /// Identifier of the second component
    pub id2: I,
    /// Binary interaction parameter(s)
    pub model_record: B,
}

impl<I, B> BinaryRecord<I, B> {
    /// Crates a new `BinaryRecord`.
    pub fn new(id1: I, id2: I, model_record: B) -> Self {
        Self {
            id1,
            id2,
            model_record,
        }
    }

    /// Read a list of `BinaryRecord`s from a JSON file.
    pub fn from_json<P: AsRef<Path>>(file: P) -> Result<Vec<Self>, ParameterError>
    where
        I: DeserializeOwned,
        B: DeserializeOwned,
    {
        Ok(serde_json::from_reader(BufReader::new(File::open(file)?))?)
    }
}

impl<I, B> std::fmt::Display for BinaryRecord<I, B>
where
    I: std::fmt::Display,
    B: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BinaryRecord(")?;
        write!(f, "\n\tid1={},", self.id1)?;
        write!(f, "\n\tid2={},", self.id2)?;
        write!(f, "\n\tmodel_record={},", self.model_record)?;
        write!(f, "\n)")
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
            "model_record": {
                "a": 0.1
            }
        }
        "#;
        let record: PureRecord<TestModelRecordSegments> =
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
                "model_record": {
                    "a": 1.0
                }
            },
            {
                "identifier": {
                    "cas": "2"
                },
                "molarweight": 2.0,
                "model_record": {
                    "a": 2.0
                }
            }
        ]"#;
        let records: Vec<PureRecord<TestModelRecordSegments>> =
            serde_json::from_str(r).expect("Unable to parse json.");
        assert_eq!(records[0].identifier.cas, Some("1".into()));
        assert_eq!(records[1].identifier.cas, Some("2".into()))
    }
}
