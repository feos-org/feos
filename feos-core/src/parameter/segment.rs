use super::ParameterError;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufReader;
use std::path::Path;

/// Parameters describing an individual segment of a molecule.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SegmentRecord<M> {
    pub identifier: String,
    pub molarweight: f64,
    pub model_record: M,
}

impl<M> SegmentRecord<M> {
    /// Creates a new `SegmentRecord`.
    pub fn new(identifier: String, molarweight: f64, model_record: M) -> Self {
        Self {
            identifier,
            molarweight,
            model_record,
        }
    }

    /// Read a list of `SegmentRecord`s from a JSON file.
    pub fn from_json<P: AsRef<Path>>(file: P) -> Result<Vec<Self>, ParameterError>
    where
        M: DeserializeOwned,
    {
        Ok(serde_json::from_reader(BufReader::new(File::open(file)?))?)
    }
}

impl<M> Hash for SegmentRecord<M> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identifier.hash(state);
    }
}

impl<M> PartialEq for SegmentRecord<M> {
    fn eq(&self, other: &Self) -> bool {
        self.identifier == other.identifier
    }
}
impl<M> Eq for SegmentRecord<M> {}

impl<M: std::fmt::Display> std::fmt::Display for SegmentRecord<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SegmentRecord(\n\tidentifier={}", self.identifier)?;
        write!(f, "\n\tmolarweight={}", self.molarweight)?;
        write!(f, "\n\tmodel_record={}", self.model_record)?;
        write!(f, "\n)")
    }
}
