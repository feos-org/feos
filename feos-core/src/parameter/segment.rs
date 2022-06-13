use super::ParameterError;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufReader;
use std::path::Path;

/// Parameters describing an individual segment of a molecule.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SegmentRecord<M, I> {
    pub identifier: String,
    pub molarweight: f64,
    pub model_record: M,
    pub ideal_gas_record: Option<I>,
}

impl<M, I> SegmentRecord<M, I> {
    /// Creates a new `SegmentRecord`.
    pub fn new(
        identifier: String,
        molarweight: f64,
        model_record: M,
        ideal_gas_record: Option<I>,
    ) -> Self {
        Self {
            identifier,
            molarweight,
            model_record,
            ideal_gas_record,
        }
    }

    /// Read a list of `SegmentRecord`s from a JSON file.
    pub fn from_json<P: AsRef<Path>>(file: P) -> Result<Vec<Self>, ParameterError>
    where
        I: DeserializeOwned,
        M: DeserializeOwned,
    {
        Ok(serde_json::from_reader(BufReader::new(File::open(file)?))?)
    }
}

impl<M, I> Hash for SegmentRecord<M, I> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identifier.hash(state);
    }
}

impl<M, I> PartialEq for SegmentRecord<M, I> {
    fn eq(&self, other: &Self) -> bool {
        self.identifier == other.identifier
    }
}
impl<M, I> Eq for SegmentRecord<M, I> {}

impl<M: std::fmt::Display, I: std::fmt::Display> std::fmt::Display for SegmentRecord<M, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SegmentRecord(\n\tidentifier={}", self.identifier)?;
        write!(f, "\n\tmolarweight={}", self.molarweight)?;
        write!(f, "\n\tmodel_record={}", self.model_record)?;
        if let Some(i) = self.ideal_gas_record.as_ref() {
            write!(f, "\n\tideal_gas_record={},", i)?;
        }
        write!(f, "\n)")
    }
}
