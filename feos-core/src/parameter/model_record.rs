use super::identifier::Identifier;
use super::segment::SegmentRecord;
use super::ParameterError;
use conv::ValueInto;
use serde::{Deserialize, Serialize};

/// A collection of parameters of a pure substance.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PureRecord<M, I> {
    pub identifier: Identifier,
    pub molarweight: f64,
    pub model_record: M,
    #[serde(default = "Default::default")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ideal_gas_record: Option<I>,
}

impl<M, I> PureRecord<M, I> {
    /// Create a new `PureRecord`.
    pub fn new(
        identifier: Identifier,
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

    /// Update the `PureRecord` from segment counts.
    ///
    /// The [FromSegments] trait needs to be implemented for both the model record
    /// and the ideal gas record.
    pub fn from_segments<S, T>(identifier: Identifier, segments: S) -> Result<Self, ParameterError>
    where
        T: Copy + ValueInto<f64>,
        M: FromSegments<T>,
        I: FromSegments<T>,
        S: IntoIterator<Item = (SegmentRecord<M, I>, T)>,
    {
        let mut molarweight = 0.0;
        let mut model_segments = Vec::new();
        let mut ideal_gas_segments = Vec::new();
        for (s, n) in segments {
            molarweight += s.molarweight * n.value_into().unwrap();
            model_segments.push((s.model_record, n));
            ideal_gas_segments.push(s.ideal_gas_record.map(|ig| (ig, n)));
        }
        let model_record = M::from_segments(&model_segments)?;

        let ideal_gas_segments: Option<Vec<_>> = ideal_gas_segments.into_iter().collect();
        let ideal_gas_record = ideal_gas_segments
            .as_deref()
            .map(I::from_segments)
            .transpose()?;

        Ok(Self::new(
            identifier,
            molarweight,
            model_record,
            ideal_gas_record,
        ))
    }
}

impl<M, I> std::fmt::Display for PureRecord<M, I>
where
    M: std::fmt::Display,
    I: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PureRecord(")?;
        write!(f, "\n\tidentifier={},", self.identifier)?;
        write!(f, "\n\tmolarweight={},", self.molarweight)?;
        write!(f, "\n\tmodel_record={},", self.model_record)?;
        if let Some(i) = self.ideal_gas_record.as_ref() {
            write!(f, "\n\tideal_gas_record={},", i)?;
        }
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
    fn from_segments_binary(segments: &[(Self, T, T)]) -> Result<Self, ParameterError>;
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
    use crate::joback::JobackRecord;

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
        let record: PureRecord<TestModelRecordSegments, JobackRecord> =
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
        let records: Vec<PureRecord<TestModelRecordSegments, JobackRecord>> =
            serde_json::from_str(r).expect("Unable to parse json.");
        assert_eq!(records[0].identifier.cas, Some("1".into()));
        assert_eq!(records[1].identifier.cas, Some("2".into()))
    }
}
