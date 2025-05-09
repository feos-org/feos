use arrayvec::ArrayString;
use num_traits::Zero;
use serde::{Deserialize, Serialize};

type SiteId = ArrayString<8>;

/// Pure component association parameters.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct AssociationRecord<A> {
    #[serde(skip_serializing_if = "SiteId::is_empty")]
    #[serde(default)]
    pub id: SiteId,
    #[serde(flatten)]
    pub parameters: Option<A>,
    /// \# of association sites of type A
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub na: f64,
    /// \# of association sites of type B
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub nb: f64,
    /// \# of association sites of type C
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub nc: f64,
}

impl<A> AssociationRecord<A> {
    pub fn new(parameters: Option<A>, na: f64, nb: f64, nc: f64) -> Self {
        Self::with_id(Default::default(), parameters, na, nb, nc)
    }

    pub fn with_id(id: SiteId, parameters: Option<A>, na: f64, nb: f64, nc: f64) -> Self {
        Self {
            id,
            parameters,
            na,
            nb,
            nc,
        }
    }
}

/// Binary association parameters.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct BinaryAssociationRecord<A> {
    // Identifier of the association site on the first molecule.
    #[serde(skip_serializing_if = "SiteId::is_empty")]
    #[serde(default)]
    pub id1: SiteId,
    // Identifier of the association site on the second molecule.
    #[serde(skip_serializing_if = "SiteId::is_empty")]
    #[serde(default)]
    pub id2: SiteId,
    // Binary association parameters
    #[serde(flatten)]
    pub parameters: A,
}

impl<A> BinaryAssociationRecord<A> {
    pub fn new(parameters: A) -> Self {
        Self::with_id(Default::default(), Default::default(), parameters)
    }

    pub fn with_id(id1: SiteId, id2: SiteId, parameters: A) -> Self {
        Self {
            id1,
            id2,
            parameters,
        }
    }
}
