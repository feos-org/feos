use crate::association::AssociationRecord;
use serde::{Deserialize, Serialize};

/// gc-PC-SAFT pure-component parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct GcPcSaftRecord {
    /// Segment shape factor
    pub m: f64,
    /// Segment diameter in units of Angstrom
    pub sigma: f64,
    /// Energetic parameter in units of Kelvin
    pub epsilon_k: f64,
    /// Dipole moment in units of Debye
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mu: Option<f64>,
    /// Association parameters
    #[serde(flatten)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub association_record: Option<AssociationRecord>,
    /// interaction range parameter for the dispersion functional
    #[serde(skip_serializing_if = "Option::is_none")]
    pub psi_dft: Option<f64>,
}

impl GcPcSaftRecord {
    pub fn new(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        mu: Option<f64>,
        association_record: Option<AssociationRecord>,
        psi_dft: Option<f64>,
    ) -> Self {
        Self {
            m,
            sigma,
            epsilon_k,
            mu,
            association_record,
            psi_dft,
        }
    }
}

impl std::fmt::Display for GcPcSaftRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GcPcSaftRecord(m={}", self.m)?;
        write!(f, ", sigma={}", self.sigma)?;
        write!(f, ", epsilon_k={}", self.epsilon_k)?;
        if let Some(n) = &self.mu {
            write!(f, ", mu={}", n)?;
        }
        if let Some(n) = &self.association_record {
            write!(f, ", association_record={}", n)?;
        }
        write!(f, ")")
    }
}
