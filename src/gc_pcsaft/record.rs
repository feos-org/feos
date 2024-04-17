use crate::association::AssociationRecord;
use num_traits::Zero;
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
    pub association_record: Option<AssociationRecord<GcPcSaftAssociationRecord>>,
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
        kappa_ab: Option<f64>,
        epsilon_k_ab: Option<f64>,
        na: Option<f64>,
        nb: Option<f64>,
        nc: Option<f64>,
        psi_dft: Option<f64>,
    ) -> Self {
        let association_record = if kappa_ab.is_none()
            && epsilon_k_ab.is_none()
            && na.is_none()
            && nb.is_none()
            && nc.is_none()
        {
            None
        } else {
            Some(AssociationRecord::new(
                GcPcSaftAssociationRecord::new(
                    kappa_ab.unwrap_or_default(),
                    epsilon_k_ab.unwrap_or_default(),
                ),
                na.unwrap_or_default(),
                nb.unwrap_or_default(),
                nc.unwrap_or_default(),
            ))
        };
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

#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct GcPcSaftAssociationRecord {
    /// Association volume parameter
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub kappa_ab: f64,
    /// Association energy parameter in units of Kelvin
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub epsilon_k_ab: f64,
}

impl GcPcSaftAssociationRecord {
    pub fn new(kappa_ab: f64, epsilon_k_ab: f64) -> Self {
        Self {
            kappa_ab,
            epsilon_k_ab,
        }
    }
}

impl std::fmt::Display for GcPcSaftAssociationRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GcPcSaftAssociationRecord(kappa_ab={}", self.kappa_ab)?;
        write!(f, ", epsilon_k_ab={})", self.epsilon_k_ab)
    }
}
