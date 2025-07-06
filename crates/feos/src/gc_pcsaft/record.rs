use feos_core::parameter::GcParameters;
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
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub mu: f64,
    /// interaction range parameter for the dispersion functional
    #[serde(skip_serializing_if = "Option::is_none")]
    pub psi_dft: Option<f64>,
}

impl GcPcSaftRecord {
    pub fn new(m: f64, sigma: f64, epsilon_k: f64, mu: f64, psi_dft: Option<f64>) -> Self {
        Self {
            m,
            sigma,
            epsilon_k,
            mu,
            psi_dft,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct GcPcSaftAssociationRecord {
    /// Association volume parameter
    pub kappa_ab: f64,
    /// Association energy parameter in units of Kelvin
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

/// Parameter set required for the gc-PC-SAFT equation of state.
pub type GcPcSaftParameters<C> =
    GcParameters<GcPcSaftRecord, f64, GcPcSaftAssociationRecord, (), C>;
