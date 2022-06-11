use serde::{Deserialize, Serialize};

/// gc-PC-SAFT pure-component parameters.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct GcPcSaftRecord {
    /// Segment shape factor
    pub m: f64,
    /// Segment diameter in units of Angstrom
    pub sigma: f64,
    /// Energetic parameter in units of Kelvin
    pub epsilon_k: f64,
    /// Dipole moment in units of Debye
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mu: Option<f64>,
    /// association volume parameter
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kappa_ab: Option<f64>,
    /// association energy parameter
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epsilon_k_ab: Option<f64>,
    /// \# of association sites of type A
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub na: Option<f64>,
    /// \# of association sites of type B
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nb: Option<f64>,
    /// interaction range parameter for the dispersion functional
    #[serde(default)]
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
        psi_dft: Option<f64>,
    ) -> Self {
        Self {
            m,
            sigma,
            epsilon_k,
            mu,
            kappa_ab,
            epsilon_k_ab,
            na,
            nb,
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
        if let Some(n) = &self.kappa_ab {
            write!(f, ", kappa_ab={}", n)?;
        }
        if let Some(n) = &self.epsilon_k_ab {
            write!(f, ", epsilon_k_ab={}", n)?;
        }
        if let Some(n) = &self.na {
            write!(f, ", na={}", n)?;
        }
        if let Some(n) = &self.nb {
            write!(f, ", nb={}", n)?;
        }
        write!(f, ")")
    }
}
