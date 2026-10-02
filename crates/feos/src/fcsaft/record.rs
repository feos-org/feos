use feos_core::FeosResult;
use feos_core::parameter::{
    BinarySegmentRecord, ChemicalRecord, CombiningRule, GcParameters, Identifier, SegmentRecord,
};
use serde::{Deserialize, Serialize};

/// FC-SAFT segment parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct FcSaftRecord {
    /// Segment diameter in units of Angstrom
    pub sigma: f64,
    /// Energetic parameter in units of Kelvin
    pub epsilon_k: f64,
    /// Interaction range parameter for the dispersion functional
    #[serde(skip_serializing_if = "Option::is_none")]
    pub psi_dft: Option<f64>,
}

impl FcSaftRecord {
    pub fn new(sigma: f64, epsilon_k: f64, psi_dft: Option<f64>) -> Self {
        Self {
            sigma,
            epsilon_k,
            psi_dft,
        }
    }
}

/// FC-SAFT association parameters.
#[derive(Serialize, Deserialize, Clone, Copy, Default, PartialEq, Debug)]
pub struct FcSaftAssociationRecord {
    /// Association volume parameter
    pub kappa_ab: f64,
    /// Association energy parameter in units of Kelvin
    pub epsilon_k_ab: f64,
}

impl FcSaftAssociationRecord {
    pub fn new(kappa_ab: f64, epsilon_k_ab: f64) -> Self {
        Self {
            kappa_ab,
            epsilon_k_ab,
        }
    }

    pub(super) fn combine(parameters_i: &Self, parameters_j: &Self) -> Self {
        Self {
            kappa_ab: (parameters_i.kappa_ab * parameters_j.kappa_ab).sqrt(),
            epsilon_k_ab: 0.5 * (parameters_i.epsilon_k_ab + parameters_j.epsilon_k_ab),
        }
    }
}

impl CombiningRule<FcSaftRecord> for FcSaftAssociationRecord {
    fn combining_rule(
        _: &FcSaftRecord,
        _: &FcSaftRecord,
        parameters_i: &Self,
        parameters_j: &Self,
    ) -> Self {
        Self::combine(parameters_i, parameters_j)
    }
}

/// FC-SAFT binary segment-segment interaction parameters.
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct FcSaftBinaryRecord {
    /// Binary dispersion interaction parameter
    #[serde(default)]
    pub k_ij: f64,
}

impl FcSaftBinaryRecord {
    pub fn new(k_ij: f64) -> Self {
        Self { k_ij }
    }
}

/// FC-SAFT bond parameters.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct FcSaftBondRecord {
    /// Distance between the centers of two bonded segments in units of Angstrom
    pub bond_length: f64,
}

impl FcSaftBondRecord {
    pub fn new(bond_length: f64) -> Self {
        Self { bond_length }
    }
}

/// Parameter set required for the FC-SAFT equation of state and Helmholtz energy functional.
pub type FcSaftParameters =
    GcParameters<FcSaftRecord, FcSaftBinaryRecord, FcSaftAssociationRecord, FcSaftBondRecord, ()>;

/// Parameters for a pure component that consists of `segments` identical
/// fused spheres with the given bond length.
pub fn new_pure_homosegmented(
    identifier: Identifier,
    segments: usize,
    sigma: f64,
    epsilon_k: f64,
    bond_length: f64,
    molarweight: f64,
    psi_dft: Option<f64>,
) -> FeosResult<FcSaftParameters> {
    let chemical = ChemicalRecord::new(identifier, vec!["S".into(); segments], None);
    let segment = SegmentRecord::new(
        "S".into(),
        molarweight / segments as f64,
        FcSaftRecord::new(sigma, epsilon_k, psi_dft),
    );
    let bond = BinarySegmentRecord::new(
        "S".into(),
        "S".into(),
        Some(FcSaftBondRecord::new(bond_length)),
    );
    FcSaftParameters::from_segments_with_bonds(vec![chemical], &[segment], None, &[bond])
}
