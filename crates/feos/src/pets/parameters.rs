use feos_core::parameter::Parameters;
use serde::{Deserialize, Serialize};

/// PeTS parameters for a pure substance.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PetsRecord {
    /// Segment diameter in units of Angstrom
    pub sigma: f64,
    /// Energetic parameter in units of Kelvin
    pub epsilon_k: f64,
}

impl PetsRecord {
    /// New PeTS parameters for a pure substance.
    ///
    /// # Example
    ///
    /// ```
    /// use feos::pets::PetsRecord;
    /// let record = PetsRecord::new(3.7, 120.0);
    /// ```
    pub fn new(sigma: f64, epsilon_k: f64) -> PetsRecord {
        PetsRecord { sigma, epsilon_k }
    }
}

/// Parameters that modify binary interactions.
///
/// $\varepsilon_{k,ij} = (1 - k_{ij})\sqrt{\varepsilon_{k,i} \varepsilon_{k,j}}$
#[derive(Serialize, Deserialize, Clone, Copy, Default, Debug)]
pub struct PetsBinaryRecord {
    pub k_ij: f64,
}

/// Parameter set for the PeTS equation of state and Helmholtz energy functional.
pub type PetsParameters = Parameters<PetsRecord, PetsBinaryRecord, ()>;

#[cfg(test)]
pub mod utils {
    use super::*;
    use feos_core::parameter::PureRecord;

    pub fn argon_parameters() -> PetsParameters {
        let argon_json = r#"
            {
                "identifier": {
                    "cas": "7440-37-1",
                    "name": "argon",
                    "iupac_name": "argon",
                    "smiles": "[Ar]",
                    "inchi": "InChI=1/Ar",
                    "formula": "Ar"
                },
                "sigma": 3.4050,
                "epsilon_k": 119.8,
                "molarweight": 39.948
            }"#;
        let argon_record: PureRecord<PetsRecord, ()> =
            serde_json::from_str(argon_json).expect("Unable to parse json.");
        PetsParameters::new_pure(argon_record)
    }

    pub fn krypton_parameters() -> PetsParameters {
        let krypton_json = r#"
            {
                "identifier": {
                    "cas": "7439-90-9",
                    "name": "krypton",
                    "iupac_name": "krypton",
                    "smiles": "[Kr]",
                    "inchi": "InChI=1S/Kr",
                    "formula": "Kr"
                },
                "sigma": 3.6300,
                "epsilon_k": 163.10,
                "molarweight": 83.798
            }"#;
        let krypton_record: PureRecord<PetsRecord, ()> =
            serde_json::from_str(krypton_json).expect("Unable to parse json.");
        PetsParameters::new_pure(krypton_record)
    }

    pub fn argon_krypton_parameters() -> PetsParameters {
        let binary_json = r#"[
            {
                "identifier": {
                    "cas": "7440-37-1",
                    "name": "argon",
                    "iupac_name": "argon",
                    "smiles": "[Ar]",
                    "inchi": "1/Ar",
                    "formula": "Ar"
                },
                "sigma": 3.4050,
                "epsilon_k": 119.8,
                "molarweight": 39.948
            },
            {
                "identifier": {
                    "cas": "7439-90-9",
                    "name": "krypton",
                    "iupac_name": "krypton",
                    "smiles": "[Kr]",
                    "inchi": "InChI=1S/Kr",
                    "formula": "Kr"
                },
                "sigma": 3.6300,
                "epsilon_k": 163.10,
                "molarweight": 83.798
            }
        ]"#;
        let binary_record: [PureRecord<PetsRecord, ()>; 2] =
            serde_json::from_str(binary_json).expect("Unable to parse json.");
        PetsParameters::new_binary(binary_record, None, vec![])
    }
}
