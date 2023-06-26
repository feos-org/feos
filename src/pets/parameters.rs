use crate::hard_sphere::{HardSphereProperties, MonomerShape};
use feos_core::joback::JobackRecord;
use feos_core::parameter::{Parameter, ParameterError, PureRecord};
use ndarray::{Array, Array1, Array2};
use num_dual::DualNum;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write;

/// PeTS parameters for a pure substance.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct PetsRecord {
    /// Segment diameter in units of Angstrom
    pub sigma: f64,
    /// Energetic parameter in units of Kelvin
    pub epsilon_k: f64,
    /// Entropy scaling parameters for viscosity
    #[serde(skip_serializing_if = "Option::is_none")]
    pub viscosity: Option<[f64; 4]>,
    /// Entropy scaling parameters for self-diffusion coefficient
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diffusion: Option<[f64; 5]>,
    /// Entropy scaling parameters for thermal conductivity
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thermal_conductivity: Option<[f64; 4]>,
}

impl std::fmt::Display for PetsRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PetsRecord(sigma={}", self.sigma)?;
        write!(f, ", epsilon_k={}", self.epsilon_k)?;
        if let Some(n) = &self.viscosity {
            write!(f, ", viscosity={:?}", n)?;
        }
        if let Some(n) = &self.diffusion {
            write!(f, ", diffusion={:?}", n)?;
        }
        if let Some(n) = &self.thermal_conductivity {
            write!(f, ", thermal_conductivity={:?}", n)?;
        }
        write!(f, ")")
    }
}

impl PetsRecord {
    /// New PeTS parameters for a pure substance.
    ///
    /// # Example
    ///
    /// ```
    /// use feos::pets::PetsRecord;
    /// let record = PetsRecord::new(3.7, 120.0, None, None, None);
    /// ```
    pub fn new(
        sigma: f64,
        epsilon_k: f64,
        viscosity: Option<[f64; 4]>,
        diffusion: Option<[f64; 5]>,
        thermal_conductivity: Option<[f64; 4]>,
    ) -> PetsRecord {
        PetsRecord {
            sigma,
            epsilon_k,
            viscosity,
            diffusion,
            thermal_conductivity,
        }
    }
}

/// Parameters that modify binary interactions.
///
/// $\varepsilon_{k,ij} = (1 - k_{ij})\sqrt{\varepsilon_{k,i} \varepsilon_{k,j}}$
#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct PetsBinaryRecord {
    k_ij: f64,
}

impl From<f64> for PetsBinaryRecord {
    fn from(k_ij: f64) -> Self {
        Self { k_ij }
    }
}

impl From<PetsBinaryRecord> for f64 {
    fn from(binary_record: PetsBinaryRecord) -> Self {
        binary_record.k_ij
    }
}

impl std::fmt::Display for PetsBinaryRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PetsBinaryRecord(k_ij={})", self.k_ij)
    }
}

/// Parameter set for the PeTS equation of state and Helmholtz energy functional.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct PetsParameters {
    /// molar weight in gram per mole
    pub molarweight: Array1<f64>,
    /// Lennard-Jones diameter in Angstrom
    pub sigma: Array1<f64>,
    /// Lennard-Jones energy parameter in Kelvin
    pub epsilon_k: Array1<f64>,
    /// binary interaction parameter
    pub k_ij: Array2<f64>,
    /// diameter matrix
    pub sigma_ij: Array2<f64>,
    /// energy parameter matrix including k_ij
    pub epsilon_k_ij: Array2<f64>,
    /// energy parameter matrix not including k_ij
    pub e_k_ij: Array2<f64>,
    /// viscosity parameters for entropy scaling
    pub viscosity: Option<Array2<f64>>,
    /// diffusion parameters for entropy scaling
    pub diffusion: Option<Array2<f64>>,
    /// thermal conductivity parameters for entropy scaling
    pub thermal_conductivity: Option<Array2<f64>>,
    /// records of all pure substances of the system
    pub pure_records: Vec<PureRecord<PetsRecord, JobackRecord>>,
    /// records of parameters for Joback method
    pub joback_records: Option<Vec<JobackRecord>>,
    /// records of all binary interaction parameters
    pub binary_records: Array2<PetsBinaryRecord>,
}

impl Parameter for PetsParameters {
    type Pure = PetsRecord;
    type IdealGas = JobackRecord;
    type Binary = PetsBinaryRecord;

    fn from_records(
        pure_records: Vec<PureRecord<Self::Pure, Self::IdealGas>>,
        binary_records: Array2<PetsBinaryRecord>,
    ) -> Result<Self, ParameterError> {
        let n = pure_records.len();

        let mut molarweight = Array::zeros(n);
        let mut sigma = Array::zeros(n);
        let mut epsilon_k = Array::zeros(n);
        let mut viscosity = Vec::with_capacity(n);
        let mut diffusion = Vec::with_capacity(n);
        let mut thermal_conductivity = Vec::with_capacity(n);

        let mut component_index = HashMap::with_capacity(n);

        for (i, record) in pure_records.iter().enumerate() {
            component_index.insert(record.identifier.clone(), i);
            let r = &record.model_record;
            sigma[i] = r.sigma;
            epsilon_k[i] = r.epsilon_k;
            viscosity.push(r.viscosity);
            diffusion.push(r.diffusion);
            thermal_conductivity.push(r.thermal_conductivity);
            molarweight[i] = record.molarweight;
        }

        let k_ij = binary_records.map(|br| br.k_ij);
        let mut epsilon_k_ij = Array::zeros((n, n));
        let mut sigma_ij = Array::zeros((n, n));
        let mut e_k_ij = Array::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                e_k_ij[[i, j]] = (epsilon_k[i] * epsilon_k[j]).sqrt();
                epsilon_k_ij[[i, j]] = (1.0 - k_ij[[i, j]]) * e_k_ij[[i, j]];
                sigma_ij[[i, j]] = 0.5 * (sigma[i] + sigma[j]);
            }
        }

        let viscosity_coefficients = if viscosity.iter().any(|v| v.is_none()) {
            None
        } else {
            let mut v = Array2::zeros((4, viscosity.len()));
            for (i, vi) in viscosity.iter().enumerate() {
                v.column_mut(i).assign(&Array1::from(vi.unwrap().to_vec()));
            }
            Some(v)
        };

        let diffusion_coefficients = if diffusion.iter().any(|v| v.is_none()) {
            None
        } else {
            let mut v = Array2::zeros((5, diffusion.len()));
            for (i, vi) in diffusion.iter().enumerate() {
                v.column_mut(i).assign(&Array1::from(vi.unwrap().to_vec()));
            }
            Some(v)
        };

        let thermal_conductivity_coefficients = if thermal_conductivity.iter().any(|v| v.is_none())
        {
            None
        } else {
            let mut v = Array2::zeros((4, thermal_conductivity.len()));
            for (i, vi) in thermal_conductivity.iter().enumerate() {
                v.column_mut(i).assign(&Array1::from(vi.unwrap().to_vec()));
            }
            Some(v)
        };

        let joback_records = pure_records
            .iter()
            .map(|r| r.ideal_gas_record.clone())
            .collect();

        Ok(Self {
            molarweight,
            sigma,
            epsilon_k,
            k_ij,
            sigma_ij,
            epsilon_k_ij,
            e_k_ij,
            viscosity: viscosity_coefficients,
            diffusion: diffusion_coefficients,
            thermal_conductivity: thermal_conductivity_coefficients,
            pure_records,
            joback_records,
            binary_records,
        })
    }

    fn records(
        &self,
    ) -> (
        &[PureRecord<PetsRecord, JobackRecord>],
        &Array2<PetsBinaryRecord>,
    ) {
        (&self.pure_records, &self.binary_records)
    }
}

impl HardSphereProperties for PetsParameters {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<N> {
        MonomerShape::Spherical(self.sigma.len())
    }

    fn hs_diameter<D: DualNum<f64>>(&self, temperature: D) -> Array1<D> {
        let ti = temperature.recip() * -3.052785558;
        Array::from_shape_fn(self.sigma.len(), |i| {
            -((ti * self.epsilon_k[i]).exp() * 0.127112544 - 1.0) * self.sigma[i]
        })
    }
}

impl PetsParameters {
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();
        let o = &mut output;
        write!(
            o,
            "|component|molarweight|$\\sigma$|$\\varepsilon$|\n|-|-|-|-|"
        )
        .unwrap();
        for i in 0..self.sigma.len() {
            let component = self.pure_records[i].identifier.name.clone();
            let component = component.unwrap_or(format!("Component {}", i + 1));
            write!(
                o,
                "\n|{}|{}|{}|{}|",
                component, self.molarweight[i], self.sigma[i], self.epsilon_k[i],
            )
            .unwrap();
        }

        output
    }
}

impl std::fmt::Display for PetsParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PetsParameters(")?;
        write!(f, "\n\tmolarweight={}", self.molarweight)?;
        write!(f, "\n\tsigma={}", self.sigma)?;
        write!(f, "\n\tepsilon_k={}", self.epsilon_k)?;
        if !self.k_ij.iter().all(|k| k.is_zero()) {
            write!(f, "\n\tk_ij=\n{}", self.k_ij)?;
        }
        write!(f, "\n)")
    }
}

#[cfg(test)]
pub mod utils {
    use super::*;
    use feos_core::joback::JobackRecord;
    use std::sync::Arc;

    pub fn argon_parameters() -> Arc<PetsParameters> {
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
                "model_record": {
                    "sigma": 3.4050,
                    "epsilon_k": 119.8,
                    "viscosity": [0.0, 0.0, 0.0, 0.0],
                    "thermal_conductivity": [0.0, 0.0, 0.0, 0.0],
                    "diffusion": [0.0, 0.0, 0.0, 0.0, 0.0]
                },
                "molarweight": 39.948
            }"#;
        let argon_record: PureRecord<PetsRecord, JobackRecord> =
            serde_json::from_str(argon_json).expect("Unable to parse json.");
        Arc::new(PetsParameters::new_pure(argon_record).unwrap())
    }

    pub fn krypton_parameters() -> Arc<PetsParameters> {
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
                "model_record": {
                    "sigma": 3.6300,
                    "epsilon_k": 163.10
                },
                "molarweight": 83.798
            }"#;
        let krypton_record: PureRecord<PetsRecord, JobackRecord> =
            serde_json::from_str(krypton_json).expect("Unable to parse json.");
        Arc::new(PetsParameters::new_pure(krypton_record).unwrap())
    }

    pub fn argon_krypton_parameters() -> Arc<PetsParameters> {
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
                "model_record": {
                    "sigma": 3.4050,
                    "epsilon_k": 119.8,
                    "viscosity": [0.0, 0.0, 0.0, 0.0],
                    "thermal_conductivity": [0.0, 0.0, 0.0, 0.0],
                    "diffusion": [0.0, 0.0, 0.0, 0.0, 0.0]
                },
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
                "model_record": {
                    "sigma": 3.6300,
                    "epsilon_k": 163.10,
                    "viscosity": [0.0, 0.0, 0.0, 0.0],
                    "thermal_conductivity": [0.0, 0.0, 0.0, 0.0],
                    "diffusion": [0.0, 0.0, 0.0, 0.0, 0.0]
                },
                "molarweight": 83.798
            }
        ]"#;
        let binary_record: Vec<PureRecord<PetsRecord, JobackRecord>> =
            serde_json::from_str(binary_json).expect("Unable to parse json.");
        Arc::new(PetsParameters::new_binary(binary_record, None).unwrap())
    }
}
