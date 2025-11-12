use feos_core::FeosResult;
use feos_core::parameter::Parameters;
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use num_traits::Zero;
use quantity::{GRAM, MOL};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Write;

use super::corresponding_states::CorrespondingParameters;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NoRecord;

impl fmt::Display for NoRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum QuantumCorrection {
    FeynmanHibbs1 {
        #[serde(skip_serializing_if = "Option::is_none")]
        c_sigma: Option<[f64; 3]>,
        #[serde(skip_serializing_if = "Option::is_none")]
        c_epsilon_k: Option<[f64; 3]>,
        #[serde(skip_serializing_if = "Option::is_none")]
        c_rep: Option<[f64; 5]>,
    },
}

impl std::fmt::Display for QuantumCorrection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QuantumCorrection(")?;
        match self {
            Self::FeynmanHibbs1 {
                c_sigma,
                c_epsilon_k,
                c_rep: c_lr,
            } => {
                write!(f, "c_sigma={:?}", c_sigma.unwrap_or([1.0; 3]))?;
                write!(f, ", c_epsilon_k={:?}", c_epsilon_k.unwrap_or([1.0; 3]))?;
                write!(f, ", c_lr={:?}", c_lr.unwrap_or([1.0; 5]))?;
            }
            _ => (),
        };
        write!(f, ")")
    }
}

/// uv-theory parameters for a pure substance
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UVCSRecord {
    pub rep: f64,
    pub att: f64,
    pub sigma: f64,
    pub epsilon_k: f64,
    #[serde(skip_serializing_if = "Option::is_none", flatten)]
    pub quantum_correction: Option<QuantumCorrection>,
}

impl UVCSRecord {
    /// Single substance record for uv-theory
    pub fn new(
        rep: f64,
        att: f64,
        sigma: f64,
        epsilon_k: f64,
        quantum_correction: Option<QuantumCorrection>,
    ) -> Self {
        Self {
            rep,
            att,
            sigma,
            epsilon_k,
            quantum_correction,
        }
    }
}

impl std::fmt::Display for UVCSRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UVCSRecord(")?;
        write!(f, ", att={}", self.att)?;
        write!(f, ", rep={}", self.rep)?;
        write!(f, ", sigma={}", self.sigma)?;
        write!(f, ", epsilon_k={}", self.epsilon_k)?;
        if let Some(qc) = self.quantum_correction.as_ref() {
            write!(f, ", quantum_correction={}", qc)?;
        }
        write!(f, ")")
    }
}

/// Binary interaction parameters
#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct UVCSBinaryRecord {
    pub k_ij: f64,
    pub l_ij: f64,
}

impl std::fmt::Display for UVCSBinaryRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UVBinaryRecord(k_ij={}, l_ij={})", self.k_ij, self.l_ij)
    }
}

#[inline]
pub fn mie_prefactor<D: DualNum<f64> + Copy>(rep: D, att: D) -> D {
    rep / (rep - att) * (rep / att).powd(att / (rep - att))
}

#[inline]
pub fn mean_field_constant<D: DualNum<f64> + Copy>(rep: D, att: D, x: D) -> D {
    mie_prefactor(rep, att) * (x.powd(-att + 3.0) / (att - 3.0) - x.powd(-rep + 3.0) / (rep - 3.0))
}

/// Parameter set required for the SAFT-VRQ Mie equation of state and Helmholtz energy functional.
pub type UVCSParameters = Parameters<UVCSRecord, UVCSBinaryRecord, ()>;

pub struct UVCSPars {
    pub ncomponents: usize,
    pub rep: DVector<f64>,
    pub att: DVector<f64>,
    pub sigma: DVector<f64>,
    pub epsilon_k: DVector<f64>,
    pub molarweight: DVector<f64>,
    pub k_ij: DMatrix<f64>,
    pub l_ij: DMatrix<f64>,
    pub quantum_correction: Vec<Option<QuantumCorrection>>,
}

impl UVCSPars {
    pub fn new(parameters: &UVCSParameters) -> Self {
        let ncomponents = parameters.pure.len();

        let [sigma, epsilon_k] = parameters.collate(|pr| [pr.sigma, pr.epsilon_k]);
        let [rep, att] = parameters.collate(|pr| [pr.rep, pr.att]);
        let [k_ij, l_ij] = parameters.collate_binary(|b| [b.k_ij, b.l_ij]);
        let molarweight = parameters.molar_weight.clone().convert_into(GRAM / MOL);
        let quantum_correction = parameters
            .pure
            .iter()
            .map(|pr| pr.model_record.quantum_correction.clone())
            .collect();

        Self {
            ncomponents,
            rep,
            att,
            sigma,
            epsilon_k,
            molarweight,
            k_ij,
            l_ij,
            quantum_correction,
        }
    }
}

// impl Parameter for UVCSPars {
//     type Pure = UVCSRecord;
//     type Binary = UVCSBinaryRecord;

//     fn from_records(
//         pure_records: Vec<PureRecord<Self::Pure>>,
//         binary_records: Option<DMatrix<Self::Binary>>,
//     ) -> Result<Self, ParameterError> {
//         let n = pure_records.len();

//         let mut molarweight = Array::zeros(n);
//         let mut rep = Array::zeros(n);
//         let mut att = Array::zeros(n);
//         let mut sigma = Array::zeros(n);
//         let mut epsilon_k = Array::zeros(n);
//         let mut component_index = HashMap::with_capacity(n);

//         for (i, record) in pure_records.iter().enumerate() {
//             component_index.insert(record.identifier.clone(), i);
//             let r = &record.model_record;
//             rep[i] = r.rep;
//             att[i] = r.att;
//             sigma[i] = r.sigma;
//             epsilon_k[i] = r.epsilon_k;
//             // construction of molar weights for GC methods, see Builder
//             molarweight[i] = record.molarweight;
//         }

//         let k_ij = binary_records.as_ref().map(|br| br.map(|br| br.k_ij));
//         let l_ij = binary_records.as_ref().map(|br| br.map(|br| br.l_ij));
//         Ok(Self {
//             ncomponents: n,
//             rep,
//             att,
//             sigma,
//             epsilon_k,
//             molarweight,
//             k_ij,
//             l_ij,
//             pure_records,
//             binary_records,
//         })
//     }

//     fn records(&self) -> (&[PureRecord<UVCSRecord>], Option<&DMatrix<UVCSBinaryRecord>>) {
//         (&self.pure_records, self.binary_records.as_ref())
//     }
// }

// impl UVCSPars {
//     /// Parameters for a single substance with molar weight one and no (default) ideal gas contributions.
//     pub fn new_simple(
//         rep: f64,
//         att: f64,
//         sigma: f64,
//         epsilon_k: f64,
//     ) -> Result<Self, ParameterError> {
//         let model_record = UVCSRecord::new(rep, att, sigma, epsilon_k, None);
//         let pure_record = PureRecord::new(Identifier::default(), 1.0, model_record);
//         Self::new_pure(pure_record)
//     }

//     /// Markdown representation of parameters.
//     pub fn to_markdown(&self) -> String {
//         let mut output = String::new();
//         let o = &mut output;
//         write!(
//             o,
//             "|component|molarweight|$\\sigma$|$\\varepsilon$|$m$|$n$|\n|-|-|-|-|-|-|"
//         )
//         .unwrap();
//         for i in 0..self.pure_records.len() {
//             let component = self.pure_records[i].identifier.name.clone();
//             let component = component.unwrap_or(format!("Component {}", i + 1));
//             write!(
//                 o,
//                 "\n|{}|{}|{}|{}|{}|{}|",
//                 component,
//                 self.molarweight[i],
//                 self.sigma[i],
//                 self.epsilon_k[i],
//                 self.rep[i],
//                 self.att[i],
//             )
//             .unwrap();
//         }
//         output
//     }

//     pub fn print_effective_parameters(&self, temperature: f64) -> String {
//         let parameters = CorrespondingParameters::new(&self, temperature);
//         let mut output = String::new();
//         let o = &mut output;
//         write!(
//             o,
//             "|component|molarweight|$\\sigma$|$\\varepsilon$|$m$|$n$|\n|-|-|-|-|-|-|"
//         )
//         .unwrap();
//         for i in 0..self.pure_records.len() {
//             let component = self.pure_records[i].identifier.name.clone();
//             let component = component.unwrap_or(format!("Component {}", i + 1));
//             write!(
//                 o,
//                 "\n|{}|{}|{}|{}|{}|{}|",
//                 component,
//                 self.molarweight[i],
//                 parameters.sigma[i],
//                 parameters.epsilon_k[i],
//                 parameters.rep[i],
//                 parameters.att[i],
//             )
//             .unwrap();
//         }
//         output
//     }
// }

impl std::fmt::Display for UVCSPars {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UVCSParameters(")?;
        write!(f, "\n\tmolarweight={}", self.molarweight)?;
        write!(f, "\n\tsigma={}", self.sigma)?;
        write!(f, "\n\tepsilon_k={}", self.epsilon_k)?;
        write!(f, "\n\trep={}", self.rep)?;
        write!(f, "\n\tatt={}", self.att)?;

        if !self.k_ij.iter().all(|k| k.is_zero()) {
            write!(f, "\n\tk_ij=\n{}", self.k_ij)?;
        }
        if !self.l_ij.iter().all(|k| k.is_zero()) {
            write!(f, "\n\tl_ij=\n{}", self.l_ij)?;
        }
        write!(f, "\n)")
    }
}

#[cfg(test)]
pub mod utils {
    use super::*;
    use feos_core::parameter::{Identifier, PureRecord};
    use std::f64;

    pub fn test_parameters(rep: f64, att: f64, sigma: f64, epsilon: f64) -> UVCSParameters {
        let identifier = Identifier::new(Some("1"), None, None, None, None, None);
        let model_record = UVCSRecord::new(rep, att, sigma, epsilon, None);
        let pr = PureRecord::new(identifier, 1.0, model_record);
        let parameters = Parameters::new_pure(pr).unwrap();
        parameters
        // UVCSPars::new(&parameters)
    }

    pub fn test_parameters_mixture(
        rep: DVector<f64>,
        att: DVector<f64>,
        sigma: DVector<f64>,
        epsilon: DVector<f64>,
    ) -> UVCSParameters {
        let identifier = Identifier::new(Some("1"), None, None, None, None, None);
        let model_record = UVCSRecord::new(rep[0], att[0], sigma[0], epsilon[0], None);
        let pr1 = PureRecord::new(identifier, 1.0, model_record);
        //
        let identifier2 = Identifier::new(Some("1"), None, None, None, None, None);
        let model_record2 = UVCSRecord::new(rep[1], att[1], sigma[1], epsilon[1], None);
        let pr2 = PureRecord::new(identifier2, 1.0, model_record2);
        let pure_records = [pr1, pr2];
        let parameters = Parameters::new_binary(pure_records, None, vec![]).unwrap();
        parameters
        // UVCSPars::new(&parameters)
    }

    pub fn methane_parameters(rep: f64, att: f64) -> UVCSParameters {
        let identifier = Identifier::new(Some("1"), None, None, None, None, None);
        let model_record = UVCSRecord::new(rep, att, 3.7039, 150.03, None);
        let pr = PureRecord::new(identifier, 1.0, model_record);
        let parameters = Parameters::new_pure(pr).unwrap();
        parameters
        // UVCSPars::new(&parameters)
    }
}
