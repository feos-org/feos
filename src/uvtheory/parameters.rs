use feos_core::parameter::{Identifier, ParameterError};
use feos_core::parameter::{Parameter, PureRecord};
use lazy_static::lazy_static;
use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray::Array2;
use num_dual::DualNum;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Write;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NoRecord;

impl fmt::Display for NoRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

/// uv-theory parameters for a pure substance
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UVRecord {
    rep: f64,
    att: f64,
    sigma: f64,
    epsilon_k: f64,
}

impl UVRecord {
    /// Single substance record for uv-theory
    pub fn new(rep: f64, att: f64, sigma: f64, epsilon_k: f64) -> Self {
        Self {
            rep,
            att,
            sigma,
            epsilon_k,
        }
    }
}

impl std::fmt::Display for UVRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UVRecord(m={}", self.rep)?;
        write!(f, ", att={}", self.att)?;
        write!(f, ", sigma={}", self.sigma)?;
        write!(f, ", epsilon_k={}", self.epsilon_k)?;
        write!(f, ")")
    }
}

/// Binary interaction parameters
#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct UVBinaryRecord {
    pub k_ij: f64,
}

impl From<f64> for UVBinaryRecord {
    fn from(k_ij: f64) -> Self {
        Self { k_ij }
    }
}

impl From<UVBinaryRecord> for f64 {
    fn from(binary_record: UVBinaryRecord) -> Self {
        binary_record.k_ij
    }
}

impl std::fmt::Display for UVBinaryRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UVBinaryRecord(k_ij={})", self.k_ij)
    }
}

lazy_static! {
/// Constants for BH temperature dependent HS diameter.
    static ref CD_BH: Array2<f64> = arr2(&[
        [0.0, 1.09360455168912E-02, 0.0],
        [-2.00897880971934E-01, -1.27074910870683E-02, 0.0],
        [
            1.40422470174053E-02,
            7.35946850956932E-02,
            1.28463973950737E-02,
        ],
        [
            3.71527116894441E-03,
            5.05384813757953E-03,
            4.91003312452622E-02,
        ],
    ]);
}

#[inline]
pub fn mie_prefactor<D: DualNum<f64>>(rep: D, att: D) -> D {
    rep / (rep - att) * (rep / att).powd(att / (rep - att))
}

#[inline]
pub fn mean_field_constant<D: DualNum<f64>>(rep: D, att: D, x: D) -> D {
    mie_prefactor(rep, att) * (x.powd(-att + 3.0) / (att - 3.0) - x.powd(-rep + 3.0) / (rep - 3.0))
}

/// Parameters for all substances for uv-theory equation of state and Helmholtz energy functional
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UVParameters {
    pub ncomponents: usize,
    pub rep: Array1<f64>,
    pub att: Array1<f64>,
    pub sigma: Array1<f64>,
    pub epsilon_k: Array1<f64>,
    pub molarweight: Array1<f64>,
    pub k_ij: Array2<f64>,
    pub rep_ij: Array2<f64>,
    pub att_ij: Array2<f64>,
    pub sigma_ij: Array2<f64>,
    pub eps_k_ij: Array2<f64>,
    pub cd_bh_pure: Vec<Array1<f64>>,
    pub cd_bh_binary: Array2<Array1<f64>>,
    pub pure_records: Vec<PureRecord<UVRecord, NoRecord>>,
    pub binary_records: Array2<UVBinaryRecord>,
}

impl Parameter for UVParameters {
    type Pure = UVRecord;
    type IdealGas = NoRecord;
    type Binary = UVBinaryRecord;

    fn from_records(
        pure_records: Vec<PureRecord<Self::Pure, Self::IdealGas>>,
        binary_records: Array2<Self::Binary>,
    ) -> Result<Self, ParameterError> {
        let n = pure_records.len();

        let mut molarweight = Array::zeros(n);
        let mut rep = Array::zeros(n);
        let mut att = Array::zeros(n);
        let mut sigma = Array::zeros(n);
        let mut epsilon_k = Array::zeros(n);
        let mut component_index = HashMap::with_capacity(n);

        for (i, record) in pure_records.iter().enumerate() {
            component_index.insert(record.identifier.clone(), i);
            let r = &record.model_record;
            rep[i] = r.rep;
            att[i] = r.att;
            sigma[i] = r.sigma;
            epsilon_k[i] = r.epsilon_k;
            // construction of molar weights for GC methods, see Builder
            molarweight[i] = record.molarweight;
        }

        let mut rep_ij = Array2::zeros((n, n));
        let mut att_ij = Array2::zeros((n, n));
        let mut sigma_ij = Array2::zeros((n, n));
        let mut eps_k_ij = Array2::zeros((n, n));
        let k_ij = binary_records.map(|br| br.k_ij);

        for i in 0..n {
            rep_ij[[i, i]] = rep[i];
            att_ij[[i, i]] = att[i];
            sigma_ij[[i, i]] = sigma[i];
            eps_k_ij[[i, i]] = epsilon_k[i];
            for j in i + 1..n {
                rep_ij[[i, j]] = (rep[i] * rep[j]).sqrt();
                rep_ij[[j, i]] = rep_ij[[i, j]];
                att_ij[[i, j]] = (att[i] * att[j]).sqrt();
                att_ij[[j, i]] = att_ij[[i, j]];
                sigma_ij[[i, j]] = 0.5 * (sigma[i] + sigma[j]);
                sigma_ij[[j, i]] = sigma_ij[[i, j]];
                eps_k_ij[[i, j]] = (1.0 - k_ij[[i, j]]) * (epsilon_k[i] * epsilon_k[j]).sqrt();
                eps_k_ij[[j, i]] = eps_k_ij[[i, j]];
            }
        }

        // BH temperature dependent HS diameter, eq. 21
        let cd_bh_pure: Vec<Array1<f64>> = rep.iter().map(|&mi| bh_coefficients(mi, 6.0)).collect();
        let cd_bh_binary =
            Array2::from_shape_fn((n, n), |(i, j)| bh_coefficients(rep_ij[[i, j]], 6.0));

        Ok(Self {
            ncomponents: n,
            rep,
            att,
            sigma,
            epsilon_k,
            molarweight,
            k_ij,
            rep_ij,
            att_ij,
            sigma_ij,
            eps_k_ij,
            cd_bh_pure,
            cd_bh_binary,
            pure_records,
            binary_records,
        })
    }

    fn records(&self) -> (&[PureRecord<UVRecord, NoRecord>], &Array2<UVBinaryRecord>) {
        (&self.pure_records, &self.binary_records)
    }
}

impl UVParameters {
    /// Parameters for a single substance with molar weight one and no (default) ideal gas contributions.
    pub fn new_simple(
        rep: f64,
        att: f64,
        sigma: f64,
        epsilon_k: f64,
    ) -> Result<Self, ParameterError> {
        let model_record = UVRecord::new(rep, att, sigma, epsilon_k);
        let pure_record = PureRecord::new(Identifier::default(), 1.0, model_record, None);
        Self::new_pure(pure_record)
    }

    /// Markdown representation of parameters.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();
        let o = &mut output;
        write!(
            o,
            "|component|molarweight|$\\sigma$|$\\varepsilon$|$m$|$n$|\n|-|-|-|-|-|-|"
        )
        .unwrap();
        for i in 0..self.pure_records.len() {
            let component = self.pure_records[i].identifier.name.clone();
            let component = component.unwrap_or(format!("Component {}", i + 1));
            write!(
                o,
                "\n|{}|{}|{}|{}|{}|{}|",
                component,
                self.molarweight[i],
                self.sigma[i],
                self.epsilon_k[i],
                self.rep[i],
                self.att[i],
            )
            .unwrap();
        }
        output
    }
}

fn bh_coefficients(rep: f64, att: f64) -> Array1<f64> {
    let inv_a76 = 1.0 / mean_field_constant(7.0, att, 1.0);
    let am6 = mean_field_constant(rep, att, 1.0);
    let alpha = 1.0 / am6 - inv_a76;
    let c0 = arr1(&[-2.0 * rep / ((att - rep) * mie_prefactor(rep, att))]);
    concatenate![Axis(0), c0, CD_BH.dot(&arr1(&[1.0, alpha, alpha * alpha]))]
}

#[cfg(test)]
pub mod utils {
    use super::*;
    use feos_core::parameter::{Identifier, PureRecord};
    use std::f64;

    pub fn test_parameters(rep: f64, att: f64, sigma: f64, epsilon: f64) -> UVParameters {
        let identifier = Identifier::new(Some("1"), None, None, None, None, None);
        let model_record = UVRecord::new(rep, att, sigma, epsilon);
        let pr = PureRecord::new(identifier, 1.0, model_record, None);
        UVParameters::new_pure(pr).unwrap()
    }

    pub fn test_parameters_mixture(
        rep: Array1<f64>,
        att: Array1<f64>,
        sigma: Array1<f64>,
        epsilon: Array1<f64>,
    ) -> UVParameters {
        let identifier = Identifier::new(Some("1"), None, None, None, None, None);
        let model_record = UVRecord::new(rep[0], att[0], sigma[0], epsilon[0]);
        let pr1 = PureRecord::new(identifier, 1.0, model_record, None);
        //
        let identifier2 = Identifier::new(Some("1"), None, None, None, None, None);
        let model_record2 = UVRecord::new(rep[1], att[1], sigma[1], epsilon[1]);
        let pr2 = PureRecord::new(identifier2, 1.0, model_record2, None);
        let pure_records = vec![pr1, pr2];
        UVParameters::new_binary(pure_records, None).unwrap()
    }

    pub fn methane_parameters(rep: f64, att: f64) -> UVParameters {
        let identifier = Identifier::new(Some("1"), None, None, None, None, None);
        let model_record = UVRecord::new(rep, att, 3.7039, 150.03);
        let pr = PureRecord::new(identifier, 1.0, model_record, None);
        UVParameters::new_pure(pr).unwrap()
    }
}
