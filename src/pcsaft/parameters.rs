use conv::ValueInto;
use feos_core::joback::JobackRecord;
use feos_core::parameter::{
    FromSegments, FromSegmentsBinary, Parameter, ParameterError, PureRecord,
};
use ndarray::{Array, Array1, Array2};
use num_traits::Zero;
use quantity::si::{JOULE, KB, KELVIN};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write;

/// PC-SAFT pure-component parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct PcSaftRecord {
    /// Segment number
    pub m: f64,
    /// Segment diameter in units of Angstrom
    pub sigma: f64,
    /// Energetic parameter in units of Kelvin
    pub epsilon_k: f64,
    /// Dipole moment in units of Debye
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mu: Option<f64>,
    /// Quadrupole moment in units of Debye
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q: Option<f64>,
    /// Association volume parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kappa_ab: Option<f64>,
    /// Association energy parameter in units of Kelvin
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epsilon_k_ab: Option<f64>,
    /// \# of association sites of type A
    #[serde(skip_serializing_if = "Option::is_none")]
    pub na: Option<f64>,
    /// \# of association sites of type B
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nb: Option<f64>,
    /// Entropy scaling coefficients for the viscosity
    #[serde(skip_serializing_if = "Option::is_none")]
    pub viscosity: Option<[f64; 4]>,
    /// Entropy scaling coefficients for the diffusion coefficient
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diffusion: Option<[f64; 5]>,
    /// Entropy scaling coefficients for the thermal conductivity
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thermal_conductivity: Option<[f64; 4]>,
}

impl FromSegments<f64> for PcSaftRecord {
    fn from_segments(segments: &[(Self, f64)]) -> Result<Self, ParameterError> {
        let mut m = 0.0;
        let mut sigma3 = 0.0;
        let mut epsilon_k = 0.0;

        segments.iter().for_each(|(s, n)| {
            m += s.m * n;
            sigma3 += s.m * s.sigma.powi(3) * n;
            epsilon_k += s.m * s.epsilon_k * n;
        });

        let q = segments
            .iter()
            .filter_map(|(s, n)| s.q.map(|q| q * n))
            .reduce(|a, b| a + b);
        let mu = segments
            .iter()
            .filter_map(|(s, n)| s.mu.map(|mu| mu * n))
            .reduce(|a, b| a + b);
        let kappa_ab = segments
            .iter()
            .filter_map(|(s, n)| s.kappa_ab.map(|k| k * n))
            .reduce(|a, b| a + b);
        let epsilon_k_ab = segments
            .iter()
            .filter_map(|(s, n)| s.epsilon_k_ab.map(|e| e * n))
            .reduce(|a, b| a + b);
        let na = segments
            .iter()
            .filter_map(|(s, n)| s.na.map(|na| na * n))
            .reduce(|a, b| a + b);
        let nb = segments
            .iter()
            .filter_map(|(s, n)| s.nb.map(|nb| nb * n))
            .reduce(|a, b| a + b);

        // entropy scaling
        let mut viscosity = if segments
            .iter()
            .all(|(record, _)| record.viscosity.is_some())
        {
            Some([0.0; 4])
        } else {
            None
        };
        let mut thermal_conductivity = if segments
            .iter()
            .all(|(record, _)| record.thermal_conductivity.is_some())
        {
            Some([0.0; 4])
        } else {
            None
        };
        let diffusion = if segments
            .iter()
            .all(|(record, _)| record.diffusion.is_some())
        {
            Some([0.0; 5])
        } else {
            None
        };

        let n_t = segments.iter().fold(0.0, |acc, (_, n)| acc + n);
        segments.iter().for_each(|(s, n)| {
            let s3 = s.m * s.sigma.powi(3) * n;
            if let Some(p) = viscosity.as_mut() {
                let [a, b, c, d] = s.viscosity.unwrap();
                p[0] += s3 * a;
                p[1] += s3 * b / sigma3.powf(0.45);
                p[2] += n * c;
                p[3] += n * d;
            }
            if let Some(p) = thermal_conductivity.as_mut() {
                let [a, b, c, d] = s.thermal_conductivity.unwrap();
                p[0] += n * a;
                p[1] += n * b;
                p[2] += n * c;
                p[3] += n_t * d;
            }
            // if let Some(p) = diffusion.as_mut() {
            //     let [a, b, c, d, e] = s.diffusion.unwrap();
            //     p[0] += s3 * a;
            //     p[1] += s3 * b / sigma3.powf(0.45);
            //     p[2] += *n * c;
            //     p[3] += *n * d;
            // }
        });
        // correction due to difference in Chapman-Enskog reference between GC and regular formulation.
        viscosity = viscosity.map(|v| [v[0] - 0.5 * m.ln(), v[1], v[2], v[3]]);

        Ok(Self {
            m,
            sigma: (sigma3 / m).cbrt(),
            epsilon_k: epsilon_k / m,
            mu,
            q,
            kappa_ab,
            epsilon_k_ab,
            na,
            nb,
            viscosity,
            diffusion,
            thermal_conductivity,
        })
    }
}

impl FromSegments<usize> for PcSaftRecord {
    fn from_segments(segments: &[(Self, usize)]) -> Result<Self, ParameterError> {
        // We do not allow more than a single segment for q, mu, kappa_ab, epsilon_k_ab
        let quadpole_comps = segments.iter().filter_map(|(s, _)| s.q).count();
        if quadpole_comps > 1 {
            return Err(ParameterError::IncompatibleParameters(format!(
                "{quadpole_comps} segments with quadrupole moment."
            )));
        };
        let dipole_comps = segments.iter().filter_map(|(s, _)| s.mu).count();
        if dipole_comps > 1 {
            return Err(ParameterError::IncompatibleParameters(format!(
                "{dipole_comps} segment with dipole moment."
            )));
        };
        let faulty_assoc_comps = segments
            .iter()
            .filter_map(|(s, _)| s.kappa_ab.xor(s.epsilon_k_ab))
            .count();
        if faulty_assoc_comps > 0 {
            return Err(ParameterError::IncompatibleParameters(format!(
                "Incorrectly specified association sites on {faulty_assoc_comps} segment(s)"
            )));
        }
        let assoc_comps = segments
            .iter()
            .filter_map(|(s, _)| s.kappa_ab.and(s.epsilon_k_ab))
            .count();
        if assoc_comps > 1 {
            return Err(ParameterError::IncompatibleParameters(format!(
                "{assoc_comps} segments with association sites."
            )));
        }
        let segments: Vec<_> = segments
            .iter()
            .cloned()
            .map(|(s, c)| (s, c as f64))
            .collect();
        Self::from_segments(&segments)
    }
}

impl std::fmt::Display for PcSaftRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PcSaftRecord(m={}", self.m)?;
        write!(f, ", sigma={}", self.sigma)?;
        write!(f, ", epsilon_k={}", self.epsilon_k)?;
        if let Some(n) = &self.mu {
            write!(f, ", mu={}", n)?;
        }
        if let Some(n) = &self.q {
            write!(f, ", q={}", n)?;
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

impl PcSaftRecord {
    pub fn new(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        mu: Option<f64>,
        q: Option<f64>,
        kappa_ab: Option<f64>,
        epsilon_k_ab: Option<f64>,
        na: Option<f64>,
        nb: Option<f64>,
        viscosity: Option<[f64; 4]>,
        diffusion: Option<[f64; 5]>,
        thermal_conductivity: Option<[f64; 4]>,
    ) -> PcSaftRecord {
        PcSaftRecord {
            m,
            sigma,
            epsilon_k,
            mu,
            q,
            kappa_ab,
            epsilon_k_ab,
            na,
            nb,
            viscosity,
            diffusion,
            thermal_conductivity,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct PcSaftBinaryRecord {
    k_ij: f64,
}

impl From<f64> for PcSaftBinaryRecord {
    fn from(k_ij: f64) -> Self {
        Self { k_ij }
    }
}

impl From<PcSaftBinaryRecord> for f64 {
    fn from(binary_record: PcSaftBinaryRecord) -> Self {
        binary_record.k_ij
    }
}

impl<T: Copy + ValueInto<f64>> FromSegmentsBinary<T> for PcSaftBinaryRecord {
    fn from_segments_binary(segments: &[(Self, T, T)]) -> Result<Self, ParameterError> {
        let k_ij = segments
            .iter()
            .map(|(br, n1, n2)| br.k_ij * (*n1).value_into().unwrap() * (*n2).value_into().unwrap())
            .sum();
        Ok(Self { k_ij })
    }
}

impl std::fmt::Display for PcSaftBinaryRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PcSaftBinaryRecord(k_ij={})", self.k_ij)
    }
}

/// Parameter set required for the PC-SAFT equation of state and Helmholtz energy functional.
pub struct PcSaftParameters {
    pub molarweight: Array1<f64>,
    pub m: Array1<f64>,
    pub sigma: Array1<f64>,
    pub epsilon_k: Array1<f64>,
    pub mu: Array1<f64>,
    pub q: Array1<f64>,
    pub mu2: Array1<f64>,
    pub q2: Array1<f64>,
    pub kappa_ab: Array1<f64>,
    pub epsilon_k_ab: Array1<f64>,
    pub na: Array1<f64>,
    pub nb: Array1<f64>,
    pub kappa_aibj: Array2<f64>,
    pub epsilon_k_aibj: Array2<f64>,
    pub k_ij: Array2<f64>,
    pub sigma_ij: Array2<f64>,
    pub epsilon_k_ij: Array2<f64>,
    pub e_k_ij: Array2<f64>,
    pub ndipole: usize,
    pub nquadpole: usize,
    pub nassoc: usize,
    pub dipole_comp: Array1<usize>,
    pub quadpole_comp: Array1<usize>,
    pub assoc_comp: Array1<usize>,
    pub viscosity: Option<Array2<f64>>,
    pub diffusion: Option<Array2<f64>>,
    pub thermal_conductivity: Option<Array2<f64>>,
    pub pure_records: Vec<PureRecord<PcSaftRecord, JobackRecord>>,
    pub binary_records: Array2<PcSaftBinaryRecord>,
    pub joback_records: Option<Vec<JobackRecord>>,
}

impl Parameter for PcSaftParameters {
    type Pure = PcSaftRecord;
    type IdealGas = JobackRecord;
    type Binary = PcSaftBinaryRecord;

    fn from_records(
        pure_records: Vec<PureRecord<Self::Pure, Self::IdealGas>>,
        binary_records: Array2<PcSaftBinaryRecord>,
    ) -> Self {
        let n = pure_records.len();

        let mut molarweight = Array::zeros(n);
        let mut m = Array::zeros(n);
        let mut sigma = Array::zeros(n);
        let mut epsilon_k = Array::zeros(n);
        let mut mu = Array::zeros(n);
        let mut q = Array::zeros(n);
        let mut na = Array::zeros(n);
        let mut nb = Array::zeros(n);
        let mut kappa_ab = Array::zeros(n);
        let mut epsilon_k_ab = Array::zeros(n);
        let mut viscosity = Vec::with_capacity(n);
        let mut diffusion = Vec::with_capacity(n);
        let mut thermal_conductivity = Vec::with_capacity(n);

        let mut component_index = HashMap::with_capacity(n);

        for (i, record) in pure_records.iter().enumerate() {
            component_index.insert(record.identifier.clone(), i);
            let r = &record.model_record;
            m[i] = r.m;
            sigma[i] = r.sigma;
            epsilon_k[i] = r.epsilon_k;
            mu[i] = r.mu.unwrap_or(0.0);
            q[i] = r.q.unwrap_or(0.0);
            na[i] = r.na.unwrap_or(1.0);
            nb[i] = r.nb.unwrap_or(1.0);
            kappa_ab[i] = r.kappa_ab.unwrap_or(0.0);
            epsilon_k_ab[i] = r.epsilon_k_ab.unwrap_or(0.0);
            viscosity.push(r.viscosity);
            diffusion.push(r.diffusion);
            thermal_conductivity.push(r.thermal_conductivity);
            molarweight[i] = record.molarweight;
        }

        let mu2 = &mu * &mu / (&m * &sigma * &sigma * &sigma * &epsilon_k)
            * 1e-19
            * (JOULE / KELVIN / KB).into_value().unwrap();
        let q2 = &q * &q / (&m * &sigma.mapv(|s| s.powi(5)) * &epsilon_k)
            * 1e-19
            * (JOULE / KELVIN / KB).into_value().unwrap();
        let dipole_comp: Array1<usize> = mu2
            .iter()
            .enumerate()
            .filter_map(|(i, &mu2)| (mu2.abs() > 0.0).then(|| i))
            .collect();
        let ndipole = dipole_comp.len();
        let quadpole_comp: Array1<usize> = q2
            .iter()
            .enumerate()
            .filter_map(|(i, &q2)| (q2.abs() > 0.0).then(|| i))
            .collect();
        let nquadpole = quadpole_comp.len();
        let assoc_comp: Array1<usize> = kappa_ab
            .iter()
            .enumerate()
            .filter_map(|(i, &k)| (k.abs() > 0.0).then(|| i))
            .collect();
        let nassoc = assoc_comp.len();

        let mut kappa_aibj = Array::zeros([n, n]);
        let mut epsilon_k_aibj = Array::zeros([n, n]);
        for i in 0..nassoc {
            for j in 0..nassoc {
                let ai = assoc_comp[i];
                let bj = assoc_comp[j];
                kappa_aibj[[ai, bj]] = (kappa_ab[ai] * kappa_ab[bj]).sqrt()
                    * (2.0 * (sigma[ai] * sigma[bj]).sqrt() / (sigma[ai] + sigma[bj])).powf(3.0);
                epsilon_k_aibj[[ai, bj]] = 0.5 * (epsilon_k_ab[ai] + epsilon_k_ab[bj]);
            }
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

        Self {
            molarweight,
            m,
            sigma,
            epsilon_k,
            mu,
            q,
            mu2,
            q2,
            kappa_ab,
            epsilon_k_ab,
            na,
            nb,
            kappa_aibj,
            epsilon_k_aibj,
            k_ij,
            sigma_ij,
            epsilon_k_ij,
            e_k_ij,
            ndipole,
            nquadpole,
            nassoc,
            dipole_comp,
            quadpole_comp,
            assoc_comp,
            viscosity: viscosity_coefficients,
            diffusion: diffusion_coefficients,
            thermal_conductivity: thermal_conductivity_coefficients,
            pure_records,
            binary_records,
            joback_records,
        }
    }

    fn records(
        &self,
    ) -> (
        &[PureRecord<PcSaftRecord, JobackRecord>],
        &Array2<PcSaftBinaryRecord>,
    ) {
        (&self.pure_records, &self.binary_records)
    }
}

impl PcSaftParameters {
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();
        let o = &mut output;
        write!(
            o,
            "|component|molarweight|$m$|$\\sigma$|$\\varepsilon$|$\\mu$|$Q$|$\\kappa_{{AB}}$|$\\varepsilon_{{AB}}$|$N_A$|$N_B$|\n|-|-|-|-|-|-|-|-|-|-|-|"
        )
        .unwrap();
        for i in 0..self.m.len() {
            let component = self.pure_records[i].identifier.name.clone();
            let component = component.unwrap_or(format!("Component {}", i + 1));
            write!(
                o,
                "\n|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|",
                component,
                self.molarweight[i],
                self.m[i],
                self.sigma[i],
                self.epsilon_k[i],
                self.mu[i],
                self.q[i],
                self.kappa_ab[i],
                self.epsilon_k_ab[i],
                self.na[i],
                self.nb[i]
            )
            .unwrap();
        }

        output
    }
}

impl std::fmt::Display for PcSaftParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PcSaftParameters(")?;
        write!(f, "\n\tmolarweight={}", self.molarweight)?;
        write!(f, "\n\tm={}", self.m)?;
        write!(f, "\n\tsigma={}", self.sigma)?;
        write!(f, "\n\tepsilon_k={}", self.epsilon_k)?;
        if !self.dipole_comp.is_empty() {
            write!(f, "\n\tmu={}", self.mu)?;
        }
        if !self.quadpole_comp.is_empty() {
            write!(f, "\n\tq={}", self.q)?;
        }
        if !self.assoc_comp.is_empty() {
            write!(f, "\n\tkappa_ab={}", self.kappa_ab)?;
            write!(f, "\n\tepsilon_k_ab={}", self.epsilon_k_ab)?;
            write!(f, "\n\tna={}", self.na)?;
            write!(f, "\n\tnb={}", self.nb)?;
        }
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
    use std::rc::Rc;
    // use feos_core::parameter::SegmentRecord;

    // pub fn pure_record_vec() -> Vec<PureRecord<PcSaftRecord, JobackRecord>> {
    //     let records = r#"[
    //         {
    //             "identifier": {
    //                 "cas": "74-98-6",
    //                 "name": "propane",
    //                 "iupac_name": "propane",
    //                 "smiles": "CCC",
    //                 "inchi": "InChI=1/C3H8/c1-3-2/h3H2,1-2H3",
    //                 "formula": "C3H8"
    //             },
    //             "model_record": {
    //                 "m": 2.0018290000000003,
    //                 "sigma": 3.618353,
    //                 "epsilon_k": 208.1101
    //             },
    //             "molarweight": 44.0962,
    //             "chemical_record": {
    //                 "segments": ["CH3", "CH2", "CH3"]
    //             }
    //         },
    //         {
    //             "identifier": {
    //                 "cas": "106-97-8",
    //                 "name": "butane",
    //                 "iupac_name": "butane",
    //                 "smiles": "CCCC",
    //                 "inchi": "InChI=1/C4H10/c1-3-4-2/h3-4H2,1-2H3",
    //                 "formula": "C4H10"
    //             },
    //             "model_record": {
    //                 "m": 2.331586,
    //                 "sigma": 3.7086010000000003,
    //                 "epsilon_k": 222.8774
    //             },
    //             "molarweight": 58.123,
    //             "chemical_record": {
    //                 "segments": ["CH3", "CH2", "CH2", "CH3"]
    //             }
    //         },
    //         {
    //             "identifier": {
    //                 "cas": "74-82-8",
    //                 "name": "methane",
    //                 "iupac_name": "methane",
    //                 "smiles": "C",
    //                 "inchi": "InChI=1/CH4/h1H4",
    //                 "formula": "CH4"
    //             },
    //             "model_record": {
    //                 "m": 1.0,
    //                 "sigma": 3.7039,
    //                 "epsilon_k": 150.034
    //             },
    //             "molarweight": 16.0426
    //         },
    //         {
    //             "identifier": {
    //                 "cas": "124-38-9",
    //                 "name": "carbon-dioxide",
    //                 "iupac_name": "carbon dioxide",
    //                 "smiles": "O=C=O",
    //                 "inchi": "InChI=1/CO2/c2-1-3",
    //                 "formula": "CO2"
    //             },
    //             "molarweight": 44.0098,
    //             "model_record": {
    //                 "m": 1.5131,
    //                 "sigma": 3.1869,
    //                 "epsilon_k": 163.333,
    //                 "q": 4.4
    //             }
    //         }
    //     ]"#;
    //     serde_json::from_str(records).expect("Unable to parse json.")
    // }

    // pub fn segments_vec() -> Vec<SegmentRecord<PcSaftRecord, JobackRecord>> {
    //     let segments_json = r#"[
    //     {
    //       "identifier": "CH3",
    //       "model_record": {
    //         "m": 0.77247,
    //         "sigma": 3.6937,
    //         "epsilon_k": 181.49
    //       },
    //       "molarweight": 15.0345
    //     },
    //     {
    //       "identifier": "CH2",
    //       "model_record": {
    //         "m": 0.7912,
    //         "sigma": 3.0207,
    //         "epsilon_k": 157.23
    //       },
    //       "molarweight": 14.02658
    //     },

    //     {
    //       "identifier": ">CH",
    //       "model_record": {
    //         "m": 0.52235,
    //         "sigma": 0.99912,
    //         "epsilon_k": 269.84
    //       },
    //       "molarweight": 13.01854
    //     },
    //     {
    //       "identifier": ">C<",
    //       "model_record": {
    //         "m": -0.70131,
    //         "sigma": 0.54350,
    //         "epsilon_k": 0.0
    //       },
    //       "molarweight": 12.0107
    //     },
    //     {
    //       "identifier": "=CH2",
    //       "model_record": {
    //         "m": 0.70581,
    //         "sigma": 3.1630,
    //         "epsilon_k": 171.34
    //       },
    //       "molarweight": 14.02658
    //     },
    //     {
    //       "identifier": "=CH",
    //       "model_record": {
    //         "m": 0.90182,
    //         "sigma": 2.8864,
    //         "epsilon_k": 158.90
    //       },
    //       "molarweight": 13.01854
    //     }
    //     ]"#;
    //     serde_json::from_str(segments_json).expect("Unable to parse json.")
    // }

    // pub fn methane_parameters() -> PcSaftParameters {
    //     let methane_json = r#"
    //         {
    //             "identifier": {
    //                 "cas": "74-82-8",
    //                 "name": "methane",
    //                 "iupac_name": "methane",
    //                 "smiles": "C",
    //                 "inchi": "InChI=1/CH4/h1H4",
    //                 "formula": "CH4"
    //             },
    //             "model_record": {
    //                 "m": 1.0,
    //                 "sigma": 3.7039,
    //                 "epsilon_k": 150.034
    //             },
    //             "molarweight": 16.0426
    //         }"#;
    //     let methane_record: PureRecord<PcSaftRecord, JobackRecord> =
    //         serde_json::from_str(methane_json).expect("Unable to parse json.");
    //     PcSaftParameters::from_records(vec![methane_record], None).unwrap()
    // }

    pub fn propane_parameters() -> Rc<PcSaftParameters> {
        let propane_json = r#"
            {
                "identifier": {
                    "cas": "74-98-6",
                    "name": "propane",
                    "iupac_name": "propane",
                    "smiles": "CCC",
                    "inchi": "InChI=1/C3H8/c1-3-2/h3H2,1-2H3",
                    "formula": "C3H8"
                },
                "model_record": {
                    "m": 2.001829,
                    "sigma": 3.618353,
                    "epsilon_k": 208.1101,
                    "viscosity": [-0.8013, -1.9972,-0.2907, -0.0467],
                    "thermal_conductivity": [-0.15348,  -0.6388, 1.21342, -0.01664],
                    "diffusion": [-0.675163251512047, 0.3212017677695878, 0.100175249144429, 0.0, 0.0]
                },
                "molarweight": 44.0962
            }"#;
        let propane_record: PureRecord<PcSaftRecord, JobackRecord> =
            serde_json::from_str(propane_json).expect("Unable to parse json.");
        Rc::new(PcSaftParameters::new_pure(propane_record))
    }

    // pub fn propane_homogc_parameters() -> PcSaftParameters {
    //     let propane_json = r#"
    //         {
    //             "identifier": {
    //                 "cas": "74-98-6",
    //                 "name": "propane",
    //                 "iupac_name": "propane",
    //                 "smiles": "CCC",
    //                 "inchi": "InChI=1/C3H8/c1-3-2/h3H2,1-2H3",
    //                 "formula": "C3H8"
    //             },
    //             "chemical_record": {
    //                 "molarweight": 44.0962,
    //                 "segments": ["CH3", "CH2", "CH3"]
    //             }
    //         }"#;
    //     let segments_json = r#"[
    //     {
    //         "identifier": "CH3",
    //         "model_record": {
    //             "m": 0.61198,
    //             "sigma": 3.7202,
    //             "epsilon_k": 229.90
    //         },
    //         "molarweight": 15.0345
    //     },
    //     {
    //         "identifier": "CH2",
    //         "model_record": {
    //             "m": 0.45606,
    //             "sigma": 3.8900,
    //             "epsilon_k": 239.01
    //             },
    //         "molarweight": 14.02658
    //     }
    // ]"#;
    //     let propane_record: PureRecord =
    //         serde_json::from_str(&propane_json).expect("Unable to parse json.");
    //     let segment_records: Vec<SegmentRecord> =
    //         serde_json::from_str(&segments_json).expect("Unable to parse json.");
    //     ParameterBuilder::new()
    //         .molecule_records(Some(&vec![propane_record]), None)
    //         .segment_records(Some(&segment_records), None)
    //         .build(BuilderOption::HomoGC)
    //         .unwrap()
    // }

    pub fn carbon_dioxide_parameters() -> PcSaftParameters {
        let co2_json = r#"
        {
            "identifier": {
                "cas": "124-38-9",
                "name": "carbon-dioxide",
                "iupac_name": "carbon dioxide",
                "smiles": "O=C=O",
                "inchi": "InChI=1/CO2/c2-1-3",
                "formula": "CO2"
            },
            "molarweight": 44.0098,
            "model_record": {
                "m": 1.5131,
                "sigma": 3.1869,
                "epsilon_k": 163.333,
                "q": 4.4
            }
        }"#;
        let co2_record: PureRecord<PcSaftRecord, JobackRecord> =
            serde_json::from_str(co2_json).expect("Unable to parse json.");
        PcSaftParameters::new_pure(co2_record)
    }

    pub fn butane_parameters() -> Rc<PcSaftParameters> {
        let butane_json = r#"
            {
                "identifier": {
                    "cas": "106-97-8",
                    "name": "butane",
                    "iupac_name": "butane",
                    "smiles": "CCCC",
                    "inchi": "InChI=1/C4H10/c1-3-4-2/h3-4H2,1-2H3",
                    "formula": "C4H10"
                },
                "model_record": {
                    "m": 2.331586,
                    "sigma": 3.7086010000000003,
                    "epsilon_k": 222.8774
                },
                "molarweight": 58.123
            }"#;
        let butane_record: PureRecord<PcSaftRecord, JobackRecord> =
            serde_json::from_str(butane_json).expect("Unable to parse json.");
        Rc::new(PcSaftParameters::new_pure(butane_record))
    }

    pub fn dme_parameters() -> PcSaftParameters {
        let dme_json = r#"
            {
                "identifier": {
                    "cas": "115-10-6",
                    "name": "dimethyl-ether",
                    "iupac_name": "methoxymethane",
                    "smiles": "COC",
                    "inchi": "InChI=1/C2H6O/c1-3-2/h1-2H3",
                    "formula": "C2H6O"
                },
                "model_record": {
                    "m": 2.2634,
                    "sigma": 3.2723,
                    "epsilon_k": 210.29,
                    "mu": 1.3
                },
                "molarweight": 46.0688
            }"#;
        let dme_record: PureRecord<PcSaftRecord, JobackRecord> =
            serde_json::from_str(dme_json).expect("Unable to parse json.");
        PcSaftParameters::new_pure(dme_record)
    }

    pub fn water_parameters() -> PcSaftParameters {
        let water_json = r#"
            {
                "identifier": {
                    "cas": "7732-18-5",
                    "name": "water_np",
                    "iupac_name": "oxidane",
                    "smiles": "O",
                    "inchi": "InChI=1/H2O/h1H2",
                    "formula": "H2O"
                },
                "model_record": {
                    "m": 1.065587,
                    "sigma": 3.000683,
                    "epsilon_k": 366.5121,
                    "kappa_ab": 0.034867983,
                    "epsilon_k_ab": 2500.6706
                },
                "molarweight": 18.0152
            }"#;
        let water_record: PureRecord<PcSaftRecord, JobackRecord> =
            serde_json::from_str(water_json).expect("Unable to parse json.");
        PcSaftParameters::new_pure(water_record)
    }

    pub fn dme_co2_parameters() -> PcSaftParameters {
        let binary_json = r#"[
            {
                "identifier": {
                    "cas": "115-10-6",
                    "name": "dimethyl-ether",
                    "iupac_name": "methoxymethane",
                    "smiles": "COC",
                    "inchi": "InChI=1/C2H6O/c1-3-2/h1-2H3",
                    "formula": "C2H6O"
                },
                "molarweight": 46.0688,
                "model_record": {
                    "m": 2.2634,
                    "sigma": 3.2723,
                    "epsilon_k": 210.29,
                    "mu": 1.3
                }
            },
            {
                "identifier": {
                    "cas": "124-38-9",
                    "name": "carbon-dioxide",
                    "iupac_name": "carbon dioxide",
                    "smiles": "O=C=O",
                    "inchi": "InChI=1/CO2/c2-1-3",
                    "formula": "CO2"
                },
                "molarweight": 44.0098,
                "model_record": {
                    "m": 1.5131,
                    "sigma": 3.1869,
                    "epsilon_k": 163.333,
                    "q": 4.4
                }
            }
        ]"#;
        let binary_record: Vec<PureRecord<PcSaftRecord, JobackRecord>> =
            serde_json::from_str(binary_json).expect("Unable to parse json.");
        PcSaftParameters::new_binary(binary_record, None)
    }

    pub fn propane_butane_parameters() -> Rc<PcSaftParameters> {
        let binary_json = r#"[
            {
                "identifier": {
                    "cas": "74-98-6",
                    "name": "propane",
                    "iupac_name": "propane",
                    "smiles": "CCC",
                    "inchi": "InChI=1/C3H8/c1-3-2/h3H2,1-2H3",
                    "formula": "C3H8"
                },
                "model_record": {
                    "m": 2.0018290000000003,
                    "sigma": 3.618353,
                    "epsilon_k": 208.1101,
                    "viscosity": [-0.8013, -1.9972, -0.2907, -0.0467],
                    "thermal_conductivity": [-0.15348, -0.6388, 1.21342, -0.01664],
                    "diffusion": [-0.675163251512047, 0.3212017677695878, 0.100175249144429, 0.0, 0.0]
                },
                "molarweight": 44.0962
            },
            {
                "identifier": {
                    "cas": "106-97-8",
                    "name": "butane",
                    "iupac_name": "butane",
                    "smiles": "CCCC",
                    "inchi": "InChI=1/C4H10/c1-3-4-2/h3-4H2,1-2H3",
                    "formula": "C4H10"
                },
                "model_record": {
                    "m": 2.331586,
                    "sigma": 3.7086010000000003,
                    "epsilon_k": 222.8774,
                    "viscosity": [-0.9763, -2.2413, -0.3690, -0.0605],
                    "diffusion": [-0.8985872992958458, 0.3428584416613513, 0.10236616087103916, 0.0, 0.0]
                },
                "molarweight": 58.123
            }
        ]"#;
        let binary_record: Vec<PureRecord<PcSaftRecord, JobackRecord>> =
            serde_json::from_str(binary_json).expect("Unable to parse json.");
        Rc::new(PcSaftParameters::new_binary(binary_record, None))
    }

    // pub fn water_hexane_parameters() -> PcSaftParameters {
    //     let binary_json = r#"[
    //     {
    //         "identifier": {
    //             "cas": "7732-18-5",
    //             "name": "water_np",
    //             "iupac_name": "oxidane",
    //             "smiles": "O",
    //             "inchi": "InChI=1/H2O/h1H2",
    //             "formula": "H2O"
    //         },
    //         "model_record": {
    //             "m": 1.065587,
    //             "sigma": 3.000683,
    //             "epsilon_k": 366.5121,
    //             "kappa_ab": 0.034867983,
    //             "epsilon_k_ab": 2500.6706
    //         },
    //         "molarweight": 18.0152
    //     },
    //     {
    //         "identifier": {
    //             "cas": "110-54-3",
    //             "name": "hexane",
    //             "iupac_name": "hexane",
    //             "smiles": "CCCCCC",
    //             "inchi": "InChI=1/C6H14/c1-3-5-6-4-2/h3-6H2,1-2H3",
    //             "formula": "C6H14"
    //         },
    //         "model_record": {
    //             "m": 3.0576,
    //             "sigma": 3.7983,
    //             "epsilon_k": 236.77
    //         },
    //         "molarweight": 86.177
    //     }
    //     ]"#;
    //     let binary_record: Vec<PureRecord<PcSaftRecord, JobackRecord>> =
    //         serde_json::from_str(binary_json).expect("Unable to parse json.");
    //     PcSaftParameters::from_records(binary_record, None).unwrap()
    // }

    // pub fn dodecane_nitrogen_parameters() -> PcSaftParameters {
    //     let binary_json = r#"[
    //     {
    //         "identifier": {
    //             "cas": "112-40-3",
    //             "name": "dodecane",
    //             "iupac_name": "dodecane",
    //             "smiles": "CCCCCCCCCCCC",
    //             "inchi": "InChI=1/C12H26/c1-3-5-7-9-11-12-10-8-6-4-2/h3-12H2,1-2H3",
    //             "formula": "C12H26"
    //         },
    //         "model_record": {
    //             "m": 5.305758999999999,
    //             "sigma": 3.895892,
    //             "epsilon_k": 249.2145,
    //             "viscosity": [
    //                 -1.6719,
    //                 -3.39020393,
    //                 -0.6956429590000001,
    //                 -0.154563667
    //             ],
    //             "diffusion": [
    //                 -1.709976456320196,
    //                 0.4350370700652692,
    //                 0.3567181896779805,
    //                 0.0,
    //                 0.0
    //             ]
    //         },
    //         "molarweight": 170.3374
    //     },
    //     {
    //         "identifier": {
    //             "cas": "7727-37-9",
    //             "name": "nitrogen",
    //             "iupac_name": "molecular nitrogen",
    //             "smiles": "N#N",
    //             "inchi": "InChI=1/N2/c1-2",
    //             "formula": "N2"
    //         },
    //         "model_record": {
    //             "m": 1.1504,
    //             "sigma": 3.3848,
    //             "epsilon_k": 91.4,
    //             "q": 1.43,
    //             "viscosity": [
    //                 -0.196376646,
    //                 -0.9460855,
    //                 -0.0309718769,
    //                 -0.0303367687
    //             ],
    //             "diffusion": [
    //                 -0.12855765455212295,
    //                 0.24885131958296933,
    //                 0.08052800000000002,
    //                 0.0,
    //                 0.0
    //             ]
    //         },
    //         "molarweight": 28.0134
    //     }
    //     ]"#;
    //     let kij_json = r#"[
    //     {
    //       "id1": {
    //               "cas": "7727-37-9",
    //               "name": "nitrogen",
    //               "iupac_name": "molecular nitrogen",
    //               "smiles": "N#N",
    //               "inchi": "InChI=1/N2/c1-2",
    //               "formula": "N2"
    //           },
    //       "id2": {
    //               "cas": "112-40-3",
    //               "name": "dodecane",
    //               "iupac_name": "dodecane",
    //               "smiles": "CCCCCCCCCCCC",
    //               "inchi": "InChI=1/C12H26/c1-3-5-7-9-11-12-10-8-6-4-2/h3-12H2,1-2H3",
    //               "formula": "C12H26"
    //           },
    //       "k_ij": 0.1661
    //     },
    //     {
    //       "id1": {
    //               "cas": "7727-37-9",
    //               "name": "nitrogen",
    //               "iupac_name": "molecular nitrogen",
    //               "smiles": "N#N",
    //               "inchi": "InChI=1/N2/c1-2",
    //               "formula": "N2"
    //           },
    //       "id2": {
    //               "cas": "74-98-6",
    //               "name": "propane",
    //               "iupac_name": "propane",
    //               "smiles": "CCC",
    //               "inchi": "InChI=1/C3H8/c1-3-2/h3H2,1-2H3",
    //               "formula": "C3H8"
    //           },
    //       "k_ij": 0.02512
    //     }
    //   ]"#;
    //     let binary_record: Vec<PureRecord<PcSaftRecord, JobackRecord>> =
    //         serde_json::from_str(binary_json).expect("Unable to parse json.");
    //     let kij_record: Vec<BinaryRecord<Identifier, f64>> =
    //         serde_json::from_str(kij_json).expect("Unable to parse binary json.");
    //     PcSaftParameters::from_records(binary_record, Some(kij_record)).unwrap()
    // }
}
