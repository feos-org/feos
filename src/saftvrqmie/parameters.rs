use crate::saftvrqmie::eos::FeynmanHibbsOrder;
use core::cmp::max;
use feos_core::parameter::{Parameter, ParameterError, PureRecord};
use feos_core::si::{Length, Temperature, CALORIE, GRAM, KELVIN, KILO, KILOGRAM, MOL, NAV, RGAS};
use ndarray::{Array, Array1, Array2};
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt::Write;
use std::fs::File;
use std::io::BufWriter;

/// SAFT-VRQ Mie pure-component parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct SaftVRQMieRecord {
    /// Segment number
    pub m: f64,
    /// Segment diameter in units of Angstrom
    pub sigma: f64,
    /// Energetic parameter in units of Kelvin
    pub epsilon_k: f64,
    /// Repulsive Mie exponent
    pub lr: f64,
    /// Attractive Mie exponent
    pub la: f64,
    /// Feynman-Hibbs order
    pub fh: usize,
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

impl std::fmt::Display for SaftVRQMieRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SaftVRQMieRecord(m={}", self.m)?;
        write!(f, ", sigma={}", self.sigma)?;
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

impl SaftVRQMieRecord {
    pub fn new(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        lr: f64,
        la: f64,
        fh: usize,
        viscosity: Option<[f64; 4]>,
        diffusion: Option<[f64; 5]>,
        thermal_conductivity: Option<[f64; 4]>,
    ) -> Result<SaftVRQMieRecord, ParameterError> {
        if m != 1.0 {
            return Err(ParameterError::IncompatibleParameters(format!(
                "Segment number `m` is not one. Chain-contributions are currently not supported."
            )));
        }
        Ok(SaftVRQMieRecord {
            m,
            sigma,
            epsilon_k,
            lr,
            la,
            fh,
            viscosity,
            diffusion,
            thermal_conductivity,
        })
    }
}

/// SAFT-VRQ Mie binary mixture parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct SaftVRQMieBinaryRecord {
    /// correction to energy parameters
    pub k_ij: f64,
    /// correction to diameter
    pub l_ij: f64,
}

impl std::fmt::Display for SaftVRQMieBinaryRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SaftVRQMieBinaryParameters(")?;
        write!(f, "\n\tk_ij={}", self.k_ij)?;
        write!(f, "\n\tl_ij={}", self.l_ij)?;
        write!(f, "\n)")
    }
}

impl TryFrom<f64> for SaftVRQMieBinaryRecord {
    type Error = ParameterError;

    fn try_from(_f: f64) -> Result<Self, Self::Error> {
        Err(ParameterError::IncompatibleParameters(
            "Cannot infer k_ij and l_ij from single float.".to_string(),
        ))
    }
}

impl TryFrom<SaftVRQMieBinaryRecord> for f64 {
    type Error = ParameterError;

    fn try_from(_f: SaftVRQMieBinaryRecord) -> Result<Self, Self::Error> {
        Err(ParameterError::IncompatibleParameters(
            "Cannot infer k_ij and l_ij from single float.".to_string(),
        ))
    }
}

/// Parameter set required for the SAFT-VRQ Mie equation of state and Helmholtz energy functional.
pub struct SaftVRQMieParameters {
    pub molarweight: Array1<f64>,
    pub m: Array1<f64>,
    pub sigma: Array1<f64>,
    pub epsilon_k: Array1<f64>,
    pub k_ij: Array2<f64>,
    pub l_ij: Array2<f64>,
    pub sigma_ij: Array2<f64>,
    pub epsilon_k_ij: Array2<f64>,
    pub e_k_ij: Array2<f64>,
    pub lr: Array1<f64>,
    pub la: Array1<f64>,
    pub fh: Array1<usize>,
    pub c_ij: Array2<f64>,
    pub lambda_r_ij: Array2<f64>,
    pub lambda_a_ij: Array2<f64>,
    pub mass_ij: Array2<f64>,
    pub viscosity: Option<Array2<f64>>,
    pub diffusion: Option<Array2<f64>>,
    pub thermal_conductivity: Option<Array2<f64>>,
    pub pure_records: Vec<PureRecord<SaftVRQMieRecord>>,
    pub binary_records: Option<Array2<SaftVRQMieBinaryRecord>>,
    pub fh_ij: Array2<FeynmanHibbsOrder>,
}

impl Parameter for SaftVRQMieParameters {
    type Pure = SaftVRQMieRecord;
    type Binary = SaftVRQMieBinaryRecord;

    fn from_records(
        pure_records: Vec<PureRecord<Self::Pure>>,
        binary_records: Option<Array2<SaftVRQMieBinaryRecord>>,
    ) -> Result<Self, ParameterError> {
        let n = pure_records.len();

        let mut fh = Array1::<usize>::zeros(n);
        let mut molarweight = Array::zeros(n);
        let mut m = Array::zeros(n);
        let mut sigma = Array::zeros(n);
        let mut epsilon_k = Array::zeros(n);
        let mut lr = Array::zeros(n);
        let mut la = Array::zeros(n);
        let mut viscosity = Vec::with_capacity(n);
        let mut diffusion = Vec::with_capacity(n);
        let mut thermal_conductivity = Vec::with_capacity(n);

        let mut component_index = HashMap::with_capacity(n);

        for (i, record) in pure_records.iter().enumerate() {
            component_index.insert(record.identifier.clone(), i);
            let r = &record.model_record;
            if r.m != 1.0 {
                return Err(
                    ParameterError::IncompatibleParameters(
                        format!(
                            "Segment number `m` for component {} is not one. Chain-contributions are currently not supported.", 
                            i
                        )
                    )
                );
            }
            m[i] = r.m;
            sigma[i] = r.sigma;
            epsilon_k[i] = r.epsilon_k;
            lr[i] = r.lr;
            la[i] = r.la;
            fh[i] = r.fh;
            viscosity.push(r.viscosity);
            diffusion.push(r.diffusion);
            thermal_conductivity.push(r.thermal_conductivity);
            molarweight[i] = record.molarweight;
        }

        let mut fh_ij: Array2<FeynmanHibbsOrder> =
            Array2::from_shape_fn((n, n), |(_i, _j)| FeynmanHibbsOrder::FH0);
        let br = binary_records.as_ref();
        let k_ij = br.map_or_else(|| Array2::zeros([n; 2]), |br| br.mapv(|br| br.k_ij));
        let l_ij = br.map_or_else(|| Array2::zeros([n; 2]), |br| br.mapv(|br| br.l_ij));
        let mut epsilon_k_ij = Array::zeros((n, n));
        let mut sigma_ij = Array::zeros((n, n));
        let mut e_k_ij = Array::zeros((n, n));
        let mut lambda_r_ij = Array::zeros((n, n));
        let mut lambda_a_ij = Array::zeros((n, n));
        let mut c_ij = Array::zeros((n, n));
        let mut mass_ij = Array::zeros((n, n));
        let to_mass_per_molecule = (GRAM / MOL / NAV / KILOGRAM).into_value();
        for i in 0..n {
            for j in 0..n {
                sigma_ij[[i, j]] = (1.0 - l_ij[[i, j]]) * 0.5 * (sigma[i] + sigma[j]);
                e_k_ij[[i, j]] = (sigma[i].powi(3) * sigma[j].powi(3)).sqrt()
                    / sigma_ij[[i, j]].powi(3)
                    * (epsilon_k[i] * epsilon_k[j]).sqrt();
                epsilon_k_ij[[i, j]] = (1.0 - k_ij[[i, j]]) * e_k_ij[[i, j]];
                lambda_r_ij[[i, j]] = ((lr[i] - 3.0) * (lr[j] - 3.0)).sqrt() + 3.0;
                lambda_a_ij[[i, j]] = ((la[i] - 3.0) * (la[j] - 3.0)).sqrt() + 3.0;
                c_ij[[i, j]] = lambda_r_ij[[i, j]] / (lambda_r_ij[[i, j]] - lambda_a_ij[[i, j]])
                    * (lambda_r_ij[[i, j]] / lambda_a_ij[[i, j]])
                        .powf(lambda_a_ij[[i, j]] / (lambda_r_ij[[i, j]] - lambda_a_ij[[i, j]]));
                mass_ij[[i, j]] = 2.0 * molarweight[i] * molarweight[j]
                    / (molarweight[i] + molarweight[j])
                    * to_mass_per_molecule;
                fh_ij[[i, j]] = FeynmanHibbsOrder::try_from(max(fh[i], fh[j]))?;
                if fh[i] * fh[j] == 2 {
                    return Err(
                        ParameterError::IncompatibleParameters(
                            format!(
                                "cannot combine Feynman-Hibbs orders 1 and 2. Component {} has order {} and component {} has order {}.", 
                                i, fh[i], j, fh[j]
                            )
                        )
                    );
                }
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

        Ok(Self {
            molarweight,
            m,
            sigma,
            epsilon_k,
            k_ij,
            l_ij,
            sigma_ij,
            epsilon_k_ij,
            e_k_ij,
            lr,
            la,
            fh,
            c_ij,
            lambda_r_ij,
            lambda_a_ij,
            mass_ij,
            viscosity: viscosity_coefficients,
            diffusion: diffusion_coefficients,
            thermal_conductivity: thermal_conductivity_coefficients,
            pure_records,
            binary_records,
            fh_ij,
        })
    }

    fn records(
        &self,
    ) -> (
        &[PureRecord<SaftVRQMieRecord>],
        Option<&Array2<SaftVRQMieBinaryRecord>>,
    ) {
        (&self.pure_records, self.binary_records.as_ref())
    }
}

impl SaftVRQMieParameters {
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();
        let o = &mut output;
        write!(
            o,
            "|component|molarweight|$\\sigma$|$\\varepsilon$|$\\lambda_r$|$\\lambda_a$|fh|\n|-|-|-|-|-|-|-|"
        )
        .unwrap();
        for i in 0..self.m.len() {
            let component = self.pure_records[i].identifier.name.clone();
            let component = component.unwrap_or(format!("Component {}", i + 1));
            write!(
                o,
                "\n|{}|{}|{}|{}|{}|{}|{}|",
                component,
                self.molarweight[i],
                self.sigma[i],
                self.epsilon_k[i],
                self.lr[i],
                self.la[i],
                self.fh[i]
            )
            .unwrap();
        }

        output
    }

    /// Generate energy and force tables to be used with LAMMPS' `pair_style table` command.
    ///
    /// For a given `temperature`, `n` values between `r_min` and `r_max` (both including) are tabulated.
    ///
    /// Files for all pure substances and all unique pairs are generated,
    /// where filenames use either the "name" field of the identifier or the index if no name is present.
    ///
    /// # Example
    ///
    /// For a hydrogen-neon mixture at 30 K, three files will be created.
    ///
    /// - "hydrogen_30K.table" for H-H interactions,
    /// - "neon_30K.table" for Ne-Ne interactions,
    /// - "hydrogen_neon_30K.table" for H-Ne interactions.
    pub fn lammps_tables(
        &self,
        temperature: Temperature,
        n: usize,
        r_min: Length,
        r_max: Length,
    ) -> std::io::Result<()> {
        let t = temperature.to_reduced();
        let rs = Array1::linspace(r_min.to_reduced(), r_max.to_reduced(), n);
        let energy_conversion = (KELVIN * RGAS / (KILO * CALORIE / MOL)).into_value();
        let force_conversion = (KELVIN * RGAS / (KILO * CALORIE / MOL)).into_value();

        let n_components = self.sigma.len();
        for i in 0..n_components {
            for j in i..n_components {
                let name_i = self.pure_records[i]
                    .identifier
                    .name
                    .clone()
                    .unwrap_or_else(|| i.to_string());
                let name_j = self.pure_records[j]
                    .identifier
                    .name
                    .clone()
                    .unwrap_or_else(|| j.to_string());

                let name = if i == j {
                    name_i
                } else {
                    format!("{}_{}", name_i, name_j)
                };
                let f = File::create(format!("{}_{}K.table", name, t))?;
                let mut stream = BufWriter::new(f);

                std::io::Write::write(
                    &mut stream,
                    b"# DATE: YYYY-MM-DD UNITS: real CONTRIBUTOR: YOUR NAME\n",
                )?;
                std::io::Write::write(
                    &mut stream,
                    format!("# FH1 potential for {} at T = {}\n", name, temperature).as_bytes(),
                )?;
                std::io::Write::write(&mut stream, format!("FH1_{}\n", name).as_bytes())?;
                std::io::Write::write(&mut stream, format!("N {}\n\n", n).as_bytes())?;

                for (k, &r) in rs.iter().enumerate() {
                    let [u, du, _] = self.qmie_potential_ij(i, j, r, t);
                    std::io::Write::write(
                        &mut stream,
                        format!(
                            "{} {:12.8} {:12.8} {:12.8}\n",
                            k + 1,
                            r,
                            u * energy_conversion,
                            -du * force_conversion
                        )
                        .as_bytes(),
                    )?;
                }
                std::io::Write::flush(&mut stream)?;
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for SaftVRQMieParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SaftVRQMieParameters(")?;
        write!(f, "\n\tmolarweight={}", self.molarweight)?;
        write!(f, "\n\tm={}", self.m)?;
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
    use std::sync::Arc;

    pub fn hydrogen_fh(fh: &str) -> Arc<SaftVRQMieParameters> {
        let hydrogen_json = &(r#"
            {
                "identifier": {
                    "cas": "1333-74-0",
                    "name": "hydrogen",
                    "iupac_name": "hydrogen",
                    "smiles": "[HH]",
                    "inchi": "InChI=1S/H2/h1H",
                    "formula": "H2"
                },
                "model_record": {
                    "m": 1.0,
                    "sigma": 3.0243,
                    "epsilon_k": 26.706,
                    "lr": 9.0,
                    "la": 6.0,
                    "fh": "#
            .to_owned()
            + fh
            + r#"
                },
                "molarweight": 2.0157309551872
            }"#);
        let hydrogen_record: PureRecord<SaftVRQMieRecord> =
            serde_json::from_str(hydrogen_json).expect("Unable to parse json.");
        Arc::new(SaftVRQMieParameters::new_pure(hydrogen_record).unwrap())
    }

    #[allow(dead_code)]
    pub fn helium_fh1() -> Arc<SaftVRQMieParameters> {
        let helium_json = r#"
            {
                "identifier": {
                    "cas": "1333-74-0",
                    "name": "helium",
                    "iupac_name": "helium",
                    "smiles": "[HH]",
                    "inchi": "InChI=1S/H2/h1H",
                    "formula": "He"
                },
                "model_record": {
                    "m": 1.0,
                    "sigma": 2.7443,
                    "epsilon_k": 5.4195,
                    "lr": 9.0,
                    "la": 6.0,
                    "fh": 1
                },
                "molarweight": 4.002601643881807
            }"#;
        let helium_record: PureRecord<SaftVRQMieRecord> =
            serde_json::from_str(helium_json).expect("Unable to parse json.");
        Arc::new(SaftVRQMieParameters::new_pure(helium_record).unwrap())
    }

    #[allow(dead_code)]
    pub fn neon_fh1() -> Arc<SaftVRQMieParameters> {
        let neon_json = r#"
            {
                "identifier": {
                    "cas": "1333-74-0",
                    "name": "neon",
                    "iupac_name": "neon",
                    "smiles": "[HH]",
                    "inchi": "InChI=1S/H2/h1H",
                    "formula": "Ne"
                },
                "model_record": {
                    "m": 1.0,
                    "sigma": 2.7778,
                    "epsilon_k": 37.501,
                    "lr": 13.0,
                    "la": 6.0,
                    "fh": 1
                },
                "molarweight": 20.17969806457545
            }"#;
        let neon_record: PureRecord<SaftVRQMieRecord> =
            serde_json::from_str(neon_json).expect("Unable to parse json.");
        Arc::new(SaftVRQMieParameters::new_pure(neon_record).unwrap())
    }

    pub fn h2_ne_fh(fh: &str) -> Arc<SaftVRQMieParameters> {
        let binary_json = &(r#"[
            {
                "identifier": {
                    "cas": "1333-74-0",
                    "name": "hydrogen",
                    "iupac_name": "hydrogen",
                    "smiles": "[HH]",
                    "inchi": "InChI=1S/H2/h1H",
                    "formula": "H2"
                },
                "model_record": {
                    "m": 1.0,
                    "sigma": 3.0243,
                    "epsilon_k": 26.706,
                    "lr": 9.0,
                    "la": 6.0,
                    "fh": "#
            .to_owned()
            + fh
            + r#"
                },
                "molarweight": 2.0157309551872
            },
            {
                "identifier": {
                    "cas": "1333-74-0",
                    "name": "neon",
                    "iupac_name": "neon",
                    "smiles": "[HH]",
                    "inchi": "InChI=1S/H2/h1H",
                    "formula": "H2"
                },
                "model_record": {
                    "m": 1.0,
                    "sigma": 2.7778,
                    "epsilon_k": 37.501,
                    "lr": 13.0,
                    "la": 6.0,
                    "fh": "#
            + fh
            + r#"
                },
                "molarweight": 20.17969806457545
            }
        ]"#);
        let binary_record: Vec<PureRecord<SaftVRQMieRecord>> =
            serde_json::from_str(binary_json).expect("Unable to parse json.");
        Arc::new(
            SaftVRQMieParameters::new_binary(
                binary_record,
                Some(SaftVRQMieBinaryRecord {
                    k_ij: 0.105,
                    l_ij: 0.0,
                }),
            )
            .unwrap(),
        )
    }
}

#[cfg(test)]
mod test {
    use super::utils::{helium_fh1, hydrogen_fh};
    use super::SaftVRQMieParameters;
    use feos_core::parameter::Parameter;

    #[test]
    #[should_panic(
        expected = "cannot combine Feynman-Hibbs orders 1 and 2. Component 0 has order 1 and component 1 has order 2."
    )]
    fn incompatible_order() {
        let order1 = helium_fh1();
        let order2 = hydrogen_fh("2");
        SaftVRQMieParameters::new_binary(
            vec![
                order1.pure_records[0].clone(),
                order2.pure_records[0].clone(),
            ],
            None,
        )
        .unwrap();
    }
}
