use crate::saftvrqmie::eos::FeynmanHibbsOrder;
use core::cmp::max;
use feos_core::parameter::Parameters;
use feos_core::{FeosError, FeosResult};
use ndarray::{Array, Array1, Array2};
use quantity::{KILOGRAM, NAV};
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

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

impl SaftVRQMieRecord {
    #[expect(clippy::too_many_arguments)]
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
    ) -> FeosResult<SaftVRQMieRecord> {
        if m != 1.0 {
            return Err(FeosError::IncompatibleParameters(
                "Segment number `m` is not one. Chain-contributions are currently not supported."
                    .to_string(),
            ));
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
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct SaftVRQMieBinaryRecord {
    /// correction to energy parameters
    pub k_ij: f64,
    /// correction to diameter
    pub l_ij: f64,
}

/// Parameter set required for the SAFT-VRQ Mie equation of state and Helmholtz energy functional.
pub type SaftVRQMieParameters = Parameters<SaftVRQMieRecord, SaftVRQMieBinaryRecord, ()>;

/// Parameter set required for the SAFT-VRQ Mie equation of state and Helmholtz energy functional.
pub struct SaftVRQMiePars {
    pub m: Array1<f64>,
    pub sigma: Array1<f64>,
    pub epsilon_k: Array1<f64>,
    pub sigma_ij: Array2<f64>,
    pub epsilon_k_ij: Array2<f64>,
    pub c_ij: Array2<f64>,
    pub lambda_r_ij: Array2<f64>,
    pub lambda_a_ij: Array2<f64>,
    pub mass_ij: Array2<f64>,
    pub viscosity: Option<Array2<f64>>,
    pub diffusion: Option<Array2<f64>>,
    pub thermal_conductivity: Option<Array2<f64>>,
    pub fh_ij: Array2<FeynmanHibbsOrder>,
}

impl SaftVRQMiePars {
    pub fn new(parameters: &SaftVRQMieParameters) -> FeosResult<Self> {
        let n = parameters.pure.len();

        let [fh] = parameters.collate(|pr| [pr.fh]);
        let [m, sigma, epsilon_k] = parameters.collate(|pr| [pr.m, pr.sigma, pr.epsilon_k]);
        let [lr, la] = parameters.collate(|pr| [pr.lr, pr.la]);
        let [viscosity, thermal_conductivity] =
            parameters.collate(|pr| [pr.viscosity, pr.thermal_conductivity]);
        let [diffusion] = parameters.collate(|pr| [pr.diffusion]);
        let molarweight = &parameters.molar_weight;

        for (i, m) in m.iter().enumerate() {
            if *m != 1.0 {
                return Err(FeosError::IncompatibleParameters(format!(
                    "Segment number `m` for component {} is not one. Chain-contributions are currently not supported.",
                    i
                )));
            }
        }

        let mut fh_ij: Array2<FeynmanHibbsOrder> =
            Array2::from_elem((n, n), FeynmanHibbsOrder::FH0);
        let [k_ij, l_ij] = parameters.collate_binary(|b| [b.k_ij, b.l_ij]);
        let mut epsilon_k_ij = Array::zeros((n, n));
        let mut sigma_ij = Array::zeros((n, n));
        let mut e_k_ij = Array::zeros((n, n));
        let mut lambda_r_ij = Array::zeros((n, n));
        let mut lambda_a_ij = Array::zeros((n, n));
        let mut c_ij = Array::zeros((n, n));
        let mut mass_ij = Array::zeros((n, n));
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
                mass_ij[[i, j]] = (2.0 * molarweight.get(i) * molarweight.get(j)
                    / (molarweight.get(i) + molarweight.get(j)))
                .convert_into(KILOGRAM * NAV);
                fh_ij[[i, j]] = FeynmanHibbsOrder::try_from(max(fh[i], fh[j]))?;
                if fh[i] * fh[j] == 2 {
                    return Err(FeosError::IncompatibleParameters(format!(
                        "cannot combine Feynman-Hibbs orders 1 and 2. Component {} has order {} and component {} has order {}.",
                        i, fh[i], j, fh[j]
                    )));
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
            m,
            sigma,
            epsilon_k,
            sigma_ij,
            epsilon_k_ij,
            c_ij,
            lambda_r_ij,
            lambda_a_ij,
            mass_ij,
            viscosity: viscosity_coefficients,
            diffusion: diffusion_coefficients,
            thermal_conductivity: thermal_conductivity_coefficients,
            fh_ij,
        })
    }
}

#[cfg(test)]
pub mod utils {
    use feos_core::parameter::PureRecord;

    use super::*;

    pub fn hydrogen_fh(fh: &str) -> SaftVRQMiePars {
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
                "m": 1.0,
                "sigma": 3.0243,
                "epsilon_k": 26.706,
                "lr": 9.0,
                "la": 6.0,
                "fh": "#
            .to_owned()
            + fh
            + r#",
                "molarweight": 2.0157309551872
            }"#);
        let hydrogen_record: PureRecord<SaftVRQMieRecord, ()> =
            serde_json::from_str(hydrogen_json).expect("Unable to parse json.");
        SaftVRQMiePars::new(&SaftVRQMieParameters::new_pure(hydrogen_record).unwrap()).unwrap()
    }

    pub fn helium_fh1() -> PureRecord<SaftVRQMieRecord, ()> {
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
                "m": 1.0,
                "sigma": 2.7443,
                "epsilon_k": 5.4195,
                "lr": 9.0,
                "la": 6.0,
                "fh": 1,
                "molarweight": 4.002601643881807
            }"#;
        serde_json::from_str(helium_json).expect("Unable to parse json.")
    }

    pub fn hydrogen_fh2() -> PureRecord<SaftVRQMieRecord, ()> {
        let helium_json = r#"
            {
                "identifier": {
                    "cas": "1333-74-0",
                    "name": "hydrogen",
                    "iupac_name": "hydrogen",
                    "smiles": "[HH]",
                    "inchi": "InChI=1S/H2/h1H",
                    "formula": "H2"
                },
                "m": 1.0,
                "sigma": 3.0243,
                "epsilon_k": 26.706,
                "lr": 9.0,
                "la": 6.0,
                "fh": 2,
                "molarweight": 2.0157309551872
            }"#;
        serde_json::from_str(helium_json).expect("Unable to parse json.")
    }

    #[expect(dead_code)]
    pub fn neon_fh1() -> SaftVRQMieParameters {
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
                "m": 1.0,
                "sigma": 2.7778,
                "epsilon_k": 37.501,
                "lr": 13.0,
                "la": 6.0,
                "fh": 1,
                "molarweight": 20.17969806457545
            }"#;
        let neon_record: PureRecord<SaftVRQMieRecord, ()> =
            serde_json::from_str(neon_json).expect("Unable to parse json.");
        SaftVRQMieParameters::new_pure(neon_record).unwrap()
    }

    pub fn h2_ne_fh(fh: &str) -> SaftVRQMiePars {
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
                "m": 1.0,
                "sigma": 3.0243,
                "epsilon_k": 26.706,
                "lr": 9.0,
                "la": 6.0,
                "fh": "#
            .to_owned()
            + fh
            + r#",
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
                "m": 1.0,
                "sigma": 2.7778,
                "epsilon_k": 37.501,
                "lr": 13.0,
                "la": 6.0,
                "fh": "#
            + fh
            + r#",
                "molarweight": 20.17969806457545
            }
        ]"#);
        let binary_record: [PureRecord<SaftVRQMieRecord, ()>; 2] =
            serde_json::from_str(binary_json).expect("Unable to parse json.");
        SaftVRQMiePars::new(
            &SaftVRQMieParameters::new_binary(
                binary_record,
                Some(SaftVRQMieBinaryRecord {
                    k_ij: 0.105,
                    l_ij: 0.0,
                }),
                vec![],
            )
            .unwrap(),
        )
        .unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::SaftVRQMieParameters;
    use super::utils::{helium_fh1, hydrogen_fh2};
    use crate::saftvrqmie::SaftVRQMie;

    #[test]
    #[should_panic(
        expected = "cannot combine Feynman-Hibbs orders 1 and 2. Component 0 has order 1 and component 1 has order 2."
    )]
    fn incompatible_order() {
        let order1 = helium_fh1();
        let order2 = hydrogen_fh2();
        SaftVRQMie::new(SaftVRQMieParameters::new_binary([order1, order2], None, vec![]).unwrap())
            .unwrap();
    }
}
