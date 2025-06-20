use feos_core::FeosResult;
use feos_core::parameter::{FromSegments, FromSegmentsBinary, Parameters};
use ndarray::{Array, Array1, Array2};
use num_traits::Zero;
use quantity::{JOULE, KB, KELVIN};
use serde::{Deserialize, Serialize};

/// PC-SAFT pure-component parameters.
#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct PcSaftRecord {
    /// Segment number
    pub m: f64,
    /// Segment diameter in units of Angstrom
    pub sigma: f64,
    /// Energetic parameter in units of Kelvin
    pub epsilon_k: f64,
    /// Dipole moment in units of Debye
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub mu: f64,
    /// Quadrupole moment in units of Debye * Angstrom
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub q: f64,
    /// Entropy scaling coefficients for the viscosity
    #[serde(skip_serializing_if = "Option::is_none")]
    viscosity: Option<[f64; 4]>,
    /// Entropy scaling coefficients for the diffusion coefficient
    #[serde(skip_serializing_if = "Option::is_none")]
    diffusion: Option<[f64; 5]>,
    /// Entropy scaling coefficients for the thermal conductivity
    #[serde(skip_serializing_if = "Option::is_none")]
    thermal_conductivity: Option<[f64; 4]>,
}

impl FromSegments for PcSaftRecord {
    fn from_segments(segments: &[(Self, f64)]) -> FeosResult<Self> {
        let mut m = 0.0;
        let mut sigma3 = 0.0;
        let mut epsilon_k = 0.0;
        let mut mu = 0.0;
        let mut q = 0.0;

        segments.iter().for_each(|(s, n)| {
            m += s.m * n;
            sigma3 += s.m * s.sigma.powi(3) * n;
            epsilon_k += s.m * s.epsilon_k * n;
            mu += s.mu * n;
            q += s.q * n;
        });

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
            viscosity,
            diffusion,
            thermal_conductivity,
        })
    }
}

impl PcSaftRecord {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        mu: f64,
        q: f64,
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
            viscosity,
            diffusion,
            thermal_conductivity,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct PcSaftAssociationRecord {
    /// Association volume parameter
    pub kappa_ab: f64,
    /// Association energy parameter in units of Kelvin
    pub epsilon_k_ab: f64,
}

impl PcSaftAssociationRecord {
    pub fn new(kappa_ab: f64, epsilon_k_ab: f64) -> Self {
        Self {
            kappa_ab,
            epsilon_k_ab,
        }
    }
}

/// PC-SAFT binary interaction parameters.
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct PcSaftBinaryRecord {
    /// Binary dispersion interaction parameter
    pub k_ij: f64,
}

impl PcSaftBinaryRecord {
    pub fn new(k_ij: f64) -> Self {
        Self { k_ij }
    }
}

impl FromSegmentsBinary for PcSaftBinaryRecord {
    fn from_segments_binary(segments: &[(Self, f64, f64)]) -> FeosResult<Self> {
        let (k_ij, n) = segments.iter().fold((0.0, 0.0), |(k_ij, n), (br, n1, n2)| {
            let nab = n1 * n2;
            (k_ij + br.k_ij * nab, n + nab)
        });
        Ok(Self { k_ij: k_ij / n })
    }
}

/// Parameter set required for the PC-SAFT equation of state and Helmholtz energy functional.
pub type PcSaftParameters = Parameters<PcSaftRecord, PcSaftBinaryRecord, PcSaftAssociationRecord>;

/// The PC-SAFT parameters in an easier accessible format.
pub struct PcSaftPars {
    pub m: Array1<f64>,
    pub sigma: Array1<f64>,
    pub epsilon_k: Array1<f64>,
    pub mu2: Array1<f64>,
    pub q2: Array1<f64>,
    pub sigma_ij: Array2<f64>,
    pub epsilon_k_ij: Array2<f64>,
    pub e_k_ij: Array2<f64>,
    pub ndipole: usize,
    pub nquadpole: usize,
    pub dipole_comp: Array1<usize>,
    pub quadpole_comp: Array1<usize>,
    pub viscosity: Option<Array2<f64>>,
    pub diffusion: Option<Array2<f64>>,
    pub thermal_conductivity: Option<Array2<f64>>,
}

impl PcSaftPars {
    pub fn new(parameters: &PcSaftParameters) -> Self {
        let n = parameters.pure.len();

        let [m, sigma, epsilon_k] = parameters.collate(|pr| [pr.m, pr.sigma, pr.epsilon_k]);
        let [mu, q] = parameters.collate(|pr| [pr.mu, pr.q]);
        let [viscosity, thermal_conductivity] =
            parameters.collate(|pr| [pr.viscosity, pr.thermal_conductivity]);
        let [diffusion] = parameters.collate(|pr| [pr.diffusion]);
        let [k_ij] = parameters.collate_binary(|br| [br.k_ij]);

        let mu2 = &mu * &mu / (&m * &sigma * &sigma * &sigma * &epsilon_k)
            * 1e-19
            * (JOULE / KELVIN / KB).into_value();
        let q2 = &q * &q / (&m * &sigma.mapv(|s| s.powi(5)) * &epsilon_k)
            * 1e-19
            * (JOULE / KELVIN / KB).into_value();
        let dipole_comp: Array1<usize> = mu2
            .iter()
            .enumerate()
            .filter_map(|(i, &mu2)| (mu2.abs() > 0.0).then_some(i))
            .collect();
        let ndipole = dipole_comp.len();
        let quadpole_comp: Array1<usize> = q2
            .iter()
            .enumerate()
            .filter_map(|(i, &q2)| (q2.abs() > 0.0).then_some(i))
            .collect();
        let nquadpole = quadpole_comp.len();

        let mut sigma_ij = Array::zeros((n, n));
        let mut e_k_ij = Array::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                e_k_ij[[i, j]] = (epsilon_k[i] * epsilon_k[j]).sqrt();
                sigma_ij[[i, j]] = 0.5 * (sigma[i] + sigma[j]);
            }
        }
        let epsilon_k_ij = (1.0 - k_ij) * &e_k_ij;

        let viscosity = if viscosity.iter().any(|v| v.is_none()) {
            None
        } else {
            let mut v = Array2::zeros((4, viscosity.len()));
            for (i, vi) in viscosity.iter().enumerate() {
                v.column_mut(i).assign(&Array1::from(vi.unwrap().to_vec()));
            }
            Some(v)
        };

        let diffusion = if diffusion.iter().any(|v| v.is_none()) {
            None
        } else {
            let mut v = Array2::zeros((5, diffusion.len()));
            for (i, vi) in diffusion.iter().enumerate() {
                v.column_mut(i).assign(&Array1::from(vi.unwrap().to_vec()));
            }
            Some(v)
        };

        let thermal_conductivity = if thermal_conductivity.iter().any(|v| v.is_none()) {
            None
        } else {
            let mut v = Array2::zeros((4, thermal_conductivity.len()));
            for (i, vi) in thermal_conductivity.iter().enumerate() {
                v.column_mut(i).assign(&Array1::from(vi.unwrap().to_vec()));
            }
            Some(v)
        };

        Self {
            m,
            sigma,
            epsilon_k,
            mu2,
            q2,
            sigma_ij,
            epsilon_k_ij,
            e_k_ij,
            ndipole,
            nquadpole,
            dipole_comp,
            quadpole_comp,
            viscosity,
            diffusion,
            thermal_conductivity,
        }
    }
}

#[cfg(test)]
pub mod utils {
    use super::*;
    use crate::pcsaft::PcSaft;
    use feos_core::parameter::{BinarySegmentRecord, ChemicalRecord, PureRecord, SegmentRecord};
    use std::sync::Arc;

    pub fn propane_parameters() -> Arc<PcSaft> {
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
                "m": 2.001829,
                "sigma": 3.618353,
                "epsilon_k": 208.1101,
                "viscosity": [-0.8013, -1.9972,-0.2907, -0.0467],
                "thermal_conductivity": [-0.15348,  -0.6388, 1.21342, -0.01664],
                "diffusion": [-0.675163251512047, 0.3212017677695878, 0.100175249144429, 0.0, 0.0],
                "molarweight": 44.0962
            }"#;
        let propane_record: PureRecord<PcSaftRecord, PcSaftAssociationRecord> =
            serde_json::from_str(propane_json).expect("Unable to parse json.");
        Arc::new(PcSaft::new(
            PcSaftParameters::new_pure(propane_record).unwrap(),
        ))
    }

    pub fn carbon_dioxide_parameters() -> PcSaftPars {
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
            "m": 1.5131,
            "sigma": 3.1869,
            "epsilon_k": 163.333,
            "q": 4.4
        }"#;
        let co2_record: PureRecord<PcSaftRecord, PcSaftAssociationRecord> =
            serde_json::from_str(co2_json).expect("Unable to parse json.");
        PcSaftPars::new(&PcSaftParameters::new_pure(co2_record).unwrap())
    }

    pub fn butane_parameters() -> Arc<PcSaft> {
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
                "m": 2.331586,
                "sigma": 3.7086010000000003,
                "epsilon_k": 222.8774,
                "molarweight": 58.123
            }"#;
        let butane_record: PureRecord<PcSaftRecord, PcSaftAssociationRecord> =
            serde_json::from_str(butane_json).expect("Unable to parse json.");
        Arc::new(PcSaft::new(
            PcSaftParameters::new_pure(butane_record).unwrap(),
        ))
    }

    pub fn dme_parameters() -> PcSaftPars {
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
                "m": 2.2634,
                "sigma": 3.2723,
                "epsilon_k": 210.29,
                "mu": 1.3,
                "molarweight": 46.0688
            }"#;
        let dme_record: PureRecord<PcSaftRecord, PcSaftAssociationRecord> =
            serde_json::from_str(dme_json).expect("Unable to parse json.");
        PcSaftPars::new(&PcSaftParameters::new_pure(dme_record).unwrap())
    }

    pub fn water_parameters(na: f64) -> PcSaftParameters {
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
                "m": 1.065587,
                "sigma": 3.000683,
                "epsilon_k": 366.5121,
                "molarweight": 18.0152,
                "association_sites": [
                    {
                        "kappa_ab": 0.034867983,
                        "epsilon_k_ab": 2500.6706,
                        "na": 1.0,
                        "nb": 1.0
                    }
                ]
            }"#;
        let mut water_record: PureRecord<PcSaftRecord, PcSaftAssociationRecord> =
            serde_json::from_str(water_json).expect("Unable to parse json.");
        water_record.association_sites[0].na = na;
        PcSaftParameters::new_pure(water_record).unwrap()
    }

    pub fn dme_co2_parameters() -> PcSaftPars {
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
                "m": 2.2634,
                "sigma": 3.2723,
                "epsilon_k": 210.29,
                "mu": 1.3
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
                "m": 1.5131,
                "sigma": 3.1869,
                "epsilon_k": 163.333,
                "q": 4.4
            }
        ]"#;
        let binary_record: [PureRecord<PcSaftRecord, PcSaftAssociationRecord>; 2] =
            serde_json::from_str(binary_json).expect("Unable to parse json.");
        PcSaftPars::new(&PcSaftParameters::new_binary(binary_record, None, vec![]).unwrap())
    }

    pub fn propane_butane_parameters() -> Arc<PcSaft> {
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
                "m": 2.0018290000000003,
                "sigma": 3.618353,
                "epsilon_k": 208.1101,
                "viscosity": [-0.8013, -1.9972, -0.2907, -0.0467],
                "thermal_conductivity": [-0.15348, -0.6388, 1.21342, -0.01664],
                "diffusion": [-0.675163251512047, 0.3212017677695878, 0.100175249144429, 0.0, 0.0],
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
                "m": 2.331586,
                "sigma": 3.7086010000000003,
                "epsilon_k": 222.8774,
                "viscosity": [-0.9763, -2.2413, -0.3690, -0.0605],
                "diffusion": [-0.8985872992958458, 0.3428584416613513, 0.10236616087103916, 0.0, 0.0],
                "molarweight": 58.123
            }
        ]"#;
        let binary_record: [PureRecord<PcSaftRecord, PcSaftAssociationRecord>; 2] =
            serde_json::from_str(binary_json).expect("Unable to parse json.");
        Arc::new(PcSaft::new(
            PcSaftParameters::new_binary(binary_record, None, vec![]).unwrap(),
        ))
    }

    #[test]
    pub fn test_kij() -> FeosResult<()> {
        let ch3: String = "CH3".into();
        let ch2: String = "CH2".into();
        let oh = "OH".into();
        let propane = ChemicalRecord::new(
            Default::default(),
            vec![ch3.clone(), ch2.clone(), ch3.clone()],
            None,
        );
        let ethanol = ChemicalRecord::new(Default::default(), vec![ch3, ch2, oh], None);
        let segment_records =
            SegmentRecord::from_json("../../parameters/pcsaft/sauer2014_homo.json")?;
        let kij = [("CH3", "OH", -0.2), ("CH2", "OH", -0.1)];
        let binary_segment_records: Vec<_> = kij
            .iter()
            .map(|&(id1, id2, k_ij)| {
                BinarySegmentRecord::new(id1.into(), id2.into(), Some(PcSaftBinaryRecord { k_ij }))
            })
            .collect();
        let params = PcSaftParameters::from_segments(
            vec![propane, ethanol],
            &segment_records,
            Some(&binary_segment_records),
        )?;
        assert_eq!(params.binary[0].id1, 0);
        assert_eq!(params.binary[0].id2, 1);
        assert_eq!(params.binary[0].model_record.k_ij, -0.5 / 9.);

        Ok(())
    }

    #[test]
    fn test_association_json() -> FeosResult<()> {
        let json1 = r#"
            {
                "identifier": {
                    "name": "comp1"
                },
                "m": 1.065587,
                "sigma": 3.000683,
                "epsilon_k": 366.5121,
                "association_sites": [
                    {
                        "kappa_ab": 0.034867983,
                        "epsilon_k_ab": 2500.6706,
                        "na": 1.0,
                        "nb": 1.0
                    }
                ]
            }"#;
        let record1: PureRecord<PcSaftRecord, PcSaftAssociationRecord> =
            serde_json::from_str(json1)?;

        let json2 = r#"
            {
                "identifier": {
                    "name": "comp2"
                },
                "m": 1.065587,
                "sigma": 3.000683,
                "epsilon_k": 366.5121,
                "association_sites": [
                    {
                        "id": "site1",
                        "kappa_ab": 0.034867983,
                        "epsilon_k_ab": 2500.6706,
                        "na": 1.0,
                        "nb": 1.0
                    },
                    {
                        "id": "site2",
                        "kappa_ab": 0.01,
                        "epsilon_k_ab": 2200.0,
                        "nb": 1.0
                    }
                ]
            }"#;
        let record2: PureRecord<PcSaftRecord, PcSaftAssociationRecord> =
            serde_json::from_str(json2)?;

        // println!("{record1}");
        // println!("{record2}");

        assert_eq!(
            record1.association_sites[0].parameters.unwrap().kappa_ab,
            record2.association_sites[0].parameters.unwrap().kappa_ab
        );

        Ok(())
    }
}
