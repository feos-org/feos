use crate::association::AssociationStrength;
use crate::epcsaft::eos::permittivity::PermittivityRecord;
use crate::hard_sphere::{HardSphereProperties, MonomerShape};
use feos_core::parameter::{AssociationParameters, CombiningRule, FromSegments, Parameters};
use feos_core::{FeosError, FeosResult, StateHD};
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use num_traits::Zero;
use serde::{Deserialize, Serialize};

/// ePC-SAFT pure-component parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct ElectrolytePcSaftRecord {
    /// Segment number
    pub m: f64,
    /// Segment diameter in units of Angstrom
    pub sigma: f64,
    /// Energetic parameter in units of Kelvin
    pub epsilon_k: f64,
    #[serde(default)]
    #[serde(skip_serializing_if = "f64::is_zero")]
    pub z: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permittivity_record: Option<PermittivityRecord>,
}

impl FromSegments for ElectrolytePcSaftRecord {
    fn from_segments(segments: &[(Self, f64)]) -> FeosResult<Self> {
        let mut m = 0.0;
        let mut sigma3 = 0.0;
        let mut epsilon_k = 0.0;
        let mut z = 0.0;

        segments.iter().for_each(|(s, n)| {
            m += s.m * n;
            sigma3 += s.m * s.sigma.powi(3) * n;
            epsilon_k += s.m * s.epsilon_k * n;
            z += s.z;
        });

        Ok(Self {
            m,
            sigma: (sigma3 / m).cbrt(),
            epsilon_k: epsilon_k / m,
            z,
            permittivity_record: None,
        })
    }
}

impl ElectrolytePcSaftRecord {
    pub fn new(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        z: f64,
        permittivity_record: Option<PermittivityRecord>,
    ) -> ElectrolytePcSaftRecord {
        ElectrolytePcSaftRecord {
            m,
            sigma,
            epsilon_k,
            z,
            permittivity_record,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct ElectrolytePcSaftAssociationRecord {
    /// Association volume parameter
    pub kappa_ab: f64,
    /// Association energy parameter in units of Kelvin
    pub epsilon_k_ab: f64,
}

impl ElectrolytePcSaftAssociationRecord {
    pub fn new(kappa_ab: f64, epsilon_k_ab: f64) -> Self {
        Self {
            kappa_ab,
            epsilon_k_ab,
        }
    }
}

impl CombiningRule<ElectrolytePcSaftRecord> for ElectrolytePcSaftAssociationRecord {
    fn combining_rule(
        _: &ElectrolytePcSaftRecord,
        _: &ElectrolytePcSaftRecord,
        parameters_i: &Self,
        parameters_j: &Self,
    ) -> Self {
        Self {
            kappa_ab: (parameters_i.kappa_ab * parameters_j.kappa_ab).sqrt(),
            epsilon_k_ab: 0.5 * (parameters_i.epsilon_k_ab + parameters_j.epsilon_k_ab),
        }
    }
}

/// ePC-SAFT binary interaction parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct ElectrolytePcSaftBinaryRecord {
    /// Binary dispersion interaction parameter
    pub k_ij: Vec<f64>,
}

impl ElectrolytePcSaftBinaryRecord {
    pub fn new(k_ij: Vec<f64>) -> Self {
        Self { k_ij }
    }
}

/// Parameter set required for the ePC-SAFT equation of state.
pub type ElectrolytePcSaftParameters = Parameters<
    ElectrolytePcSaftRecord,
    ElectrolytePcSaftBinaryRecord,
    ElectrolytePcSaftAssociationRecord,
>;

/// Parameter set required for the ePC-SAFT equation of state.
pub struct ElectrolytePcSaftPars {
    pub m: DVector<f64>,
    pub sigma: DVector<f64>,
    pub epsilon_k: DVector<f64>,
    pub z: DVector<f64>,
    pub k_ij: DMatrix<Vec<f64>>,
    pub sigma_ij: DMatrix<f64>,
    pub e_k_ij: DMatrix<f64>,
    pub nionic: usize,
    pub nsolvent: usize,
    pub water_sigma_t_comp: Option<usize>,
    pub ionic_comp: DVector<usize>,
    pub solvent_comp: DVector<usize>,
    pub permittivity: Vec<Option<PermittivityRecord>>,
}

impl ElectrolytePcSaftPars {
    pub fn sigma_t<D: DualNum<f64> + Copy>(&self, temperature: D) -> DVector<D> {
        let mut sigma_t = DVector::from_fn(self.sigma.len(), |i, _| D::from(self.sigma[i]));

        if let Some(i) = self.water_sigma_t_comp {
            sigma_t[i] +=
                (temperature * -0.01775).exp() * 10.11 - (temperature * -0.01146).exp() * 1.417;
        }

        sigma_t
    }

    pub fn sigma_ij_t<D: DualNum<f64> + Copy>(&self, temperature: D) -> DMatrix<D> {
        let diameter = self.sigma_t(temperature);
        let n = diameter.len();

        let mut sigma_ij_t = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                sigma_ij_t[(i, j)] = (diameter[i] + diameter[j]) * 0.5;
            }
        }
        sigma_ij_t
    }
}

impl ElectrolytePcSaftPars {
    pub fn new(parameters: &ElectrolytePcSaftParameters) -> FeosResult<Self> {
        let n = parameters.pure.len();

        let [m, sigma, epsilon_k, z] =
            parameters.collate(|pr| [pr.m, pr.sigma, pr.epsilon_k, pr.z]);

        let mut water_sigma_t_comp = None;

        for (i, _) in parameters.pure.iter().enumerate() {
            // check if component i is water with temperature-dependent sigma
            if (m[i] * 1000.0).round() / 1000.0 == 1.205 && epsilon_k[i].round() == 354.0 {
                // Godforsaken code below
                //
                // if let Some(record) = record.association_sites.first() {
                //     if let Some(assoc) = record.parameters {
                //         if (assoc.kappa_ab * 1000.0).round() / 1000.0 == 0.045
                //             && assoc.epsilon_k_ab.round() == 2426.0
                //         {
                water_sigma_t_comp = Some(i);
                // }
                // }
                // }
            }
        }

        let ionic_comp: DVector<usize> = z
            .iter()
            .enumerate()
            .filter_map(|(i, &zi)| (zi.abs() > 0.0).then_some(i))
            .collect::<Vec<_>>()
            .into();

        let nionic = ionic_comp.len();

        let solvent_comp: DVector<usize> = z
            .iter()
            .enumerate()
            .filter_map(|(i, &zi)| (zi.abs() == 0.0).then_some(i))
            .collect::<Vec<_>>()
            .into();
        let nsolvent = solvent_comp.len();

        let mut k_ij: DMatrix<Vec<f64>> = DMatrix::from_element(n, n, vec![0., 0., 0., 0.]);

        for br in &parameters.binary {
            let i = br.id1;
            let j = br.id2;
            let r = &br.model_record;
            if r.k_ij.len() > 4 {
                return Err(FeosError::IncompatibleParameters(format!(
                    "Binary interaction for component {i} with {j} is parametrized with more than 4 k_ij coefficients."
                )));
            } else {
                (0..r.k_ij.len()).for_each(|k| {
                    k_ij[(i, j)][k] = r.k_ij[k];
                    k_ij[(j, i)][k] = r.k_ij[k];
                });
            }
        }
        // No binary interaction between charged species of same kind (+/+ and -/-)
        ionic_comp.iter().for_each(|&ai| {
            k_ij[(ai, ai)][0] = 1.0;
            for k in 1..4usize {
                k_ij[(ai, ai)][k] = 0.0;
            }
        });

        let mut sigma_ij = DMatrix::zeros(n, n);
        let mut e_k_ij = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                e_k_ij[(i, j)] = (epsilon_k[i] * epsilon_k[j]).sqrt();
                sigma_ij[(i, j)] = 0.5 * (sigma[i] + sigma[j]);
            }
        }

        // Permittivity records
        let mut permittivity_records: Vec<Option<PermittivityRecord>> = parameters
            .pure
            .iter()
            .map(|record| record.clone().model_record.permittivity_record)
            .collect();

        // Check if permittivity_records contains maximum one record for each solvent
        // Permittivity
        if nionic != 0
            && permittivity_records
                .iter()
                .enumerate()
                .any(|(i, record)| record.is_none() && z[i] == 0.0)
        {
            return Err(FeosError::IncompatibleParameters(
                "Provide permittivity record for all solvent components.".to_string(),
            ));
        }

        let mut modeltypes: Vec<usize> = vec![];

        permittivity_records
            .iter()
            .filter(|&record| record.is_some())
            .for_each(|record| match record.as_ref().unwrap() {
                PermittivityRecord::PerturbationTheory { .. } => {
                    modeltypes.push(1);
                }
                PermittivityRecord::ExperimentalData { .. } => {
                    modeltypes.push(2);
                }
            });

        // check if modeltypes contains a mix of 1 and 2
        if modeltypes.contains(&1) && modeltypes.contains(&2) {
            return Err(FeosError::IncompatibleParameters(
                "Inconsistent models for permittivity.".to_string(),
            ));
        }

        if !modeltypes.is_empty() && modeltypes[0] == 2 {
            for permittivity_record in &permittivity_records {
                if let Some(PermittivityRecord::ExperimentalData { data }) =
                    permittivity_record.as_ref()
                {
                    // check if length of data is greater than 0
                    if data.is_empty() {
                        return Err(FeosError::IncompatibleParameters(
                                "Experimental data for permittivity must contain at least one data point.".to_string(),
                            ));
                    }
                }
            }
        }

        if !modeltypes.is_empty() && modeltypes[0] == 2 {
            // order points in data by increasing temperature
            let mut permittivity_records_clone = permittivity_records.clone();
            permittivity_records_clone
                .iter_mut()
                .filter(|record| record.is_some())
                .enumerate()
                .for_each(|(i, record)| {
                    if let PermittivityRecord::ExperimentalData { data } = record.as_mut().unwrap()
                    {
                        let mut data = data.clone();
                        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        // check if all temperatures a.0 in data are finite, if not, make them finite by rounding to four digits
                        data.iter_mut().for_each(|a| {
                            if !a.0.is_finite() {
                                a.0 = (a.0 * 1e4).round() / 1e4;
                            }
                        });
                        // save data again in record
                        permittivity_records[i] =
                            Some(PermittivityRecord::ExperimentalData { data });
                    }
                });
        }

        Ok(Self {
            m,
            sigma,
            epsilon_k,
            z,
            k_ij,
            sigma_ij,
            e_k_ij,
            nionic,
            nsolvent,
            ionic_comp,
            solvent_comp,
            water_sigma_t_comp,
            permittivity: permittivity_records,
        })
    }
}

impl HardSphereProperties for ElectrolytePcSaftPars {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<'_, N> {
        MonomerShape::NonSpherical(self.m.map(N::from))
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> DVector<D> {
        let sigma_t = self.sigma_t(temperature);

        let ti = temperature.recip() * -3.0;
        let mut d = DVector::from_fn(sigma_t.len(), |i, _| {
            -((ti * self.epsilon_k[i]).exp() * 0.12 - 1.0) * sigma_t[i]
        });
        for i in 0..self.nionic {
            let ai = self.ionic_comp[i];
            d[ai] = D::one() * sigma_t[ai] * (1.0 - 0.12);
        }
        d
    }
}

impl AssociationStrength for ElectrolytePcSaftPars {
    type Record = ElectrolytePcSaftAssociationRecord;

    fn association_strength<D: DualNum<f64> + Copy>(
        &self,
        parameters: &AssociationParameters<Self::Record>,
        state: &StateHD<D>,
        diameter: &DVector<D>,
        xi: D,
    ) -> [DMatrix<D>; 2] {
        let p = parameters;
        let t_inv = state.temperature.recip();
        let [zeta2, n3] = self.zeta(state.temperature, &state.partial_density, [2, 3]);
        let n2 = zeta2 * 6.0;
        let n3i = (-n3 + 1.0).recip();

        let mut delta_ab = DMatrix::zeros(p.sites_a.len(), p.sites_b.len());
        for b in &p.binary_ab {
            let [i, j] = [b.id1, b.id2];
            let [comp_i, comp_j] = [p.sites_a[i].assoc_comp, p.sites_b[j].assoc_comp];
            let f_ab_ij = (t_inv * b.model_record.epsilon_k_ab).exp_m1();
            let k_ab_ij =
                b.model_record.kappa_ab * (self.sigma[comp_i] * self.sigma[comp_j]).powf(1.5);

            // g_HS(d)
            let di = diameter[comp_i];
            let dj = diameter[comp_j];
            let k = di * dj / (di + dj) * (n2 * n3i);
            let g_contact = n3i * (k * xi * (k / 18.0 + 0.5) + 1.0);

            delta_ab[(i, j)] = g_contact * f_ab_ij * k_ab_ij;
        }

        let mut delta_cc = DMatrix::zeros(p.sites_c.len(), p.sites_c.len());
        for b in &p.binary_cc {
            let [i, j] = [b.id1, b.id2];
            let [comp_i, comp_j] = [p.sites_c[i].assoc_comp, p.sites_c[j].assoc_comp];
            let f_ab_ij = (t_inv * b.model_record.epsilon_k_ab).exp_m1();
            let k_ab_ij =
                b.model_record.kappa_ab * (self.sigma[comp_i] * self.sigma[comp_j]).powf(1.5);

            // g_HS(d)
            let di = diameter[comp_i];
            let dj = diameter[comp_j];
            let k = di * dj / (di + dj) * (n2 * n3i);
            let g_contact = n3i * (k * xi * (k / 18.0 + 0.5) + 1.0);

            delta_cc[(i, j)] = g_contact * f_ab_ij * k_ab_ij;
        }
        [delta_ab, delta_cc]
    }
}

#[cfg(test)]
pub mod utils {
    use super::*;
    use feos_core::parameter::{BinaryRecord, Identifier, IdentifierOption, PureRecord};

    type Pure = PureRecord<ElectrolytePcSaftRecord, ElectrolytePcSaftAssociationRecord>;
    type Binary =
        BinaryRecord<Identifier, ElectrolytePcSaftBinaryRecord, ElectrolytePcSaftAssociationRecord>;

    pub fn propane_parameters() -> ElectrolytePcSaftParameters {
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
                "molarweight": 44.0962
            }"#;
        let propane_record: Pure =
            serde_json::from_str(propane_json).expect("Unable to parse json.");
        ElectrolytePcSaftParameters::new_pure(propane_record).unwrap()
    }

    pub fn butane_parameters() -> ElectrolytePcSaftParameters {
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
        let butane_record: Pure = serde_json::from_str(butane_json).expect("Unable to parse json.");
        ElectrolytePcSaftParameters::new_pure(butane_record).unwrap()
    }

    pub fn water_nacl_parameters_perturb() -> ElectrolytePcSaftParameters {
        // Water parameters from Held et al. (2014), originally from Fuchs et al. (2006)
        let pure_json = r#"[
            {
                "identifier": {
                    "cas": "7732-18-5",
                    "name": "water_np_sigma_t",
                    "iupac_name": "oxidane",
                    "smiles": "O",
                    "inchi": "InChI=1/H2O/h1H2",
                    "formula": "H2O"
                },
                "m": 1.2047,
                "sigma": 2.7927,
                "epsilon_k": 353.95,
                "kappa_ab": 0.04509,
                "epsilon_k_ab": 2425.7,
                "permittivity_record": {
                    "PerturbationTheory": {
                        "dipole_scaling": 
                            5.199
                        ,
                        "polarizability_scaling": 
                            0.0
                        ,
                        "correlation_integral_parameter": 
                            0.1276
                        
                    }
                },
                "molarweight": 18.0152
            },
            {
                "identifier": {
                    "cas": "110-54-3",
                    "name": "na+",
                    "formula": "na+"
                },
                "m": 1,
                "sigma": 2.8232,
                "epsilon_k": 230.0,
                "z": 1,
                "permittivity_record": {
                    "PerturbationTheory": {
                        "dipole_scaling": 
                            0.0,
                        "polarizability_scaling": 
                            0.0,
                        "correlation_integral_parameter": 
                            0.0658      
                    }
                },
                "molarweight": 22.98977
            },
            {
                "identifier": {
                    "cas": "7782-50-5",
                    "name": "cl-",
                    "formula": "cl-"
                },
                "m": 1,
                "sigma": 2.7560,
                "epsilon_k": 170,
                "z": -1,
                "permittivity_record": {
                    "PerturbationTheory": {
                        "dipole_scaling": 
                            7.3238,
                        "polarizability_scaling": 
                            0.0,
                        "correlation_integral_parameter": 
                            0.2620 
                    }
                },
                "molarweight": 35.45
            }
            ]"#;
        let binary_json = r#"[
            {
                "id1": {
                    "cas": "7732-18-5",
                    "name": "water_np_sigma_t",
                    "iupac_name": "oxidane",
                    "smiles": "O",
                    "inchi": "InChI=1/H2O/h1H2",
                    "formula": "H2O"
                },
                "id2": {
                    "cas": "110-54-3",
                    "name": "sodium ion",
                    "formula": "na+"
                },
                "k_ij": [
                    0.0045,
                    0.0,
                    0.0,
                    0.0
                ]
            },
            {
                "id1": {
                    "cas": "7732-18-5",
                    "name": "water_np_sigma_t",
                    "iupac_name": "oxidane",
                    "smiles": "O",
                    "inchi": "InChI=1/H2O/h1H2",
                    "formula": "H2O"
                },
                "id2": {
                    "cas": "7782-50-5",
                    "name": "chloride ion",
                    "formula": "cl-"
                },
                "k_ij": [
                    -0.25,
                    0.0,
                    0.0,
                    0.0
                ]
            },
            {
                "id1": {
                    "cas": "110-54-3",
                    "name": "sodium ion",
                    "formula": "na+"
                },
                "id2": {
                    "cas": "7782-50-5",
                    "name": "chloride ion",
                    "formula": "cl-"
                },
                "k_ij": [
                    0.317,
                    0.0,
                    0.0,
                    0.0
                ]
            }
            ]"#;
        let pure_records: Vec<Pure> =
            serde_json::from_str(pure_json).expect("Unable to parse json.");
        let binary_records: Vec<Binary> =
            serde_json::from_str(binary_json).expect("Unable to parse json.");
        ElectrolytePcSaftParameters::from_records(
            pure_records,
            binary_records,
            IdentifierOption::Name,
        )
        .unwrap()
    }

    pub fn water_nacl_parameters() -> ElectrolytePcSaftParameters {
        // Water parameters from Held et al. (2014), originally from Fuchs et al. (2006)
        let pure_json = r#"[
            {
                "identifier": {
                    "cas": "7732-18-5",
                    "name": "water_np_sigma_t",
                    "iupac_name": "oxidane",
                    "smiles": "O",
                    "inchi": "InChI=1/H2O/h1H2",
                    "formula": "H2O"
                },
                "m": 1.2047,
                "sigma": 2.7927,
                "epsilon_k": 353.95,
                "kappa_ab": 0.04509,
                "epsilon_k_ab": 2425.7,
                "permittivity_record": {
                    "ExperimentalData": {
                    "data": 
                        [
                            [
                                280.15,
                                84.89
                            ],
                            [
                                298.15,
                                78.39
                            ],
                            [
                                360.15,
                                58.73
                            ]
                        ]
                    }
                },
                "molarweight": 18.0152
            },
            {
                "identifier": {
                    "cas": "110-54-3",
                    "name": "na+",
                    "formula": "na+"
                },
                "m": 1,
                "sigma": 2.8232,
                "epsilon_k": 230.0,
                "z": 1,
                "permittivity_record": {
                    "ExperimentalData": {
                    "data":  
                        [
                            [
                                298.15,
                                8.0
                            ]
                        ]
                    }
                },
                "molarweight": 22.98977
            },
            {
                "identifier": {
                    "cas": "7782-50-5",
                    "name": "cl-",
                    "formula": "cl-"
                },
                "m": 1,
                "sigma": 2.7560,
                "epsilon_k": 170,
                "z": -1,
                "permittivity_record": {
                    "ExperimentalData": {
                    "data": 
                        [
                            [
                                298.15,
                                8.0
                            ]
                        ]
                    }
                },
                "molarweight": 35.45
            }
            ]"#;
        let binary_json = r#"[
            {
                "id1": {
                    "cas": "7732-18-5",
                    "name": "water_np_sigma_t",
                    "iupac_name": "oxidane",
                    "smiles": "O",
                    "inchi": "InChI=1/H2O/h1H2",
                    "formula": "H2O"
                },
                "id2": {
                    "cas": "110-54-3",
                    "name": "sodium ion",
                    "formula": "na+"
                },
                "k_ij": [
                    0.0045,
                    0.0,
                    0.0,
                    0.0
                ]
            },
            {
                "id1": {
                    "cas": "7732-18-5",
                    "name": "water_np_sigma_t",
                    "iupac_name": "oxidane",
                    "smiles": "O",
                    "inchi": "InChI=1/H2O/h1H2",
                    "formula": "H2O"
                },
                "id2": {
                    "cas": "7782-50-5",
                    "name": "chloride ion",
                    "formula": "cl-"
                },
                "k_ij": [
                    -0.25,
                    0.0,
                    0.0,
                    0.0
                ]
            },
            {
                "id1": {
                    "cas": "110-54-3",
                    "name": "sodium ion",
                    "formula": "na+"
                },
                "id2": {
                    "cas": "7782-50-5",
                    "name": "chloride ion",
                    "formula": "cl-"
                },
                "k_ij": [
                    0.317,
                    0.0,
                    0.0,
                    0.0
                ]
            }
            ]"#;
        let pure_records: Vec<Pure> =
            serde_json::from_str(pure_json).expect("Unable to parse json.");
        let binary_records: Vec<Binary> =
            serde_json::from_str(binary_json).expect("Unable to parse json.");
        ElectrolytePcSaftParameters::from_records(
            pure_records,
            binary_records,
            IdentifierOption::Name,
        )
        .unwrap()
    }

    pub fn propane_butane_parameters() -> ElectrolytePcSaftParameters {
        let records_json = r#"[
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
                "molarweight": 58.123
            }
        ]"#;
        let records: [Pure; 2] = serde_json::from_str(records_json).expect("Unable to parse json.");
        ElectrolytePcSaftParameters::new_binary(records, None, vec![]).unwrap()
    }
}
