use crate::association::{
    AssociationParameters, AssociationRecord, AssociationStrength, BinaryAssociationRecord,
};
use crate::hard_sphere::{HardSphereProperties, MonomerShape};
use feos_core::parameter::{FromSegments, Parameter, PureRecord};
use feos_core::{FeosError, FeosResult};
use ndarray::{Array, Array1, Array2};
use num_dual::DualNum;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Arc;

use crate::epcsaft::eos::permittivity::PermittivityRecord;

/// ePC-SAFT pure-component parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct ElectrolytePcSaftRecord {
    /// Segment number
    pub m: f64,
    /// Segment diameter in units of Angstrom
    pub sigma: f64,
    /// Energetic parameter in units of Kelvin
    pub epsilon_k: f64,
    /// Association parameters
    #[serde(flatten)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub association_record: Option<AssociationRecord<ElectrolytePcSaftAssociationRecord>>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permittivity_record: Option<PermittivityRecord>,
}

impl FromSegments<f64> for ElectrolytePcSaftRecord {
    fn from_segments(segments: &[(Self, f64)]) -> FeosResult<Self> {
        let mut m = 0.0;
        let mut sigma3 = 0.0;
        let mut epsilon_k = 0.0;
        let mut z = 0.0;

        segments.iter().for_each(|(s, n)| {
            m += s.m * n;
            sigma3 += s.m * s.sigma.powi(3) * n;
            epsilon_k += s.m * s.epsilon_k * n;
            z += s.z.unwrap_or(0.0);
        });

        let association_record = segments
            .iter()
            .filter_map(|(s, n)| {
                s.association_record.as_ref().map(|record| {
                    [
                        record.parameters.kappa_ab * n,
                        record.parameters.epsilon_k_ab * n,
                        record.na * n,
                        record.nb * n,
                        record.nc * n,
                    ]
                })
            })
            .reduce(|a, b| {
                [
                    a[0] + b[0],
                    a[1] + b[1],
                    a[2] + b[2],
                    a[3] + b[3],
                    a[4] + b[4],
                ]
            })
            .map(|[kappa_ab, epsilon_k_ab, na, nb, nc]| {
                AssociationRecord::new(
                    ElectrolytePcSaftAssociationRecord::new(kappa_ab, epsilon_k_ab),
                    na,
                    nb,
                    nc,
                )
            });

        Ok(Self {
            m,
            sigma: (sigma3 / m).cbrt(),
            epsilon_k: epsilon_k / m,
            association_record,
            z: Some(z),
            permittivity_record: None,
        })
    }
}

impl FromSegments<usize> for ElectrolytePcSaftRecord {
    fn from_segments(segments: &[(Self, usize)]) -> FeosResult<Self> {
        // We do not allow more than a single segment for q, mu, kappa_ab, epsilon_k_ab
        let segments: Vec<_> = segments
            .iter()
            .cloned()
            .map(|(s, c)| (s, c as f64))
            .collect();
        Self::from_segments(&segments)
    }
}

impl std::fmt::Display for ElectrolytePcSaftRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ElectrolytePcSaftRecord(m={}", self.m)?;
        write!(f, ", sigma={}", self.sigma)?;
        write!(f, ", epsilon_k={}", self.epsilon_k)?;
        if let Some(n) = &self.association_record {
            write!(f, ", association_record={}", n)?;
        }
        if let Some(n) = &self.z {
            write!(f, ", z={}", n)?;
        }
        if let Some(n) = &self.permittivity_record {
            write!(f, ", permittivity_record={:?}", n)?;
        }
        write!(f, ")")
    }
}

impl ElectrolytePcSaftRecord {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        kappa_ab: Option<f64>,
        epsilon_k_ab: Option<f64>,
        na: Option<f64>,
        nb: Option<f64>,
        nc: Option<f64>,
        z: Option<f64>,
        permittivity_record: Option<PermittivityRecord>,
    ) -> ElectrolytePcSaftRecord {
        let association_record =
            if let (Some(kappa_ab), Some(epsilon_k_ab)) = (kappa_ab, epsilon_k_ab) {
                Some(AssociationRecord::new(
                    ElectrolytePcSaftAssociationRecord::new(kappa_ab, epsilon_k_ab),
                    na.unwrap_or_default(),
                    nb.unwrap_or_default(),
                    nc.unwrap_or_default(),
                ))
            } else {
                None
            };
        ElectrolytePcSaftRecord {
            m,
            sigma,
            epsilon_k,
            association_record,
            z,
            permittivity_record,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Default)]
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

impl std::fmt::Display for ElectrolytePcSaftAssociationRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ElectrolytePcSaftAssociationRecord(kappa_ab={}",
            self.kappa_ab
        )?;
        write!(f, ", epsilon_k_ab={})", self.epsilon_k_ab)
    }
}

/// ePC-SAFT binary interaction parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct ElectrolytePcSaftBinaryRecord {
    /// Binary dispersion interaction parameter
    #[serde(default)]
    pub k_ij: Vec<f64>,
    /// Binary association parameters
    #[serde(flatten)]
    association: Option<BinaryAssociationRecord<ElectrolytePcSaftBinaryAssociationRecord>>,
}

impl ElectrolytePcSaftBinaryRecord {
    pub fn new(k_ij: Option<Vec<f64>>, kappa_ab: Option<f64>, epsilon_k_ab: Option<f64>) -> Self {
        let k_ij = k_ij.unwrap_or_default();
        let association = if kappa_ab.is_none() && epsilon_k_ab.is_none() {
            None
        } else {
            Some(BinaryAssociationRecord::new(
                ElectrolytePcSaftBinaryAssociationRecord::new(kappa_ab, epsilon_k_ab),
                None,
            ))
        };
        Self { k_ij, association }
    }
}

impl From<f64> for ElectrolytePcSaftBinaryRecord {
    fn from(k_ij: f64) -> Self {
        Self {
            k_ij: vec![k_ij, 0., 0., 0.],
            association: None,
        }
    }
}

impl From<Vec<f64>> for ElectrolytePcSaftBinaryRecord {
    fn from(k_ij: Vec<f64>) -> Self {
        Self {
            k_ij,
            association: None,
        }
    }
}

impl From<ElectrolytePcSaftBinaryRecord> for f64 {
    fn from(binary_record: ElectrolytePcSaftBinaryRecord) -> Self {
        match binary_record.k_ij.first() {
            Some(&k_ij) => k_ij,
            None => 0.0,
        }
    }
}

impl From<ElectrolytePcSaftBinaryRecord> for Vec<f64> {
    fn from(binary_record: ElectrolytePcSaftBinaryRecord) -> Self {
        binary_record.k_ij
    }
}

impl std::fmt::Display for ElectrolytePcSaftBinaryRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut tokens = vec![];
        if !self.k_ij[0].is_zero() {
            tokens.push(format!("k_ij_0={}", self.k_ij[0]));
            tokens.push(format!(", k_ij_1={}", self.k_ij[1]));
            tokens.push(format!(", k_ij_2={}", self.k_ij[2]));
            tokens.push(format!(", k_ij_3={})", self.k_ij[3]));
        }
        if let Some(association) = self.association {
            if let Some(kappa_ab) = association.parameters.kappa_ab {
                tokens.push(format!("kappa_ab={}", kappa_ab));
            }
            if let Some(epsilon_k_ab) = association.parameters.epsilon_k_ab {
                tokens.push(format!("epsilon_k_ab={}", epsilon_k_ab));
            }
        }
        write!(f, "ElectrolytePcSaftBinaryRecord({})", tokens.join(""))
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct ElectrolytePcSaftBinaryAssociationRecord {
    /// Cross-association association volume parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kappa_ab: Option<f64>,
    /// Cross-association energy parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epsilon_k_ab: Option<f64>,
}

impl ElectrolytePcSaftBinaryAssociationRecord {
    pub fn new(kappa_ab: Option<f64>, epsilon_k_ab: Option<f64>) -> Self {
        Self {
            kappa_ab,
            epsilon_k_ab,
        }
    }
}

/// Parameter set required for the ePC-SAFT equation of state.
pub struct ElectrolytePcSaftParameters {
    pub molarweight: Array1<f64>,
    pub m: Array1<f64>,
    pub sigma: Array1<f64>,
    pub epsilon_k: Array1<f64>,
    pub association: Arc<AssociationParameters<Self>>,
    pub z: Array1<f64>,
    pub k_ij: Array2<Vec<f64>>,
    pub sigma_ij: Array2<f64>,
    pub e_k_ij: Array2<f64>,
    pub nionic: usize,
    pub nsolvent: usize,
    pub water_sigma_t_comp: Option<usize>,
    pub ionic_comp: Array1<usize>,
    pub solvent_comp: Array1<usize>,
    pub permittivity: Array1<Option<PermittivityRecord>>,
    pub pure_records: Vec<PureRecord<ElectrolytePcSaftRecord>>,
    pub binary_records: Option<Array2<ElectrolytePcSaftBinaryRecord>>,
}

impl ElectrolytePcSaftParameters {
    pub fn sigma_t<D: DualNum<f64>>(&self, temperature: D) -> Array1<f64> {
        let mut sigma_t: Array1<f64> = Array::from_shape_fn(self.sigma.len(), |i| self.sigma[i]);

        if let Some(i) = self.water_sigma_t_comp {
            sigma_t[i] = (sigma_t[i] + (temperature.re() * -0.01775).exp() * 10.11
                - (temperature.re() * -0.01146).exp() * 1.417)
                .re();
        }

        sigma_t
    }

    pub fn sigma_ij_t<D: DualNum<f64>>(&self, temperature: D) -> Array2<f64> {
        let diameter = self.sigma_t(temperature);
        let n = diameter.len();

        let mut sigma_ij_t = Array::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                sigma_ij_t[[i, j]] = (diameter[i] + diameter[j]) * 0.5;
            }
        }
        sigma_ij_t
    }
}

impl Parameter for ElectrolytePcSaftParameters {
    type Pure = ElectrolytePcSaftRecord;
    type Binary = ElectrolytePcSaftBinaryRecord;

    fn from_records(
        pure_records: Vec<PureRecord<Self::Pure>>,
        binary_records: Option<Array2<Self::Binary>>,
    ) -> FeosResult<Self> {
        let n = pure_records.len();

        let mut molarweight = Array::zeros(n);
        let mut m = Array::zeros(n);
        let mut sigma = Array::zeros(n);
        let mut epsilon_k = Array::zeros(n);
        let mut z = Array::zeros(n);
        let mut association_records = Vec::with_capacity(n);
        let mut water_sigma_t_comp = None;

        let mut component_index = HashMap::with_capacity(n);

        for (i, record) in pure_records.iter().enumerate() {
            component_index.insert(record.identifier.clone(), i);
            let r = &record.model_record;
            m[i] = r.m;
            sigma[i] = r.sigma;
            epsilon_k[i] = r.epsilon_k;
            z[i] = r.z.unwrap_or(0.0);
            association_records.push(r.association_record.into_iter().collect());
            molarweight[i] = record.molarweight;
            // check if component i is water with temperature-dependent sigma
            if (m[i] * 1000.0).round() / 1000.0 == 1.205 && epsilon_k[i].round() == 354.0 {
                if let Some(record) = r.association_record {
                    if (record.parameters.kappa_ab * 1000.0).round() / 1000.0 == 0.045
                        && record.parameters.epsilon_k_ab.round() == 2426.0
                    {
                        water_sigma_t_comp = Some(i);
                    }
                }
            }
        }

        let binary_association: Vec<_> = binary_records
            .iter()
            .flat_map(|r| {
                r.indexed_iter()
                    .filter_map(|((i, j), record)| record.association.map(|r| ([i, j], r)))
            })
            .collect();
        let association =
            AssociationParameters::new(&association_records, &binary_association, None);

        let ionic_comp: Array1<usize> = z
            .iter()
            .enumerate()
            .filter_map(|(i, &zi)| (zi.abs() > 0.0).then_some(i))
            .collect();

        let nionic = ionic_comp.len();

        let solvent_comp: Array1<usize> = z
            .iter()
            .enumerate()
            .filter_map(|(i, &zi)| (zi.abs() == 0.0).then_some(i))
            .collect();
        let nsolvent = solvent_comp.len();

        let mut k_ij: Array2<Vec<f64>> = Array2::from_elem((n, n), vec![0., 0., 0., 0.]);

        if let Some(binary_records) = binary_records.as_ref() {
            for i in 0..n {
                for j in 0..n {
                    let temp_kij = binary_records[[i, j]].k_ij.clone();
                    if temp_kij.len() > 4 {
                        return Err(FeosError::IncompatibleParameters(format!(
                            "Binary interaction for component {} with {} is parametrized with more than 4 k_ij coefficients.",
                            i, j
                        )));
                    } else {
                        (0..temp_kij.len()).for_each(|k| {
                            k_ij[[i, j]][k] = temp_kij[k];
                        });
                    }
                }
            }

            // No binary interaction between charged species of same kind (+/+ and -/-)
            ionic_comp.iter().for_each(|ai| {
                k_ij[[*ai, *ai]][0] = 1.0;
                for k in 1..4usize {
                    k_ij[[*ai, *ai]][k] = 0.0;
                }
            });
        }

        let mut sigma_ij = Array::zeros((n, n));
        let mut e_k_ij = Array::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                e_k_ij[[i, j]] = (epsilon_k[i] * epsilon_k[j]).sqrt();
                sigma_ij[[i, j]] = 0.5 * (sigma[i] + sigma[j]);
            }
        }

        // Permittivity records
        let mut permittivity_records: Array1<Option<PermittivityRecord>> = pure_records
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
            .filter(|&record| (record.is_some()))
            .for_each(|record| match record.as_ref().unwrap() {
                PermittivityRecord::PerturbationTheory { .. } => {
                    modeltypes.push(1);
                }
                PermittivityRecord::ExperimentalData { .. } => {
                    modeltypes.push(2);
                }
            });

        // check if modeltypes contains a mix of 1 and 2
        if modeltypes.iter().any(|&x| x == 1) && modeltypes.iter().any(|&x| x == 2) {
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
                .filter(|record| (record.is_some()))
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
            molarweight,
            m,
            sigma,
            epsilon_k,
            association: Arc::new(association),
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
            pure_records,
            binary_records,
        })
    }

    fn records(
        &self,
    ) -> (
        &[PureRecord<ElectrolytePcSaftRecord>],
        Option<&Array2<ElectrolytePcSaftBinaryRecord>>,
    ) {
        (&self.pure_records, self.binary_records.as_ref())
    }
}

impl HardSphereProperties for ElectrolytePcSaftParameters {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<N> {
        MonomerShape::NonSpherical(self.m.mapv(N::from))
    }

    fn hs_diameter<D: DualNum<f64>>(&self, temperature: D) -> Array1<D> {
        let sigma_t = self.sigma_t(temperature.clone());

        let ti = temperature.recip() * -3.0;
        let mut d = Array::from_shape_fn(sigma_t.len(), |i| {
            -((ti.clone() * self.epsilon_k[i]).exp() * 0.12 - 1.0) * sigma_t[i]
        });
        for i in 0..self.nionic {
            let ai = self.ionic_comp[i];
            d[ai] = D::one() * sigma_t[ai] * (1.0 - 0.12);
        }
        d
    }
}

impl AssociationStrength for ElectrolytePcSaftParameters {
    type Record = ElectrolytePcSaftAssociationRecord;
    type BinaryRecord = ElectrolytePcSaftBinaryAssociationRecord;

    fn association_strength<D: DualNum<f64> + Copy>(
        &self,
        temperature: D,
        comp_i: usize,
        comp_j: usize,
        assoc_ij: Self::Record,
    ) -> D {
        let sigma_t = self.sigma_t(temperature);
        let si = sigma_t[comp_i];
        let sj = sigma_t[comp_j];
        (temperature.recip() * assoc_ij.epsilon_k_ab).exp_m1()
            * assoc_ij.kappa_ab
            * (si * sj).powf(1.5)
    }

    fn combining_rule(parameters_i: Self::Record, parameters_j: Self::Record) -> Self::Record {
        Self::Record {
            kappa_ab: (parameters_i.kappa_ab * parameters_j.kappa_ab).sqrt(),
            epsilon_k_ab: 0.5 * (parameters_i.epsilon_k_ab + parameters_j.epsilon_k_ab),
        }
    }

    fn update_binary(parameters_ij: &mut Self::Record, binary_parameters: Self::BinaryRecord) {
        if let Some(kappa_ab) = binary_parameters.kappa_ab {
            parameters_ij.kappa_ab = kappa_ab
        }
        if let Some(epsilon_k_ab) = binary_parameters.epsilon_k_ab {
            parameters_ij.epsilon_k_ab = epsilon_k_ab
        }
    }
}

impl ElectrolytePcSaftParameters {
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();
        let o = &mut output;
        write!(
            o,
            "|component|molarweight|$m$|$\\sigma$|$\\varepsilon$|$z$|$\\kappa_{{AB}}$|$\\varepsilon_{{AB}}$|$N_A$|$N_B$|\n|-|-|-|-|-|-|-|-|-|-|-|-|"
        )
        .unwrap();
        for (i, record) in self.pure_records.iter().enumerate() {
            let component = record.identifier.name.clone();
            let component = component.unwrap_or(format!("Component {}", i + 1));
            let association = record.model_record.association_record.unwrap_or_else(|| {
                AssociationRecord::new(
                    ElectrolytePcSaftAssociationRecord::new(0.0, 0.0),
                    0.0,
                    0.0,
                    0.0,
                )
            });
            write!(
                o,
                "\n|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|",
                component,
                record.molarweight,
                record.model_record.m,
                record.model_record.sigma,
                record.model_record.epsilon_k,
                record.model_record.z.unwrap_or(0.0),
                association.parameters.kappa_ab,
                association.parameters.epsilon_k_ab,
                association.na,
                association.nb,
                association.nc
            )
            .unwrap();
        }

        output
    }
}

#[cfg(test)]
pub mod utils {
    use feos_core::parameter::BinaryRecord;

    use super::*;
    use std::sync::Arc;

    pub fn propane_parameters() -> Arc<ElectrolytePcSaftParameters> {
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
                    "epsilon_k": 208.1101
                },
                "molarweight": 44.0962
            }"#;
        let propane_record: PureRecord<ElectrolytePcSaftRecord> =
            serde_json::from_str(propane_json).expect("Unable to parse json.");
        Arc::new(ElectrolytePcSaftParameters::new_pure(propane_record).unwrap())
    }

    pub fn butane_parameters() -> Arc<ElectrolytePcSaftParameters> {
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
        let butane_record: PureRecord<ElectrolytePcSaftRecord> =
            serde_json::from_str(butane_json).expect("Unable to parse json.");
        Arc::new(ElectrolytePcSaftParameters::new_pure(butane_record).unwrap())
    }

    pub fn water_nacl_parameters_perturb() -> Arc<ElectrolytePcSaftParameters> {
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
                "model_record": {
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
                "model_record": {
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
                "model_record": {
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
                "model_record": {
                    "k_ij": [
                        0.0045,
                        0.0,
                        0.0,
                        0.0
                    ]
                }
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
                "model_record": {
                    "k_ij": [
                        -0.25,
                        0.0,
                        0.0,
                        0.0
                    ]
                }
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
                "model_record": {
                    "k_ij": [
                        0.317,
                        0.0,
                        0.0,
                        0.0
                    ]
                }
            }
            ]"#;
        let pure_records: Vec<PureRecord<ElectrolytePcSaftRecord>> =
            serde_json::from_str(pure_json).expect("Unable to parse json.");
        let binary_records: Vec<BinaryRecord<ElectrolytePcSaftBinaryRecord>> =
            serde_json::from_str(binary_json).expect("Unable to parse json.");
        let binary_matrix = ElectrolytePcSaftParameters::binary_matrix_from_records(
            &pure_records,
            &binary_records,
            feos_core::parameter::IdentifierOption::Name,
        )
        .unwrap();
        Arc::new(
            ElectrolytePcSaftParameters::from_records(pure_records, Some(binary_matrix)).unwrap(),
        )
    }

    pub fn water_nacl_parameters() -> Arc<ElectrolytePcSaftParameters> {
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
                "model_record": {
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
                "model_record": {
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
                "model_record": {
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
                "model_record": {
                    "k_ij": [
                        0.0045,
                        0.0,
                        0.0,
                        0.0
                    ]
                }
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
                "model_record": {
                    "k_ij": [
                        -0.25,
                        0.0,
                        0.0,
                        0.0
                    ]
                }
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
                "model_record": {
                    "k_ij": [
                        0.317,
                        0.0,
                        0.0,
                        0.0
                    ]
                }
            }
            ]"#;
        let pure_records: Vec<PureRecord<ElectrolytePcSaftRecord>> =
            serde_json::from_str(pure_json).expect("Unable to parse json.");
        let binary_records: Vec<BinaryRecord<ElectrolytePcSaftBinaryRecord>> =
            serde_json::from_str(binary_json).expect("Unable to parse json.");
        let binary_matrix = ElectrolytePcSaftParameters::binary_matrix_from_records(
            &pure_records,
            &binary_records,
            feos_core::parameter::IdentifierOption::Name,
        )
        .unwrap();
        Arc::new(
            ElectrolytePcSaftParameters::from_records(pure_records, Some(binary_matrix)).unwrap(),
        )
    }

    pub fn water_parameters() -> Arc<ElectrolytePcSaftParameters> {
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
                    "epsilon_k_ab": 2500.6706,
                    "na": 1.0,
                    "nb": 1.0
                },
                "molarweight": 18.0152
            }"#;
        let water_record: PureRecord<ElectrolytePcSaftRecord> =
            serde_json::from_str(water_json).expect("Unable to parse json.");
        Arc::new(ElectrolytePcSaftParameters::new_pure(water_record).unwrap())
    }

    pub fn propane_butane_parameters() -> Arc<ElectrolytePcSaftParameters> {
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
                    "epsilon_k": 208.1101
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
                    "epsilon_k": 222.8774
                },
                "molarweight": 58.123
            }
        ]"#;
        let binary_record: Vec<PureRecord<ElectrolytePcSaftRecord>> =
            serde_json::from_str(binary_json).expect("Unable to parse json.");
        Arc::new(ElectrolytePcSaftParameters::new_binary(binary_record, None).unwrap())
    }
}
