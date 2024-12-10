use crate::association::{
    AssociationParameters, AssociationRecord, AssociationStrength, BinaryAssociationRecord,
};
use crate::hard_sphere::{HardSphereProperties, MonomerShape};
use feos_core::parameter::{Parameter, ParameterError, PureRecord};
use ndarray::{Array, Array1, Array2};
use num_dual::DualNum;
use num_traits::Zero;
use quantity::{JOULE, KB, KELVIN};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Arc;

/// PC-SAFT pure-component parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct FcSaftHomoRecord {
    /// Segment number
    pub s: usize,
    /// Bond length
    pub l: f64,
    /// Segment diameter in units of Angstrom
    pub sigma: f64,
    /// Energetic parameter in units of Kelvin
    pub epsilon_k: f64,
    /// Dipole moment in units of Debye
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mu: Option<f64>,
    /// Association parameters
    #[serde(flatten)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub association_record: Option<AssociationRecord<FcSaftAssociationRecord>>,
}

impl std::fmt::Display for FcSaftHomoRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FcSaftHomoRecord(s={}", self.s)?;
        write!(f, ", l={}", self.l)?;
        write!(f, ", sigma={}", self.sigma)?;
        write!(f, ", epsilon_k={}", self.epsilon_k)?;
        if let Some(n) = &self.mu {
            write!(f, ", mu={}", n)?;
        }
        if let Some(n) = &self.association_record {
            write!(f, ", association_record={}", n)?;
        }
        write!(f, ")")
    }
}

impl FcSaftHomoRecord {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        s: usize,
        l: f64,
        sigma: f64,
        epsilon_k: f64,
        mu: Option<f64>,
        kappa_ab: Option<f64>,
        epsilon_k_ab: Option<f64>,
        na: Option<f64>,
        nb: Option<f64>,
        nc: Option<f64>,
    ) -> FcSaftHomoRecord {
        let association_record =
            if let (Some(kappa_ab), Some(epsilon_k_ab)) = (kappa_ab, epsilon_k_ab) {
                Some(AssociationRecord::new(
                    FcSaftAssociationRecord::new(kappa_ab, epsilon_k_ab),
                    na.unwrap_or_default(),
                    nb.unwrap_or_default(),
                    nc.unwrap_or_default(),
                ))
            } else {
                None
            };
        FcSaftHomoRecord {
            s,
            l,
            sigma,
            epsilon_k,
            mu,
            association_record,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct FcSaftAssociationRecord {
    /// Association volume parameter
    pub kappa_ab: f64,
    /// Association energy parameter in units of Kelvin
    pub epsilon_k_ab: f64,
}

impl FcSaftAssociationRecord {
    pub fn new(kappa_ab: f64, epsilon_k_ab: f64) -> Self {
        Self {
            kappa_ab,
            epsilon_k_ab,
        }
    }
}

impl std::fmt::Display for FcSaftAssociationRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FcSaftAssociationRecord(kappa_ab={}", self.kappa_ab)?;
        write!(f, ", epsilon_k_ab={})", self.epsilon_k_ab)
    }
}

/// PC-SAFT binary interaction parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct FcSaftHomoBinaryRecord {
    /// Binary dispersion interaction parameter
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub k_ij: f64,
    /// Binary association parameters
    #[serde(flatten)]
    association: Option<BinaryAssociationRecord<FcSaftBinaryAssociationRecord>>,
}

impl From<f64> for FcSaftHomoBinaryRecord {
    fn from(k_ij: f64) -> Self {
        Self {
            k_ij,
            association: None,
        }
    }
}

impl From<FcSaftHomoBinaryRecord> for f64 {
    fn from(binary_record: FcSaftHomoBinaryRecord) -> Self {
        binary_record.k_ij
    }
}

impl FcSaftHomoBinaryRecord {
    pub fn new(k_ij: Option<f64>, kappa_ab: Option<f64>, epsilon_k_ab: Option<f64>) -> Self {
        let k_ij = k_ij.unwrap_or_default();
        let association = if kappa_ab.is_none() && epsilon_k_ab.is_none() {
            None
        } else {
            Some(BinaryAssociationRecord::new(
                FcSaftBinaryAssociationRecord::new(kappa_ab, epsilon_k_ab),
                None,
            ))
        };
        Self { k_ij, association }
    }
}

impl std::fmt::Display for FcSaftHomoBinaryRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut tokens = vec![];
        if !self.k_ij.is_zero() {
            tokens.push(format!("k_ij={}", self.k_ij));
        }
        if let Some(association) = self.association {
            if let Some(kappa_ab) = association.parameters.kappa_ab {
                tokens.push(format!("kappa_ab={}", kappa_ab));
            }
            if let Some(epsilon_k_ab) = association.parameters.epsilon_k_ab {
                tokens.push(format!("epsilon_k_ab={}", epsilon_k_ab));
            }
        }
        write!(f, "FcSaftHomoBinaryRecord({})", tokens.join(", "))
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct FcSaftBinaryAssociationRecord {
    /// Cross-association association volume parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kappa_ab: Option<f64>,
    /// Cross-association energy parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epsilon_k_ab: Option<f64>,
}

impl FcSaftBinaryAssociationRecord {
    pub fn new(kappa_ab: Option<f64>, epsilon_k_ab: Option<f64>) -> Self {
        Self {
            kappa_ab,
            epsilon_k_ab,
        }
    }
}

/// Parameter set required for the PC-SAFT equation of state and Helmholtz energy functional.
pub struct FcSaftHomoParameters {
    pub molarweight: Array1<f64>,
    pub s: Array1<f64>,
    pub l: Array1<f64>,
    pub sigma: Array1<f64>,
    pub epsilon_k: Array1<f64>,
    pub mu: Array1<f64>,
    pub mu2: Array1<f64>,
    pub association: Arc<AssociationParameters<Self>>,
    pub sigma_ij: Array2<f64>,
    pub epsilon_k_ij: Array2<f64>,
    pub e_k_ij: Array2<f64>,
    pub ndipole: usize,
    pub dipole_comp: Array1<usize>,
    pub pure_records: Vec<PureRecord<FcSaftHomoRecord>>,
    pub binary_records: Option<Array2<FcSaftHomoBinaryRecord>>,
}

impl Parameter for FcSaftHomoParameters {
    type Pure = FcSaftHomoRecord;
    type Binary = FcSaftHomoBinaryRecord;

    fn from_records(
        pure_records: Vec<PureRecord<Self::Pure>>,
        binary_records: Option<Array2<Self::Binary>>,
    ) -> Result<Self, ParameterError> {
        let n = pure_records.len();

        let mut molarweight = Array::zeros(n);
        let mut s = Array::zeros(n);
        let mut l = Array::zeros(n);
        let mut sigma = Array::zeros(n);
        let mut epsilon_k = Array::zeros(n);
        let mut mu = Array::zeros(n);
        let mut association_records = Vec::with_capacity(n);

        let mut component_index = HashMap::with_capacity(n);

        for (i, record) in pure_records.iter().enumerate() {
            component_index.insert(record.identifier.clone(), i);
            let r = &record.model_record;
            s[i] = r.s as f64;
            l[i] = r.l;
            sigma[i] = r.sigma;
            epsilon_k[i] = r.epsilon_k;
            mu[i] = r.mu.unwrap_or(0.0);
            association_records.push(r.association_record.into_iter().collect());
            molarweight[i] = record.molarweight;
        }

        let ld = &l / &sigma;
        let m_star = 1.0 + 0.5 * &ld * (3.0 - &ld * &ld) * (&s - 1.0);
        let mu2 = &mu * &mu / (m_star * &sigma * &sigma * &sigma * &epsilon_k)
            * 1e-19
            * (JOULE / KELVIN / KB).into_value();
        let dipole_comp: Array1<usize> = mu2
            .iter()
            .enumerate()
            .filter_map(|(i, &mu2)| (mu2.abs() > 0.0).then_some(i))
            .collect();
        let ndipole = dipole_comp.len();

        let binary_association: Vec<_> = binary_records
            .iter()
            .flat_map(|r| {
                r.indexed_iter()
                    .filter_map(|((i, j), record)| record.association.map(|r| ([i, j], r)))
            })
            .collect();
        let association =
            AssociationParameters::new(&association_records, &binary_association, None);

        let k_ij = binary_records.as_ref().map(|br| br.map(|br| br.k_ij));
        let mut sigma_ij = Array::zeros((n, n));
        let mut e_k_ij = Array::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                e_k_ij[[i, j]] = (epsilon_k[i] * epsilon_k[j]).sqrt();
                sigma_ij[[i, j]] = 0.5 * (sigma[i] + sigma[j]);
            }
        }
        let mut epsilon_k_ij = e_k_ij.clone();
        if let Some(k_ij) = k_ij.as_ref() {
            epsilon_k_ij *= &(1.0 - k_ij)
        };

        Ok(Self {
            molarweight,
            s,
            l,
            sigma,
            epsilon_k,
            mu,
            mu2,
            association: Arc::new(association),
            sigma_ij,
            epsilon_k_ij,
            e_k_ij,
            ndipole,
            dipole_comp,
            pure_records,
            binary_records,
        })
    }

    fn records(
        &self,
    ) -> (
        &[PureRecord<FcSaftHomoRecord>],
        Option<&Array2<FcSaftHomoBinaryRecord>>,
    ) {
        (&self.pure_records, self.binary_records.as_ref())
    }
}

impl FcSaftHomoParameters {
    pub fn m_values<D: DualNum<f64> + Copy>(&self, diameter: &Array1<D>) -> [Array1<D>; 2] {
        let ld = Array1::from_shape_fn(diameter.len(), |i| diameter[i].recip() * self.l[i]);
        let m = ld.clone() * (&self.s - 1.0) + 1.0;
        let m_star = (-(&ld * &ld) + 3.0) * ld * 0.5 * (&self.s - 1.0) + 1.0;
        [m, m_star]
    }
}

impl HardSphereProperties for FcSaftHomoParameters {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<N> {
        unreachable!()
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D> {
        let ti = temperature.recip() * -3.0;
        Array::from_shape_fn(self.sigma.len(), |i| {
            -((ti * self.epsilon_k[i]).exp() * 0.12 - 1.0) * self.sigma[i]
        })
    }

    fn component_index(&self) -> Cow<Array1<usize>> {
        Cow::Owned(Array1::from_shape_fn(self.s.len(), |i| i))
    }

    fn geometry_coefficients<D: DualNum<f64> + Copy>(&self, temperature: D) -> [Array1<D>; 4] {
        let diameter = self.hs_diameter(temperature);
        let [m, m_star] = self.m_values(&diameter);
        [self.s.mapv(D::from), m.clone(), m, m_star]
    }
}

impl AssociationStrength for FcSaftHomoParameters {
    type Record = FcSaftAssociationRecord;
    type BinaryRecord = FcSaftBinaryAssociationRecord;

    fn association_strength<D: DualNum<f64> + Copy>(
        &self,
        temperature: D,
        comp_i: usize,
        comp_j: usize,
        assoc_ij: Self::Record,
    ) -> D {
        let si = self.sigma[comp_i];
        let sj = self.sigma[comp_j];
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

impl FcSaftHomoParameters {
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();
        let o = &mut output;
        write!(
            o,
            "|component|molarweight|$s$|$l$|$\\sigma$|$\\varepsilon$|$\\mu$|$\\kappa_{{AB}}$|$\\varepsilon_{{AB}}$|$N_A$|$N_B$|$N_C$|\n|-|-|-|-|-|-|-|-|-|-|-|-|"
        )
        .unwrap();
        for (i, record) in self.pure_records.iter().enumerate() {
            let component = record.identifier.name.clone();
            let component = component.unwrap_or(format!("Component {}", i + 1));
            let association = record.model_record.association_record.unwrap_or_else(|| {
                AssociationRecord::new(FcSaftAssociationRecord::new(0.0, 0.0), 0.0, 0.0, 0.0)
            });
            write!(
                o,
                "\n|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|",
                component,
                record.molarweight,
                record.model_record.s,
                record.model_record.l,
                record.model_record.sigma,
                record.model_record.epsilon_k,
                record.model_record.mu.unwrap_or(0.0),
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
