use crate::association::AssociationParameters;
use crate::gc_pcsaft::record::GcPcSaftRecord;
use crate::hard_sphere::{HardSphereProperties, MonomerShape};
use feos_core::parameter::{
    BinaryRecord, ChemicalRecord, Identifier, ParameterError, ParameterHetero, SegmentCount,
    SegmentRecord,
};
use feos_core::si::{JOULE, KB, KELVIN};
use indexmap::IndexMap;
use ndarray::{Array1, Array2};
use num_dual::DualNum;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Write;

#[derive(Clone)]
pub struct GcPcSaftChemicalRecord {
    pub identifier: Identifier,
    pub segments: HashMap<String, f64>,
    pub bonds: HashMap<[String; 2], f64>,
    phi: f64,
}

impl GcPcSaftChemicalRecord {
    pub fn new(
        identifier: Identifier,
        segments: HashMap<String, f64>,
        bonds: HashMap<[String; 2], f64>,
        phi: f64,
    ) -> Self {
        Self {
            identifier,
            segments,
            bonds,
            phi,
        }
    }
}

impl SegmentCount for GcPcSaftChemicalRecord {
    type Count = f64;

    fn identifier(&self) -> Cow<Identifier> {
        Cow::Borrowed(&self.identifier)
    }

    fn segment_count(&self) -> Cow<HashMap<String, f64>> {
        Cow::Borrowed(&self.segments)
    }
}

impl From<ChemicalRecord> for GcPcSaftChemicalRecord {
    fn from(chemical_record: ChemicalRecord) -> Self {
        Self::new(
            chemical_record.identifier.clone(),
            chemical_record.segment_count(),
            chemical_record.bond_count(),
            1.0,
        )
    }
}

/// Parameter set required for the gc-PC-SAFT equation of state.
#[derive(Clone)]
pub struct GcPcSaftEosParameters {
    pub molarweight: Array1<f64>,
    pub component_index: Array1<usize>,
    identifiers: Vec<String>,

    pub m: Array1<f64>,
    pub sigma: Array1<f64>,
    pub epsilon_k: Array1<f64>,
    pub bonds: IndexMap<[usize; 2], f64>,

    pub association: AssociationParameters,

    pub dipole_comp: Array1<usize>,
    mu: Array1<f64>,
    pub mu2: Array1<f64>,
    pub m_mix: Array1<f64>,
    pub s_ij: Array2<f64>,
    pub e_k_ij: Array2<f64>,

    pub k_ij: Array2<f64>,
    pub sigma_ij: Array2<f64>,
    pub epsilon_k_ij: Array2<f64>,

    pub chemical_records: Vec<GcPcSaftChemicalRecord>,
    segment_records: Vec<SegmentRecord<GcPcSaftRecord>>,
    binary_segment_records: Option<Vec<BinaryRecord<String, f64>>>,
}

impl ParameterHetero for GcPcSaftEosParameters {
    type Chemical = GcPcSaftChemicalRecord;
    type Pure = GcPcSaftRecord;
    type Binary = f64;

    fn from_segments<C: Clone + Into<GcPcSaftChemicalRecord>>(
        chemical_records: Vec<C>,
        segment_records: Vec<SegmentRecord<GcPcSaftRecord>>,
        binary_segment_records: Option<Vec<BinaryRecord<String, f64>>>,
    ) -> Result<Self, ParameterError> {
        let chemical_records: Vec<_> = chemical_records.into_iter().map(|c| c.into()).collect();

        let mut molarweight = Array1::zeros(chemical_records.len());
        let mut component_index = Vec::new();
        let mut identifiers = Vec::new();
        let mut m = Vec::new();
        let mut sigma = Vec::new();
        let mut epsilon_k = Vec::new();
        let mut bonds = IndexMap::with_capacity(segment_records.len());
        let mut association_records = Vec::new();

        let mut dipole_comp = Vec::new();
        let mut mu = Vec::new();
        let mut mu2 = Vec::new();
        let mut m_mix = Vec::new();
        let mut sigma_mix = Vec::new();
        let mut epsilon_k_mix = Vec::new();

        let mut phi = Vec::new();

        for (i, chemical_record) in chemical_records.iter().cloned().enumerate() {
            let mut segment_indices = IndexMap::with_capacity(segment_records.len());
            let segment_map = chemical_record.segment_map(&segment_records)?;
            phi.push(chemical_record.phi);

            let mut m_i = 0.0;
            let mut sigma_i = 0.0;
            let mut epsilon_k_i = 0.0;
            let mut mu2_i = 0.0;

            for (segment, &count) in segment_map.iter() {
                segment_indices.insert(segment.identifier.clone(), m.len());

                molarweight[i] += segment.molarweight * count;

                component_index.push(i);
                identifiers.push(segment.identifier.clone());
                m.push(segment.model_record.m * count);
                sigma.push(segment.model_record.sigma);
                epsilon_k.push(segment.model_record.epsilon_k);

                let mut assoc = segment.model_record.association_record;
                if let Some(assoc) = assoc.as_mut() {
                    assoc.na *= count;
                    assoc.nb *= count;
                };
                association_records.push(assoc.into_iter().collect());

                m_i += segment.model_record.m * count;
                sigma_i += segment.model_record.m * segment.model_record.sigma.powi(3) * count;
                epsilon_k_i += segment.model_record.m * segment.model_record.epsilon_k * count;
                if let Some(mu) = segment.model_record.mu {
                    mu2_i += mu.powi(2) * count;
                }
            }

            if mu2_i > 0.0 {
                dipole_comp.push(i);
                mu.push(mu2_i.sqrt());
                mu2.push(mu2_i / m_i * (1e-19 * (JOULE / KELVIN / KB).into_value()));
                m_mix.push(m_i);
                sigma_mix.push((sigma_i / m_i).cbrt());
                epsilon_k_mix.push(epsilon_k_i / m_i);
            }

            for (b, &count) in chemical_record.bonds.iter() {
                let i1 = segment_indices.get(&b[0]);
                let i2 = segment_indices.get(&b[1]);
                if let (Some(&i1), Some(&i2)) = (i1, i2) {
                    let indices = if i1 > i2 { [i2, i1] } else { [i1, i2] };
                    let bond = bonds.entry(indices).or_insert(0.0);
                    *bond += count;
                }
            }
        }

        // Binary interaction parameter
        let mut k_ij = Array2::zeros([epsilon_k.len(); 2]);
        if let Some(binary_segment_records) = binary_segment_records.as_ref() {
            let mut binary_segment_records_map = IndexMap::new();
            for binary_record in binary_segment_records {
                binary_segment_records_map.insert(
                    (binary_record.id1.clone(), binary_record.id2.clone()),
                    binary_record.model_record,
                );
                binary_segment_records_map.insert(
                    (binary_record.id2.clone(), binary_record.id1.clone()),
                    binary_record.model_record,
                );
            }
            for (i, id1) in identifiers.iter().enumerate() {
                for (j, id2) in identifiers.iter().cloned().enumerate() {
                    if component_index[i] != component_index[j] {
                        if let Some(k) = binary_segment_records_map.get(&(id1.clone(), id2)) {
                            k_ij[(i, j)] = *k;
                        }
                    }
                }
            }
        }

        // Combining rules dispersion
        let sigma_ij =
            Array2::from_shape_fn([sigma.len(); 2], |(i, j)| 0.5 * (sigma[i] + sigma[j]));
        let epsilon_k_ij = Array2::from_shape_fn([epsilon_k.len(); 2], |(i, j)| {
            (epsilon_k[i] * phi[component_index[i]] * epsilon_k[j] * phi[component_index[j]]).sqrt()
        }) * (1.0 - &k_ij);

        // Combining rules polar
        let s_ij = Array2::from_shape_fn([dipole_comp.len(); 2], |(i, j)| {
            0.5 * (sigma_mix[i] + sigma_mix[j])
        });
        let e_k_ij = Array2::from_shape_fn([dipole_comp.len(); 2], |(i, j)| {
            (epsilon_k_mix[i] * epsilon_k_mix[j]).sqrt()
        });

        // Association
        let sigma = Array1::from_vec(sigma);
        let component_index = Array1::from_vec(component_index);
        let association =
            AssociationParameters::new(&association_records, &sigma, &[], Some(&component_index));

        Ok(Self {
            molarweight,
            component_index,
            identifiers,
            m: Array1::from_vec(m),
            sigma,
            epsilon_k: Array1::from_vec(epsilon_k),
            bonds,
            association,
            dipole_comp: Array1::from_vec(dipole_comp),
            mu: Array1::from_vec(mu),
            mu2: Array1::from_vec(mu2),
            m_mix: Array1::from_vec(m_mix),
            s_ij,
            e_k_ij,
            k_ij,
            sigma_ij,
            epsilon_k_ij,
            chemical_records,
            segment_records,
            binary_segment_records,
        })
    }

    fn records(
        &self,
    ) -> (
        &[Self::Chemical],
        &[SegmentRecord<Self::Pure>],
        &Option<Vec<BinaryRecord<String, Self::Binary>>>,
    ) {
        (
            &self.chemical_records,
            &self.segment_records,
            &self.binary_segment_records,
        )
    }
}

impl GcPcSaftEosParameters {
    pub fn phi(self, phi: &[f64]) -> Result<Self, ParameterError> {
        let mut cr = self.chemical_records;
        cr.iter_mut().zip(phi.iter()).for_each(|(c, &p)| c.phi = p);
        Self::from_segments(cr, self.segment_records, self.binary_segment_records)
    }
}

impl HardSphereProperties for GcPcSaftEosParameters {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<N> {
        let m = self.m.mapv(N::from);
        MonomerShape::Heterosegmented([m.clone(), m.clone(), m.clone(), m], &self.component_index)
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D> {
        let ti = temperature.recip() * -3.0;
        Array1::from_shape_fn(self.sigma.len(), |i| {
            -((ti * self.epsilon_k[i]).exp() * 0.12 - 1.0) * self.sigma[i]
        })
    }
}

impl GcPcSaftEosParameters {
    pub fn to_markdown(&self) -> String {
        let gorup_dict: HashMap<&String, &GcPcSaftRecord> = self
            .segment_records
            .iter()
            .map(|r| (&r.identifier, &r.model_record))
            .collect();

        let mut output = String::new();
        let o = &mut output;
        write!(
            o,
            "|component|molarweight|dipole moment|group|$m$|$\\sigma$|$\\varepsilon$|$\\kappa_{{AB}}$|$\\varepsilon_{{AB}}$|$N_A$|$N_B$|$N_C$|\n|-|-|-|-|-|-|-|-|-|-|-|-|"
        )
        .unwrap();
        for i in 0..self.m.len() {
            let component = if i > 0 && self.component_index[i] == self.component_index[i - 1] {
                "||".to_string()
            } else {
                let pure = &self.chemical_records[self.component_index[i]].identifier;
                format!(
                    "{}|{}|{}",
                    pure.name
                        .as_ref()
                        .unwrap_or(&format!("Component {}", self.component_index[i] + 1)),
                    self.molarweight[self.component_index[i]],
                    if let Some(d) = self
                        .dipole_comp
                        .iter()
                        .position(|&d| d == self.component_index[i])
                    {
                        format!("{}", self.mu[d])
                    } else {
                        "".into()
                    }
                )
            };
            let record = gorup_dict[&self.identifiers[i]];
            let association = if let Some(a) = record.association_record {
                format!(
                    "{}|{}|{}|{}|{}",
                    a.kappa_ab, a.epsilon_k_ab, a.na, a.nb, a.nc
                )
            } else {
                "||||".to_string()
            };
            write!(
                o,
                "\n|{}|{}|{}|{}|{}|{}|",
                component,
                self.identifiers[i],
                record.m,
                record.sigma,
                record.epsilon_k,
                association
            )
            .unwrap();
        }

        write!(o, "\n\n|component|group 1|group 2|bonds|\n|-|-|-|-|").unwrap();

        let mut last_component = None;
        for ([c1, c2], &c) in &self.bonds {
            let next = self.component_index[*c1];
            let pure = &self.chemical_records[next].identifier;
            let component = if let Some(last) = last_component {
                if last == next {
                    "".into()
                } else {
                    last_component = Some(next);
                    pure.name
                        .clone()
                        .unwrap_or(format!("Component {}", next + 1))
                }
            } else {
                last_component = Some(next);
                pure.name
                    .clone()
                    .unwrap_or(format!("Component {}", next + 1))
            };
            write!(
                o,
                "\n|{}|{}|{}|{}|",
                component, self.identifiers[*c1], self.identifiers[*c2], c
            )
            .unwrap();
        }

        output
    }
}

impl std::fmt::Display for GcPcSaftEosParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GcPcSaftParameters(")?;
        write!(f, "\n\tmolarweight={}", self.molarweight)?;
        write!(f, "\n\tcomponent_index={}", self.component_index)?;
        write!(f, "\n\tm={}", self.m)?;
        write!(f, "\n\tsigma={}", self.sigma)?;
        write!(f, "\n\tepsilon_k={}", self.epsilon_k)?;
        write!(f, "\n\tbonds={:?}", self.bonds)?;
        // if !self.assoc_segment.is_empty() {
        //     write!(f, "\n\tassoc_segment={}", self.assoc_segment)?;
        //     write!(f, "\n\tkappa_ab={}", self.kappa_ab)?;
        //     write!(f, "\n\tepsilon_k_ab={}", self.epsilon_k_ab)?;
        //     write!(f, "\n\tna={}", self.na)?;
        //     write!(f, "\n\tnb={}", self.nb)?;
        // }
        if !self.dipole_comp.is_empty() {
            write!(f, "\n\tdipole_comp={}", self.dipole_comp)?;
            write!(f, "\n\tmu={}", self.mu)?;
        }
        write!(f, "\n)")
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::association::AssociationRecord;
    use feos_core::parameter::{ChemicalRecord, Identifier};

    fn ch3() -> SegmentRecord<GcPcSaftRecord> {
        SegmentRecord::new(
            "CH3".into(),
            15.0,
            GcPcSaftRecord::new(0.77247, 3.6937, 181.49, None, None, None),
        )
    }

    fn ch2() -> SegmentRecord<GcPcSaftRecord> {
        SegmentRecord::new(
            "CH2".into(),
            14.0,
            GcPcSaftRecord::new(0.7912, 3.0207, 157.23, None, None, None),
        )
    }

    fn oh() -> SegmentRecord<GcPcSaftRecord> {
        SegmentRecord::new(
            "OH".into(),
            0.0,
            GcPcSaftRecord::new(
                1.0231,
                2.7702,
                334.29,
                None,
                Some(AssociationRecord::new(0.009583, 2575.9, 1.0, 1.0, 0.0)),
                None,
            ),
        )
    }

    pub fn ch3_oh() -> BinaryRecord<String, f64> {
        BinaryRecord::new("CH3".to_string(), "OH".to_string(), -0.0087)
    }

    pub fn propane() -> GcPcSaftEosParameters {
        let pure = ChemicalRecord::new(
            Identifier::new(Some("74-98-6"), Some("propane"), None, None, None, None),
            vec!["CH3".into(), "CH2".into(), "CH3".into()],
            None,
        );
        GcPcSaftEosParameters::from_segments(vec![pure], vec![ch3(), ch2()], None).unwrap()
    }

    pub fn propanol() -> GcPcSaftEosParameters {
        let pure = ChemicalRecord::new(
            Identifier::new(Some("71-23-8"), Some("1-propanol"), None, None, None, None),
            vec!["CH3".into(), "CH2".into(), "CH2".into(), "OH".into()],
            None,
        );
        GcPcSaftEosParameters::from_segments(vec![pure], vec![ch3(), ch2(), oh()], None).unwrap()
    }

    pub fn ethanol_propanol(binary: bool) -> GcPcSaftEosParameters {
        let ethanol = ChemicalRecord::new(
            Identifier::new(Some("64-17-5"), Some("ethanol"), None, None, None, None),
            vec!["CH3".into(), "CH2".into(), "OH".into()],
            None,
        );
        let propanol = ChemicalRecord::new(
            Identifier::new(Some("71-23-8"), Some("1-propanol"), None, None, None, None),
            vec!["CH3".into(), "CH2".into(), "CH2".into(), "OH".into()],
            None,
        );
        let binary = if binary { Some(vec![ch3_oh()]) } else { None };
        GcPcSaftEosParameters::from_segments(
            vec![ethanol, propanol],
            vec![ch3(), ch2(), oh()],
            binary,
        )
        .unwrap()
    }

    #[test]
    fn test_kij() {
        let params = ethanol_propanol(true);
        let identifiers: Vec<_> = params.identifiers.iter().enumerate().collect();
        let ch3 = identifiers.iter().find(|&&(_, id)| id == "CH3").unwrap();
        let ch2 = identifiers
            .iter()
            .skip(3)
            .find(|&&(_, id)| id == "CH2")
            .unwrap();
        let oh = identifiers
            .iter()
            .skip(3)
            .find(|&&(_, id)| id == "OH")
            .unwrap();
        println!("{:?}", params.identifiers);
        println!("{}", params.k_ij);
        // CH3 - CH2
        assert_eq!(
            params.epsilon_k_ij[(ch3.0, ch2.0)],
            (181.49f64 * 157.23).sqrt()
        );
        // CH3 - OH
        assert_eq!(
            params.epsilon_k_ij[(ch3.0, oh.0)],
            (181.49f64 * 334.29).sqrt() * 1.0087
        );
    }
}
