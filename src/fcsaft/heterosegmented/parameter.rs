use crate::association::{AssociationParameters, AssociationRecord, AssociationStrength};
use crate::fcsaft::homosegmented::parameters::FcSaftAssociationRecord;
use feos_core::parameter::{
    BinaryRecord, ChemicalRecord, Identifier, IdentifierOption, ParameterError, SegmentRecord,
};
use indexmap::{IndexMap, IndexSet};
use ndarray::{Array1, Array2};
use num_dual::DualNum;
use petgraph::dot::{Config, Dot};
use petgraph::graph::{Graph, UnGraph};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

const PSI_FUSED_CHAINS: f64 = 1.5;

/// FC-SAFT parameter set.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct FcSaftRecord {
    /// Segment diameter in units of Angstrom
    pub sigma: f64,
    /// Energetic parameter in units of Kelvin
    pub epsilon_k: f64,
    /// association record
    #[serde(skip_serializing_if = "Option::is_none")]
    pub association_record: Option<AssociationRecord<FcSaftAssociationRecord>>,
    /// Interaction range parameter for the dispersion functional
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub psi_dft: Option<f64>,
}

impl FcSaftRecord {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        sigma: f64,
        epsilon_k: f64,
        kappa_ab: Option<f64>,
        epsilon_k_ab: Option<f64>,
        na: Option<f64>,
        nb: Option<f64>,
        nc: Option<f64>,
        psi_dft: Option<f64>,
    ) -> Self {
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
        Self {
            sigma,
            epsilon_k,
            association_record,
            psi_dft,
        }
    }
}

impl std::fmt::Display for FcSaftRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FcSaftRecord(sigma={}, epsilon_k={}",
            self.sigma, self.epsilon_k
        )?;
        if let Some(a) = &self.association_record {
            write!(f, ", association_record={a}")?;
        }
        if let Some(p) = &self.psi_dft {
            write!(f, ", psi_dft={p}")?;
        }
        write!(f, ")")
    }
}

pub struct FcSaftParameters {
    pub molarweight: Array1<f64>,
    pub component_index: Array1<usize>,
    identifiers: Vec<String>,

    pub sigma: Array1<f64>,
    pub epsilon_k: Array1<f64>,
    pub bonds: UnGraph<(), f64>,

    pub association: Arc<AssociationParameters<Self>>,

    pub k_ij: Array2<f64>,
    pub sigma_ij: Array2<f64>,
    pub epsilon_k_ij: Array2<f64>,

    pub psi_dft: Array1<f64>,

    pub chemical_records: Vec<ChemicalRecord>,
    segment_records: Vec<SegmentRecord<FcSaftRecord>>,
    bond_records: Vec<BinaryRecord<String, f64>>,
    binary_segment_records: Option<Vec<BinaryRecord<String, f64>>>,
}

impl FcSaftParameters {
    pub fn new_pure_homosegmented(
        identifier: Identifier,
        segments: usize,
        sigma: f64,
        epsilon_k: f64,
        bond_length: f64,
        molarweight: f64,
        psi_dft: Option<f64>,
    ) -> Result<Self, ParameterError> {
        let chemical = ChemicalRecord::new(identifier, vec!["S".into(); segments], None);
        let segment = SegmentRecord::new(
            "S".into(),
            molarweight / segments as f64,
            FcSaftRecord::new(sigma, epsilon_k, None, None, None, None, None, psi_dft),
        );
        let bond = BinaryRecord::new("S".into(), "S".into(), bond_length);
        Self::from_segments(vec![chemical], vec![segment], vec![bond], None)
    }

    pub fn from_segments(
        chemical_records: Vec<ChemicalRecord>,
        segment_records: Vec<SegmentRecord<FcSaftRecord>>,
        bond_records: Vec<BinaryRecord<String, f64>>,
        binary_segment_records: Option<Vec<BinaryRecord<String, f64>>>,
    ) -> Result<Self, ParameterError> {
        let segment_map: IndexMap<_, _> = segment_records
            .iter()
            .map(|r| (r.identifier.clone(), r.clone()))
            .collect();

        let mut molarweight = Array1::zeros(chemical_records.len());
        let mut component_index = Vec::new();
        let mut identifiers = Vec::new();
        let mut sigma = Vec::new();
        let mut epsilon_k = Vec::new();
        let mut bonds = Graph::default();
        let mut association_records = Vec::new();
        let mut psi_dft = Vec::new();

        let mut bond_map = IndexMap::new();
        for bond_record in bond_records.iter() {
            bond_map.insert(
                (bond_record.id1.clone(), bond_record.id2.clone()),
                bond_record.model_record,
            );
            bond_map.insert(
                (bond_record.id2.clone(), bond_record.id1.clone()),
                bond_record.model_record,
            );
        }

        let mut segment_index = 0;
        for (i, chemical_record) in chemical_records.iter().enumerate() {
            if chemical_record.bonds.is_empty() {
                bonds.add_node(());
            } else {
                bonds.extend_with_edges(chemical_record.bonds.iter().map(|&[c1, c2]| {
                    (
                        (segment_index + c1) as u32,
                        (segment_index + c2) as u32,
                        bond_map
                            .get(&(
                                chemical_record.segments[c1].clone(),
                                chemical_record.segments[c2].clone(),
                            ))
                            .unwrap(),
                    )
                }));
            }

            for id in &chemical_record.segments {
                let segment = segment_map
                    .get(id)
                    .ok_or_else(|| ParameterError::ComponentsNotFound(id.to_string()))?;
                molarweight[i] += segment.molarweight;
                component_index.push(i);
                identifiers.push(id.clone());
                sigma.push(segment.model_record.sigma);
                epsilon_k.push(segment.model_record.epsilon_k);

                association_records.push(
                    segment
                        .model_record
                        .association_record
                        .into_iter()
                        .collect(),
                );

                psi_dft.push(segment.model_record.psi_dft.unwrap_or(PSI_FUSED_CHAINS));

                segment_index += 1;
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
            (epsilon_k[i] * epsilon_k[j]).sqrt() * (1.0 - k_ij[(i, j)])
        });

        // Association
        let sigma = Array1::from_vec(sigma);
        let component_index = Array1::from_vec(component_index);
        let association =
            AssociationParameters::new(&association_records, &[], Some(&component_index));

        Ok(Self {
            molarweight,
            component_index,
            identifiers,
            sigma,
            epsilon_k: Array1::from_vec(epsilon_k),
            bonds,
            association: Arc::new(association),
            psi_dft: Array1::from_vec(psi_dft),
            k_ij,
            sigma_ij,
            epsilon_k_ij,
            chemical_records,
            segment_records,
            bond_records,
            binary_segment_records,
        })
    }

    pub fn from_json_segments<P>(
        substances: &[&str],
        file_pure: P,
        file_segments: P,
        file_bonds: P,
        file_binary: Option<P>,
        search_option: IdentifierOption,
    ) -> Result<Self, ParameterError>
    where
        P: AsRef<Path>,
    {
        let queried: IndexSet<String> = substances
            .iter()
            .map(|identifier| identifier.to_string())
            .collect();

        let reader = BufReader::new(File::open(file_pure)?);
        let chemical_records: Vec<ChemicalRecord> = serde_json::from_reader(reader)?;
        let mut record_map: IndexMap<_, _> = chemical_records
            .into_iter()
            .filter_map(|record| {
                record
                    .identifier
                    .as_string(search_option)
                    .map(|i| (i, record))
            })
            .collect();

        // Compare queried components and available components
        let available: IndexSet<String> = record_map
            .keys()
            .map(|identifier| identifier.to_string())
            .collect();
        if !queried.is_subset(&available) {
            let missing: Vec<String> = queried.difference(&available).cloned().collect();
            return Err(ParameterError::ComponentsNotFound(format!("{:?}", missing)));
        };

        // Collect all pure records that were queried
        let chemical_records: Vec<_> = queried
            .iter()
            .filter_map(|identifier| record_map.shift_remove(&identifier.clone()))
            .collect();

        // Read segment records
        let segment_records: Vec<SegmentRecord<FcSaftRecord>> =
            serde_json::from_reader(BufReader::new(File::open(file_segments)?))?;

        // Read bond records
        let bond_records: Vec<BinaryRecord<String, f64>> =
            serde_json::from_reader(BufReader::new(File::open(file_bonds)?))?;

        // Read binary records
        let binary_records = file_binary
            .map(|file_binary| {
                let reader = BufReader::new(File::open(file_binary)?);
                let binary_records: Result<Vec<BinaryRecord<String, f64>>, ParameterError> =
                    Ok(serde_json::from_reader(reader)?);
                binary_records
            })
            .transpose()?;

        Self::from_segments(
            chemical_records,
            segment_records,
            bond_records,
            binary_records,
        )
    }

    pub fn subset(&self, component_list: &[usize]) -> Self {
        let chemical_records: Vec<_> = component_list
            .iter()
            .map(|&i| self.chemical_records[i].clone())
            .collect();
        Self::from_segments(
            chemical_records,
            self.segment_records.clone(),
            self.bond_records.clone(),
            self.binary_segment_records.clone(),
        )
        .unwrap()
    }

    // pub fn to_markdown(&self) -> String {
    //     let mut output = String::new();
    //     let o = &mut output;
    //     write!(
    //         o,
    //         "|component|molarweight|segment|$m$|$\\sigma$|$\\varepsilon$|$\\kappa_{{AB}}$|$\\varepsilon_{{AB}}$|$N_A$|$N_B$|\n|-|-|-|-|-|-|-|-|-|-|"
    //     )
    //     .unwrap();
    //     for i in 0..self.m.len() {
    //         let component = if i > 0 && self.component_index[i] == self.component_index[i - 1] {
    //             "|".to_string()
    //         } else {
    //             let pure = self.chemical_records[self.component_index[i]].identifier();
    //             format!(
    //                 "{}|{}",
    //                 pure.name.as_ref().unwrap_or(&pure.cas),
    //                 self.molarweight[self.component_index[i]]
    //             )
    //         };
    //         let association = if let Some(a) = self.assoc_segment.iter().position(|&a| a == i) {
    //             format!(
    //                 "{}|{}|{}|{}",
    //                 self.kappa_ab[a], self.epsilon_k_ab[a], self.na[a], self.nb[a]
    //             )
    //         } else {
    //             "|||".to_string()
    //         };
    //         write!(
    //             o,
    //             "\n|{}|{}|{}|{}|{}|{}|||",
    //             component,
    //             self.identifiers[i],
    //             self.m[i],
    //             self.sigma[i],
    //             self.epsilon_k[i],
    //             association
    //         )
    //         .unwrap();
    //     }

    //     output
    // }

    pub fn graph(&self) -> String {
        let graph = self
            .bonds
            .map(|i, _| &self.identifiers[i.index()], |_, _| ());
        format!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel]))
    }
}

impl AssociationStrength for FcSaftParameters {
    type Record = FcSaftAssociationRecord;
    type BinaryRecord = ();

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
}

impl std::fmt::Display for FcSaftParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FcSaftParameters(")?;
        write!(f, "\n\tmolarweight={}", self.molarweight)?;
        write!(f, "\n\tcomponent_index={}", self.component_index)?;
        write!(f, "\n\tsigma={}", self.sigma)?;
        write!(f, "\n\tepsilon_k={}", self.epsilon_k)?;
        write!(f, "\n\tbonds={:?}", self.bonds)?;
        write!(f, "\n)")
    }
}
