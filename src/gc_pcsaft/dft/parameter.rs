use crate::association::AssociationParameters;
use crate::gc_pcsaft::record::GcPcSaftRecord;
use feos_core::parameter::{
    BinaryRecord, ChemicalRecord, ParameterError, ParameterHetero, SegmentRecord,
};
use indexmap::IndexMap;
use ndarray::{Array1, Array2};
use petgraph::dot::{Config, Dot};
use petgraph::graph::{Graph, UnGraph};

/// psi Parameter for heterosegmented DFT (Mairhofer2018)
const PSI_GC_DFT: f64 = 1.5357;

/// Parameter set required for the gc-PC-SAFT Helmholtz energy functional.
pub struct GcPcSaftFunctionalParameters {
    pub molarweight: Array1<f64>,
    pub component_index: Array1<usize>,
    identifiers: Vec<String>,
    pub m: Array1<f64>,
    pub sigma: Array1<f64>,
    pub epsilon_k: Array1<f64>,
    pub bonds: UnGraph<(), ()>,
    pub association: AssociationParameters,
    pub psi_dft: Array1<f64>,
    pub k_ij: Array2<f64>,
    pub sigma_ij: Array2<f64>,
    pub epsilon_k_ij: Array2<f64>,
    pub chemical_records: Vec<ChemicalRecord>,
    segment_records: Vec<SegmentRecord<GcPcSaftRecord>>,
    binary_segment_records: Option<Vec<BinaryRecord<String, f64>>>,
}

impl ParameterHetero for GcPcSaftFunctionalParameters {
    type Chemical = ChemicalRecord;
    type Pure = GcPcSaftRecord;
    type Binary = f64;

    fn from_segments<C: Into<ChemicalRecord>>(
        chemical_records: Vec<C>,
        segment_records: Vec<SegmentRecord<GcPcSaftRecord>>,
        binary_segment_records: Option<Vec<BinaryRecord<String, f64>>>,
    ) -> Result<Self, ParameterError> {
        let chemical_records: Vec<_> = chemical_records.into_iter().map(|cr| cr.into()).collect();

        let segment_map: IndexMap<_, _> = segment_records
            .iter()
            .map(|r| (r.identifier.clone(), r.clone()))
            .collect();

        let mut molarweight = Array1::zeros(chemical_records.len());
        let mut component_index = Vec::new();
        let mut identifiers = Vec::new();
        let mut m = Vec::new();
        let mut sigma = Vec::new();
        let mut epsilon_k = Vec::new();
        let mut bonds = Graph::default();
        let mut association_records = Vec::new();
        let mut psi_dft = Vec::new();

        let mut segment_index = 0;
        for (i, chemical_record) in chemical_records.iter().enumerate() {
            bonds.extend_with_edges(chemical_record.bonds.iter().map(|x| {
                (
                    (segment_index + x[0]) as u32,
                    (segment_index + x[1]) as u32,
                    (),
                )
            }));

            for id in &chemical_record.segments {
                let segment = segment_map
                    .get(id)
                    .ok_or_else(|| ParameterError::ComponentsNotFound(id.to_string()))?;
                molarweight[i] += segment.molarweight;
                component_index.push(i);
                identifiers.push(id.clone());
                m.push(segment.model_record.m);
                sigma.push(segment.model_record.sigma);
                epsilon_k.push(segment.model_record.epsilon_k);

                association_records.push(
                    segment
                        .model_record
                        .association_record
                        .into_iter()
                        .collect(),
                );

                psi_dft.push(segment.model_record.psi_dft.unwrap_or(PSI_GC_DFT));

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
            psi_dft: Array1::from_vec(psi_dft),
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

impl GcPcSaftFunctionalParameters {
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
    //             let pure = &self.chemical_records[self.component_index[i]].identifier;
    //             format!(
    //                 "{}|{}",
    //                 pure.name
    //                     .as_ref()
    //                     .unwrap_or(&format!("Component {}", i + 1)),
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

impl std::fmt::Display for GcPcSaftFunctionalParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GcPcSaftFunctionalParameters(")?;
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
        write!(f, "\n)")
    }
}
