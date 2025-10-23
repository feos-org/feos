use super::{BinaryParameters, BinaryRecord, GroupCount, PureParameters};
use crate::{FeosResult, parameter::PureRecord};
use nalgebra::DVector;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pure component association parameters.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AssociationRecord<A> {
    #[serde(skip_serializing_if = "String::is_empty")]
    #[serde(default)]
    pub id: String,
    #[serde(flatten)]
    pub parameters: Option<A>,
    /// \# of association sites of type A
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub na: f64,
    /// \# of association sites of type B
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub nb: f64,
    /// \# of association sites of type C
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub nc: f64,
}

impl<A> AssociationRecord<A> {
    pub fn new(parameters: Option<A>, na: f64, nb: f64, nc: f64) -> Self {
        Self::with_id(Default::default(), parameters, na, nb, nc)
    }

    pub fn with_id(id: String, parameters: Option<A>, na: f64, nb: f64, nc: f64) -> Self {
        Self {
            id,
            parameters,
            na,
            nb,
            nc,
        }
    }
}

/// Binary association parameters.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BinaryAssociationRecord<A> {
    // Identifier of the association site on the first molecule.
    #[serde(skip_serializing_if = "String::is_empty")]
    #[serde(default)]
    pub id1: String,
    // Identifier of the association site on the second molecule.
    #[serde(skip_serializing_if = "String::is_empty")]
    #[serde(default)]
    pub id2: String,
    // Binary association parameters
    #[serde(flatten)]
    pub parameters: A,
}

impl<A> BinaryAssociationRecord<A> {
    pub fn new(parameters: A) -> Self {
        Self::with_id(Default::default(), Default::default(), parameters)
    }

    pub fn with_id(id1: String, id2: String, parameters: A) -> Self {
        Self {
            id1,
            id2,
            parameters,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AssociationSite {
    pub assoc_comp: usize,
    pub id: String,
    pub n: f64,
}

impl AssociationSite {
    fn new(assoc_comp: usize, id: String, n: f64) -> Self {
        Self { assoc_comp, id, n }
    }
}

pub trait CombiningRule<P> {
    fn combining_rule(comp_i: &P, comp_j: &P, parameters_i: &Self, parameters_j: &Self) -> Self;
}

impl<P> CombiningRule<P> for () {
    fn combining_rule(_: &P, _: &P, _: &Self, _: &Self) {}
}

/// Parameter set required for the SAFT association Helmoltz energy
/// contribution and functional.
#[derive(Clone)]
pub struct AssociationParameters<A> {
    pub component_index: DVector<usize>,
    pub sites_a: Vec<AssociationSite>,
    pub sites_b: Vec<AssociationSite>,
    pub sites_c: Vec<AssociationSite>,
    pub binary_ab: Vec<BinaryParameters<A, ()>>,
    pub binary_cc: Vec<BinaryParameters<A, ()>>,
}

impl<A: Clone> AssociationParameters<A> {
    pub fn new<P, B>(
        pure_records: &[PureRecord<P, A>],
        binary_records: &[BinaryRecord<usize, B, A>],
    ) -> FeosResult<Self>
    where
        A: CombiningRule<P>,
    {
        let mut sites_a = Vec::new();
        let mut sites_b = Vec::new();
        let mut sites_c = Vec::new();
        let mut pars_a = Vec::new();
        let mut pars_b = Vec::new();
        let mut pars_c = Vec::new();

        for (i, record) in pure_records.iter().enumerate() {
            for site in record.association_sites.iter() {
                if site.na > 0.0 {
                    sites_a.push(AssociationSite::new(i, site.id.clone(), site.na));
                    pars_a.push(&site.parameters);
                }
                if site.nb > 0.0 {
                    sites_b.push(AssociationSite::new(i, site.id.clone(), site.nb));
                    pars_b.push(&site.parameters);
                }
                if site.nc > 0.0 {
                    sites_c.push(AssociationSite::new(i, site.id.clone(), site.nc));
                    pars_c.push(&site.parameters);
                }
            }
        }

        let record_map: HashMap<_, _> = binary_records
            .iter()
            .flat_map(|br| {
                br.association_sites.iter().flat_map(|a| {
                    [
                        ((br.id1, br.id2, &a.id1, &a.id2), &a.parameters),
                        ((br.id2, br.id1, &a.id2, &a.id1), &a.parameters),
                    ]
                })
            })
            .collect();

        let mut binary_ab = Vec::new();
        for ((a, site_a), pa) in sites_a.iter().enumerate().zip(&pars_a) {
            for ((b, site_b), pb) in sites_b.iter().enumerate().zip(&pars_b) {
                if let Some(&record) =
                    record_map.get(&(site_a.assoc_comp, site_b.assoc_comp, &site_a.id, &site_b.id))
                {
                    binary_ab.push(BinaryParameters::new(a, b, record.clone(), ()));
                } else if let (Some(pa), Some(pb)) = (pa, pb) {
                    binary_ab.push(BinaryParameters::new(
                        a,
                        b,
                        A::combining_rule(
                            &pure_records[site_a.assoc_comp].model_record,
                            &pure_records[site_b.assoc_comp].model_record,
                            pa,
                            pb,
                        ),
                        (),
                    ));
                }
            }
        }

        let mut binary_cc = Vec::new();
        for ((a, site_a), pa) in sites_c.iter().enumerate().zip(&pars_c) {
            for ((b, site_b), pb) in sites_c.iter().enumerate().zip(&pars_c) {
                if let Some(&record) =
                    record_map.get(&(site_a.assoc_comp, site_b.assoc_comp, &site_a.id, &site_b.id))
                {
                    binary_cc.push(BinaryParameters::new(a, b, record.clone(), ()));
                } else if let (Some(pa), Some(pb)) = (pa, pb) {
                    binary_cc.push(BinaryParameters::new(
                        a,
                        b,
                        A::combining_rule(
                            &pure_records[site_a.assoc_comp].model_record,
                            &pure_records[site_b.assoc_comp].model_record,
                            pa,
                            pb,
                        ),
                        (),
                    ));
                }
            }
        }

        let component_index = DVector::from_vec((0..pure_records.len()).collect());

        Ok(Self {
            component_index,
            sites_a,
            sites_b,
            sites_c,
            binary_ab,
            binary_cc,
        })
    }

    pub fn new_hetero<P, C: GroupCount>(
        groups: &[PureParameters<P, C>],
        association_sites: &[Vec<AssociationRecord<A>>],
        binary_records: &[BinaryParameters<Vec<BinaryAssociationRecord<A>>, ()>],
    ) -> FeosResult<Self>
    where
        A: CombiningRule<P>,
    {
        let mut sites_a = Vec::new();
        let mut sites_b = Vec::new();
        let mut sites_c = Vec::new();
        let mut pars_a = Vec::new();
        let mut pars_b = Vec::new();
        let mut pars_c = Vec::new();

        for (i, (record, sites)) in groups.iter().zip(association_sites).enumerate() {
            for site in sites.iter() {
                if site.na > 0.0 {
                    let na = site.na * record.count.into_f64();
                    sites_a.push(AssociationSite::new(i, site.id.clone(), na));
                    pars_a.push(&site.parameters)
                }
                if site.nb > 0.0 {
                    let nb = site.nb * record.count.into_f64();
                    sites_b.push(AssociationSite::new(i, site.id.clone(), nb));
                    pars_b.push(&site.parameters)
                }
                if site.nc > 0.0 {
                    let nc = site.nc * record.count.into_f64();
                    sites_c.push(AssociationSite::new(i, site.id.clone(), nc));
                    pars_c.push(&site.parameters)
                }
            }
        }

        let record_map: HashMap<_, _> = binary_records
            .iter()
            .flat_map(|br| {
                br.model_record.iter().flat_map(|a| {
                    [
                        ((br.id1, br.id2, &a.id1, &a.id2), &a.parameters),
                        ((br.id2, br.id1, &a.id2, &a.id1), &a.parameters),
                    ]
                })
            })
            .collect();

        let mut binary_ab = Vec::new();
        for ((a, site_a), pa) in sites_a.iter().enumerate().zip(&pars_a) {
            for ((b, site_b), pb) in sites_b.iter().enumerate().zip(&pars_b) {
                if let Some(&record) =
                    record_map.get(&(site_a.assoc_comp, site_b.assoc_comp, &site_a.id, &site_b.id))
                {
                    binary_ab.push(BinaryParameters::new(a, b, record.clone(), ()));
                } else if let (Some(pa), Some(pb)) = (pa, pb) {
                    binary_ab.push(BinaryParameters::new(
                        a,
                        b,
                        A::combining_rule(
                            &groups[site_a.assoc_comp].model_record,
                            &groups[site_b.assoc_comp].model_record,
                            pa,
                            pb,
                        ),
                        (),
                    ));
                }
            }
        }

        let mut binary_cc = Vec::new();
        for ((a, site_a), pa) in sites_c.iter().enumerate().zip(&pars_c) {
            for ((b, site_b), pb) in sites_c.iter().enumerate().zip(&pars_c) {
                if let Some(&record) =
                    record_map.get(&(site_a.assoc_comp, site_b.assoc_comp, &site_a.id, &site_b.id))
                {
                    binary_cc.push(BinaryParameters::new(a, b, record.clone(), ()));
                } else if let (Some(pa), Some(pb)) = (pa, pb) {
                    binary_cc.push(BinaryParameters::new(
                        a,
                        b,
                        A::combining_rule(
                            &groups[site_a.assoc_comp].model_record,
                            &groups[site_b.assoc_comp].model_record,
                            pa,
                            pb,
                        ),
                        (),
                    ));
                }
            }
        }

        let component_index =
            DVector::from_vec(groups.iter().map(|pr| pr.component_index).collect());

        Ok(Self {
            component_index,
            sites_a,
            sites_b,
            sites_c,
            binary_ab,
            binary_cc,
        })
    }

    pub fn is_empty(&self) -> bool {
        (self.sites_a.is_empty() | self.sites_b.is_empty()) & self.sites_c.is_empty()
    }
}
