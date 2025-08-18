use super::{BinaryParameters, BinaryRecord, GroupCount, PureParameters};
use crate::{FeosError, FeosResult, parameter::PureRecord};
use nalgebra::DVector;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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
pub struct AssociationSite<A> {
    pub assoc_comp: usize,
    pub id: String,
    pub n: f64,
    pub parameters: A,
}

impl<A> AssociationSite<A> {
    fn new(assoc_comp: usize, id: String, n: f64, parameters: A) -> Self {
        Self {
            assoc_comp,
            id,
            n,
            parameters,
        }
    }
}

/// Parameter set required for the SAFT association Helmoltz energy
/// contribution and functional.
#[derive(Clone)]
pub struct AssociationParameters<A> {
    pub component_index: DVector<usize>,
    pub sites_a: Vec<AssociationSite<Option<A>>>,
    pub sites_b: Vec<AssociationSite<Option<A>>>,
    pub sites_c: Vec<AssociationSite<Option<A>>>,
    pub binary_ab: Vec<BinaryParameters<A, ()>>,
    pub binary_cc: Vec<BinaryParameters<A, ()>>,
}

impl<A: Clone> AssociationParameters<A> {
    pub fn new<P, B>(
        pure_records: &[PureRecord<P, A>],
        binary_records: &[BinaryRecord<usize, B, A>],
    ) -> FeosResult<Self> {
        let mut sites_a = Vec::new();
        let mut sites_b = Vec::new();
        let mut sites_c = Vec::new();

        for (i, record) in pure_records.iter().enumerate() {
            for site in record.association_sites.iter() {
                let par = &site.parameters;
                if site.na > 0.0 {
                    sites_a.push(AssociationSite::new(
                        i,
                        site.id.clone(),
                        site.na,
                        par.clone(),
                    ));
                }
                if site.nb > 0.0 {
                    sites_b.push(AssociationSite::new(
                        i,
                        site.id.clone(),
                        site.nb,
                        par.clone(),
                    ));
                }
                if site.nc > 0.0 {
                    sites_c.push(AssociationSite::new(
                        i,
                        site.id.clone(),
                        site.nc,
                        par.clone(),
                    ));
                }
            }
        }

        let indices_a: HashMap<_, _> = sites_a
            .iter()
            .enumerate()
            .map(|(i, site)| ((site.assoc_comp, &site.id), i))
            .collect();

        let indices_b: HashMap<_, _> = sites_b
            .iter()
            .enumerate()
            .map(|(i, site)| ((site.assoc_comp, &site.id), i))
            .collect();

        let indices_c: HashMap<_, _> = sites_c
            .iter()
            .enumerate()
            .map(|(i, site)| ((site.assoc_comp, &site.id), i))
            .collect();

        let index_set: HashSet<_> = indices_a
            .keys()
            .chain(indices_b.keys())
            .chain(indices_c.keys())
            .copied()
            .collect();

        let mut binary_ab = Vec::new();
        let mut binary_cc = Vec::new();
        for br in binary_records {
            let i = br.id1;
            let j = br.id2;
            for record in &br.association_sites {
                let a = &record.id1;
                let b = &record.id2;
                if !index_set.contains(&(i, a)) {
                    return Err(FeosError::IncompatibleParameters(format!(
                        "No association site {a} on component {i}"
                    )));
                }
                if !index_set.contains(&(j, b)) {
                    return Err(FeosError::IncompatibleParameters(format!(
                        "No association site {b} on component {j}"
                    )));
                }
                if let (Some(x), Some(y)) = (indices_a.get(&(i, a)), indices_b.get(&(j, b))) {
                    binary_ab.push(BinaryParameters::new(*x, *y, record.parameters.clone(), ()));
                }
                if let (Some(y), Some(x)) = (indices_b.get(&(i, a)), indices_a.get(&(j, b))) {
                    binary_ab.push(BinaryParameters::new(*x, *y, record.parameters.clone(), ()));
                }
                if let (Some(x), Some(y)) = (indices_c.get(&(i, a)), indices_c.get(&(j, b))) {
                    binary_cc.push(BinaryParameters::new(*x, *y, record.parameters.clone(), ()));
                    binary_cc.push(BinaryParameters::new(*y, *x, record.parameters.clone(), ()));
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
        pure_records: &[PureParameters<P, C>],
        association_sites: &[Vec<AssociationRecord<A>>],
        binary_records: &[BinaryParameters<Vec<BinaryAssociationRecord<A>>, ()>],
    ) -> FeosResult<Self> {
        let mut sites_a = Vec::new();
        let mut sites_b = Vec::new();
        let mut sites_c = Vec::new();

        for (i, (record, sites)) in pure_records.iter().zip(association_sites).enumerate() {
            for site in sites.iter() {
                let par = &site.parameters;
                if site.na > 0.0 {
                    let na = site.na * record.count.into_f64();
                    sites_a.push(AssociationSite::new(i, site.id.clone(), na, par.clone()));
                }
                if site.nb > 0.0 {
                    let nb = site.nb * record.count.into_f64();
                    sites_b.push(AssociationSite::new(i, site.id.clone(), nb, par.clone()));
                }
                if site.nc > 0.0 {
                    let nc = site.nc * record.count.into_f64();
                    sites_c.push(AssociationSite::new(i, site.id.clone(), nc, par.clone()));
                }
            }
        }

        let indices_a: HashMap<_, _> = sites_a
            .iter()
            .enumerate()
            .map(|(i, site)| ((site.assoc_comp, &site.id), i))
            .collect();

        let indices_b: HashMap<_, _> = sites_b
            .iter()
            .enumerate()
            .map(|(i, site)| ((site.assoc_comp, &site.id), i))
            .collect();

        let indices_c: HashMap<_, _> = sites_c
            .iter()
            .enumerate()
            .map(|(i, site)| ((site.assoc_comp, &site.id), i))
            .collect();

        let index_set: HashSet<_> = indices_a
            .keys()
            .chain(indices_b.keys())
            .chain(indices_c.keys())
            .copied()
            .collect();

        let mut binary_ab = Vec::new();
        let mut binary_cc = Vec::new();
        for br in binary_records {
            let i = br.id1;
            let j = br.id2;
            for record in &br.model_record {
                let a = &record.id1;
                let b = &record.id2;
                if !index_set.contains(&(i, a)) {
                    return Err(FeosError::IncompatibleParameters(format!(
                        "No association site {a} on component {i}"
                    )));
                }
                if !index_set.contains(&(j, b)) {
                    return Err(FeosError::IncompatibleParameters(format!(
                        "No association site {b} on component {j}"
                    )));
                }
                if let (Some(x), Some(y)) = (indices_a.get(&(i, a)), indices_b.get(&(j, b))) {
                    binary_ab.push(BinaryParameters::new(*x, *y, record.parameters.clone(), ()));
                }
                if let (Some(y), Some(x)) = (indices_b.get(&(i, a)), indices_a.get(&(j, b))) {
                    binary_ab.push(BinaryParameters::new(*x, *y, record.parameters.clone(), ()));
                }
                if let (Some(x), Some(y)) = (indices_c.get(&(i, a)), indices_c.get(&(j, b))) {
                    binary_cc.push(BinaryParameters::new(*x, *y, record.parameters.clone(), ()));
                    binary_cc.push(BinaryParameters::new(*y, *x, record.parameters.clone(), ()));
                }
            }
        }

        let component_index =
            DVector::from_vec(pure_records.iter().map(|pr| pr.component_index).collect());

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
