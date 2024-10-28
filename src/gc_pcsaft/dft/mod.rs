use super::eos::GcPcSaftOptions;
use super::record::GcPcSaftAssociationRecord;
use crate::association::{Association, AssociationStrength};
use crate::hard_sphere::{FMTContribution, FMTVersion, HardSphereProperties, MonomerShape};
use feos_core::parameter::ParameterHetero;
use feos_core::{Components, EosResult, Molarweight};
use feos_derive::FunctionalContribution;
use feos_dft::adsorption::FluidParameters;
use feos_dft::{
    FunctionalContribution, HelmholtzEnergyFunctional, MoleculeShape, WeightFunctionInfo, DFT,
};
use ndarray::{Array1, ArrayView2, ScalarOperand};
use num_dual::DualNum;
use petgraph::graph::UnGraph;
use quantity::{MolarWeight, GRAM, MOL};
use std::f64::consts::FRAC_PI_6;
use std::sync::Arc;

mod dispersion;
mod hard_chain;
mod parameter;
use dispersion::AttractiveFunctional;
use hard_chain::ChainFunctional;
pub use parameter::GcPcSaftFunctionalParameters;

/// gc-PC-SAFT Helmholtz energy functional.
pub struct GcPcSaftFunctional {
    pub parameters: Arc<GcPcSaftFunctionalParameters>,
    fmt_version: FMTVersion,
    options: GcPcSaftOptions,
}

impl GcPcSaftFunctional {
    pub fn new(parameters: Arc<GcPcSaftFunctionalParameters>) -> DFT<Self> {
        Self::with_options(
            parameters,
            FMTVersion::WhiteBear,
            GcPcSaftOptions::default(),
        )
    }

    pub fn with_options(
        parameters: Arc<GcPcSaftFunctionalParameters>,
        fmt_version: FMTVersion,
        saft_options: GcPcSaftOptions,
    ) -> DFT<Self> {
        DFT(Self {
            parameters,
            fmt_version,
            options: saft_options,
        })
    }
}

impl Components for GcPcSaftFunctional {
    fn components(&self) -> usize {
        self.parameters.chemical_records.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            Arc::new(self.parameters.subset(component_list)),
            self.fmt_version,
            self.options,
        )
        .0
    }
}

impl HelmholtzEnergyFunctional for GcPcSaftFunctional {
    type Contribution = GcPcSaftFunctionalContribution;

    fn molecule_shape(&self) -> MoleculeShape {
        MoleculeShape::Heterosegmented(&self.parameters.component_index)
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        let p = &self.parameters;
        let moles_segments: Array1<f64> = p.component_index.iter().map(|&i| moles[i]).collect();
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * &p.m * p.sigma.mapv(|v| v.powi(3)) * moles_segments).sum()
    }

    fn contributions(&self) -> Box<dyn Iterator<Item = GcPcSaftFunctionalContribution>> {
        let mut contributions = Vec::with_capacity(4);

        // Hard sphere contribution
        let hs = FMTContribution::new(&self.parameters, self.fmt_version);
        contributions.push(hs.into());

        // Hard chains
        let chain = ChainFunctional::new(&self.parameters);
        contributions.push(chain.into());

        // Dispersion
        let att = AttractiveFunctional::new(&self.parameters);
        contributions.push(att.into());

        // Association
        if !self.parameters.association.is_empty() {
            let assoc = Association::new(
                &self.parameters,
                &self.parameters.association,
                self.options.max_iter_cross_assoc,
                self.options.tol_cross_assoc,
            );
            contributions.push(Box::new(assoc).into());
        }

        Box::new(contributions.into_iter())
    }

    fn bond_lengths(&self, temperature: f64) -> UnGraph<(), f64> {
        // temperature dependent segment diameter
        let d = self.parameters.hs_diameter(temperature);

        self.parameters.bonds.map(
            |_, _| (),
            |e, _| {
                let (i, j) = self.parameters.bonds.edge_endpoints(e).unwrap();
                let di = d[i.index()];
                let dj = d[j.index()];
                0.5 * (di + dj)
            },
        )
    }
}

impl Molarweight for GcPcSaftFunctional {
    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}

impl HardSphereProperties for GcPcSaftFunctionalParameters {
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

impl AssociationStrength for GcPcSaftFunctionalParameters {
    type Record = GcPcSaftAssociationRecord;
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

impl FluidParameters for GcPcSaftFunctional {
    fn epsilon_k_ff(&self) -> Array1<f64> {
        self.parameters.epsilon_k.clone()
    }

    fn sigma_ff(&self) -> &Array1<f64> {
        &self.parameters.sigma
    }
}

/// Individual contributions for the gc-PC-SAFT Helmholtz energy functional.
#[derive(FunctionalContribution)]
pub enum GcPcSaftFunctionalContribution {
    Fmt(FMTContribution<GcPcSaftFunctionalParameters>),
    Chain(ChainFunctional),
    Attractive(AttractiveFunctional),
    Association(Box<Association<GcPcSaftFunctionalParameters>>),
}
