use super::eos::GcPcSaftOptions;
use super::record::GcPcSaftAssociationRecord;
use crate::association::{Association, AssociationFunctional, AssociationStrength};
use crate::gc_pcsaft::{GcPcSaftParameters, GcPcSaftRecord};
use crate::hard_sphere::{FMTContribution, FMTVersion, HardSphereProperties, MonomerShape};
use feos_core::{FeosResult, Molarweight, ResidualDyn, StateHD, Subset};
use feos_derive::FunctionalContribution;
use feos_dft::adsorption::FluidParameters;
use feos_dft::{
    FunctionalContribution, HelmholtzEnergyFunctional, HelmholtzEnergyFunctionalDyn, MoleculeShape,
};
use nalgebra::DVector;
use ndarray::Array1;
use num_dual::DualNum;
use petgraph::graph::UnGraph;
use quantity::MolarWeight;
use std::f64::consts::FRAC_PI_6;

mod dispersion;
mod hard_chain;
mod parameter;
use dispersion::AttractiveFunctional;
use hard_chain::ChainFunctional;
pub use parameter::GcPcSaftFunctionalParameters;

/// gc-PC-SAFT Helmholtz energy functional.
pub struct GcPcSaftFunctional {
    parameters: GcPcSaftParameters<()>,
    params: GcPcSaftFunctionalParameters,
    association: Option<Association<GcPcSaftFunctionalParameters>>,
    fmt_version: FMTVersion,
    options: GcPcSaftOptions,
}

impl GcPcSaftFunctional {
    pub fn new(parameters: GcPcSaftParameters<()>) -> Self {
        Self::with_options(
            parameters,
            FMTVersion::WhiteBear,
            GcPcSaftOptions::default(),
        )
    }

    pub fn with_options(
        parameters: GcPcSaftParameters<()>,
        fmt_version: FMTVersion,
        saft_options: GcPcSaftOptions,
    ) -> Self {
        let params = GcPcSaftFunctionalParameters::new(&parameters);
        let association = Association::new(
            &parameters,
            saft_options.max_iter_cross_assoc,
            saft_options.tol_cross_assoc,
        )
        .unwrap();
        Self {
            parameters,
            params,
            association,
            fmt_version,
            options: saft_options,
        }
    }
}

impl Subset for GcPcSaftFunctional {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            self.parameters.subset(component_list),
            self.fmt_version,
            self.options,
        )
    }
}

impl ResidualDyn for GcPcSaftFunctional {
    fn components(&self) -> usize {
        self.parameters.molar_weight.len()
    }

    fn compute_max_density<D: DualNum<f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        let p = &self.params;
        let msigma3 = p.m.zip_map(&p.sigma, |m, s| m * s.powi(3) * m);
        let molefracs_segments: DVector<_> = p
            .component_index
            .iter()
            .map(|&i| molefracs[i])
            .collect::<Vec<_>>()
            .into();
        (msigma3.map(D::from).dot(&molefracs_segments) * FRAC_PI_6).recip() * self.options.max_eta
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)> {
        self.evaluate_bulk(state)
    }
}

impl HelmholtzEnergyFunctionalDyn for GcPcSaftFunctional {
    type Contribution<'a> = GcPcSaftFunctionalContribution<'a>;

    fn molecule_shape(&self) -> MoleculeShape<'_> {
        MoleculeShape::Heterosegmented(&self.params.component_index)
    }

    fn contributions<'a>(&'a self) -> impl Iterator<Item = GcPcSaftFunctionalContribution<'a>> {
        let mut contributions = Vec::with_capacity(4);

        let assoc = AssociationFunctional::new(&self.params, &self.parameters, &self.association);

        // Hard sphere contribution
        let hs = FMTContribution::new(&self.params, self.fmt_version);
        contributions.push(hs.into());

        // Hard chains
        let chain = ChainFunctional::new(&self.params);
        contributions.push(chain.into());

        // Dispersion
        let att = AttractiveFunctional::new(&self.params);
        contributions.push(att.into());

        // Association
        if let Some(assoc) = assoc {
            contributions.push(assoc.into());
        }

        contributions.into_iter()
    }

    fn bond_lengths<N: DualNum<f64> + Copy>(&self, temperature: N) -> UnGraph<(), N> {
        // temperature dependent segment diameter
        let d = self.params.hs_diameter(temperature);

        self.params.bonds.map(
            |_, _| (),
            |e, _| {
                let (i, j) = self.params.bonds.edge_endpoints(e).unwrap();
                let di = d[i.index()];
                let dj = d[j.index()];
                (di + dj) * 0.5
            },
        )
    }
}

impl Molarweight for GcPcSaftFunctional {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

impl HardSphereProperties for GcPcSaftFunctionalParameters {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<'_, N> {
        let m = self.m.map(N::from);
        MonomerShape::Heterosegmented([m.clone(), m.clone(), m.clone(), m], &self.component_index)
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> DVector<D> {
        let ti = temperature.recip() * -3.0;
        DVector::from_fn(self.sigma.len(), |i, _| {
            -((ti * self.epsilon_k[i]).exp() * 0.12 - 1.0) * self.sigma[i]
        })
    }
}

impl AssociationStrength for GcPcSaftFunctionalParameters {
    type Pure = GcPcSaftRecord;
    type Record = GcPcSaftAssociationRecord;

    fn association_strength<D: DualNum<f64> + Copy>(
        &self,
        temperature: D,
        comp_i: usize,
        comp_j: usize,
        assoc_ij: &Self::Record,
    ) -> D {
        let si = self.sigma[comp_i];
        let sj = self.sigma[comp_j];
        (temperature.recip() * assoc_ij.epsilon_k_ab).exp_m1()
            * assoc_ij.kappa_ab
            * (si * sj).powf(1.5)
    }

    fn combining_rule(
        _: &Self::Pure,
        _: &Self::Pure,
        parameters_i: &Self::Record,
        parameters_j: &Self::Record,
    ) -> Self::Record {
        Self::Record {
            kappa_ab: (parameters_i.kappa_ab * parameters_j.kappa_ab).sqrt(),
            epsilon_k_ab: 0.5 * (parameters_i.epsilon_k_ab + parameters_j.epsilon_k_ab),
        }
    }
}

impl FluidParameters for GcPcSaftFunctional {
    fn epsilon_k_ff(&self) -> DVector<f64> {
        self.params.epsilon_k.clone()
    }

    fn sigma_ff(&self) -> DVector<f64> {
        self.params.sigma.clone()
    }
}

/// Individual contributions for the gc-PC-SAFT Helmholtz energy functional.
#[derive(FunctionalContribution)]
pub enum GcPcSaftFunctionalContribution<'a> {
    Fmt(FMTContribution<'a, GcPcSaftFunctionalParameters>),
    Chain(ChainFunctional<'a>),
    Attractive(AttractiveFunctional<'a>),
    Association(AssociationFunctional<'a, GcPcSaftFunctionalParameters>),
}
