use super::{FcSaftOptions, FcSaftParameters};
use crate::association::Association;
use crate::hard_sphere::{FMTContribution, FMTVersion};
use feos_core::{Components, EosResult, Molarweight, Residual};
use feos_derive::FunctionalContribution;
use feos_dft::adsorption::FluidParameters;
use feos_dft::{FunctionalContribution, HelmholtzEnergyFunctional, MoleculeShape};
use ndarray::{Array1, ScalarOperand};
use num_dual::DualNum;
use petgraph::graph::UnGraph;
use quantity::{MolarWeight, GRAM, MOL};
use std::f64::consts::FRAC_PI_6;
use std::sync::Arc;

mod dispersion;
mod fused_chain;
use dispersion::DispersionFunctional;
use fused_chain::FusedHardChainFunctional;

/// gc-PC-SAFT Helmholtz energy functional.
pub struct FcSaftFunctional {
    pub parameters: Arc<FcSaftParameters>,
    fmt_version: FMTVersion,
    options: FcSaftOptions,
}

impl FcSaftFunctional {
    pub fn new(parameters: Arc<FcSaftParameters>) -> Self {
        Self::with_options(parameters, FMTVersion::WhiteBear, FcSaftOptions::default())
    }

    pub fn with_options(
        parameters: Arc<FcSaftParameters>,
        fmt_version: FMTVersion,
        options: FcSaftOptions,
    ) -> Self {
        Self {
            parameters,
            fmt_version,
            options,
        }
    }
}

impl Components for FcSaftFunctional {
    fn components(&self) -> usize {
        self.parameters.chemical_records.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            Arc::new(self.parameters.subset(component_list)),
            self.fmt_version,
            self.options,
        )
    }
}

impl Residual for FcSaftFunctional {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        let p = &self.parameters;
        let moles_segments: Array1<f64> = p.component_index.iter().map(|&i| moles[i]).collect();
        let [_, v] = p.geometry_coefficients(&p.sigma);
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * p.sigma.mapv(|v| v.powi(3)) * v * moles_segments).sum()
    }

    fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        state: &feos_core::StateHD<D>,
    ) -> Vec<(String, D)> {
        self.evaluate_bulk(state)
    }
}

impl HelmholtzEnergyFunctional for FcSaftFunctional {
    type Contribution = FcSaftFunctionalContribution;

    fn molecule_shape(&self) -> MoleculeShape {
        MoleculeShape::Heterosegmented(&self.parameters.component_index)
    }

    fn contributions(&self) -> Box<dyn Iterator<Item = FcSaftFunctionalContribution>> {
        let mut contributions: Vec<FcSaftFunctionalContribution> = Vec::with_capacity(4);

        // Hard sphere contribution
        let hs = FMTContribution::new(&self.parameters, self.fmt_version);
        contributions.push(FcSaftFunctionalContribution::Fmt(hs));

        // Hard chains
        let chain = FusedHardChainFunctional::new(&self.parameters);
        contributions.push(FcSaftFunctionalContribution::FusedChain(chain));

        // Dispersion
        let disp = DispersionFunctional::new(&self.parameters);
        contributions.push(FcSaftFunctionalContribution::Dispersion(disp));

        // Association
        if !self.parameters.association.is_empty() {
            let assoc = Association::new(
                &self.parameters,
                &self.parameters.association,
                self.options.max_iter_cross_assoc,
                self.options.tol_cross_assoc,
            );
            contributions.push(FcSaftFunctionalContribution::Association(assoc));
        }

        Box::new(contributions.into_iter())
    }

    fn bond_lengths<N: DualNum<f64> + Copy>(&self, _: N) -> UnGraph<(), N> {
        self.parameters.bonds.map(|_, _| (), |_, &x| N::from(x))
    }
}

impl Molarweight for FcSaftFunctional {
    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}

impl FluidParameters for FcSaftFunctional {
    fn epsilon_k_ff(&self) -> Array1<f64> {
        self.parameters.epsilon_k.clone()
    }

    fn sigma_ff(&self) -> &Array1<f64> {
        &self.parameters.sigma
    }
}

/// Individual contributions for the PC-SAFT Helmholtz energy functional.
#[derive(FunctionalContribution)]
pub enum FcSaftFunctionalContribution {
    Fmt(FMTContribution<FcSaftParameters>),
    FusedChain(FusedHardChainFunctional),
    Dispersion(DispersionFunctional),
    Association(Association<FcSaftParameters>),
}
