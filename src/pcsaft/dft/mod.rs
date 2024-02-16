use super::PcSaftParameters;
use crate::association::Association;
use crate::hard_sphere::{FMTContribution, FMTVersion};
use crate::pcsaft::eos::PcSaftOptions;
use feos_core::parameter::Parameter;
use feos_core::si::{MolarWeight, GRAM, MOL};
use feos_core::{Components, EosResult};
use feos_derive::FunctionalContribution;
use feos_dft::adsorption::FluidParameters;
use feos_dft::solvation::PairPotential;
use feos_dft::{
    FunctionalContribution, HelmholtzEnergyFunctional, MoleculeShape, WeightFunctionInfo, DFT,
};
use ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use num_dual::DualNum;
use num_traits::One;
use std::f64::consts::FRAC_PI_6;
use std::sync::Arc;

mod dispersion;
mod hard_chain;
mod polar;
mod pure_saft_functional;
use dispersion::AttractiveFunctional;
use hard_chain::ChainFunctional;
use pure_saft_functional::*;

/// PC-SAFT Helmholtz energy functional.
pub struct PcSaftFunctional {
    pub parameters: Arc<PcSaftParameters>,
    fmt_version: FMTVersion,
    options: PcSaftOptions,
}

impl PcSaftFunctional {
    pub fn new(parameters: Arc<PcSaftParameters>) -> DFT<Self> {
        Self::with_options(parameters, FMTVersion::WhiteBear, PcSaftOptions::default())
    }

    pub fn new_full(parameters: Arc<PcSaftParameters>, fmt_version: FMTVersion) -> DFT<Self> {
        Self::with_options(parameters, fmt_version, PcSaftOptions::default())
    }

    pub fn with_options(
        parameters: Arc<PcSaftParameters>,
        fmt_version: FMTVersion,
        saft_options: PcSaftOptions,
    ) -> DFT<Self> {
        DFT(Self {
            parameters,
            fmt_version,
            options: saft_options,
        })
    }
}

impl Components for PcSaftFunctional {
    fn components(&self) -> usize {
        self.parameters.pure_records.len()
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

impl HelmholtzEnergyFunctional for PcSaftFunctional {
    type Contribution = PcSaftFunctionalContribution;

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * &self.parameters.m * self.parameters.sigma.mapv(|v| v.powi(3)) * moles)
                .sum()
    }

    fn contributions(&self) -> Box<dyn Iterator<Item = PcSaftFunctionalContribution>> {
        let mut contributions = Vec::with_capacity(4);

        if matches!(
            self.fmt_version,
            FMTVersion::WhiteBear | FMTVersion::AntiSymWhiteBear
        ) && self.parameters.m.len() == 1
        {
            let fmt_assoc = PureFMTAssocFunctional::new(self.parameters.clone(), self.fmt_version);
            contributions.push(fmt_assoc.into());
            if self.parameters.m.iter().any(|&mi| mi > 1.0) {
                let chain = PureChainFunctional::new(self.parameters.clone());
                contributions.push(chain.into());
            }
            let att = PureAttFunctional::new(self.parameters.clone());
            contributions.push(att.into());
        } else {
            // Hard sphere contribution
            let hs = FMTContribution::new(&self.parameters, self.fmt_version);
            contributions.push(hs.into());

            // Hard chains
            if self.parameters.m.iter().any(|&mi| !mi.is_one()) {
                let chain = ChainFunctional::new(self.parameters.clone());
                contributions.push(chain.into());
            }

            // Dispersion
            let att = AttractiveFunctional::new(self.parameters.clone());
            contributions.push(att.into());

            // Association
            if !self.parameters.association.is_empty() {
                let assoc = Association::new(
                    &self.parameters,
                    &self.parameters.association,
                    self.options.max_iter_cross_assoc,
                    self.options.tol_cross_assoc,
                );
                contributions.push(assoc.into());
            }
        }
        Box::new(contributions.into_iter())
    }

    fn molecule_shape(&self) -> MoleculeShape {
        MoleculeShape::NonSpherical(&self.parameters.m)
    }

    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}

impl FluidParameters for PcSaftFunctional {
    fn epsilon_k_ff(&self) -> Array1<f64> {
        self.parameters.epsilon_k.clone()
    }

    fn sigma_ff(&self) -> &Array1<f64> {
        &self.parameters.sigma
    }
}

impl PairPotential for PcSaftFunctional {
    fn pair_potential(&self, i: usize, r: &Array1<f64>, _: f64) -> Array2<f64> {
        let sigma_ij = &self.parameters.sigma_ij;
        let eps_ij_4 = 4.0 * &self.parameters.epsilon_k_ij;
        Array2::from_shape_fn((self.parameters.m.len(), r.len()), |(j, k)| {
            let att = (sigma_ij[[i, j]] / r[k]).powi(6);
            eps_ij_4[[i, j]] * att * (att - 1.0)
        })
    }
}

#[derive(FunctionalContribution)]
pub enum PcSaftFunctionalContribution {
    PureFMTAssoc(PureFMTAssocFunctional),
    PureChain(PureChainFunctional),
    PureAtt(PureAttFunctional),
    Fmt(FMTContribution<PcSaftParameters>),
    Chain(ChainFunctional),
    Attractive(AttractiveFunctional),
    Association(Association<PcSaftParameters>),
}
