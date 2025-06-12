use super::PcSaftParameters;
use super::parameters::PcSaftPars;
use crate::association::{AssociationFunctional, AssociationParameters};
use crate::hard_sphere::{FMTContribution, FMTVersion};
use crate::pcsaft::eos::PcSaftOptions;
use feos_core::{Components, FeosResult, Molarweight, Residual, StateHD};
use feos_derive::FunctionalContribution;
use feos_dft::adsorption::FluidParameters;
use feos_dft::solvation::PairPotential;
use feos_dft::{FunctionalContribution, HelmholtzEnergyFunctional, MoleculeShape};
use ndarray::{Array1, Array2, ScalarOperand};
use num_dual::DualNum;
use num_traits::One;
use quantity::MolarWeight;
use std::f64::consts::FRAC_PI_6;

mod dispersion;
mod hard_chain;
mod polar;
mod pure_saft_functional;
use dispersion::AttractiveFunctional;
use hard_chain::ChainFunctional;
use pure_saft_functional::*;

/// PC-SAFT Helmholtz energy functional.
pub struct PcSaftFunctional {
    parameters: PcSaftParameters,
    association_parameters: AssociationParameters<PcSaftPars>,
    params: PcSaftPars,
    fmt_version: FMTVersion,
    options: PcSaftOptions,
}

impl PcSaftFunctional {
    pub fn new(parameters: PcSaftParameters) -> Self {
        Self::with_options(parameters, FMTVersion::WhiteBear, PcSaftOptions::default())
    }

    pub fn new_full(parameters: PcSaftParameters, fmt_version: FMTVersion) -> Self {
        Self::with_options(parameters, fmt_version, PcSaftOptions::default())
    }

    pub fn with_options(
        parameters: PcSaftParameters,
        fmt_version: FMTVersion,
        saft_options: PcSaftOptions,
    ) -> Self {
        let params = PcSaftPars::new(&parameters);
        let association_parameters = AssociationParameters::new(&parameters).unwrap();
        Self {
            parameters,
            association_parameters,
            params,
            fmt_version,
            options: saft_options,
        }
    }
}

impl Components for PcSaftFunctional {
    fn components(&self) -> usize {
        self.parameters.pure_records.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            self.parameters.subset(component_list),
            self.fmt_version,
            self.options,
        )
    }
}

impl Residual for PcSaftFunctional {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * &self.params.m * self.params.sigma.mapv(|v| v.powi(3)) * moles).sum()
    }

    fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)> {
        self.evaluate_bulk(state)
    }
}

impl HelmholtzEnergyFunctional for PcSaftFunctional {
    type Contribution<'a> = PcSaftFunctionalContribution<'a>;

    fn contributions<'a>(&'a self) -> Vec<PcSaftFunctionalContribution<'a>> {
        let mut contributions = Vec::with_capacity(4);

        let assoc = AssociationFunctional::new(
            &self.params,
            &self.association_parameters,
            self.options.max_iter_cross_assoc,
            self.options.tol_cross_assoc,
        );

        if matches!(
            self.fmt_version,
            FMTVersion::WhiteBear | FMTVersion::AntiSymWhiteBear
        ) && self.params.m.len() == 1
        {
            let fmt_assoc = PureFMTAssocFunctional::new(&self.params, assoc, self.fmt_version);
            contributions.push(fmt_assoc.into());
            if self.params.m.iter().any(|&mi| mi > 1.0) {
                let chain = PureChainFunctional::new(&self.params);
                contributions.push(chain.into());
            }
            let att = PureAttFunctional::new(&self.params);
            contributions.push(att.into());
        } else {
            // Hard sphere contribution
            let hs = FMTContribution::new(&self.params, self.fmt_version);
            contributions.push(hs.into());

            // Hard chains
            if self.params.m.iter().any(|&mi| !mi.is_one()) {
                let chain = ChainFunctional::new(&self.params);
                contributions.push(chain.into());
            }

            // Dispersion
            let att = AttractiveFunctional::new(&self.params);
            contributions.push(att.into());

            // Association
            if let Some(assoc) = assoc {
                contributions.push(assoc.into());
            }
        }
        contributions
    }

    fn molecule_shape(&self) -> MoleculeShape {
        MoleculeShape::NonSpherical(&self.params.m)
    }
}

impl Molarweight for PcSaftFunctional {
    fn molar_weight(&self) -> &MolarWeight<Array1<f64>> {
        &self.parameters.molar_weight
    }
}

impl FluidParameters for PcSaftFunctional {
    fn epsilon_k_ff(&self) -> Array1<f64> {
        self.params.epsilon_k.clone()
    }

    fn sigma_ff(&self) -> &Array1<f64> {
        &self.params.sigma
    }
}

impl PairPotential for PcSaftFunctional {
    fn pair_potential(&self, i: usize, r: &Array1<f64>, _: f64) -> Array2<f64> {
        let sigma_ij = &self.params.sigma_ij;
        let eps_ij_4 = 4.0 * &self.params.epsilon_k_ij;
        Array2::from_shape_fn((self.params.m.len(), r.len()), |(j, k)| {
            let att = (sigma_ij[[i, j]] / r[k]).powi(6);
            eps_ij_4[[i, j]] * att * (att - 1.0)
        })
    }
}

/// Individual contributions for the PC-SAFT Helmholtz energy functional.
#[derive(FunctionalContribution)]
#[expect(private_interfaces)]
pub enum PcSaftFunctionalContribution<'a> {
    PureFMTAssoc(PureFMTAssocFunctional<'a>),
    PureChain(PureChainFunctional<'a>),
    PureAtt(PureAttFunctional<'a>),
    Fmt(FMTContribution<'a, PcSaftPars>),
    Chain(ChainFunctional<'a>),
    Attractive(AttractiveFunctional<'a>),
    Association(AssociationFunctional<'a, PcSaftPars>),
}
