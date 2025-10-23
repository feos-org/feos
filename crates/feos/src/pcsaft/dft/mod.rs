use super::PcSaftParameters;
use super::parameters::PcSaftPars;
use crate::association::{Association, AssociationFunctional};
use crate::hard_sphere::{FMTContribution, FMTVersion};
use crate::pcsaft::eos::PcSaftOptions;
use feos_core::{FeosResult, Molarweight, ResidualDyn, StateHD, Subset};
use feos_derive::FunctionalContribution;
use feos_dft::adsorption::FluidParameters;
use feos_dft::solvation::PairPotential;
use feos_dft::{
    FunctionalContribution, HelmholtzEnergyFunctional, HelmholtzEnergyFunctionalDyn, MoleculeShape,
};
use nalgebra::DVector;
use ndarray::{Array1, Array2};
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
    pub parameters: PcSaftParameters,
    association: Option<Association>,
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
        let association = (!parameters.association.is_empty()).then(|| {
            Association::new(
                saft_options.max_iter_cross_assoc,
                saft_options.tol_cross_assoc,
            )
        });
        Self {
            parameters,
            association,
            params,
            fmt_version,
            options: saft_options,
        }
    }
}

impl Subset for PcSaftFunctional {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            self.parameters.subset(component_list),
            self.fmt_version,
            self.options,
        )
    }
}

impl ResidualDyn for PcSaftFunctional {
    fn components(&self) -> usize {
        self.parameters.pure.len()
    }

    fn compute_max_density<D: DualNum<f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        let p = &self.params;
        let msigma3 = p.m.zip_map(&p.sigma, |m, s| m * s.powi(3));
        (msigma3.map(D::from).dot(molefracs) * FRAC_PI_6).recip() * self.options.max_eta
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        self.evaluate_bulk(state)
    }
}

impl HelmholtzEnergyFunctionalDyn for PcSaftFunctional {
    type Contribution<'a>
        = PcSaftFunctionalContribution<'a>
    where
        Self: 'a;

    fn contributions<'a>(&'a self) -> impl Iterator<Item = PcSaftFunctionalContribution<'a>> {
        let mut contributions = Vec::with_capacity(4);

        let assoc = AssociationFunctional::new(&self.params, &self.parameters, self.association);

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
        contributions.into_iter()
    }

    fn molecule_shape(&self) -> MoleculeShape<'_> {
        MoleculeShape::NonSpherical(&self.params.m)
    }
}

impl Molarweight for PcSaftFunctional {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

impl FluidParameters for PcSaftFunctional {
    fn epsilon_k_ff(&self) -> DVector<f64> {
        self.params.epsilon_k.clone()
    }

    fn sigma_ff(&self) -> DVector<f64> {
        self.params.sigma.clone()
    }
}

impl PairPotential for PcSaftFunctional {
    fn pair_potential(&self, i: usize, r: &Array1<f64>, _: f64) -> Array2<f64> {
        let sigma_ij = &self.params.sigma_ij;
        let eps_ij_4 = 4.0 * &self.params.epsilon_k_ij;
        Array2::from_shape_fn((self.params.m.len(), r.len()), |(j, k)| {
            let att = (sigma_ij[(i, j)] / r[k]).powi(6);
            eps_ij_4[(i, j)] * att * (att - 1.0)
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
