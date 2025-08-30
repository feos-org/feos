use super::Pets;
use super::eos::PetsOptions;
use super::parameters::PetsParameters;
use crate::hard_sphere::{FMTContribution, FMTVersion};
use dispersion::AttractiveFunctional;
use feos_core::FeosResult;
use feos_derive::FunctionalContribution;
use feos_dft::adsorption::FluidParameters;
use feos_dft::solvation::PairPotential;
use feos_dft::{FunctionalContribution, HelmholtzEnergyFunctionalDyn, MoleculeShape};
use nalgebra::DVector;
use ndarray::{Array1, Array2};
use num_dual::DualNum;
use pure_pets_functional::*;

mod dispersion;
mod pure_pets_functional;

impl Pets {
    /// PeTS model with default options and provided FMT version.
    pub fn with_fmt_version(parameters: PetsParameters, fmt_version: FMTVersion) -> Self {
        let options = PetsOptions {
            fmt_version,
            ..Default::default()
        };
        Self::with_options(parameters, options)
    }
}

impl HelmholtzEnergyFunctionalDyn for Pets {
    type Contribution<'a> = PetsFunctionalContribution<'a>;

    fn molecule_shape(&self) -> MoleculeShape<'_> {
        MoleculeShape::Spherical(self.sigma.len())
    }

    fn contributions<'a>(&'a self) -> impl Iterator<Item = PetsFunctionalContribution<'a>> {
        let mut contributions = Vec::with_capacity(2);

        if matches!(
            self.options.fmt_version,
            FMTVersion::WhiteBear | FMTVersion::AntiSymWhiteBear
        ) && self.sigma.len() == 1
        // Pure substance or mixture
        {
            // Hard-sphere contribution pure substance
            let fmt = PureFMTFunctional::new(self, self.options.fmt_version);
            contributions.push(fmt.into());

            // Dispersion contribution pure substance
            let att = PureAttFunctional::new(self);
            contributions.push(att.into());
        } else {
            // Hard-sphere contribution mixtures
            let hs = FMTContribution::new(self, self.options.fmt_version);
            contributions.push(hs.into());

            // Dispersion contribution mixtures
            let att = AttractiveFunctional::new(self);
            contributions.push(att.into());
        }

        contributions.into_iter()
    }
}

impl FluidParameters for Pets {
    fn epsilon_k_ff(&self) -> DVector<f64> {
        self.epsilon_k.clone()
    }

    fn sigma_ff(&self) -> DVector<f64> {
        self.sigma.clone()
    }
}

impl PairPotential for Pets {
    fn pair_potential(&self, i: usize, r: &Array1<f64>, _: f64) -> Array2<f64> {
        let eps_ij_4 = 4.0 * self.epsilon_k_ij.clone();
        let shift_ij = &eps_ij_4 * (2.5.powi(-12) - 2.5.powi(-6));
        let rc_ij = 2.5 * &self.sigma_ij;
        Array2::from_shape_fn((self.sigma.len(), r.len()), |(j, k)| {
            if r[k] > rc_ij[(i, j)] {
                0.0
            } else {
                let att = (self.sigma_ij[(i, j)] / r[k]).powi(6);
                eps_ij_4[(i, j)] * att * (att - 1.0) - shift_ij[(i, j)]
            }
        })
    }
}

#[derive(FunctionalContribution)]
pub enum PetsFunctionalContribution<'a> {
    PureFMT(PureFMTFunctional<'a>),
    PureAtt(PureAttFunctional<'a>),
    Fmt(FMTContribution<'a, Pets>),
    Attractive(AttractiveFunctional<'a>),
}
