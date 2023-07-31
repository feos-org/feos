use super::eos::PetsOptions;
use super::parameters::PetsParameters;
use crate::hard_sphere::{FMTContribution, FMTVersion};
use dispersion::AttractiveFunctional;
use feos_core::parameter::Parameter;
use feos_core::si::{MolarWeight, GRAM, MOL};
use feos_core::Components;
use feos_dft::adsorption::FluidParameters;
use feos_dft::solvation::PairPotential;
use feos_dft::{FunctionalContribution, HelmholtzEnergyFunctional, MoleculeShape, DFT};
use ndarray::{Array1, Array2};
use num_dual::DualNum;
use pure_pets_functional::*;
use std::f64::consts::FRAC_PI_6;
use std::sync::Arc;

mod dispersion;
mod pure_pets_functional;

/// PeTS Helmholtz energy functional.
pub struct PetsFunctional {
    /// PeTS parameters of all substances in the system
    pub parameters: Arc<PetsParameters>,
    fmt_version: FMTVersion,
    options: PetsOptions,
    contributions: Vec<Box<dyn FunctionalContribution>>,
}

impl PetsFunctional {
    /// PeTS functional with default options.
    ///
    /// # Defaults
    /// `FMTVersion`: `FMTVersion::WhiteBear`
    pub fn new(parameters: Arc<PetsParameters>) -> DFT<Self> {
        Self::with_options(parameters, FMTVersion::WhiteBear, PetsOptions::default())
    }

    /// PeTS functional with default options for and provided FMT version.
    #[allow(non_snake_case)]
    pub fn new_full(parameters: Arc<PetsParameters>, fmt_Version: FMTVersion) -> DFT<Self> {
        Self::with_options(parameters, fmt_Version, PetsOptions::default())
    }

    /// PeTS functional with provided options for FMT and equation of state options.
    pub fn with_options(
        parameters: Arc<PetsParameters>,
        fmt_version: FMTVersion,
        pets_options: PetsOptions,
    ) -> DFT<Self> {
        let mut contributions: Vec<Box<dyn FunctionalContribution>> = Vec::with_capacity(2);

        if matches!(
            fmt_version,
            FMTVersion::WhiteBear | FMTVersion::AntiSymWhiteBear
        ) && parameters.sigma.len() == 1
        // Pure substance or mixture
        {
            // Hard-sphere contribution pure substance
            let fmt = PureFMTFunctional::new(parameters.clone(), fmt_version);
            contributions.push(Box::new(fmt));

            // Dispersion contribution pure substance
            let att = PureAttFunctional::new(parameters.clone());
            contributions.push(Box::new(att));
        } else {
            // Hard-sphere contribution mixtures
            let hs = FMTContribution::new(&parameters, fmt_version);
            contributions.push(Box::new(hs));

            // Dispersion contribution mixtures
            let att = AttractiveFunctional::new(parameters.clone());
            contributions.push(Box::new(att));
        }

        DFT(Self {
            parameters,
            fmt_version,
            options: pets_options,
            contributions,
        })
    }
}

impl Components for PetsFunctional {
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

impl HelmholtzEnergyFunctional for PetsFunctional {
    fn molecule_shape(&self) -> MoleculeShape {
        MoleculeShape::Spherical(self.parameters.sigma.len())
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * self.parameters.sigma.mapv(|v| v.powi(3)) * moles).sum()
    }

    fn contributions(&self) -> &[Box<dyn FunctionalContribution>] {
        &self.contributions
    }

    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}

impl FluidParameters for PetsFunctional {
    fn epsilon_k_ff(&self) -> Array1<f64> {
        self.parameters.epsilon_k.clone()
    }

    fn sigma_ff(&self) -> &Array1<f64> {
        &self.parameters.sigma
    }
}

impl PairPotential for PetsFunctional {
    fn pair_potential(&self, i: usize, r: &Array1<f64>, _: f64) -> Array2<f64> {
        let eps_ij_4 = 4.0 * self.parameters.epsilon_k_ij.clone();
        let shift_ij = &eps_ij_4 * (2.5.powi(-12) - 2.5.powi(-6));
        let rc_ij = 2.5 * &self.parameters.sigma_ij;
        Array2::from_shape_fn((self.parameters.sigma.len(), r.len()), |(j, k)| {
            if r[k] > rc_ij[[i, j]] {
                0.0
            } else {
                let att = (self.parameters.sigma_ij[[i, j]] / r[k]).powi(6);
                eps_ij_4[[i, j]] * att * (att - 1.0) - shift_ij[[i, j]]
            }
        })
    }
}
