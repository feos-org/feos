use super::eos::PetsOptions;
use super::parameters::PetsParameters;
use dispersion::AttractiveFunctional;
use feos_core::joback::Joback;
use feos_core::parameter::Parameter;
use feos_core::{IdealGasContribution, MolarWeight};
use feos_dft::adsorption::FluidParameters;
use feos_dft::fundamental_measure_theory::{
    FMTContribution, FMTProperties, FMTVersion, MonomerShape,
};
use feos_dft::solvation::PairPotential;
use feos_dft::{FunctionalContribution, HelmholtzEnergyFunctional, MoleculeShape, DFT};
use ndarray::{Array, Array1, Array2};
use num_dual::DualNum;
use pure_pets_functional::*;
use quantity::si::*;
use std::f64::consts::FRAC_PI_6;
use std::rc::Rc;

mod dispersion;
mod pure_pets_functional;

pub struct PetsFunctional {
    pub parameters: Rc<PetsParameters>,
    fmt_version: FMTVersion,
    options: PetsOptions,
    contributions: Vec<Box<dyn FunctionalContribution>>,
    joback: Joback,
}

impl PetsFunctional {
    pub fn new(parameters: Rc<PetsParameters>) -> DFT<Self> {
        Self::with_options(parameters, FMTVersion::WhiteBear, PetsOptions::default())
    }

    #[allow(non_snake_case)]
    pub fn new_full(parameters: Rc<PetsParameters>, fmt_Version: FMTVersion) -> DFT<Self> {
        Self::with_options(parameters, fmt_Version, PetsOptions::default())
    }

    pub fn with_options(
        parameters: Rc<PetsParameters>,
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

        let joback = match &parameters.joback_records {
            Some(joback_records) => Joback::new(joback_records.clone()),
            None => Joback::default(parameters.sigma.len()),
        };

        Self {
            parameters,
            fmt_version,
            options: pets_options,
            contributions,
            joback,
        }
        .into()
    }
}

impl HelmholtzEnergyFunctional for PetsFunctional {
    fn subset(&self, component_list: &[usize]) -> DFT<Self> {
        Self::with_options(
            Rc::new(self.parameters.subset(component_list)),
            self.fmt_version,
            self.options,
        )
    }

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

    fn ideal_gas(&self) -> &dyn IdealGasContribution {
        &self.joback
    }
}

impl MolarWeight<SIUnit> for PetsFunctional {
    fn molar_weight(&self) -> SIArray1 {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}

impl FMTProperties for PetsParameters {
    fn component_index(&self) -> Array1<usize> {
        Array::from_shape_fn(self.sigma.len(), |i| i)
    }

    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<N> {
        MonomerShape::Spherical(self.sigma.len())
    }

    fn hs_diameter<D: DualNum<f64>>(&self, temperature: D) -> Array1<D> {
        self.hs_diameter(temperature)
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
        let eps_ij_4 = 4.0 * self.epsilon_k_ij;
        let shift_ij = eps_ij_4 * (2.5.powi(-12) - 2.5.powi(-6));
        let rc_ij = 2.5 * self.parameters.sigma_ij;
        Array::from_shape_fn((self.parameters.m.len(), r.len()), |(j, k)| {
            if r[k] > rc_ij[[i, j]] {
                0.0
            } else {
                let att = (self.parameters.sigma_ij[[i, j]] / r[k]).powi(6);
                eps_ij_4[[i, j]] * att * (att - 1.0) - shift_ij[[i, j]]
            }
        })
    }
}
