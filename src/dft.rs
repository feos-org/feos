#[cfg(feature = "pcsaft")]
use crate::gc_pcsaft::GcPcSaftFunctional;
#[cfg(feature = "gc_pcsaft")]
use crate::hard_sphere::FMTFunctional;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::PcSaftFunctional;
#[cfg(feature = "pets")]
use crate::pets::PetsFunctional;
use feos_core::*;
use feos_dft::adsorption::*;
use feos_dft::solvation::*;
use feos_dft::*;
use ndarray::{Array1, Array2};
#[cfg(feature = "gc_pcsaft")]
use petgraph::graph::UnGraph;
#[cfg(feature = "gc_pcsaft")]
use petgraph::Graph;
#[cfg(feature = "estimator")]
use quantity::si::*;

pub enum FunctionalVariant {
    #[cfg(feature = "pcsaft")]
    PcSaft(PcSaftFunctional),
    #[cfg(feature = "gc_pcsaft")]
    GcPcSaft(GcPcSaftFunctional),
    #[cfg(feature = "pets")]
    Pets(PetsFunctional),
    Fmt(FMTFunctional),
}

#[cfg(feature = "pcsaft")]
impl From<PcSaftFunctional> for FunctionalVariant {
    fn from(f: PcSaftFunctional) -> Self {
        Self::PcSaft(f)
    }
}

#[cfg(feature = "gc_pcsaft")]
impl From<GcPcSaftFunctional> for FunctionalVariant {
    fn from(f: GcPcSaftFunctional) -> Self {
        Self::GcPcSaft(f)
    }
}

#[cfg(feature = "pets")]
impl From<PetsFunctional> for FunctionalVariant {
    fn from(f: PetsFunctional) -> Self {
        Self::Pets(f)
    }
}

impl From<FMTFunctional> for FunctionalVariant {
    fn from(f: FMTFunctional) -> Self {
        Self::Fmt(f)
    }
}

impl HelmholtzEnergyFunctional for FunctionalVariant {
    fn subset(&self, component_list: &[usize]) -> DFT<Self> {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.subset(component_list).into(),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(functional) => functional.subset(component_list).into(),
            #[cfg(feature = "pets")]
            FunctionalVariant::Pets(functional) => functional.subset(component_list).into(),
            FunctionalVariant::Fmt(functional) => functional.subset(component_list).into(),
        }
    }

    fn molecule_shape(&self) -> MoleculeShape {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.molecule_shape(),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(functional) => functional.molecule_shape(),
            #[cfg(feature = "pets")]
            FunctionalVariant::Pets(functional) => functional.molecule_shape(),
            FunctionalVariant::Fmt(functional) => functional.molecule_shape(),
        }
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.compute_max_density(moles),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(functional) => functional.compute_max_density(moles),
            #[cfg(feature = "pets")]
            FunctionalVariant::Pets(functional) => functional.compute_max_density(moles),
            FunctionalVariant::Fmt(functional) => functional.compute_max_density(moles),
        }
    }

    fn contributions(&self) -> &[Box<dyn FunctionalContribution>] {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.contributions(),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(functional) => functional.contributions(),
            #[cfg(feature = "pets")]
            FunctionalVariant::Pets(functional) => functional.contributions(),
            FunctionalVariant::Fmt(functional) => functional.contributions(),
        }
    }

    fn ideal_gas(&self) -> &dyn IdealGasContribution {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.ideal_gas(),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(functional) => functional.ideal_gas(),
            #[cfg(feature = "pets")]
            FunctionalVariant::Pets(functional) => functional.ideal_gas(),
            FunctionalVariant::Fmt(functional) => functional.ideal_gas(),
        }
    }

    #[cfg(feature = "gc_pcsaft")]
    fn bond_lengths(&self, temperature: f64) -> UnGraph<(), f64> {
        match self {
            FunctionalVariant::GcPcSaft(functional) => functional.bond_lengths(temperature),
            _ => Graph::with_capacity(0, 0),
        }
    }
}

impl MolarWeight<SIUnit> for FunctionalVariant {
    fn molar_weight(&self) -> SIArray1 {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.molar_weight(),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(functional) => functional.molar_weight(),
            #[cfg(feature = "pets")]
            FunctionalVariant::Pets(functional) => functional.molar_weight(),
            _ => unimplemented!(),
        }
    }
}

impl FluidParameters for FunctionalVariant {
    fn epsilon_k_ff(&self) -> Array1<f64> {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.epsilon_k_ff(),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(functional) => functional.epsilon_k_ff(),
            #[cfg(feature = "pets")]
            FunctionalVariant::Pets(functional) => functional.epsilon_k_ff(),
            FunctionalVariant::Fmt(functional) => functional.epsilon_k_ff(),
        }
    }

    fn sigma_ff(&self) -> &Array1<f64> {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.sigma_ff(),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(functional) => functional.sigma_ff(),
            #[cfg(feature = "pets")]
            FunctionalVariant::Pets(functional) => functional.sigma_ff(),
            FunctionalVariant::Fmt(functional) => functional.sigma_ff(),
        }
    }
}

impl PairPotential for FunctionalVariant {
    fn pair_potential(&self, i: usize, r: &Array1<f64>, temperature: f64) -> Array2<f64> {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.pair_potential(i, r, temperature),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(_) => unimplemented!(),
            #[cfg(feature = "pets")]
            FunctionalVariant::Pets(functional) => functional.pair_potential(i, r, temperature),
            FunctionalVariant::Fmt(functional) => functional.pair_potential(i, r, temperature),
        }
    }
}
