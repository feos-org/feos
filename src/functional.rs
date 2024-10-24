#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::{GcPcSaftFunctional, GcPcSaftFunctionalContribution};
use crate::hard_sphere::{FMTContribution, FMTFunctional, HardSphereParameters};
#[cfg(feature = "pcsaft")]
use crate::pcsaft::{PcSaftFunctional, PcSaftFunctionalContribution};
#[cfg(feature = "pets")]
use crate::pets::{PetsFunctional, PetsFunctionalContribution};
#[cfg(feature = "saftvrqmie")]
use crate::saftvrqmie::{SaftVRQMieFunctional, SaftVRQMieFunctionalContribution};
use quantity::MolarWeight;
use feos_core::*;
use feos_derive::FunctionalContribution;
use feos_derive::{Components, HelmholtzEnergyFunctional};
use feos_dft::adsorption::*;
use feos_dft::solvation::*;
use feos_dft::*;
use ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use num_dual::DualNum;
use petgraph::graph::UnGraph;
use petgraph::Graph;

/// Collection of different [HelmholtzEnergyFunctional] implementations.
///
/// Particularly relevant for situations in which generic types
/// are undesirable (e.g. FFI).
#[derive(Components, HelmholtzEnergyFunctional)]
pub enum FunctionalVariant {
    #[cfg(feature = "pcsaft")]
    #[implement(fluid_parameters, molar_weight, pair_potential)]
    PcSaft(PcSaftFunctional),
    #[cfg(feature = "gc_pcsaft")]
    #[implement(fluid_parameters, molar_weight, bond_lengths)]
    GcPcSaft(GcPcSaftFunctional),
    #[cfg(feature = "pets")]
    #[implement(fluid_parameters, molar_weight, pair_potential)]
    Pets(PetsFunctional),
    #[implement(fluid_parameters, pair_potential)]
    Fmt(FMTFunctional),
    #[cfg(feature = "saftvrqmie")]
    #[implement(fluid_parameters, molar_weight, pair_potential)]
    SaftVRQMie(SaftVRQMieFunctional),
}

#[derive(FunctionalContribution)]
pub enum FunctionalContributionVariant {
    #[cfg(feature = "pcsaft")]
    PcSaftFunctional(PcSaftFunctionalContribution),
    #[cfg(feature = "gc_pcsaft")]
    GcPcSaftFunctional(GcPcSaftFunctionalContribution),
    #[cfg(feature = "pets")]
    PetsFunctional(PetsFunctionalContribution),
    Fmt(FMTContribution<HardSphereParameters>),
    #[cfg(feature = "saftvrqmie")]
    SaftVRQMieFunctional(SaftVRQMieFunctionalContribution),
}
