#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::GcPcSaftFunctional;
use crate::hard_sphere::FMTFunctional;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::PcSaftFunctional;
#[cfg(feature = "pets")]
use crate::pets::PetsFunctional;
#[cfg(feature = "saftvrqmie")]
use crate::saftvrqmie::SaftVRQMieFunctional;
use feos_core::si::MolarWeight;
use feos_core::*;
use feos_derive::{Components, HelmholtzEnergyFunctional};
use feos_dft::adsorption::*;
use feos_dft::solvation::*;
use feos_dft::*;
use ndarray::{Array1, Array2};
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
