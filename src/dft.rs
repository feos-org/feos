#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::GcPcSaftFunctional;
use crate::hard_sphere::FMTFunctional;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::PcSaftFunctional;
#[cfg(feature = "pets")]
use crate::pets::PetsFunctional;
use feos_core::*;
use feos_derive::HelmholtzEnergyFunctional;
use feos_dft::adsorption::*;
use feos_dft::solvation::*;
use feos_dft::*;
use ndarray::{Array1, Array2};
use petgraph::graph::UnGraph;
use petgraph::Graph;
use quantity::si::*;

#[derive(HelmholtzEnergyFunctional)]
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
}
