use feos_core::cubic::PengRobinson;
use feos_core::*;
#[cfg(feature = "dft")]
use feos_derive::{FunctionalContribution, HelmholtzEnergyFunctionalDyn};
use feos_derive::{ResidualDyn, Subset};
#[cfg(feature = "dft")]
use feos_dft::{FunctionalContribution, HelmholtzEnergyFunctionalDyn};
use nalgebra::DVector;
#[cfg(feature = "dft")]
use ndarray::Array1;
use num_dual::DualNum;
use quantity::*;

/// Collection of different [Residual] implementations.
///
/// Particularly relevant for situations in which generic types
/// are undesirable (e.g. FFI).
#[cfg_attr(feature = "dft", derive(HelmholtzEnergyFunctionalDyn))]
#[derive(ResidualDyn, Subset)]
#[expect(clippy::large_enum_variant)]
pub enum ResidualModel {
    // Equations of state
    NoResidual(NoResidual),
    #[cfg(feature = "pcsaft")]
    #[implement(molar_weight)]
    PcSaft(feos::pcsaft::PcSaft),

    #[cfg(feature = "epcsaft")]
    #[implement(molar_weight)]
    ElectrolytePcSaft(feos::epcsaft::ElectrolytePcSaft),

    #[cfg(feature = "gc_pcsaft")]
    #[implement(molar_weight)]
    GcPcSaft(feos::gc_pcsaft::GcPcSaft),

    #[implement(molar_weight)]
    PengRobinson(PengRobinson),

    #[implement(molar_weight)]
    Python(crate::user_defined::PyResidual),

    #[cfg(feature = "saftvrqmie")]
    #[implement(molar_weight)]
    SaftVRQMie(feos::saftvrqmie::SaftVRQMie),

    #[cfg(feature = "saftvrmie")]
    #[implement(molar_weight)]
    SaftVRMie(feos::saftvrmie::SaftVRMie),

    #[cfg(feature = "pets")]
    #[implement(molar_weight)]
    Pets(feos::pets::Pets),

    #[cfg(feature = "uvtheory")]
    #[implement(molar_weight)]
    UVTheory(feos::uvtheory::UVTheory),

    // Helmholtz energy functionals
    #[cfg(all(feature = "dft", feature = "pcsaft"))]
    #[implement(molar_weight, functional, fluid_parameters, pair_potential)]
    PcSaftFunctional(feos::pcsaft::PcSaftFunctional),

    #[cfg(all(feature = "dft", feature = "gc_pcsaft"))]
    #[implement(molar_weight, functional, fluid_parameters, bond_lengths)]
    GcPcSaftFunctional(feos::gc_pcsaft::GcPcSaftFunctional),

    #[cfg(all(feature = "dft", feature = "pets"))]
    #[implement(molar_weight, functional, fluid_parameters, pair_potential)]
    PetsFunctional(feos::pets::Pets),

    #[cfg(feature = "dft")]
    #[implement(functional, fluid_parameters, pair_potential)]
    FmtFunctional(feos::hard_sphere::FMTFunctional),

    #[cfg(all(feature = "dft", feature = "saftvrqmie"))]
    #[implement(molar_weight, functional, fluid_parameters, pair_potential)]
    SaftVRQMieFunctional(feos::saftvrqmie::SaftVRQMie),
}

#[cfg(feature = "dft")]
#[derive(FunctionalContribution)]
pub enum FunctionalContributionVariant<'a> {
    #[cfg(feature = "pcsaft")]
    PcSaftFunctional(feos::pcsaft::PcSaftFunctionalContribution<'a>),
    #[cfg(feature = "gc_pcsaft")]
    GcPcSaftFunctional(feos::gc_pcsaft::GcPcSaftFunctionalContribution<'a>),
    #[cfg(feature = "pets")]
    PetsFunctional(feos::pets::PetsFunctionalContribution<'a>),
    Fmt(feos::hard_sphere::FMTContribution<'a, feos::hard_sphere::HardSphereParameters>),
    #[cfg(feature = "saftvrqmie")]
    SaftVRQMieFunctional(feos::saftvrqmie::SaftVRQMieFunctionalContribution<'a>),
}
