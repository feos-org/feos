use feos_core::cubic::PengRobinson;
use feos_core::*;
use feos_derive::{Components, Residual};
#[cfg(feature = "dft")]
use feos_derive::{FunctionalContribution, HelmholtzEnergyFunctional};
#[cfg(feature = "dft")]
use feos_dft::{FunctionalContribution, HelmholtzEnergyFunctional};
use ndarray::{Array1, ScalarOperand};
use num_dual::DualNum;
use quantity::*;

/// Collection of different [Residual] implementations.
///
/// Particularly relevant for situations in which generic types
/// are undesirable (e.g. FFI).
#[cfg_attr(feature = "dft", derive(HelmholtzEnergyFunctional))]
#[derive(Components, Residual)]
pub enum ResidualModel {
    // Equations of state
    NoResidual(NoResidual),
    #[cfg(feature = "pcsaft")]
    #[implement(entropy_scaling, molar_weight)]
    PcSaft(crate::pcsaft::PcSaft),

    #[cfg(feature = "epcsaft")]
    #[implement(molar_weight)]
    ElectrolytePcSaft(crate::epcsaft::ElectrolytePcSaft),

    #[cfg(feature = "gc_pcsaft")]
    #[implement(molar_weight)]
    GcPcSaft(crate::gc_pcsaft::GcPcSaft),

    #[implement(molar_weight)]
    PengRobinson(PengRobinson),

    #[cfg(feature = "python")]
    #[implement(molar_weight)]
    Python(feos_core::python::user_defined::PyResidual),

    #[cfg(feature = "saftvrqmie")]
    #[implement(entropy_scaling, molar_weight)]
    SaftVRQMie(crate::saftvrqmie::SaftVRQMie),

    #[cfg(feature = "saftvrmie")]
    #[implement(molar_weight)]
    SaftVRMie(crate::saftvrmie::SaftVRMie),

    #[cfg(feature = "pets")]
    #[implement(molar_weight)]
    Pets(crate::pets::Pets),

    #[cfg(feature = "uvtheory")]
    #[implement(molar_weight)]
    UVTheory(crate::uvtheory::UVTheory),

    // Helmholtz energy functionals
    #[cfg(all(feature = "dft", feature = "pcsaft"))]
    #[implement(molar_weight, functional, fluid_parameters, pair_potential)]
    PcSaftFunctional(crate::pcsaft::PcSaftFunctional),

    #[cfg(all(feature = "dft", feature = "gc_pcsaft"))]
    #[implement(molar_weight, functional, fluid_parameters, bond_lengths)]
    GcPcSaftFunctional(crate::gc_pcsaft::GcPcSaftFunctional),

    #[cfg(all(feature = "dft", feature = "pets"))]
    #[implement(molar_weight, functional, fluid_parameters, pair_potential)]
    PetsFunctional(crate::pets::PetsFunctional),

    #[cfg(feature = "dft")]
    #[implement(functional, fluid_parameters, pair_potential)]
    FmtFunctional(crate::hard_sphere::FMTFunctional),

    #[cfg(all(feature = "dft", feature = "saftvrqmie"))]
    #[implement(molar_weight, functional, fluid_parameters, pair_potential)]
    SaftVRQMieFunctional(crate::saftvrqmie::SaftVRQMieFunctional),
}

#[cfg(feature = "dft")]
#[derive(FunctionalContribution)]
pub enum FunctionalContributionVariant {
    #[cfg(feature = "pcsaft")]
    PcSaftFunctional(crate::pcsaft::PcSaftFunctionalContribution),
    #[cfg(feature = "gc_pcsaft")]
    GcPcSaftFunctional(crate::gc_pcsaft::GcPcSaftFunctionalContribution),
    #[cfg(feature = "pets")]
    PetsFunctional(crate::pets::PetsFunctionalContribution),
    Fmt(crate::hard_sphere::FMTContribution<crate::hard_sphere::HardSphereParameters>),
    #[cfg(feature = "saftvrqmie")]
    SaftVRQMieFunctional(crate::saftvrqmie::SaftVRQMieFunctionalContribution),
}
