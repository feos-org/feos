#[cfg(feature = "epcsaft")]
use crate::epcsaft::ElectrolytePcSaft;
#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::GcPcSaft;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::PcSaft;
#[cfg(feature = "pets")]
use crate::pets::Pets;
#[cfg(feature = "saftvrmie")]
use crate::saftvrmie::SaftVRMie;
#[cfg(feature = "saftvrqmie")]
use crate::saftvrqmie::SaftVRQMie;
#[cfg(feature = "uvtheory")]
use crate::uvtheory::UVTheory;
use feos_core::cubic::PengRobinson;
#[cfg(feature = "python")]
use feos_core::python::user_defined::PyResidual;
use feos_core::*;
use feos_derive::{Components, Residual};
use ndarray::{Array1, ScalarOperand};
use num_dual::DualNum;
use quantity::*;

/// Collection of different [Residual] implementations.
///
/// Particularly relevant for situations in which generic types
/// are undesirable (e.g. FFI).
#[derive(Components, Residual)]
pub enum ResidualModel {
    NoResidual(NoResidual),
    #[cfg(feature = "pcsaft")]
    #[implement(entropy_scaling, molar_weight)]
    PcSaft(PcSaft),
    #[cfg(feature = "epcsaft")]
    #[implement(molar_weight)]
    ElectrolytePcSaft(ElectrolytePcSaft),
    #[cfg(feature = "gc_pcsaft")]
    #[implement(molar_weight)]
    GcPcSaft(GcPcSaft),
    #[implement(molar_weight)]
    PengRobinson(PengRobinson),
    #[cfg(feature = "python")]
    #[implement(molar_weight)]
    Python(PyResidual),
    #[cfg(feature = "saftvrqmie")]
    #[implement(entropy_scaling, molar_weight)]
    SaftVRQMie(SaftVRQMie),
    #[cfg(feature = "saftvrmie")]
    #[implement(molar_weight)]
    SaftVRMie(SaftVRMie),
    #[cfg(feature = "pets")]
    #[implement(molar_weight)]
    Pets(Pets),
    #[cfg(feature = "uvtheory")]
    #[implement(molar_weight)]
    UVTheory(UVTheory),
}
