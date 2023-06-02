#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::GcPcSaft;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::PcSaft;
#[cfg(feature = "pets")]
use crate::pets::Pets;
#[cfg(feature = "saftvrqmie")]
use crate::saftvrqmie::SaftVRQMie;
#[cfg(feature = "uvtheory")]
use crate::uvtheory::UVTheory;
use feos_core::cubic::PengRobinson;
#[cfg(feature = "python")]
use feos_core::python::user_defined::PyEoSObj;
use feos_core::*;
use feos_derive;
use feos_core::{DeBroglieWavelength, IdealGas, Residual};
use feos_core::joback::Joback;
use ndarray::Array1;
use quantity::si::*;
use std::fmt;

/// Collection of different [EquationOfState] implementations.
///
/// Particularly relevant for situations in which generic types
/// are undesirable (e.g. FFI).
#[derive(feos_derive::Residual)]
pub enum ResidualModel {
    #[cfg(feature = "pcsaft")]
    // #[implement(entropy_scaling, molar_weight)]
    #[implement(molar_weight)]
    PcSaft(PcSaft),
    #[cfg(feature = "gc_pcsaft")]
    #[implement(molar_weight)]
    GcPcSaft(GcPcSaft),
    #[implement(molar_weight)]
    PengRobinson(PengRobinson),
    #[cfg(feature = "python")]
    #[implement(molar_weight)]
    Python(PyResidual),
    #[cfg(feature = "saftvrqmie")]
    #[implement(molar_weight)]
    SaftVRQMie(SaftVRQMie),
    #[cfg(feature = "pets")]
    #[implement(molar_weight)]
    Pets(Pets),
    #[cfg(feature = "uvtheory")]
    UVTheory(UVTheory),
}

#[derive(feos_derive::IdealGas)]
pub enum IdealGasModel {
    Joback(Joback),
    #[cfg(feature = "python")]
    Python(PyIdealGas),
}
