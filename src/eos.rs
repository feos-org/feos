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
use feos_core::joback::Joback;
#[cfg(feature = "python")]
use feos_core::python::user_defined::{PyIdealGas, PyResidual};
use feos_core::*;
use feos_derive::{Components, IdealGas, Residual};
use ndarray::Array1;
use num_dual::DualNum;
use quantity::si::*;
use std::fmt;

/// Collection of different [EquationOfState] implementations.
///
/// Particularly relevant for situations in which generic types
/// are undesirable (e.g. FFI).
#[derive(Components, Residual)]
pub enum ResidualModel {
    #[cfg(feature = "pcsaft")]
    #[implement(entropy_scaling, molar_weight)]
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

#[derive(Components, IdealGas)]
pub enum IdealGasModel {
    NoModel(usize),
    Joback(Joback),
    #[cfg(feature = "python")]
    Python(PyIdealGas),
}
