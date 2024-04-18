//! Electrolyte Perturbed-Chain Statistical Associating Fluid Theory (e12PC-SAFT)
//!

#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

mod eos;
pub(crate) mod parameters;

pub use eos::{ElectrolytePcSaft, ElectrolytePcSaftOptions, ElectrolytePcSaftVariants};
pub use parameters::{
    ElectrolytePcSaftBinaryRecord, ElectrolytePcSaftParameters, ElectrolytePcSaftRecord,
};

#[cfg(feature = "python")]
pub mod python;
