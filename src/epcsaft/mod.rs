//! Electrolyte Perturbed-Chain Statistical Associating Fluid Theory (ePC-SAFT)

mod eos;
pub(crate) mod parameters;

pub use eos::{ElectrolytePcSaft, ElectrolytePcSaftOptions, ElectrolytePcSaftVariants};
pub use parameters::{
    ElectrolytePcSaftBinaryRecord, ElectrolytePcSaftParameters, ElectrolytePcSaftRecord,
};

#[cfg(feature = "python")]
pub mod python;
