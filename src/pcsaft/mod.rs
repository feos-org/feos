//! Perturbed-Chain Statistical Associating Fluid Theory (PC-SAFT)
//!
//! [Gross et al. (2001)](https://doi.org/10.1021/ie0003887)
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "dft")]
mod dft;
mod eos;
pub(crate) mod parameters;

#[cfg(feature = "dft")]
pub use dft::PcSaftFunctional;
pub use eos::{DQVariants, PcSaft, PcSaftOptions};
pub use parameters::{PcSaftBinaryRecord, PcSaftParameters, PcSaftRecord};

#[cfg(feature = "python")]
pub mod python;
