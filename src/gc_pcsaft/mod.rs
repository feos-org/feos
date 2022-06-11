#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "dft")]
mod dft;
mod eos;
#[cfg(feature = "micelles")]
pub mod micelles;
mod record;
#[cfg(feature = "dft")]
pub use dft::{GcPcSaftFunctional, GcPcSaftFunctionalParameters};
pub use eos::{GcPcSaft, GcPcSaftChemicalRecord, GcPcSaftEosParameters, GcPcSaftOptions};
pub use record::GcPcSaftRecord;

#[cfg(feature = "python")]
pub mod python;
