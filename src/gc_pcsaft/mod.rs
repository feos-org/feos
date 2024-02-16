//! Heterosegmented Group Contribution PC-SAFT
//!
//! - [Gross et al. (2003)](https://doi.org/10.1021/ie020509y)
//! - [Sauer et al. (2014)](https://doi.org/10.1021/ie502203w)
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "dft")]
mod dft;
pub(crate) mod eos;
#[cfg(feature = "micelles")]
pub mod micelles;
mod record;
#[cfg(feature = "dft")]
pub use dft::{GcPcSaftFunctional, GcPcSaftFunctionalContribution, GcPcSaftFunctionalParameters};
pub use eos::{GcPcSaft, GcPcSaftChemicalRecord, GcPcSaftEosParameters, GcPcSaftOptions};
pub use record::GcPcSaftRecord;

#[cfg(feature = "python")]
pub mod python;
