//! uv-theory equation of state
//!
//! [van Westen et al. (2021)](https://doi.org/10.1063/5.0073572)
mod eos;
mod parameters;

pub use eos::{Perturbation, UVTheory, UVTheoryOptions};
pub use parameters::{UVBinaryRecord, UVParameters, UVRecord};

#[cfg(feature = "python")]
pub mod python;
