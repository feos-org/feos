//! Statistical Associating Fluid Theory for Variable Range interactions of the generic Mie form (SAFT-VR Mie)
//!
//! [Lafitte et al. (2013)](https://doi.org/10.1063/1.4819786)
mod eos;
pub(crate) mod parameters;

pub use eos::{SaftVRMie, SaftVRMieOptions};
pub use parameters::{test_utils, SaftVRMieBinaryRecord, SaftVRMieParameters, SaftVRMieRecord};
