//! Heterosegmented Group Contribution PC-SAFT
//!
//! - [Gross et al. (2003)](https://doi.org/10.1021/ie020509y)
//! - [Sauer et al. (2014)](https://doi.org/10.1021/ie502203w)
#![expect(unexpected_cfgs)]

#[cfg(feature = "dft")]
mod dft;
pub(crate) mod eos;
#[cfg(feature = "micelles")]
pub mod micelles;
mod record;
#[cfg(feature = "dft")]
pub use dft::{GcPcSaftFunctional, GcPcSaftFunctionalContribution, GcPcSaftFunctionalParameters};
pub use eos::{GcPcSaft, GcPcSaftAD, GcPcSaftEosParameters, GcPcSaftOptions};
pub use record::{GcPcSaftParameters, GcPcSaftRecord};
