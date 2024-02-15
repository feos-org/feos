//! uv-theory for fluids interacting with a Mie potential.
//!
//! # Implementations
//!
//! ## uv-theory
//!
//! [van Westen et al. (2021)](https://doi.org/10.1063/5.0073572): utilizing second virial coeffients and Barker-Henderson or Weeks-Chandler-Andersen perturbation.
//!
#![cfg_attr(not(feature = "uvtheory"), doc = "```ignore")]
#![cfg_attr(feature = "uvtheory", doc = "```")]
//! # use feos_core::EosError;
//! use feos::uvtheory::{Perturbation, UVTheory, UVTheoryOptions, UVTheoryParameters};
//! use std::sync::Arc;
//!
//! let parameters = Arc::new(
//!     UVTheoryParameters::new_simple(24.0, 7.0, 3.0, 150.0)?
//! );
//!
//! let default_options = UVTheoryOptions {
//!     max_eta: 0.5,
//!     perturbation: Perturbation::WeeksChandlerAndersen,
//! };
//! // Define equation of state.
//! let uv_wca = Arc::new(UVTheory::new(parameters.clone()));
//! // this is identical to above
//! let uv_wca = Arc::new(
//!     UVTheory::with_options(parameters.clone(), default_options)
//! );
//!
//! // use Barker-Henderson perturbation
//! let options = UVTheoryOptions {
//!     max_eta: 0.5,
//!     perturbation: Perturbation::BarkerHenderson,
//! };
//! let uv_bh = Arc::new(
//!     UVTheory::with_options(parameters, options)
//! );
//! # Ok::<(), EosError>(())
//! ```
//!
//! ## uv-B3-theory
//!
//! - utilizing third virial coefficients for pure fluids with attractive exponent of 6 and Weeks-Chandler-Andersen perturbation. Manuscript submitted.
//!
#![cfg_attr(not(feature = "uvtheory"), doc = "```ignore")]
#![cfg_attr(feature = "uvtheory", doc = "```")]
//! # use feos_core::EosError;
//! use feos::uvtheory::{Perturbation, UVTheory, UVTheoryOptions, UVTheoryParameters};
//! use std::sync::Arc;
//!
//! let parameters = Arc::new(
//!     UVTheoryParameters::new_simple(24.0, 6.0, 3.0, 150.0)?
//! );
//!
//! // use uv-B3-theory
//! let options = UVTheoryOptions {
//!     max_eta: 0.5,
//!     perturbation: Perturbation::WeeksChandlerAndersenB3,
//! };
//! // Define equation of state.
//! let uv_b3 = Arc::new(
//!     UVTheory::with_options(parameters, options)
//! );
//! # Ok::<(), EosError>(())
//! ```
mod eos;
mod parameters;

pub use eos::{Perturbation, UVTheory, UVTheoryOptions};
pub use parameters::{UVTheoryBinaryRecord, UVTheoryParameters, UVTheoryRecord};

#[cfg(feature = "python")]
pub mod python;
