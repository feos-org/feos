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
//! # use feos_core::FeosError;
//! use feos::uvtheory::{Perturbation, UVTheory, UVTheoryOptions, UVTheoryParameters, UVTheoryRecord};
//! use std::sync::Arc;
//!
//! let params = UVTheoryRecord::new(24.0, 7.0, 3.0, 150.0);
//!
//! let default_options = UVTheoryOptions {
//!     max_eta: 0.5,
//!     perturbation: Perturbation::WeeksChandlerAndersen,
//! };
//! // Define equation of state.
//! let uv_wca = Arc::new(UVTheory::new(UVTheoryParameters::from_model_records(vec![params])?));
//! // this is identical to above
//! let uv_wca = Arc::new(
//!     UVTheory::with_options(UVTheoryParameters::from_model_records(vec![params])?, default_options)
//! );
//!
//! // use Barker-Henderson perturbation
//! let options = UVTheoryOptions {
//!     max_eta: 0.5,
//!     perturbation: Perturbation::BarkerHenderson,
//! };
//! let uv_bh = Arc::new(
//!     UVTheory::with_options(UVTheoryParameters::from_model_records(vec![params])?, options)
//! );
//! # Ok::<(), FeosError>(())
//! ```
//!
//! ## uv-B3-theory
//!
//! - utilizing third virial coefficients for pure fluids with attractive exponent of 6 and Weeks-Chandler-Andersen perturbation. Manuscript submitted.
//!
#![cfg_attr(not(feature = "uvtheory"), doc = "```ignore")]
#![cfg_attr(feature = "uvtheory", doc = "```")]
//! # use feos_core::FeosError;
//! use feos::uvtheory::{Perturbation, UVTheory, UVTheoryOptions, UVTheoryParameters, UVTheoryRecord};
//! use std::sync::Arc;
//!
//! let params = UVTheoryRecord::new(24.0, 6.0, 3.0, 150.0);
//!
//! let parameters = UVTheoryParameters::from_model_records(vec![params])?;
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
//! # Ok::<(), FeosError>(())
//! ```
mod eos;
mod parameters;

pub use eos::{
    BarkerHenderson, Perturbation, UVTheory, UVTheoryOptions, WeeksChandlerAndersen,
    WeeksChandlerAndersenB3,
};
pub use parameters::{UVTheoryParameters, UVTheoryRecord};
