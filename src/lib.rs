//! FeOs - An open-source framework for equations of state and classical functional density theory.
//!
//! # Example: critical point of a pure substance using PC-SAFT
//!
//! ```ignore
//! # use feos_core::EosError;
//! use feos::pcsaft::{PcSaft, PcSaftParameters};
//! use feos_core::parameter::{IdentifierOption, Parameter};
//! use feos_core::{Contributions, State};
//! use quantity::si::KELVIN;
//! use std::sync::Arc;
//!
//! // Read parameters from json file.
//! let parameters = PcSaftParameters::from_json(
//!     vec!["propane"],
//!     "tests/pcsaft/test_parameters.json",
//!     None,
//!     IdentifierOption::Name,
//! )?;
//!
//! // Define equation of state.
//! let saft = Arc::new(PcSaft::new(Arc::new(parameters)));
//!
//! // Define thermodynamic conditions.
//! let critical_point = State::critical_point(&saft, None, None, Default::default())?;
//!
//! // Compute properties.
//! let p = critical_point.pressure(Contributions::Total);
//! let t = critical_point.temperature;
//! println!("Critical point: T={}, p={}.", t, p);
//! # Ok::<(), EosError>(())
//! ```

#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]
#[cfg(feature = "dft")]
mod dft;
#[cfg(feature = "dft")]
pub use dft::FunctionalVariant;
mod eos;
pub use eos::EosVariant;

#[cfg(feature = "estimator")]
pub mod estimator;

#[cfg(feature = "association")]
pub mod association;
pub mod hard_sphere;

// models
#[cfg(feature = "gc_pcsaft")]
pub mod gc_pcsaft;
#[cfg(feature = "pcsaft")]
pub mod pcsaft;
#[cfg(feature = "pets")]
pub mod pets;
#[cfg(feature = "uvtheory")]
pub mod uvtheory;

#[cfg(feature = "python")]
mod python;
