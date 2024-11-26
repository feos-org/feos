//! FeOs - An open-source framework for equations of state and classical functional density theory.
//!
//! # Example: critical point of a pure substance using PC-SAFT
//!
#![cfg_attr(not(feature = "pcsaft"), doc = "```ignore")]
#![cfg_attr(feature = "pcsaft", doc = "```")]
//! # use feos_core::EosError;
//! use feos::pcsaft::{PcSaft, PcSaftParameters};
//! use feos_core::parameter::{IdentifierOption, Parameter};
//! use feos_core::{Contributions, State};
//! use quantity::KELVIN;
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
#![warn(clippy::allow_attributes)]

mod allocator;
#[cfg(feature = "dft")]
mod functional;
#[cfg(feature = "dft")]
pub use functional::FunctionalVariant;
mod eos;
pub use eos::ResidualModel;

#[cfg(feature = "estimator")]
pub mod estimator;

#[cfg(feature = "association")]
pub mod association;
pub mod hard_sphere;

// models
#[cfg(feature = "epcsaft")]
pub mod epcsaft;
#[cfg(feature = "gc_pcsaft")]
pub mod gc_pcsaft;
#[cfg(feature = "pcsaft")]
pub mod pcsaft;
#[cfg(feature = "pets")]
pub mod pets;
#[cfg(feature = "saftvrmie")]
pub mod saftvrmie;
#[cfg(feature = "saftvrqmie")]
pub mod saftvrqmie;
#[cfg(feature = "uvtheory")]
pub mod uvtheory;

pub mod ideal_gas;

#[cfg(feature = "python")]
mod python;

pub mod core {
    //! Re-export of all functionalities in [feos_core].
    pub use feos_core::*;
}

#[cfg(feature = "dft")]
pub mod dft {
    //! Re-export of all functionalities in [feos_dft].
    pub use feos_dft::*;
}
