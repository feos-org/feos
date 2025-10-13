//! FeOs - An open-source framework for equations of state and classical functional density theory.
//!
//! # Example: critical point of a pure substance using PC-SAFT
//!
#![cfg_attr(not(feature = "pcsaft"), doc = "```ignore")]
#![cfg_attr(feature = "pcsaft", doc = "```")]
//! # use feos_core::FeosError;
//! use feos::pcsaft::{PcSaft, PcSaftParameters};
//! use feos_core::parameter::{IdentifierOption};
//! use feos_core::{Contributions, State};
//! use quantity::KELVIN;
//! use nalgebra::dvector;
//!
//! // Read parameters from json file.
//! let parameters = PcSaftParameters::from_json(
//!     vec!["propane"],
//!     "../../parameters/pcsaft/esper2023.json",
//!     None,
//!     IdentifierOption::Name,
//! )?;
//!
//! // Define equation of state.
//! let saft = PcSaft::new(parameters);
//!
//! // Define thermodynamic conditions.
//! let critical_point = State::critical_point(&&saft, Some(&dvector![1.0]), None, None, Default::default())?;
//!
//! // Compute properties.
//! let p = critical_point.pressure(Contributions::Total);
//! let t = critical_point.temperature;
//! println!("Critical point: T={}, p={}.", t, p);
//! # Ok::<(), FeosError>(())
//! ```

#![warn(clippy::all)]
#![warn(clippy::allow_attributes)]

// pub mod estimator;

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

pub mod core {
    //! Re-export of all functionalities in [feos_core].
    pub use feos_core::*;
}

#[cfg(feature = "dft")]
pub mod dft {
    //! Re-export of all functionalities in [feos_dft].
    pub use feos_dft::*;
}
