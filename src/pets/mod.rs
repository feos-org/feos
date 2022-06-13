#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::suspicious_operation_groupings)]

#[cfg(feature = "dft")]
mod dft;
mod eos;
mod parameters;

#[cfg(feature = "dft")]
pub use dft::PetsFunctional;
pub use eos::{Pets, PetsOptions};
pub use parameters::{PetsParameters, PetsRecord};

#[cfg(feature = "python")]
pub mod python;
