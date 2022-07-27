//! Perturbed Truncated and Shifted (PeTS) equation of state
//! 
//! [Heier et al. (2018)](https://doi.org/10.1080/00268976.2018.1447153)
//! 
//! PeTS is an equation of state for the truncated and shifted Lennar-Jones fluid with cut-off
//! distance 2.5 $\sigma$. 
//! It utilizes a hard-sphere fluid as reference with an attactive perturbation.
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
pub use parameters::{PetsParameters, PetsRecord, PetsBinaryRecord};

#[cfg(feature = "python")]
pub mod python;
