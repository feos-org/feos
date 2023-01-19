//! SAFT-VRQ Mie equation of state.
//!
//! Quantum effects are described by the first order Feynman–Hibbs corrections to Mie fluids.
//! The model accurately predicts properties for pure substances and mixtures down to 20K.
//! For mixtures, the additive hard-sphere reference contribution is extended with a non-additive correction.
//!
//! # Literature
//! - Pure substances: [Aasen et al. (2019)](https://aip.scitation.org/doi/10.1063/1.5111364)  
//! - Binary mixtures: [Aasen et al. (2020)](https://aip.scitation.org/doi/10.1063/1.5136079)
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]
#[cfg(feature = "dft")]
mod dft;
mod eos;
mod parameters;

#[cfg(feature = "dft")]
pub use dft::SaftVRQMieFunctional;
pub use eos::{FeynmanHibbsOrder, SaftVRQMie, SaftVRQMieOptions};
pub use parameters::{SaftVRQMieBinaryRecord, SaftVRQMieParameters, SaftVRQMieRecord};

#[cfg(feature = "python")]
pub mod python;
