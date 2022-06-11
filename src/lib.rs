#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "fit")]
pub mod fit;

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
