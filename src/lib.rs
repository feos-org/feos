#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "fit")]
pub mod fit;
#[cfg(feature = "pcsaft")]
pub mod pcsaft;
#[cfg(feature = "python")]
mod python;
// mod fcsaft;
// use fcsaft::__PYO3_PYMODULE_DEF_FCSAFT;
// mod gc_pcsaft;
// use gc_pcsaft::__PYO3_PYMODULE_DEF_GC_PCSAFT;
// mod pets;
// use pets::__PYO3_PYMODULE_DEF_PETS;
// mod uvtheory;
// use uvtheory::__PYO3_PYMODULE_DEF_UVTHEORY;
