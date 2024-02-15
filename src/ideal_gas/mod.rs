//! Collection of ideal gas models.
use std::fmt;

#[cfg(feature = "python")]
use feos_core::python::user_defined::PyIdealGas;
use feos_core::{Components, IdealGas};
use feos_derive::{Components, IdealGas};
use std::sync::Arc;
use ndarray::Array1;
use num_dual::DualNum;

mod dippr;
mod joback;
pub use dippr::{Dippr, DipprRecord};
pub use joback::{Joback, JobackRecord};

/// Collection of different [IdealGas] implementations.
///
/// Particularly relevant for situations in which generic types
/// are undesirable (e.g. FFI).
#[derive(Components, IdealGas)]
pub enum IdealGasModel {
    NoModel(usize),
    Joback(Arc<Joback>),
    Dippr(Arc<Dippr>),
    #[cfg(feature = "python")]
    Python(PyIdealGas),
}
