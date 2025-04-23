//! Collection of ideal gas models.
use crate::user_defined::PyIdealGas;
use feos::ideal_gas::{Dippr, Joback};
use feos_core::{Components, IdealGas};
use feos_derive::{Components, IdealGas};
use ndarray::Array1;
use num_dual::DualNum;
use std::sync::Arc;

/// Collection of different [IdealGas] implementations.
///
/// Particularly relevant for situations in which generic types
/// are undesirable (e.g. FFI).
#[derive(Components, IdealGas)]
pub enum IdealGasModel {
    NoModel(usize),
    Joback(Arc<Joback>),
    Dippr(Arc<Dippr>),
    Python(PyIdealGas),
}
