//! Collection of ideal gas models.
use crate::user_defined::PyIdealGas;
use feos::ideal_gas::{Dippr, Joback};
use feos_core::IdealGas;
use feos_derive::IdealGas;
use num_dual::DualNum;
use std::sync::Arc;

/// Collection of different [IdealGas] implementations.
///
/// Particularly relevant for situations in which generic types
/// are undesirable (e.g. FFI).
#[derive(IdealGas, Clone)]
pub enum IdealGasModel {
    NoModel,
    Joback(Joback),
    Dippr(Dippr),
    #[cfg(feature = "multiparameter")]
    MultiParameter(feos::multiparameter::MultiParameterIdealGas),
    Python(Arc<PyIdealGas>),
}
