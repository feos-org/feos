use crate::EosError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

mod equation_of_state;
pub mod parameter;
mod phase_equilibria;
mod state;

impl From<EosError> for PyErr {
    fn from(e: EosError) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }
}
