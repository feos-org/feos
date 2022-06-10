use crate::EosError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;

pub mod cubic;
mod equation_of_state;
pub mod joback;
pub mod parameter;
mod phase_equilibria;
mod state;
pub mod statehd;
pub mod user_defined;

impl From<EosError> for PyErr {
    fn from(e: EosError) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }
}
