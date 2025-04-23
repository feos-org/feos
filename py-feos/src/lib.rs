use feos_core::Verbosity;
use pyo3::prelude::*;

pub(crate) mod eos;
pub(crate) mod error;
pub(crate) mod ideal_gas;
pub(crate) mod parameter;
pub(crate) mod residual;
pub(crate) mod state;
pub(crate) mod user_defined;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(name = "Verbosity", eq, eq_int)]
pub(crate) enum PyVerbosity {
    /// Do not print output.
    None,
    /// Print information about the success of failure of the iteration.
    Result,
    /// Print a detailed outpur for every iteration.
    Iter,
}

impl From<Verbosity> for PyVerbosity {
    fn from(value: Verbosity) -> Self {
        use Verbosity::*;
        match value {
            None => Self::None,
            Result => Self::Result,
            Iter => Self::Iter,
        }
    }
}

impl From<PyVerbosity> for Verbosity {
    fn from(value: PyVerbosity) -> Self {
        use PyVerbosity::*;
        match value {
            None => Self::None,
            Result => Self::Result,
            Iter => Self::Iter,
        }
    }
}

#[pymodule]
fn _feos(m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
