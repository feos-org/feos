use feos_core::Contributions;
use pyo3::prelude::*;

mod state;
pub use state::{PyState, PyStateVec};

/// Possible contributions that can be computed.
#[derive(Clone, Copy, PartialEq)]
#[pyclass(name = "Contributions", eq, eq_int)]
pub enum PyContributions {
    /// Only compute the ideal gas contribution
    IdealGas,
    /// Only compute the difference between the total and the ideal gas contribution
    Residual,
    // /// Compute the differnce between the total and the ideal gas contribution for a (N,p,T) reference state
    // ResidualNpt,
    /// Compute ideal gas and residual contributions
    Total,
}

impl From<Contributions> for PyContributions {
    fn from(value: Contributions) -> Self {
        use Contributions::*;
        match value {
            IdealGas => Self::IdealGas,
            Residual => Self::Residual,
            Total => Self::Total,
        }
    }
}

impl From<PyContributions> for Contributions {
    fn from(value: PyContributions) -> Self {
        use PyContributions::*;
        match value {
            IdealGas => Self::IdealGas,
            Residual => Self::Residual,
            Total => Self::Total,
        }
    }
}
