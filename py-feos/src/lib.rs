#![warn(clippy::all)]
#![warn(clippy::allow_attributes)]
use feos_core::Verbosity;
use pyo3::prelude::*;

#[cfg(feature = "ad")]
pub(crate) mod ad;
#[cfg(feature = "dft")]
pub(crate) mod dft;
pub(crate) mod eos;
pub(crate) mod error;
// pub(crate) mod estimator;
pub(crate) mod ideal_gas;
pub(crate) mod parameter;
pub(crate) mod phase_equilibria;
pub(crate) mod residual;
pub(crate) mod state;
pub(crate) mod user_defined;

/// Output level for phase equilibrium solvers.
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
pub mod feos {
    pub const __version__: &'static str = env!("CARGO_PKG_VERSION");

    #[pymodule_export]
    use super::PyVerbosity;
    #[pymodule_export]
    use crate::phase_equilibria::{PyPhaseDiagram, PyPhaseDiagramHetero, PyPhaseEquilibrium};
    #[pymodule_export]
    use crate::state::{PyContributions, PyState, PyStateVec};

    #[pymodule_export]
    use crate::eos::PyEquationOfState;
    #[pymodule_export]
    use crate::parameter::{
        PyBinaryRecord, PyBinarySegmentRecord, PyChemicalRecord, PyGcParameters, PyIdentifier,
        PyIdentifierOption, PyParameters, PyPureRecord, PySegmentRecord, PySmartsRecord,
    };

    #[cfg(feature = "ad")]
    #[pymodule_export]
    use crate::ad::{
        bubble_point_pressure_derivatives, dew_point_pressure_derivatives,
        equilibrium_liquid_density_derivatives, liquid_density_derivatives,
        vapor_pressure_derivatives,
    };

    #[cfg(feature = "dft")]
    #[pymodule_export]
    use crate::dft::{
        // Adsorption
        PyAdsorption1D,
        PyAdsorption3D,
        // Solver
        PyDFTSolver,
        PyDFTSolverLog,

        PyExternalPotential,
        PyFMTVersion,
        PyGeometry,

        PyHelmholtzEnergyFunctional,
        // Solvation
        PyPairCorrelation,
        PyPlanarInterface,

        PyPore1D,
        PyPore2D,
        PyPore3D,

        PySolvationProfile,
        // Interface
        PySurfaceTensionDiagram,
    };
}
