#![warn(clippy::all)]
#![warn(clippy::allow_attributes)]
use feos_core::Verbosity;
use pyo3::prelude::*;

#[cfg(feature = "dft")]
pub(crate) mod dft;
pub(crate) mod eos;
pub(crate) mod error;
pub(crate) mod estimator;
pub(crate) mod ideal_gas;
pub(crate) mod parameter;
pub(crate) mod phase_equilibria;
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
fn feos(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("version", env!("CARGO_PKG_VERSION"))?;

    // Utility
    m.add_class::<PyVerbosity>()?;

    // State & phase equilibria.
    m.add_class::<state::PyContributions>()?;
    m.add_class::<state::PyState>()?;
    m.add_class::<state::PyStateVec>()?;
    m.add_class::<phase_equilibria::PyPhaseDiagram>()?;
    m.add_class::<phase_equilibria::PyPhaseDiagramHetero>()?;
    m.add_class::<phase_equilibria::PyPhaseEquilibrium>()?;

    // Parameter
    m.add_class::<parameter::PyIdentifier>()?;
    m.add_class::<parameter::PyIdentifierOption>()?;
    m.add_class::<parameter::PyChemicalRecord>()?;
    m.add_class::<parameter::PySmartsRecord>()?;
    m.add_class::<parameter::PyPureRecord>()?;
    m.add_class::<parameter::PySegmentRecord>()?;
    m.add_class::<parameter::PyBinaryRecord>()?;
    m.add_class::<parameter::PyBinarySegmentRecord>()?;
    m.add_class::<parameter::PyParameters>()?;
    m.add_class::<parameter::PyGcParameters>()?;

    // Equation of state
    m.add_class::<eos::PyEquationOfState>()?;

    // Estimator
    m.add_class::<estimator::PyDataSet>()?;
    m.add_class::<estimator::PyEstimator>()?;
    m.add_class::<estimator::PyLoss>()?;
    m.add_class::<estimator::PyPhase>()?;

    #[cfg(not(feature = "dft"))]
    m.add("__dft__", false)?;

    #[cfg(feature = "dft")]
    {
        m.add("__dft__", true)?;
        m.add_class::<dft::PyHelmholtzEnergyFunctional>()?;
        m.add_class::<dft::PyFMTVersion>()?;
        m.add_class::<dft::PyGeometry>()?;

        // Solver
        m.add_class::<dft::PyDFTSolver>()?;
        m.add_class::<dft::PyDFTSolverLog>()?;

        // Adsorption
        m.add_class::<dft::PyAdsorption1D>()?;
        m.add_class::<dft::PyAdsorption3D>()?;
        m.add_class::<dft::PyExternalPotential>()?;
        m.add_class::<dft::PyPore1D>()?;
        m.add_class::<dft::PyPore2D>()?;
        m.add_class::<dft::PyPore3D>()?;

        // Interface
        m.add_class::<dft::PySurfaceTensionDiagram>()?;
        m.add_class::<dft::PyPlanarInterface>()?;

        // Solvation
        m.add_class::<dft::PyPairCorrelation>()?;
        m.add_class::<dft::PySolvationProfile>()?;
    }
    Ok(())
}
