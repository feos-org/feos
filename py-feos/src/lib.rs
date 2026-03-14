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

#[cfg(feature = "rayon")]
mod rayon_features {
    use pyo3::exceptions::{PyRuntimeError, PyUserWarning};
    use pyo3::prelude::*;
    use std::ffi::CString;

    /// Reads the `FEOS_MAX_THREADS` environment variable and, if present,
    /// initializes the global Rayon thread pool with that many threads.
    /// Called automatically at module import time.
    pub fn rayon_threads_from_env() {
        if let Some(n) = std::env::var("FEOS_MAX_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global();
        }
    }

    #[pyfunction]
    /// Set the number of threads used for any parallel calculations.
    ///
    /// Must be called before any parallel computation is performed and
    /// before the `FEOS_MAX_THREADS` environment variable takes effect.
    /// If the thread pool has already been initialized — either
    /// because `FEOS_MAX_THREADS` was set at import time or because a
    /// parallel function has already run — this call has no effect and
    /// a warning is emitted.
    ///
    /// Args:
    ///     n (int): Number of threads. Pass `0` to use the default
    ///         (number of logical CPUs).
    ///
    /// Example:
    ///     >>> import feos
    ///     >>> feos.set_num_threads(4)
    pub fn set_num_threads(py: Python<'_>, n: usize) -> PyResult<()> {
        match rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
        {
            Ok(_) => Ok(()),
            Err(_) => {
                // build useful warning
                let current = rayon::current_num_threads();
                let reason = if std::env::var("FEOS_MAX_THREADS").is_ok() {
                    format!(
                        "FEOS_MAX_THREADS is set. \
                        The thread pool was already initialized with {} thread(s) \
                        (probably configured at import time). \
                        To change this, set FEOS_MAX_THREADS before starting Python.",
                        current
                    )
                } else {
                    format!(
                        "The thread pool was already initialized with {} thread(s) \
                        Call set_num_threads() before any parallel work or set \
                        FEOS_MAX_THREADS before starting Python.",
                        current
                    )
                };

                let msg =
                    CString::new(format!("set_num_threads({}) without effect: {}", n, reason))
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                PyErr::warn(py, &py.get_type::<PyUserWarning>(), &msg, 1)
            }
        }
    }

    #[pyfunction]
    /// Return the number of threads in the thread pool.
    ///
    /// If the thread pool has not yet been initialized, calling this
    /// function will trigger initialization with the default
    /// (number of logical CPUs), making any subsequent call to
    /// `set_num_threads()` ineffective.
    ///
    /// Returns:
    ///     int: Number of threads currently configured.
    ///
    /// Example:
    ///     >>> import feos
    ///     >>> feos.get_num_threads()
    ///     8
    pub fn get_num_threads() -> usize {
        rayon::current_num_threads()
    }
}

#[pymodule]
fn feos(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    #[cfg(feature = "rayon")]
    {
        rayon_features::rayon_threads_from_env();
        m.add_function(wrap_pyfunction!(rayon_features::set_num_threads, m)?)?;
        m.add_function(wrap_pyfunction!(rayon_features::get_num_threads, m)?)?;
    }

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

    // // Estimator
    // m.add_class::<estimator::PyDataSet>()?;
    // m.add_class::<estimator::PyEstimator>()?;
    // m.add_class::<estimator::PyLoss>()?;
    // m.add_class::<estimator::PyPhase>()?;

    // AD
    #[cfg(feature = "ad")]
    {
        m.add_function(wrap_pyfunction!(ad::vapor_pressure_derivatives, m)?)?;
        m.add_function(wrap_pyfunction!(ad::liquid_density_derivatives, m)?)?;
        m.add_function(wrap_pyfunction!(
            ad::equilibrium_liquid_density_derivatives,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(ad::bubble_point_pressure_derivatives, m)?)?;
        m.add_function(wrap_pyfunction!(ad::dew_point_pressure_derivatives, m)?)?;
        m.add_class::<ad::PyEquationOfStateAD>()?;
    }

    #[cfg(feature = "dft")]
    {
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
