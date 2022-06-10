use crate::DFTSolver;
use feos_core::Verbosity;
use pyo3::prelude::*;

/// Settings for the DFT solver.
///
/// Parameters
/// ----------
/// verbosity: Verbosity, optional
///     The verbosity level of the solver.
///
/// Returns
/// -------
/// DFTSolver
#[pyclass(name = "DFTSolver")]
#[derive(Clone)]
#[pyo3(text_signature = "(verbosity=None)")]
pub struct PyDFTSolver(pub DFTSolver);

#[pymethods]
impl PyDFTSolver {
    #[new]
    fn new(verbosity: Option<Verbosity>) -> Self {
        Self(DFTSolver::new(verbosity.unwrap_or_default()))
    }

    /// The default solver.
    ///
    /// Returns
    /// -------
    /// DFTSolver
    #[classattr]
    fn default() -> Self {
        Self(DFTSolver::default())
    }

    /// Add a picard iteration to the solver object.
    ///
    /// Parameters
    /// ----------
    /// log: bool, optional
    ///     Iterate the logarithm of the density profile
    /// max_iter: int, optional
    ///     The maximum number of iterations.
    /// tol: float, optional
    ///     The tolerance.
    /// beta: float, optional
    ///     The damping factor.
    ///
    /// Returns
    /// -------
    /// DFTSolver
    #[pyo3(text_signature = "($self, log=None, max_iter=None, tol=None, beta=None)")]
    fn picard_iteration(
        &self,
        max_rel: Option<f64>,
        log: Option<bool>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        beta: Option<f64>,
    ) -> Self {
        let mut solver = self.0.clone().picard_iteration(max_rel);
        if let Some(log) = log {
            if log {
                solver = solver.log();
            }
        }
        if let Some(max_iter) = max_iter {
            solver = solver.max_iter(max_iter);
        }
        if let Some(tol) = tol {
            solver = solver.tol(tol);
        }
        if let Some(beta) = beta {
            solver = solver.beta(beta);
        }
        Self(solver)
    }

    fn log_iter(&self) -> Self {
        let mut solver = self.0.clone();
        solver.verbosity = Verbosity::Iter;
        Self(solver)
    }

    fn log_result(&self) -> Self {
        let mut solver = self.0.clone();
        solver.verbosity = Verbosity::Result;
        Self(solver)
    }

    /// Add Anderson mixing to the solver object.
    ///
    /// Parameters
    /// ----------
    /// mmax: int, optional
    ///     The maximum number of old solutions that are used.
    /// log: bool, optional
    ///     Iterate the logarithm of the density profile
    /// max_iter: int, optional
    ///     The maximum number of iterations.
    /// tol: float, optional
    ///     The tolerance.
    /// beta: float, optional
    ///     The damping factor.
    ///
    /// Returns
    /// -------
    /// DFTSolver
    #[pyo3(text_signature = "($self, mmax=None, log=None, max_iter=None, tol=None, beta=None)")]
    fn anderson_mixing(
        &self,
        mmax: Option<usize>,
        log: Option<bool>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        beta: Option<f64>,
    ) -> Self {
        let mut solver = self.0.clone().anderson_mixing(mmax);
        if let Some(log) = log {
            if log {
                solver = solver.log();
            }
        }
        if let Some(max_iter) = max_iter {
            solver = solver.max_iter(max_iter);
        }
        if let Some(tol) = tol {
            solver = solver.tol(tol);
        }
        if let Some(beta) = beta {
            solver = solver.beta(beta);
        }
        Self(solver)
    }

    fn _repr_markdown_(&self) -> String {
        self.0._repr_markdown_()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}
