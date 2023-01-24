use crate::{DFTSolver, DFTSolverLog};
use feos_core::Verbosity;
use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use quantity::python::PySIArray1;

/// Settings for the DFT solver.
///
/// Parameters
/// ----------
/// verbosity: Verbosity, optional
///     The verbosity level of the solver.
///     Defaults to Verbosity.None.
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
        Self(DFTSolver::new(verbosity))
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
    ///     Iterate the logarithm of the density profile.
    ///     Defaults to False.
    /// max_iter: int, optional
    ///     The maximum number of iterations.
    ///     Defaults to 500.
    /// tol: float, optional
    ///     The tolerance.
    ///     Defaults to 1e-11.
    /// damping_coefficient: float, optional
    ///     Constant damping coefficient.
    ///     If no damping coefficient is provided, a line
    ///     search is used to determine the step size.
    ///
    /// Returns
    /// -------
    /// DFTSolver
    #[pyo3(text_signature = "($self, log=None, max_iter=None, tol=None, damping_coefficient=None)")]
    fn picard_iteration(
        &self,
        log: Option<bool>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        damping_coefficient: Option<f64>,
    ) -> Self {
        Self(
            self.0
                .clone()
                .picard_iteration(log, max_iter, tol, damping_coefficient),
        )
    }

    /// Add Anderson mixing to the solver object.
    ///
    /// Parameters
    /// ----------
    /// log: bool, optional
    ///     Iterate the logarithm of the density profile
    ///     Defaults to False.
    /// max_iter: int, optional
    ///     The maximum number of iterations.
    ///     Defaults to 150.
    /// tol: float, optional
    ///     The tolerance.
    ///     Defaults to 1e-11.
    /// damping_coefficient: float, optional
    ///     The damping coefficient.
    ///     Defaults to 0.15.
    /// mmax: int, optional
    ///     The maximum number of old solutions that are used.
    ///     Defaults to 100.
    ///
    /// Returns
    /// -------
    /// DFTSolver
    #[pyo3(
        text_signature = "($self, log=None, max_iter=None, tol=None, damping_coefficient=None, mmax=None)"
    )]
    fn anderson_mixing(
        &self,
        log: Option<bool>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        damping_coefficient: Option<f64>,
        mmax: Option<usize>,
    ) -> Self {
        Self(
            self.0
                .clone()
                .anderson_mixing(log, max_iter, tol, damping_coefficient, mmax),
        )
    }

    /// Add Newton solver to the solver object.
    ///
    /// Parameters
    /// ----------
    /// log: bool, optional
    ///     Iterate the logarithm of the density profile
    ///     Defaults to False.
    /// max_iter: int, optional
    ///     The maximum number of iterations.
    ///     Defaults to 50.
    /// max_iter_gmres: int, optional
    ///     The maximum number of iterations for the GMRES solver.
    ///     Defaults to 200.
    /// tol: float, optional
    ///     The tolerance.
    ///     Defaults to 1e-11.
    ///
    /// Returns
    /// -------
    /// DFTSolver
    #[pyo3(text_signature = "($self, log=None, max_iter=None, max_iter_gmres=None, tol=None)")]
    fn newton(
        &self,
        log: Option<bool>,
        max_iter: Option<usize>,
        max_iter_gmres: Option<usize>,
        tol: Option<f64>,
    ) -> Self {
        Self(self.0.clone().newton(log, max_iter, max_iter_gmres, tol))
    }

    fn _repr_markdown_(&self) -> String {
        self.0._repr_markdown_()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

#[pyclass(name = "DFTSolverLog")]
#[derive(Clone)]
pub struct PyDFTSolverLog(pub DFTSolverLog);

#[pymethods]
impl PyDFTSolverLog {
    #[getter]
    fn get_residual<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.0.residual().to_pyarray(py)
    }

    #[getter]
    fn get_time(&self) -> PySIArray1 {
        self.0.time().into()
    }

    #[getter]
    fn get_solver(&self) -> Vec<&'static str> {
        self.0.solver().to_vec()
    }
}
