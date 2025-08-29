use super::profile::{impl_1d_profile, impl_profile};
use super::{PyDFTSolver, PyDFTSolverLog};
use crate::error::PyFeosError;
use crate::ideal_gas::IdealGasModel;
use crate::phase_equilibria::PyPhaseEquilibrium;
use crate::residual::ResidualModel;
use crate::state::{PyContributions, PyState};
use feos_core::{EquationOfState, ReferenceSystem};
use feos_dft::interface::PlanarInterface;
use nalgebra::{DMatrix, DVector};
use ndarray::*;
use numpy::*;
use pyo3::*;
use quantity::*;
use std::sync::Arc;

mod surface_tension_diagram;
pub use surface_tension_diagram::PySurfaceTensionDiagram;

/// A one-dimensional density profile of a vapor-liquid or liquid-liquid interface.
#[pyclass(name = "PlanarInterface")]
pub struct PyPlanarInterface(
    PlanarInterface<Arc<EquationOfState<Vec<IdealGasModel>, ResidualModel>>>,
);

impl_1d_profile!(PyPlanarInterface, [get_z]);

#[pymethods]
impl PyPlanarInterface {
    /// Initialize a planar interface with a hyperbolic tangent.
    ///
    /// Parameters
    /// ----------
    /// vle : PhaseEquilibrium
    ///     The bulk phase equilibrium.
    /// n_grid : int
    ///     The number of grid points.
    /// l_grid: SINumber
    ///     The width of the calculation domain.
    /// critical_temperature: SINumber
    ///     An estimate for the critical temperature of the system.
    ///     Used to guess the width of the interface.
    /// fix_equimolar_surface: bool, optional
    ///     If True use additional constraints to fix the
    ///     equimolar surface of the system.
    ///     Defaults to False.
    ///
    /// Returns
    /// -------
    /// PlanarInterface
    ///
    #[staticmethod]
    #[pyo3(
        text_signature = "(vle, n_grid, l_grid, critical_temperature, fix_equimolar_surface=None)"
    )]
    #[pyo3(signature = (vle, n_grid, l_grid, critical_temperature, fix_equimolar_surface=None))]
    fn from_tanh(
        vle: &PyPhaseEquilibrium,
        n_grid: usize,
        l_grid: Length,
        critical_temperature: Temperature,
        fix_equimolar_surface: Option<bool>,
    ) -> Self {
        let profile = PlanarInterface::from_tanh(
            &vle.0,
            n_grid,
            l_grid,
            critical_temperature,
            fix_equimolar_surface.unwrap_or(false),
        );
        PyPlanarInterface(profile)
    }

    /// Initialize a planar interface with a pDGT calculation.
    ///
    /// Parameters
    /// ----------
    /// vle : PhaseEquilibrium
    ///     The bulk phase equilibrium.
    /// n_grid : int
    ///     The number of grid points.
    /// fix_equimolar_surface: bool, optional
    ///     If True use additional constraints to fix the
    ///     equimolar surface of the system.
    ///     Defaults to False.
    ///
    /// Returns
    /// -------
    /// PlanarInterface
    ///
    #[staticmethod]
    #[pyo3(text_signature = "(vle, n_grid, fix_equimolar_surface=None)")]
    #[pyo3(signature = (vle, n_grid, fix_equimolar_surface=None))]
    fn from_pdgt(
        vle: &PyPhaseEquilibrium,
        n_grid: usize,
        fix_equimolar_surface: Option<bool>,
    ) -> PyResult<Self> {
        let profile =
            PlanarInterface::from_pdgt(&vle.0, n_grid, fix_equimolar_surface.unwrap_or(false))
                .map_err(PyFeosError::from)?;
        Ok(PyPlanarInterface(profile))
    }

    /// Initialize a planar interface with a provided density profile.
    ///
    /// Parameters
    /// ----------
    /// vle : PhaseEquilibrium
    ///     The bulk phase equilibrium.
    /// n_grid : int
    ///     The number of grid points.
    /// l_grid: SINumber
    ///     The width of the calculation domain.
    /// density_profile: SIArray2
    ///     Initial condition for the density profile iterations
    ///
    /// Returns
    /// -------
    /// PlanarInterface
    ///
    #[staticmethod]
    fn from_density_profile(
        vle: &PyPhaseEquilibrium,
        n_grid: usize,
        l_grid: Length,
        density_profile: Density<Array2<f64>>,
    ) -> Self {
        let mut profile = PlanarInterface::new(&vle.0, n_grid, l_grid);
        profile.profile.density = density_profile;
        PyPlanarInterface(profile)
    }
}

#[pymethods]
impl PyPlanarInterface {
    #[getter]
    fn get_surface_tension(&mut self) -> Option<SurfaceTension> {
        self.0.surface_tension
    }

    #[getter]
    fn get_equimolar_radius(&mut self) -> Option<Length> {
        self.0.equimolar_radius
    }

    #[getter]
    fn get_vle(&self) -> PyPhaseEquilibrium {
        PyPhaseEquilibrium(self.0.vle.clone())
    }

    /// Calculates the Gibbs' relative adsorption of component `i' with
    /// respect to `j': \Gamma_i^(j)
    ///
    /// Returns
    /// -------
    /// SIArray2
    ///
    fn relative_adsorption(&self) -> Moles<Array2<f64>> {
        self.0.relative_adsorption()
    }

    /// Calculates the interfacial enrichment E_i.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///
    fn interfacial_enrichment<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.0.interfacial_enrichment().to_pyarray(py)
    }

    /// Calculates the interfacial thickness (90-10 number density difference)
    ///
    /// Returns
    /// -------
    /// SINumber
    ///
    fn interfacial_thickness(&self) -> PyResult<Length> {
        Ok(self.0.interfacial_thickness().map_err(PyFeosError::from)?)
    }
}
