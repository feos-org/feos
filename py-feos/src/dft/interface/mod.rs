use super::PyGeometry;
use super::adsorption::PyGrid;
use super::profile::impl_profile;
use super::{PyDFTSolver, PyDFTSolverLog};
use crate::error::PyFeosError;
use crate::ideal_gas::IdealGasModel;
use crate::phase_equilibria::PyPhaseEquilibrium;
use crate::residual::ResidualModel;
use crate::state::{PyContributions, PyState};
use feos_core::{EquationOfState, FeosError};
use feos_dft::Grid;
use feos_dft::interface::Interface;
use nalgebra::{DMatrix, DVector};
use ndarray::*;
use numpy::*;
use pyo3::*;
use quantity::*;
use std::sync::Arc;

mod surface_tension_diagram;
pub use surface_tension_diagram::PySurfaceTensionDiagram;

/// A one-dimensional density profile of a planar, cylindrical or spherical
/// vapor-liquid or liquid-liquid interface.
///
/// Parameters
/// ----------
/// grid : Grid
///     The grid on which the density is calculated
///     (`Grid.cartesian_1d`, `Grid.polar`, or `Grid.spherical`).
/// vle : PhaseEquilibrium
///     The bulk phase equilibrium.
/// density : SIArray2
///     The initial density profile.
///
/// Returns
/// -------
/// Interface
///
#[pyclass(name = "Interface")]
pub struct PyInterface(Interface<Arc<EquationOfState<Vec<IdealGasModel>, ResidualModel>>>);

impl_profile!(PyInterface);

#[pymethods]
impl PyInterface {
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
    ///
    /// Returns
    /// -------
    /// Interface
    ///
    #[staticmethod]
    #[pyo3(text_signature = "(vle, n_grid, l_grid, critical_temperature)")]
    #[pyo3(signature = (vle, n_grid, l_grid, critical_temperature))]
    fn planar_from_tanh(
        vle: &PyPhaseEquilibrium,
        n_grid: usize,
        l_grid: Length,
        critical_temperature: Temperature,
    ) -> Self {
        let profile = Interface::planar_from_tanh(&vle.0, n_grid, l_grid, critical_temperature);
        PyInterface(profile)
    }

    /// Initialize a planar interface with a pDGT calculation.
    ///
    /// Parameters
    /// ----------
    /// vle : PhaseEquilibrium
    ///     The bulk phase equilibrium.
    /// n_grid : int
    ///     The number of grid points.
    ///
    /// Returns
    /// -------
    /// Interface
    ///
    #[staticmethod]
    #[pyo3(text_signature = "(vle, n_grid)")]
    #[pyo3(signature = (vle, n_grid))]
    fn planar_from_pdgt(vle: &PyPhaseEquilibrium, n_grid: usize) -> PyResult<Self> {
        let profile = Interface::planar_from_pdgt(&vle.0, n_grid).map_err(PyFeosError::from)?;
        Ok(Self(profile))
    }

    /// Initialize a curved interface from a converged planar interface.
    ///
    /// The planar density profile is interpolated onto the new grid so that
    /// its equimolar surface is located at the specified radius.
    ///
    /// Parameters
    /// ----------
    /// planar_interface : Interface
    ///     A (converged) planar interface.
    /// n_grid : int
    ///     The number of grid points.
    /// l_grid: SINumber
    ///     The width of the calculation domain.
    /// radius: SINumber
    ///     The initial radius of the equimolar surface. A positive radius
    ///     corresponds to a droplet, a negative radius to a bubble.
    /// geometry: Geometry
    ///     The geometry of the interface (cylindrical or spherical).
    ///
    /// Returns
    /// -------
    /// Interface
    ///
    #[staticmethod]
    #[pyo3(text_signature = "(planar_interface, n_grid, l_grid, radius, geometry)")]
    fn curved(
        planar_interface: &PyInterface,
        n_grid: usize,
        l_grid: Length,
        radius: Length,
        geometry: PyGeometry,
    ) -> PyResult<Self> {
        let profile =
            Interface::curved(&planar_interface.0, n_grid, l_grid, radius, geometry.into())
                .map_err(PyFeosError::from)?;
        Ok(Self(profile))
    }

    #[new]
    fn new(
        grid: PyGrid,
        vle: &PyPhaseEquilibrium,
        density: Density<Array2<f64>>,
    ) -> PyResult<Self> {
        let (Grid::Cartesian1(axis) | Grid::Polar(axis) | Grid::Spherical(axis)) = grid.0 else {
            return Err(PyFeosError::from(FeosError::Error(
                "An interface requires a 1D cartesian, polar, or spherical grid.".into(),
            ))
            .into());
        };
        Ok(Self(Interface::new(&vle.0, axis, density)))
    }
}

#[pymethods]
impl PyInterface {
    #[getter]
    fn get_surface_tension(&mut self) -> Option<SurfaceTension> {
        self.0.surface_tension
    }

    #[getter]
    fn get_surface_of_tension(&mut self) -> Option<Length> {
        self.0.surface_of_tension
    }

    #[getter]
    fn get_vle(&self) -> PyPhaseEquilibrium {
        PyPhaseEquilibrium(self.0.vle.clone())
    }

    /// Calculates the Gibbs relative adsorption of component i with
    /// respect to j: \Gamma_i^(j)
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
