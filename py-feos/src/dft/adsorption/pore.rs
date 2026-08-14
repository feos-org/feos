use super::PyExternalPotential;
use crate::dft::profile::*;
use crate::dft::{PyDFTSolver, PyDFTSolverLog, PyGeometry};
use crate::eos::PyEquationOfState;
use crate::error::PyFeosError;
use crate::ideal_gas::IdealGasModel;
use crate::residual::ResidualModel;
use crate::state::{PyContributions, PyState};
use feos_core::{EquationOfState, ReferenceSystem};
use feos_dft::{Axis as AxisDFT, Grid, adsorption::*};
use nalgebra::{DMatrix, DVector};
use ndarray::*;
use numpy::*;
use pyo3::prelude::*;
use quantity::*;
use std::sync::Arc;

#[pyclass(name = "Grid", from_py_object)]
#[derive(Clone)]
pub struct PyGrid(pub Grid);

#[pymethods]
impl PyGrid {
    /// Generate a 1D Cartesian grid with mirror boundary conditions on both sides.
    #[staticmethod]
    pub fn cartesian_1d(n_points: usize, length: Length) -> Self {
        let x = AxisDFT::new_cartesian(n_points, length, None);
        Self(Grid::Cartesian1(x))
    }

    /// Generate a 1D Cartesian grid with periodic boundary conditions on both sides.
    #[staticmethod]
    pub fn periodical_1d(n_points: usize, length: Length) -> Self {
        let x = AxisDFT::new_cartesian(n_points, length, None);
        Self(Grid::Periodical1(x))
    }

    /// Generate a polar grid with radial axis.
    #[staticmethod]
    pub fn polar(n_points: usize, length: Length) -> Self {
        let x = AxisDFT::new_polar(n_points, length);
        Self(Grid::Polar(x))
    }

    /// Generate a spherical grid with radial axis.
    #[staticmethod]
    pub fn spherical(n_points: usize, length: Length) -> Self {
        let x = AxisDFT::new_spherical(n_points, length);
        Self(Grid::Spherical(x))
    }

    /// Generate a 2D Cartesian grid with mirror boundary conditions on all sides.
    #[staticmethod]
    pub fn cartesian_2d(n_points: [usize; 2], length: [Length; 2]) -> Self {
        let [n_x, n_y] = n_points;
        let [l_x, l_y] = length;
        let x = AxisDFT::new_cartesian(n_x, l_x, None);
        let y = AxisDFT::new_cartesian(n_y, l_y, None);
        Self(Grid::Cartesian2(x, y))
    }

    /// Generate a 2D Cartesian (possibly oblique) grid with periodic boundary conditions on all sides.
    #[staticmethod]
    pub fn periodical_2d(n_points: [usize; 2], length: [Length; 2], alpha: Angle) -> Self {
        let [n_x, n_y] = n_points;
        let [l_x, l_y] = length;
        let x = AxisDFT::new_cartesian(n_x, l_x, None);
        let y = AxisDFT::new_cartesian(n_y, l_y, None);
        Self(Grid::Periodical2(x, y, alpha))
    }

    /// Generate a cylindrical grid with axes (in this order) r and z.
    #[staticmethod]
    pub fn cylindrical(n_points: [usize; 2], length: [Length; 2]) -> Self {
        let [n_r, n_z] = n_points;
        let [l_r, l_z] = length;
        let r = AxisDFT::new_polar(n_r, l_r);
        let z = AxisDFT::new_cartesian(n_z, l_z, None);
        Self(Grid::Cylindrical { r, z })
    }

    /// Generate a 3D Cartesian grid with mirror boundary conditions on all sides.
    #[staticmethod]
    pub fn cartesian_3d(n_points: [usize; 3], length: [Length; 3]) -> Self {
        let [n_x, n_y, n_z] = n_points;
        let [l_x, l_y, l_z] = length;
        let x = AxisDFT::new_cartesian(n_x, l_x, None);
        let y = AxisDFT::new_cartesian(n_y, l_y, None);
        let z = AxisDFT::new_cartesian(n_z, l_z, None);
        Self(Grid::Cartesian3(x, y, z))
    }

    /// Generate a 3D Cartesian (possibly oblique) grid with periodic boundary conditions on all sides.
    #[staticmethod]
    pub fn periodical_3d(n_points: [usize; 3], length: [Length; 3], angles: [Angle; 3]) -> Self {
        let [n_x, n_y, n_z] = n_points;
        let [l_x, l_y, l_z] = length;
        let x = AxisDFT::new_cartesian(n_x, l_x, None);
        let y = AxisDFT::new_cartesian(n_y, l_y, None);
        let z = AxisDFT::new_cartesian(n_z, l_z, None);
        Self(Grid::Periodical3(x, y, z, angles))
    }

    #[getter]
    pub fn get_axes(&self) -> Vec<Length<Array1<f64>>> {
        self.0
            .grids()
            .into_iter()
            .map(|ax| Length::from_reduced(ax.clone()))
            .collect()
    }

    #[getter]
    pub fn get_grid(&self) -> Vec<Length<ArrayD<f64>>> {
        self.0.mesh()
    }
}

/// The base class for studying adsorption phenomena.
///
/// Parameters
/// ----------
/// grid : Grid
///     The grid on which the density is calculated.
/// bulk : State
///     The (initial) bulk state in equilibrium with the pore.
/// external_potential : SIArray
///     The external potential used to model wall-fluid interactions.
/// density : SIArray, optional
///     The initial density distribution.
/// specification : PoreSpecification
///     The external constraint that specifies the state
///     in the pore.
///
/// Returns
/// -------
/// PoreProfile
///
#[pyclass(name = "PoreProfile")]
pub struct PyPoreProfile(
    pub PoreProfile<IxDyn, Arc<EquationOfState<Vec<IdealGasModel>, ResidualModel>>>,
);

#[pymethods]
impl PyPoreProfile {
    #[new]
    #[pyo3(
        text_signature = "(grid, bulk, external_potential, density=None, specification=PyPoreSpecification.ChemicalPotential)"
    )]
    #[pyo3(signature = (grid, bulk, external_potential, density=None, specification=PyPoreSpecification::ChemicalPotential()))]
    fn new(
        grid: PyGrid,
        bulk: &PyState,
        external_potential: Energy<ArrayD<f64>>,
        density: Option<Density<ArrayD<f64>>>,
        specification: PyPoreSpecification,
    ) -> Self {
        Self(PoreProfile::new(
            grid.0,
            &bulk.0,
            &external_potential,
            density.as_ref(),
            specification.0,
        ))
    }

    #[getter]
    fn get_grand_potential(&self) -> Option<Energy> {
        self.0.grand_potential
    }

    #[getter]
    fn get_interfacial_tension(&self) -> Option<Energy> {
        self.0.interfacial_tension
    }

    #[getter]
    fn get_partial_molar_enthalpy_of_adsorption(&self) -> PyResult<MolarEnergy<DVector<f64>>> {
        Ok(self
            .0
            .partial_molar_enthalpy_of_adsorption()
            .map_err(PyFeosError::from)?)
    }

    #[getter]
    fn get_enthalpy_of_adsorption(&self) -> PyResult<MolarEnergy> {
        Ok(self.0.enthalpy_of_adsorption().map_err(PyFeosError::from)?)
    }

    #[getter]
    fn get_henry_coefficients(&self) -> HenryCoefficient<DVector<f64>> {
        self.0.henry_coefficients()
    }

    #[getter]
    fn get_ideal_gas_enthalpy_of_adsorption(&self) -> MolarEnergy<DVector<f64>> {
        self.0.ideal_gas_enthalpy_of_adsorption()
    }
}

impl_profile!(PyPoreProfile);

/// Different ways that the thermodynamic state of the fluid in the pore
/// can be specified.
#[pyclass(name = "PoreSpecification", from_py_object)]
#[derive(Clone)]
pub struct PyPoreSpecification(PoreSpecification);

#[pymethods]
impl PyPoreSpecification {
    /// Specify the chemical potential (via the bulk state).
    #[classattr]
    #[expect(non_snake_case)]
    fn ChemicalPotential() -> Self {
        Self(PoreSpecification::ChemicalPotential)
    }

    /// Specify the amount of moles of every component.
    #[staticmethod]
    #[expect(non_snake_case)]
    fn Moles(moles: Moles<Array1<f64>>) -> Self {
        Self(PoreSpecification::Moles(moles))
    }

    /// Fix the amount of moles of every component based on the initial density profile.
    #[classattr]
    #[expect(non_snake_case)]
    fn FixedMoles() -> Self {
        Self(PoreSpecification::FixedMoles)
    }
}

/// Parameters required to specify a 1D pore.
///
/// Parameters
/// ----------
/// geometry : Geometry
///     The pore geometry.
/// pore_size : SINumber
///     The width of the slit pore in cartesian coordinates,
///     or the pore radius in spherical and cylindrical coordinates.
/// potential : ExternalPotential
///     The potential used to model wall-fluid interactions.
/// n_grid : int, optional
///     The number of grid points.
/// potential_cutoff : float, optional
///     Maximum value for the external potential.
///
/// Returns
/// -------
/// Pore1D
///
#[pyclass(name = "Pore1D")]
pub struct PyPore1D(pub Pore1D);

#[pymethods]
impl PyPore1D {
    #[new]
    #[pyo3(text_signature = "(functional, geometry, pore_size, external_potential, n_grid=None)")]
    #[pyo3(signature = (functional, geometry, pore_size, external_potential, n_grid=None))]
    fn new(
        functional: &PyEquationOfState,
        geometry: PyGeometry,
        pore_size: Length,
        external_potential: PyExternalPotential,
        n_grid: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self(Pore1D::new(
            &functional.0,
            geometry.into(),
            pore_size,
            external_potential.0,
            n_grid,
        )))
    }

    /// Initialize the pore for the given bulk state.
    ///
    /// Parameters
    /// ----------
    /// bulk : State
    ///     The bulk state in equilibrium with the pore.
    /// density : SIArray2, optional
    ///     Initial values for the density profile.
    /// specification : PoreSpecification
    ///     The external constraint that specifies the state
    ///     in the pore.
    ///
    /// Returns
    /// -------
    /// PoreProfile1D
    #[pyo3(
        text_signature = "($self, bulk, density=None, specification=PoreSpecification.ChemicalPotential)"
    )]
    #[pyo3(signature = (bulk, density=None, specification=PyPoreSpecification::ChemicalPotential()))]
    fn initialize(
        &self,
        bulk: &PyState,
        density: Option<Density<Array2<f64>>>,
        specification: PyPoreSpecification,
    ) -> PyResult<PyPoreProfile> {
        Ok(PyPoreProfile(
            self.0
                .initialize(&bulk.0, density.as_ref(), specification.0)
                .map_err(PyFeosError::from)?
                .into_dyn(),
        ))
    }

    #[getter]
    fn get_pore_size(&self) -> Length {
        self.0.pore_size
    }

    #[getter]
    fn get_external_potential(&self) -> Energy<Array2<f64>> {
        self.0.external_potential.clone()
    }

    #[getter]
    fn get_grid(&self) -> PyGrid {
        PyGrid(self.0.grid.clone())
    }

    /// The pore volume using Helium at 298 K as reference.
    #[getter]
    fn get_pore_volume(&self) -> PyResult<Volume> {
        Ok(self.0.pore_volume().map_err(PyFeosError::from)?)
    }
}
