use super::profile::{impl_1d_profile, impl_3d_profile, impl_profile};
use super::{PyDFTSolver, PyDFTSolverLog};
use crate::residual::ResidualModel;
use crate::state::{PyContributions, PyState};
use crate::{error::PyFeosError, ideal_gas::IdealGasModel};
use feos_core::{EquationOfState, ReferenceSystem};
use feos_dft::solvation::{PairCorrelation, SolvationProfile};
use ndarray::*;
use numpy::*;
use pyo3::*;
use quantity::*;

/// Density profile and properties of a solute in an inhomogeneous fluid.
///
/// Parameters
/// ----------
/// bulk : State
///     The bulk state of the surrounding solvent.
/// n_grid : [int, int, int]
///     The number of grid points in x-, y- and z-direction.
/// coordinates : SIArray2
///     The cartesian coordinates of all N interaction sites.
/// sigma : numpy.ndarray[float]
///     The size parameters of all N interaction sites in units of Angstrom.
/// epsilon_k : numpy.ndarray[float]
///     The reduced energy parameters epsilon / kB of all N interaction sites in units of Kelvin.
/// system_size : [SINumber, SINumber, SINumber], optional
///     The box length in x-, y- and z-direction (default: [40.0 * ANGSTROM, 40.0 * ANGSTROM, 40.0 * ANGSTROM]).
/// cutoff_radius : SINumber, optional
///      The cut-off radius up to which the dispersive solute-solvent interactions are evaluated (default: 14.0 * ANGSTROM).
/// potential_cutoff: float, optional
///     Maximum value for the external potential.
///
/// Returns
/// -------
/// SolvationProfile
///
#[pyclass(name = "SolvationProfile")]
pub struct PySolvationProfile(SolvationProfile<EquationOfState<IdealGasModel, ResidualModel>>);

impl_3d_profile!(PySolvationProfile, get_x, get_y, get_z);

#[pymethods]
impl PySolvationProfile {
    #[new]
    #[pyo3(
        text_signature = "(bulk, n_grid, coordinates, sigma, epsilon_k, system_size=None, cutoff_radius=None, potential_cutoff=None)"
    )]
    #[pyo3(signature = (bulk, n_grid, coordinates, sigma, epsilon_k, system_size=None, cutoff_radius=None, potential_cutoff=None))]
    #[expect(clippy::too_many_arguments)]
    fn new<'py>(
        bulk: &PyState,
        n_grid: [usize; 3],
        coordinates: Length<Array2<f64>>,
        sigma: &Bound<'py, PyArray1<f64>>,
        epsilon_k: &Bound<'py, PyArray1<f64>>,
        system_size: Option<[Length; 3]>,
        cutoff_radius: Option<Length>,
        potential_cutoff: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self(
            SolvationProfile::new(
                &bulk.0,
                n_grid,
                coordinates,
                sigma.to_owned_array(),
                epsilon_k.to_owned_array(),
                system_size,
                cutoff_radius,
                potential_cutoff,
            )
            .map_err(PyFeosError::from)?,
        ))
    }

    #[getter]
    fn get_grand_potential(&self) -> Option<Energy> {
        self.0.grand_potential
    }

    #[getter]
    fn get_solvation_free_energy(&self) -> Option<MolarEnergy> {
        self.0.solvation_free_energy
    }
}

/// Density profile and properties of a test particle system.
///
/// Parameters
/// ----------
/// bulk : State
///     The bulk state in equilibrium with the profile.
/// test_particle : int
///     The index of the test particle.
/// n_grid : int
///     The number of grid points.
/// width: SINumber
///     The width of the system.
///
/// Returns
/// -------
/// PairCorrelation
///
#[pyclass(name = "PairCorrelation")]
pub struct PyPairCorrelation(PairCorrelation<EquationOfState<IdealGasModel, ResidualModel>>);

impl_1d_profile!(PyPairCorrelation, [get_r]);

#[pymethods]
impl PyPairCorrelation {
    #[new]
    fn new(bulk: PyState, test_particle: usize, n_grid: usize, width: Length) -> Self {
        Self(PairCorrelation::new(&bulk.0, test_particle, n_grid, width))
    }

    #[getter]
    fn get_pair_correlation_function<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<Bound<'py, PyArray2<f64>>> {
        self.0
            .pair_correlation_function
            .as_ref()
            .map(|g| g.view().to_pyarray(py))
    }

    #[getter]
    fn get_self_solvation_free_energy(&self) -> Option<Energy> {
        self.0.self_solvation_free_energy
    }

    #[getter]
    fn get_structure_factor(&self) -> Option<f64> {
        self.0.structure_factor
    }
}
