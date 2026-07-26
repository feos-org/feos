use super::PyDFTSolver;
use crate::PyVerbosity;
use crate::eos::{Compositions, PyEquationOfState};
use crate::error::PyFeosError;
use crate::ideal_gas::IdealGasModel;
use crate::residual::ResidualModel;
use feos_core::EquationOfState;
use feos_dft::adsorption::Adsorption;
use nalgebra::DMatrix;
use ndarray::*;
use numpy::*;
use pyo3::prelude::*;
use quantity::*;
use std::sync::Arc;

mod external_potential;
mod pore;

pub use external_potential::PyExternalPotential;
pub use pore::{PyGrid, PyPore1D, PyPoreProfile, PyPoreSpecification};

/// Container structure for adsorption isotherms.
#[pyclass(name = "Adsorption")]
pub struct PyAdsorption(Adsorption<IxDyn, Arc<EquationOfState<Vec<IdealGasModel>, ResidualModel>>>);

#[pymethods]
impl PyAdsorption {
    /// Calculate an adsorption isotherm for the given pressure range.
    /// The profiles are evaluated starting from the lowest pressure.
    /// The resulting density profiles can be metastable.
    ///
    /// Parameters
    /// ----------
    /// functional : HelmholtzEnergyFunctional
    ///     The Helmholtz energy functional.
    /// temperature : SINumber
    ///     The temperature.
    /// pressure : SIArray1
    ///     The pressures for which the profiles are calculated.
    /// grid : Grid
    ///     The grid on which the density is calculated.
    /// external_potential : SIArray
    ///     The external potential used to model wall-fluid interactions.
    /// composition : float | SINumber | numpy.ndarray[float] | SIArray1 | list[float], optional
    ///     The composition of the mixture.
    /// solver: DFTSolver, optional
    ///     Custom solver options.
    ///
    /// Returns
    /// -------
    /// Adsorption
    ///
    #[staticmethod]
    #[pyo3(
        text_signature = "(functional, temperature, pressure, grid, external_potential, composition=None, solver=None)"
    )]
    #[pyo3(signature = (functional, temperature, pressure, grid, external_potential, composition=None, solver=None))]
    fn adsorption_isotherm(
        functional: &PyEquationOfState,
        temperature: Temperature,
        pressure: Pressure<Array1<f64>>,
        grid: PyGrid,
        external_potential: Energy<ArrayD<f64>>,
        composition: Option<&Bound<'_, PyAny>>,
        solver: Option<PyDFTSolver>,
    ) -> PyResult<Self> {
        Ok(Self(
            Adsorption::adsorption_isotherm(
                &functional.0,
                temperature,
                &pressure,
                &grid.0,
                &external_potential,
                Compositions::try_from(composition)?,
                solver.map(|s| s.0).as_ref(),
            )
            .map_err(PyFeosError::from)?,
        ))
    }

    /// Calculate a desorption isotherm for the given pressure range.
    /// The profiles are evaluated starting from the highest pressure.
    /// The resulting density profiles can be metastable.
    ///
    /// Parameters
    /// ----------
    /// functional : HelmholtzEnergyFunctional
    ///     The Helmholtz energy functional.
    /// temperature : SINumber
    ///     The temperature.
    /// pressure : SIArray1
    ///     The pressures for which the profiles are calculated.
    /// grid : Grid
    ///     The grid on which the density is calculated.
    /// external_potential : SIArray
    ///     The external potential used to model wall-fluid interactions.
    /// composition : float | SINumber | numpy.ndarray[float] | SIArray1 | list[float], optional
    ///     The composition of the mixture.
    /// solver: DFTSolver, optional
    ///     Custom solver options.
    ///
    /// Returns
    /// -------
    /// Adsorption
    ///
    #[staticmethod]
    #[pyo3(
        text_signature = "(functional, temperature, pressure, grid, external_potential, composition=None, solver=None)"
    )]
    #[pyo3(signature = (functional, temperature, pressure, grid, external_potential, composition=None, solver=None))]
    fn desorption_isotherm(
        functional: &PyEquationOfState,
        temperature: Temperature,
        pressure: Pressure<Array1<f64>>,
        grid: PyGrid,
        external_potential: Energy<ArrayD<f64>>,
        composition: Option<&Bound<'_, PyAny>>,
        solver: Option<PyDFTSolver>,
    ) -> PyResult<Self> {
        Ok(Self(
            Adsorption::desorption_isotherm(
                &functional.0,
                temperature,
                &pressure,
                &grid.0,
                &external_potential,
                Compositions::try_from(composition)?,
                solver.map(|s| s.0).as_ref(),
            )
            .map_err(PyFeosError::from)?,
        ))
    }

    /// Calculate an equilibrium isotherm for the given pressure range.
    /// A phase equilibrium in the pore is calculated to determine the
    /// stable phases for every pressure. If no phase equilibrium can be
    /// calculated, the isotherm is calculated twice, one in the adsorption
    /// direction and once in the desorption direction to determine the
    /// stability of the profiles.
    ///
    /// Parameters
    /// ----------
    /// functional : HelmholtzEnergyFunctional
    ///     The Helmholtz energy functional.
    /// temperature : SINumber
    ///     The temperature.
    /// pressure : SIArray1
    ///     The pressures for which the profiles are calculated.
    /// grid : Grid
    ///     The grid on which the density is calculated.
    /// external_potential : SIArray
    ///     The external potential used to model wall-fluid interactions.
    /// composition : float | SINumber | numpy.ndarray[float] | SIArray1 | list[float], optional
    ///     The composition of the mixture.
    /// solver: DFTSolver, optional
    ///     Custom solver options.
    ///
    /// Returns
    /// -------
    /// Adsorption
    ///
    #[staticmethod]
    #[pyo3(
        text_signature = "(functional, temperature, pressure, grid, external_potential, composition=None, solver=None)"
    )]
    #[pyo3(signature = (functional, temperature, pressure, grid, external_potential, composition=None, solver=None))]
    fn equilibrium_isotherm(
        functional: &PyEquationOfState,
        temperature: Temperature,
        pressure: Pressure<Array1<f64>>,
        grid: PyGrid,
        external_potential: Energy<ArrayD<f64>>,
        composition: Option<&Bound<'_, PyAny>>,
        solver: Option<PyDFTSolver>,
    ) -> PyResult<Self> {
        Ok(Self(
            Adsorption::equilibrium_isotherm(
                &functional.0,
                temperature,
                &pressure,
                &grid.0,
                &external_potential,
                Compositions::try_from(composition)?,
                solver.map(|s| s.0).as_ref(),
            )
            .map_err(PyFeosError::from)?,
        ))
    }

    /// Calculate a phase equilibrium in a pore.
    ///
    /// Parameters
    /// ----------
    /// functional : HelmholtzEnergyFunctional
    ///     The Helmholtz energy functional.
    /// temperature : SINumber
    ///     The temperature.
    /// p_min : SINumber
    ///     A suitable lower limit for the pressure.
    /// p_max : SINumber
    ///     A suitable upper limit for the pressure.
    /// grid : Grid
    ///     The grid on which the density is calculated.
    /// external_potential : SIArray
    ///     The external potential used to model wall-fluid interactions.
    /// composition : float | SINumber | numpy.ndarray[float] | SIArray1 | list[float], optional
    ///     The composition of the mixture.
    /// solver: DFTSolver, optional
    ///     Custom solver options.
    /// max_iter : int, optional
    ///     The maximum number of iterations of the phase equilibrium calculation.
    /// tol: float, optional
    ///     The tolerance of the phase equilibrium calculation.
    /// verbosity: Verbosity, optional
    ///     The verbosity of the phase equilibrium calculation.
    ///
    /// Returns
    /// -------
    /// Adsorption
    ///
    #[staticmethod]
    #[pyo3(
        text_signature = "(functional, temperature, p_min, p_max, grid, external_potential, composition=None, solver=None, max_iter=None, tol=None, verbosity=None)"
    )]
    #[pyo3(signature = (functional, temperature, p_min, p_max, grid, external_potential, composition=None, solver=None, max_iter=None, tol=None, verbosity=None))]
    #[expect(clippy::too_many_arguments)]
    fn phase_equilibrium(
        functional: &PyEquationOfState,
        temperature: Temperature,
        p_min: Pressure,
        p_max: Pressure,
        grid: PyGrid,
        external_potential: Energy<ArrayD<f64>>,
        composition: Option<&Bound<'_, PyAny>>,
        solver: Option<PyDFTSolver>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        verbosity: Option<PyVerbosity>,
    ) -> PyResult<Self> {
        Ok(Self(
            Adsorption::phase_equilibrium(
                &functional.0,
                temperature,
                p_min,
                p_max,
                &grid.0,
                &external_potential,
                Compositions::try_from(composition)?,
                solver.map(|s| s.0).as_ref(),
                (max_iter, tol, verbosity.map(|v| v.into())).into(),
            )
            .map_err(PyFeosError::from)?,
        ))
    }

    #[getter]
    fn get_profiles(&self) -> Vec<PyPoreProfile> {
        self.0
            .profiles
            .iter()
            .filter_map(|p| p.as_ref().ok().map(|p| PyPoreProfile(p.clone())))
            .collect()
    }

    #[getter]
    fn get_pressure(&self) -> Pressure<Array1<f64>> {
        self.0.pressure()
    }

    #[getter]
    fn get_adsorption(&self) -> Moles<Array2<f64>> {
        self.0.adsorption()
    }

    #[getter]
    fn get_total_adsorption(&self) -> Moles<Array1<f64>> {
        self.0.total_adsorption()
    }

    #[getter]
    fn get_grand_potential(&mut self) -> Energy<Array1<f64>> {
        self.0.grand_potential()
    }

    #[getter]
    fn get_partial_molar_enthalpy_of_adsorption(&mut self) -> MolarEnergy<DMatrix<f64>> {
        self.0.partial_molar_enthalpy_of_adsorption()
    }

    #[getter]
    fn get_enthalpy_of_adsorption(&mut self) -> MolarEnergy<Array1<f64>> {
        self.0.enthalpy_of_adsorption()
    }
}
