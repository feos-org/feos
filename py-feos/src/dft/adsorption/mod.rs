use super::PyDFTSolver;
use crate::eos::{parse_molefracs, PyEquationOfState};
use crate::error::PyFeosError;
use crate::ideal_gas::IdealGasModel;
use crate::residual::ResidualModel;
use crate::PyVerbosity;
use feos_core::EquationOfState;
use feos_dft::adsorption::{Adsorption, Adsorption1D, Adsorption3D};
use nalgebra::DMatrix;
use ndarray::*;
use numpy::*;
use pyo3::prelude::*;
use quantity::*;
use std::sync::Arc;

mod external_potential;
mod pore;

pub use external_potential::PyExternalPotential;
pub use pore::{PyPore1D, PyPore2D, PyPore3D, PyPoreProfile1D, PyPoreProfile3D};

/// Container structure for adsorption isotherms in 1D pores.
#[pyclass(name = "Adsorption1D")]
pub struct PyAdsorption1D(Adsorption1D<Arc<EquationOfState<Vec<IdealGasModel>, ResidualModel>>>);

/// Container structure for adsorption isotherms in 3D pores.
#[pyclass(name = "Adsorption3D")]
pub struct PyAdsorption3D(Adsorption3D<Arc<EquationOfState<Vec<IdealGasModel>, ResidualModel>>>);

macro_rules! impl_adsorption_isotherm {
    ($py_adsorption:ty, $py_pore:ty, $py_pore_profile:ident) => {
        #[pymethods]
        impl $py_adsorption {
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
            /// pore : Pore
            ///     The pore parameters.
            /// molefracs: numpy.ndarray[float], optional
            ///     For a mixture, the molefracs of the bulk system.
            /// solver: DFTSolver, optional
            ///     Custom solver options.
            ///
            /// Returns
            /// -------
            /// Adsorption
            ///
            #[staticmethod]
            #[pyo3(text_signature = "(functional, temperature, pressure, pore, molefracs=None, solver=None)")]
            #[pyo3(signature = (functional, temperature, pressure, pore, molefracs=None, solver=None))]
            fn adsorption_isotherm(
                functional: &PyEquationOfState,
                temperature: Temperature,
                pressure: Pressure<Array1<f64>>,
                pore: &$py_pore,
                molefracs: Option<PyReadonlyArray1<'_, f64>>,
                solver: Option<PyDFTSolver>,
            ) -> PyResult<Self> {
                Ok(Self(Adsorption::adsorption_isotherm(
                    &functional.0,
                    temperature,
                    &pressure,
                    &pore.0,
                    &parse_molefracs(molefracs),
                    solver.map(|s| s.0).as_ref(),
                ).map_err(PyFeosError::from)?))
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
            /// pore : Pore
            ///     The pore parameters.
            /// molefracs: numpy.ndarray[float], optional
            ///     For a mixture, the molefracs of the bulk system.
            /// solver: DFTSolver, optional
            ///     Custom solver options.
            ///
            /// Returns
            /// -------
            /// Adsorption
            ///
            #[staticmethod]
            #[pyo3(text_signature = "(functional, temperature, pressure, pore, molefracs=None, solver=None)")]
            #[pyo3(signature = (functional, temperature, pressure, pore, molefracs=None, solver=None))]
            fn desorption_isotherm(
                functional: &PyEquationOfState,
                temperature: Temperature,
                pressure: Pressure<Array1<f64>>,
                pore: &$py_pore,
                molefracs: Option<PyReadonlyArray1<'_, f64>>,
                solver: Option<PyDFTSolver>,
            ) -> PyResult<Self> {
                Ok(Self(Adsorption::desorption_isotherm(
                    &functional.0,
                    temperature,
                    &pressure,
                    &pore.0,
                    &parse_molefracs(molefracs),
                    solver.map(|s| s.0).as_ref(),
                ).map_err(PyFeosError::from)?))
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
            /// pore : Pore
            ///     The pore parameters.
            /// molefracs: numpy.ndarray[float], optional
            ///     For a mixture, the molefracs of the bulk system.
            /// solver: DFTSolver, optional
            ///     Custom solver options.
            ///
            /// Returns
            /// -------
            /// Adsorption
            ///
            #[staticmethod]
            #[pyo3(text_signature = "(functional, temperature, pressure, pore, molefracs=None, solver=None)")]
            #[pyo3(signature = (functional, temperature, pressure, pore, molefracs=None, solver=None))]
            fn equilibrium_isotherm(
                functional: &PyEquationOfState,
                temperature: Temperature,
                pressure: Pressure<Array1<f64>>,
                pore: &$py_pore,
                molefracs: Option<PyReadonlyArray1<'_, f64>>,
                solver: Option<PyDFTSolver>,
            ) -> PyResult<Self> {
                Ok(Self(Adsorption::equilibrium_isotherm(
                    &functional.0,
                    temperature,
                    &pressure,
                    &pore.0,
                    &parse_molefracs(molefracs),
                    solver.map(|s| s.0).as_ref(),
                ).map_err(PyFeosError::from)?))
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
            /// pore : Pore
            ///     The pore parameters.
            /// molefracs: numpy.ndarray[float], optional
            ///     For a mixture, the molefracs of the bulk system.
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
            #[pyo3(text_signature = "(functional, temperature, p_min, p_max, pore, molefracs=None, solver=None, max_iter=None, tol=None, verbosity=None)")]
            #[pyo3(signature = (functional, temperature, p_min, p_max, pore, molefracs=None, solver=None, max_iter=None, tol=None, verbosity=None))]
            #[expect(clippy::too_many_arguments)]
            fn phase_equilibrium(
                functional: &PyEquationOfState,
                temperature: Temperature,
                p_min: Pressure,
                p_max: Pressure,
                pore: &$py_pore,
                molefracs: Option<PyReadonlyArray1<'_, f64>>,
                solver: Option<PyDFTSolver>,
                max_iter: Option<usize>,
                tol: Option<f64>,
                verbosity: Option<PyVerbosity>,
            ) -> PyResult<Self> {
                Ok(Self(Adsorption::phase_equilibrium(
                    &functional.0,
                    temperature,
                    p_min,
                    p_max,
                    &pore.0,
                    &parse_molefracs(molefracs),
                    solver.map(|s| s.0).as_ref(),
                    (max_iter, tol, verbosity.map(|v| v.into())).into(),
                ).map_err(PyFeosError::from)?))
            }

            #[getter]
            fn get_profiles(&self) -> Vec<$py_pore_profile> {
                self.0
                    .profiles
                    .iter()
                    .filter_map(|p| {
                        p.as_ref()
                            .ok()
                            .map(|p| $py_pore_profile(p.clone()))
                    })
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
            fn get_partial_molar_enthalpy_of_adsorption(&self) -> MolarEnergy<DMatrix<f64>> {
                self.0.partial_molar_enthalpy_of_adsorption()
            }

            #[getter]
            fn get_enthalpy_of_adsorption(&self) -> MolarEnergy<Array1<f64>> {
                self.0.enthalpy_of_adsorption()
            }
        }
    };
}

impl_adsorption_isotherm!(PyAdsorption1D, PyPore1D, PyPoreProfile1D);
impl_adsorption_isotherm!(PyAdsorption3D, PyPore3D, PyPoreProfile3D);
