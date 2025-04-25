use super::PyEquationOfState;
use crate::{ideal_gas::IdealGasModel, parameter::PyParameters, residual::ResidualModel};
use feos::saftvrmie::{SaftVRMie, SaftVRMieOptions};
use feos_core::{Components, EquationOfState};
use pyo3::prelude::*;
use std::sync::Arc;

#[pymethods]
impl PyEquationOfState {
    /// SAFT-VR Mie equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : SaftVRMieParameters
    ///     The parameters of the PC-SAFT equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// max_iter_cross_assoc : unsigned integer, optional
    ///     Maximum number of iterations for cross association. Defaults to 50.
    /// tol_cross_assoc : float
    ///     Tolerance for convergence of cross association. Defaults to 1e-10.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The SAFT-VR Mie equation of state that can be used to compute thermodynamic
    ///     states.
    #[cfg(feature = "saftvrmie")]
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10),
        text_signature = "(parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10)"
    )]
    pub fn saftvrmie(
        parameters: PyParameters,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
    ) -> PyResult<Self> {
        let options = SaftVRMieOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
        };
        let residual = Arc::new(ResidualModel::SaftVRMie(SaftVRMie::with_options(
            Arc::new(parameters.try_convert()?),
            options,
        )));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }
}
