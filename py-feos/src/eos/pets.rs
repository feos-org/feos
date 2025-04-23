use super::PyEquationOfState;
use crate::{ideal_gas::IdealGasModel, parameter::PyParameters, residual::ResidualModel};
use feos::pets::{Pets, PetsOptions};
use feos_core::{Components, EquationOfState};
use pyo3::prelude::*;
use std::sync::Arc;

#[pymethods]
impl PyEquationOfState {
    /// PeTS equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : PetsParameters
    ///     The parameters of the PeTS equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The PeTS equation of state that can be used to compute thermodynamic
    ///     states.
    #[cfg(feature = "pets")]
    #[staticmethod]
    #[pyo3(signature = (parameters, max_eta=0.5), text_signature = "(parameters, max_eta=0.5)")]
    fn pets(parameters: PyParameters, max_eta: f64) -> PyResult<Self> {
        let options = PetsOptions { max_eta };
        let residual = Arc::new(ResidualModel::Pets(Pets::with_options(
            Arc::new(parameters.try_convert()?),
            options,
        )));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }
}
