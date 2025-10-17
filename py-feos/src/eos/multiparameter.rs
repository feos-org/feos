use super::PyEquationOfState;
use crate::ideal_gas::IdealGasModel;
use crate::parameter::PyParameters;
use crate::residual::ResidualModel;
use feos::multiparameter::MultiParameter;
use feos_core::EquationOfState;
use pyo3::prelude::*;
use std::sync::Arc;

#[pymethods]
impl PyEquationOfState {
    /// Multiparameter Helmholtz energy equations of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : Parameters
    ///     The parameters of the multiparameter Helmholtz equation of state.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    #[staticmethod]
    pub fn multiparameter(parameters: PyParameters) -> PyResult<Self> {
        let eos = MultiParameter::new(parameters.try_convert()?);
        let residual = ResidualModel::MultiParameter(eos.residual);
        let ideal_gas = eos
            .ideal_gas
            .into_iter()
            .map(IdealGasModel::MultiParameter)
            .collect();
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }

    /// Ideal gas model used in multiparameter Helmholtz energy equations of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : Parameters
    ///     The parameters of the multiparameter Helmholtz equation of state.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    pub fn multiparameter_ideal_gas(
        slf: Bound<'_, Self>,
        parameters: PyParameters,
    ) -> PyResult<Bound<'_, Self>> {
        let eos = MultiParameter::new(parameters.try_convert()?);
        let ideal_gas = eos
            .ideal_gas
            .into_iter()
            .map(IdealGasModel::MultiParameter)
            .collect();
        slf.borrow_mut().add_ideal_gas(ideal_gas);
        Ok(slf)
    }
}
