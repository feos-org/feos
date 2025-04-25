use super::PyEquationOfState;
use crate::ideal_gas::IdealGasModel;
use crate::parameter::{PyGcParameters, PyParameters};
use crate::residual::ResidualModel;
use crate::user_defined::{PyIdealGas, PyResidual};
use feos_core::cubic::PengRobinson;
use feos_core::*;
use pyo3::prelude::*;
use std::sync::Arc;

#[pymethods]
impl PyEquationOfState {
    /// Peng-Robinson equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : PengRobinsonParameters
    ///     The parameters of the PR equation of state to use.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The PR equation of state that can be used to compute thermodynamic
    ///     states.
    #[staticmethod]
    pub fn peng_robinson(parameters: PyParameters) -> PyResult<Self> {
        let residual = Arc::new(ResidualModel::PengRobinson(PengRobinson::new(Arc::new(
            parameters.try_convert()?,
        ))));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }

    /// Residual Helmholtz energy model from a Python class.
    ///
    /// Parameters
    /// ----------
    /// residual : Class
    ///     A python class implementing the necessary methods
    ///     to be used as residual equation of state.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    #[staticmethod]
    fn python_residual(residual: Bound<'_, PyAny>) -> PyResult<Self> {
        let residual = Arc::new(ResidualModel::Python(PyResidual::new(residual)?));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }

    /// Equation of state that only contains an ideal gas contribution.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    #[staticmethod]
    fn ideal_gas() -> Self {
        let residual = Arc::new(ResidualModel::NoResidual(NoResidual(0)));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(0));
        Self(Arc::new(EquationOfState::new(ideal_gas, residual)))
    }

    /// Ideal gas equation of state from a Python class.
    ///
    /// Parameters
    /// ----------
    /// ideal_gas : Class
    ///     A python class implementing the necessary methods
    ///     to be used as an ideal gas model.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    fn python_ideal_gas(&self, ideal_gas: Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.add_ideal_gas(IdealGasModel::Python(PyIdealGas::new(ideal_gas)?)))
    }

    /// Ideal gas model of Joback and Reid.
    ///
    /// Parameters
    /// ----------
    /// joback : Joback
    ///     The parametrized Joback model.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    fn joback(&self, joback: PyGcParameters) -> PyResult<Self> {
        Ok(self.add_ideal_gas(IdealGasModel::Joback(Arc::new(
            joback.try_convert_homosegmented()?,
        ))))
    }

    /// Ideal gas model based on DIPPR equations for the ideal
    /// gas heat capacity.
    ///
    /// Parameters
    /// ----------
    /// dippr : Dippr
    ///     The parametrized Dippr model.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    fn dippr(&self, dippr: PyParameters) -> PyResult<Self> {
        Ok(self.add_ideal_gas(IdealGasModel::Dippr(Arc::new(dippr.try_convert()?))))
    }
}
