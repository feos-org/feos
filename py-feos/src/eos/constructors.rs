use super::PyEquationOfState;
use crate::ideal_gas::IdealGasModel;
use crate::parameter::{PyGcParameters, PyParameters};
use crate::residual::ResidualModel;
use crate::user_defined::{PyIdealGas, PyResidual};
use feos::ideal_gas::{Dippr, Joback};
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
        let residual = ResidualModel::PengRobinson(PengRobinson::new(parameters.try_convert()?));
        let ideal_gas = vec![IdealGasModel::NoModel; residual.components()];
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
        let residual = ResidualModel::Python(PyResidual::new(residual)?);
        let ideal_gas = vec![IdealGasModel::NoModel; residual.components()];
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }

    /// Equation of state that only contains an ideal gas contribution.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    #[staticmethod]
    fn ideal_gas() -> Self {
        let residual = ResidualModel::NoResidual(NoResidual(0));
        let ideal_gas = vec![IdealGasModel::NoModel; 0];
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
    fn python_ideal_gas<'py>(
        slf: Bound<'py, Self>,
        ideal_gas: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.borrow_mut().add_ideal_gas(
            ideal_gas
                .into_iter()
                .map(|i| Ok(IdealGasModel::Python(Arc::new(PyIdealGas::new(i)?))))
                .collect::<PyResult<_>>()?,
        );
        Ok(slf)
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
    fn joback(slf: Bound<'_, Self>, joback: PyGcParameters) -> PyResult<Bound<'_, Self>> {
        slf.borrow_mut().add_ideal_gas(
            Joback::new(joback.try_convert_homosegmented()?)
                .into_iter()
                .map(IdealGasModel::Joback)
                .collect(),
        );
        Ok(slf)
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
    fn dippr(slf: Bound<'_, Self>, dippr: PyParameters) -> PyResult<Bound<'_, Self>> {
        slf.borrow_mut().add_ideal_gas(
            Dippr::new(dippr.try_convert()?)
                .into_iter()
                .map(IdealGasModel::Dippr)
                .collect(),
        );
        Ok(slf)
    }
}
