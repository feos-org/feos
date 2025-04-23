use crate::ideal_gas::IdealGasModel;
use crate::parameter::{PyGcParameters, PyParameters};
use crate::residual::ResidualModel;
#[cfg(feature = "epcsaft")]
use feos::epcsaft::{ElectrolytePcSaft, ElectrolytePcSaftOptions, ElectrolytePcSaftVariants};
#[cfg(feature = "estimator")]
use feos::estimator::*;
#[cfg(feature = "gc_pcsaft")]
use feos::gc_pcsaft::{GcPcSaft, GcPcSaftOptions};
// #[cfg(feature = "estimator")]
// use feos::impl_estimator;
// #[cfg(all(feature = "estimator", feature = "pcsaft"))]
// use feos::impl_estimator_entropy_scaling;
#[cfg(feature = "pcsaft")]
use feos::pcsaft::{DQVariants, PcSaft, PcSaftOptions};
#[cfg(feature = "pets")]
use feos::pets::{Pets, PetsOptions};
#[cfg(feature = "saftvrmie")]
use feos::saftvrmie::{SaftVRMie, SaftVRMieOptions};
#[cfg(feature = "saftvrqmie")]
use feos::saftvrqmie::{SaftVRQMie, SaftVRQMieOptions};
#[cfg(feature = "uvtheory")]
use feos::uvtheory::{Perturbation, UVTheory, UVTheoryOptions};

use crate::user_defined::{PyIdealGas, PyResidual};
use feos_core::cubic::PengRobinson;
use feos_core::*;
use ndarray::{Array1, Array2};
use numpy::prelude::*;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
#[cfg(feature = "estimator")]
use pyo3::wrap_pymodule;
use quantity::*;
use std::collections::HashMap;
use std::convert::TryInto;
use std::sync::Arc;
use typenum::{Quot, P3};

use super::PyEquationOfState;

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
