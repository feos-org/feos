use super::PyEquationOfState;
use crate::{ideal_gas::IdealGasModel, parameter::PyParameters, residual::ResidualModel};
use feos::uvtheory::{Perturbation, UVTheory, UVTheoryOptions};
use feos_core::{Components, EquationOfState};
use pyo3::prelude::*;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq)]
#[pyclass(name = "Perturbation", eq, eq_int)]
pub enum PyPerturbation {
    BarkerHenderson,
    WeeksChandlerAndersen,
    WeeksChandlerAndersenB3,
}

impl From<Perturbation> for PyPerturbation {
    fn from(value: Perturbation) -> Self {
        use Perturbation::*;
        match value {
            BarkerHenderson => Self::BarkerHenderson,
            WeeksChandlerAndersen => Self::WeeksChandlerAndersen,
            WeeksChandlerAndersenB3 => Self::WeeksChandlerAndersenB3,
        }
    }
}

impl From<PyPerturbation> for Perturbation {
    fn from(value: PyPerturbation) -> Self {
        use PyPerturbation::*;
        match value {
            BarkerHenderson => Self::BarkerHenderson,
            WeeksChandlerAndersen => Self::WeeksChandlerAndersen,
            WeeksChandlerAndersenB3 => Self::WeeksChandlerAndersenB3,
        }
    }
}

#[pymethods]
impl PyEquationOfState {
    /// UV-Theory equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : UVTheoryParameters
    ///     The parameters of the UV-theory equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// perturbation : Perturbation, optional
    ///     Division type of the Mie potential. Defaults to WCA division.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The UV-Theory equation of state that can be used to compute thermodynamic
    ///     states.
    #[cfg(feature = "uvtheory")]
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, perturbation=PyPerturbation::WeeksChandlerAndersen),
        text_signature = "(parameters, max_eta=0.5, perturbation)"
    )]
    fn uvtheory(
        parameters: PyParameters,
        max_eta: f64,
        perturbation: PyPerturbation,
    ) -> PyResult<Self> {
        let options = UVTheoryOptions {
            max_eta,
            perturbation: perturbation.into(),
        };
        let residual = Arc::new(ResidualModel::UVTheory(UVTheory::with_options(
            Arc::new(parameters.try_convert()?),
            options,
        )));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }
}
