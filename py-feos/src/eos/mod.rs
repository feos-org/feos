use crate::error::PyFeosError;
use crate::ideal_gas::IdealGasModel;
use crate::residual::ResidualModel;

use feos_core::*;
use ndarray::Array1;
use pyo3::prelude::*;
use quantity::*;
use std::sync::Arc;
use typenum::Quot;

mod constructors;
#[cfg(feature = "epcsaft")]
mod epcsaft;
#[cfg(feature = "gc_pcsaft")]
mod gc_pcsaft;
#[cfg(feature = "pcsaft")]
mod pcsaft;
#[cfg(feature = "pets")]
mod pets;
#[cfg(feature = "saftvrmie")]
mod saftvrmie;
#[cfg(feature = "saftvrqmie")]
mod saftvrqmie;
#[cfg(feature = "uvtheory")]
mod uvtheory;

/// Collection of equations of state.
#[pyclass(name = "EquationOfState")]
#[derive(Clone)]
pub struct PyEquationOfState(pub Arc<EquationOfState<IdealGasModel, ResidualModel>>);

#[pymethods]
impl PyEquationOfState {
    /// Return maximum density for given amount of substance of each component.
    ///
    /// Parameters
    /// ----------
    /// moles : SIArray1, optional
    ///     The amount of substance in mol for each component.
    ///
    /// Returns
    /// -------
    /// SINumber
    #[pyo3(text_signature = "(moles=None)", signature = (moles=None))]
    fn max_density(&self, moles: Option<Moles<Array1<f64>>>) -> PyResult<Density> {
        Ok(self
            .0
            .max_density(moles.as_ref())
            .map_err(PyFeosError::from)?)
    }

    /// Calculate the second Virial coefficient B(T,x).
    ///
    /// Parameters
    /// ----------
    /// temperature : SINumber
    ///     The temperature for which B should be computed.
    /// moles : SIArray1, optional
    ///     The amount of substance in mol for each component.
    ///
    /// Returns
    /// -------
    /// SINumber
    #[pyo3(text_signature = "(temperature, moles=None)", signature = (temperature, moles=None))]
    fn second_virial_coefficient(
        &self,
        temperature: Temperature,
        moles: Option<Moles<Array1<f64>>>,
    ) -> PyResult<Quot<f64, Density>> {
        Ok(self
            .0
            .second_virial_coefficient(temperature, moles.as_ref())
            .map_err(PyFeosError::from)?)
    }

    /// Calculate the third Virial coefficient C(T,x).
    ///
    /// Parameters
    /// ----------
    /// temperature : SINumber
    ///     The temperature for which C should be computed.
    /// moles : SIArray1, optional
    ///     The amount of substance in mol for each component.
    ///
    /// Returns
    /// -------
    /// SINumber
    #[pyo3(text_signature = "(temperature, moles=None)", signature = (temperature, moles=None))]
    fn third_virial_coefficient(
        &self,
        temperature: Temperature,
        moles: Option<Moles<Array1<f64>>>,
    ) -> PyResult<Quot<Quot<f64, Density>, Density>> {
        Ok(self
            .0
            .third_virial_coefficient(temperature, moles.as_ref())
            .map_err(PyFeosError::from)?)
    }

    /// Calculate the derivative of the second Virial coefficient B(T,x)
    /// with respect to temperature.
    ///
    /// Parameters
    /// ----------
    /// temperature : SINumber
    ///     The temperature for which B' should be computed.
    /// moles : SIArray1, optional
    ///     The amount of substance in mol for each component.
    ///
    /// Returns
    /// -------
    /// SINumber
    #[pyo3(text_signature = "(temperature, moles=None)", signature = (temperature, moles=None))]
    fn second_virial_coefficient_temperature_derivative(
        &self,
        temperature: Temperature,
        moles: Option<Moles<Array1<f64>>>,
    ) -> PyResult<Quot<Quot<f64, Density>, Temperature>> {
        Ok(self
            .0
            .second_virial_coefficient_temperature_derivative(temperature, moles.as_ref())
            .map_err(PyFeosError::from)?)
    }

    /// Calculate the derivative of the third Virial coefficient C(T,x)
    /// with respect to temperature.
    ///
    /// Parameters
    /// ----------
    /// temperature : SINumber
    ///     The temperature for which C' should be computed.
    /// moles : SIArray1, optional
    ///     The amount of substance in mol for each component.
    ///
    /// Returns
    /// -------
    /// SINumber
    #[pyo3(text_signature = "(temperature, moles=None)", signature = (temperature, moles=None))]
    #[expect(clippy::type_complexity)]
    fn third_virial_coefficient_temperature_derivative(
        &self,
        temperature: Temperature,
        moles: Option<Moles<Array1<f64>>>,
    ) -> PyResult<Quot<Quot<Quot<f64, Density>, Density>, Temperature>> {
        Ok(self
            .0
            .third_virial_coefficient_temperature_derivative(temperature, moles.as_ref())
            .map_err(PyFeosError::from)?)
    }
}

impl PyEquationOfState {
    fn add_ideal_gas(&self, ideal_gas: IdealGasModel) -> Self {
        let residual = match self.0.residual.as_ref() {
            ResidualModel::NoResidual(_) => Arc::new(ResidualModel::NoResidual(NoResidual(
                ideal_gas.components(),
            ))),
            _ => self.0.residual.clone(),
        };
        Self(Arc::new(EquationOfState::new(
            Arc::new(ideal_gas),
            residual,
        )))
    }
}

// impl_state_entropy_scaling!(EquationOfState<IdealGasModel, ResidualModel>, PyEquationOfState);
// impl_phase_equilibrium!(EquationOfState<IdealGasModel, ResidualModel>, PyEquationOfState);

// #[cfg(feature = "estimator")]
// impl_estimator!(EquationOfState<IdealGasModel, ResidualModel>, PyEquationOfState);
// #[cfg(all(feature = "estimator", feature = "pcsaft"))]
// impl_estimator_entropy_scaling!(EquationOfState<IdealGasModel, ResidualModel>, PyEquationOfState);

// #[pymodule]
// pub fn eos(m: &Bound<'_, PyModule>) -> PyResult<()> {
//     m.add_class::<Contributions>()?;
//     m.add_class::<Verbosity>()?;

//     m.add_class::<PyEquationOfState>()?;
//     m.add_class::<PyState>()?;
//     m.add_class::<PyStateVec>()?;
//     m.add_class::<PyPhaseDiagram>()?;
//     m.add_class::<PyPhaseEquilibrium>()?;

//     #[cfg(feature = "estimator")]
//     m.add_wrapped(wrap_pymodule!(estimator_eos))?;

//     Ok(())
// }

// #[cfg(feature = "estimator")]
// #[pymodule]
// pub fn estimator_eos(m: &Bound<'_, PyModule>) -> PyResult<()> {
//     m.add_class::<PyDataSet>()?;
//     m.add_class::<PyEstimator>()?;
//     m.add_class::<PyLoss>()?;
//     m.add_class::<Phase>()
// }
