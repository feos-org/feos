use crate::error::PyFeosError;
use crate::ideal_gas::IdealGasModel;
use crate::residual::ResidualModel;

use feos_core::*;
use nshare::IntoNalgebra;
use numpy::{PyArray1, PyArrayMethods};
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
pub struct PyEquationOfState(pub Arc<EquationOfState<Vec<IdealGasModel>, ResidualModel>>);

#[pymethods]
impl PyEquationOfState {
    /// Return maximum density for given amount of substance of each component.
    ///
    /// Parameters
    /// ----------
    /// molefracs : np.ndarray[float], optional
    ///     The composition of the mixture.
    ///
    /// Returns
    /// -------
    /// SINumber
    #[pyo3(text_signature = "(molefracs=None)", signature = (molefracs=None))]
    fn max_density<'py>(&self, molefracs: Option<Bound<'py, PyArray1<f64>>>) -> PyResult<Density> {
        Ok(self
            .0
            .max_density(&molefracs.map(|x| x.to_owned_array().into_nalgebra()))
            .map_err(PyFeosError::from)?)
    }

    /// Calculate the second Virial coefficient B(T,x).
    ///
    /// Parameters
    /// ----------
    /// temperature : SINumber
    ///     The temperature for which B should be computed.
    /// molefracs : np.ndarray[float], optional
    ///     The composition of the mixture.
    ///
    /// Returns
    /// -------
    /// SINumber
    #[pyo3(text_signature = "(temperature, molefracs=None)", signature = (temperature, molefracs=None))]
    fn second_virial_coefficient<'py>(
        &self,
        temperature: Temperature,
        molefracs: Option<Bound<'py, PyArray1<f64>>>,
    ) -> Quot<f64, Density> {
        self.0.second_virial_coefficient(
            temperature,
            &molefracs.map(|x| x.to_owned_array().into_nalgebra()),
        )
    }

    /// Calculate the third Virial coefficient C(T,x).
    ///
    /// Parameters
    /// ----------
    /// temperature : SINumber
    ///     The temperature for which C should be computed.
    /// molefracs : np.ndarray[float], optional
    ///     The composition of the mixture.
    ///
    /// Returns
    /// -------
    /// SINumber
    #[pyo3(text_signature = "(temperature, molefracs=None)", signature = (temperature, molefracs=None))]
    fn third_virial_coefficient<'py>(
        &self,
        temperature: Temperature,
        molefracs: Option<Bound<'py, PyArray1<f64>>>,
    ) -> Quot<Quot<f64, Density>, Density> {
        self.0.third_virial_coefficient(
            temperature,
            &molefracs.map(|x| x.to_owned_array().into_nalgebra()),
        )
    }

    /// Calculate the derivative of the second Virial coefficient B(T,x)
    /// with respect to temperature.
    ///
    /// Parameters
    /// ----------
    /// temperature : SINumber
    ///     The temperature for which B' should be computed.
    /// molefracs : np.ndarray[float], optional
    ///     The composition of the mixture.
    ///
    /// Returns
    /// -------
    /// SINumber
    #[pyo3(text_signature = "(temperature, molefracs=None)", signature = (temperature, molefracs=None))]
    fn second_virial_coefficient_temperature_derivative<'py>(
        &self,
        temperature: Temperature,
        molefracs: Option<Bound<'py, PyArray1<f64>>>,
    ) -> Quot<Quot<f64, Density>, Temperature> {
        self.0.second_virial_coefficient_temperature_derivative(
            temperature,
            &molefracs.map(|x| x.to_owned_array().into_nalgebra()),
        )
    }

    /// Calculate the derivative of the third Virial coefficient C(T,x)
    /// with respect to temperature.
    ///
    /// Parameters
    /// ----------
    /// temperature : SINumber
    ///     The temperature for which C' should be computed.
    /// molefracs : np.ndarray[float], optional
    ///     The composition of the mixture.
    ///
    /// Returns
    /// -------
    /// SINumber
    #[pyo3(text_signature = "(temperature, molefracs=None)", signature = (temperature, molefracs=None))]
    fn third_virial_coefficient_temperature_derivative<'py>(
        &self,
        temperature: Temperature,
        molefracs: Option<Bound<'py, PyArray1<f64>>>,
    ) -> Quot<Quot<Quot<f64, Density>, Density>, Temperature> {
        self.0.third_virial_coefficient_temperature_derivative(
            temperature,
            &molefracs.map(|x| x.to_owned_array().into_nalgebra()),
        )
    }
}

impl PyEquationOfState {
    fn add_ideal_gas(&mut self, ideal_gas: Vec<IdealGasModel>) {
        let Some(eos) = Arc::get_mut(&mut self.0) else {
            panic!("Cannot change equation of state after using it!")
        };
        if let ResidualModel::NoResidual(c) = &mut eos.residual {
            c.0 = ideal_gas.len()
        }
        eos.ideal_gas = ideal_gas;
    }
}

// impl_state_entropy_scaling!(EquationOfState<Vec<IdealGasModel>, ResidualModel>, PyEquationOfState);
// impl_phase_equilibrium!(EquationOfState<Vec<IdealGasModel>, ResidualModel>, PyEquationOfState);

// #[cfg(feature = "estimator")]
// impl_estimator!(EquationOfState<Vec<IdealGasModel>, ResidualModel>, PyEquationOfState);
// #[cfg(all(feature = "estimator", feature = "pcsaft"))]
// impl_estimator_entropy_scaling!(EquationOfState<Vec<IdealGasModel>, ResidualModel>, PyEquationOfState);

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
