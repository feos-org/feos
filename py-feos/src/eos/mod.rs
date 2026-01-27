use crate::error::PyFeosError;
use crate::ideal_gas::IdealGasModel;
use crate::residual::ResidualModel;
use feos_core::*;
use indexmap::IndexMap;
use nalgebra::{DVector, DVectorView, Dyn, U1};
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::{exceptions::PyValueError, prelude::*};
use quantity::*;
use std::ops::Div;
use std::sync::Arc;

type Quot<T1, T2> = <T1 as Div<T2>>::Output;

mod constructors;
#[cfg(feature = "epcsaft")]
mod epcsaft;
#[cfg(feature = "gc_pcsaft")]
mod gc_pcsaft;
#[cfg(feature = "multiparameter")]
mod multiparameter;
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
    #[getter]
    fn get_parameters<'py>(&self, py: Python<'py>) -> IndexMap<String, Bound<'py, PyAny>> {
        let pure = self.0.pure_parameters();
        let binary = self.0.binary_parameters();
        let association_ab = self.0.association_parameters_ab();
        let association_cc = self.0.association_parameters_cc();
        pure.into_iter()
            .map(|(k, v)| (k, PyArray1::from_slice(py, v.as_slice()).into_any()))
            .chain(
                binary
                    .into_iter()
                    .chain(association_ab)
                    .chain(association_cc)
                    .map(|(k, v)| (k, v.to_pyarray(py).into_any())),
            )
            .collect()
    }

    /// Return maximum density for given amount of substance of each component.
    ///
    /// Parameters
    /// ----------
    /// composition : float | SINumber | numpy.ndarray[float] | SIArray1 | list[float], optional
    ///     The composition of the mixture.
    ///
    /// Returns
    /// -------
    /// SINumber
    #[pyo3(text_signature = "(composition=None)", signature = (composition=None))]
    fn max_density(&self, composition: Option<&Bound<'_, PyAny>>) -> PyResult<Density> {
        Ok(self
            .0
            .max_density(Compositions::try_from(composition)?)
            .map_err(PyFeosError::from)?)
    }

    /// Calculate the second Virial coefficient B(T,x).
    ///
    /// Parameters
    /// ----------
    /// temperature : SINumber
    ///     The temperature for which B should be computed.
    /// composition : float | SINumber | numpy.ndarray[float] | SIArray1 | list[float], optional
    ///     The composition of the mixture.
    ///
    /// Returns
    /// -------
    /// SINumber
    #[pyo3(text_signature = "(temperature, composition=None)", signature = (temperature, composition=None))]
    fn second_virial_coefficient(
        &self,
        temperature: Temperature,
        composition: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Quot<f64, Density>> {
        Ok(self
            .0
            .second_virial_coefficient(temperature, Compositions::try_from(composition)?)
            .map_err(PyFeosError::from)?)
    }

    /// Calculate the third Virial coefficient C(T,x).
    ///
    /// Parameters
    /// ----------
    /// temperature : SINumber
    ///     The temperature for which C should be computed.
    /// composition : float | SINumber | numpy.ndarray[float] | SIArray1 | list[float], optional
    ///     The composition of the mixture.
    ///
    /// Returns
    /// -------
    /// SINumber
    #[pyo3(text_signature = "(temperature, composition=None)", signature = (temperature, composition=None))]
    fn third_virial_coefficient(
        &self,
        temperature: Temperature,
        composition: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Quot<Quot<f64, Density>, Density>> {
        Ok(self
            .0
            .third_virial_coefficient(temperature, Compositions::try_from(composition)?)
            .map_err(PyFeosError::from)?)
    }

    /// Calculate the derivative of the second Virial coefficient B(T,x)
    /// with respect to temperature.
    ///
    /// Parameters
    /// ----------
    /// temperature : SINumber
    ///     The temperature for which B' should be computed.
    /// composition : float | SINumber | numpy.ndarray[float] | SIArray1 | list[float], optional
    ///     The composition of the mixture.
    ///
    /// Returns
    /// -------
    /// SINumber
    #[pyo3(text_signature = "(temperature, composition=None)", signature = (temperature, composition=None))]
    fn second_virial_coefficient_temperature_derivative(
        &self,
        temperature: Temperature,
        composition: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Quot<Quot<f64, Density>, Temperature>> {
        Ok(self
            .0
            .second_virial_coefficient_temperature_derivative(
                temperature,
                Compositions::try_from(composition)?,
            )
            .map_err(PyFeosError::from)?)
    }

    /// Calculate the derivative of the third Virial coefficient C(T,x)
    /// with respect to temperature.
    ///
    /// Parameters
    /// ----------
    /// temperature : SINumber
    ///     The temperature for which C' should be computed.
    /// composition : float | SINumber | numpy.ndarray[float] | SIArray1 | list[float], optional
    ///     The composition of the mixture.
    ///
    /// Returns
    /// -------
    /// SINumber
    #[expect(clippy::type_complexity)]
    #[pyo3(text_signature = "(temperature, composition=None)", signature = (temperature, composition=None))]
    fn third_virial_coefficient_temperature_derivative(
        &self,
        temperature: Temperature,
        composition: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Quot<Quot<Quot<f64, Density>, Density>, Temperature>> {
        Ok(self
            .0
            .third_virial_coefficient_temperature_derivative(
                temperature,
                Compositions::try_from(composition)?,
            )
            .map_err(PyFeosError::from)?)
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
        if eos.ideal_gas.is_empty() || matches!(eos.ideal_gas[0], IdealGasModel::NoModel) {
            eos.ideal_gas = ideal_gas;
        } else {
            panic!("There is already an ideal gas model initialized for the equation of state!")
        }
    }
}

pub(crate) fn parse_molefracs(molefracs: Option<PyReadonlyArray1<f64>>) -> Option<DVector<f64>> {
    molefracs.map(|x| {
        let x: DVectorView<f64, Dyn, Dyn> = x
            .try_as_matrix()
            .expect("molefracs are in an invalid format!");
        x.clone_owned()
    })
}

#[derive(Clone)]
pub enum Compositions {
    None,
    Scalar(f64),
    TotalMoles(Moles<f64>),
    Molefracs(DVector<f64>),
    Moles(Moles<DVector<f64>>),
    PartialDensity(Density<DVector<f64>>),
}

impl Composition<f64, Dyn> for Compositions {
    fn into_molefracs<E: Residual<Dyn, f64>>(
        self,
        eos: &E,
    ) -> FeosResult<(DVector<f64>, Option<Moles<f64>>)> {
        match self {
            Self::None => ().into_molefracs(eos),
            Self::Scalar(x) => x.into_molefracs(eos),
            Self::TotalMoles(total_moles) => total_moles.into_molefracs(eos),
            Self::Molefracs(molefracs) => molefracs.into_molefracs(eos),
            Self::Moles(moles) => moles.into_molefracs(eos),
            Self::PartialDensity(partial_density) => partial_density.into_molefracs(eos),
        }
    }

    fn density(&self) -> Option<Density<f64>> {
        if let Self::PartialDensity(partial_density) = self {
            partial_density.density()
        } else {
            None
        }
    }
}

impl TryFrom<Option<&Bound<'_, PyAny>>> for Compositions {
    type Error = PyErr;
    fn try_from(composition: Option<&Bound<'_, PyAny>>) -> PyResult<Compositions> {
        let Some(composition) = composition else {
            return Ok(Compositions::None);
        };
        if let Ok(x) = composition.extract::<PyReadonlyArray1<f64>>()
            && let Some(x) = x.try_as_matrix::<Dyn, U1, Dyn, Dyn>()
        {
            Ok(Compositions::Molefracs(x.clone_owned()))
        } else if let Ok(x) = composition.extract::<Vec<f64>>() {
            Ok(Compositions::Molefracs(DVector::from_vec(x)))
        } else if let Ok(x) = composition.extract::<f64>() {
            Ok(Compositions::Scalar(x))
        } else if let Ok(n) = composition.extract::<Moles<DVector<f64>>>() {
            Ok(Compositions::Moles(n))
        } else if let Ok(n) = composition.extract::<Moles>() {
            Ok(Compositions::TotalMoles(n))
        } else if let Ok(rho) = composition.extract::<Density<DVector<f64>>>() {
            Ok(Compositions::PartialDensity(rho))
        } else {
            Err(PyErr::new::<PyValueError, _>(format!(
                "failed to parse value '{composition}' as composition."
            )))
        }
    }
}
