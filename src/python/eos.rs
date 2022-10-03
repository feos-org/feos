use crate::eos::EosVariant;
#[cfg(feature = "estimator")]
use crate::estimator::*;
#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::python::PyGcPcSaftEosParameters;
#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::{GcPcSaft, GcPcSaftOptions};
#[cfg(feature = "estimator")]
use crate::impl_estimator;
#[cfg(all(feature = "estimator", feature = "pcsaft"))]
use crate::impl_estimator_entropy_scaling;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::python::PyPcSaftParameters;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::{DQVariants, PcSaft, PcSaftOptions};
#[cfg(feature = "pets")]
use crate::pets::python::PyPetsParameters;
#[cfg(feature = "pets")]
use crate::pets::{Pets, PetsOptions};
#[cfg(feature = "uvtheory")]
use crate::uvtheory::python::PyUVParameters;
#[cfg(feature = "uvtheory")]
use crate::uvtheory::{Perturbation, UVTheory, UVTheoryOptions};
use feos_core::cubic::PengRobinson;
use feos_core::python::cubic::PyPengRobinsonParameters;
use feos_core::python::user_defined::PyEoSObj;
use feos_core::*;
use numpy::convert::ToPyArray;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
#[cfg(feature = "estimator")]
use pyo3::wrap_pymodule;
use quantity::python::*;
use quantity::si::*;
use std::collections::HashMap;
use std::sync::Arc;

#[pyclass(name = "EquationOfState", unsendable)]
#[derive(Clone)]
pub struct PyEosVariant(pub Arc<EosVariant>);

#[pymethods]
impl PyEosVariant {
    /// Initialize PC-SAFT equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : PcSaftParameters
    ///     The parameters of the PC-Saft equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// max_iter_cross_assoc : unsigned integer, optional
    ///     Maximum number of iterations for cross association. Defaults to 50.
    /// tol_cross_assoc : float
    ///     Tolerance for convergence of cross association. Defaults to 1e-10.
    /// dq_variant : DQVariants, optional
    ///     Combination rule used in the dipole/quadrupole term. Defaults to 'DQVariants.DQ35'
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The PC-SAFT equation of state that can be used to compute thermodynamic
    ///     states.
    #[cfg(feature = "pcsaft")]
    #[args(
        max_eta = "0.5",
        max_iter_cross_assoc = "50",
        tol_cross_assoc = "1e-10",
        dq_variant = "DQVariants::DQ35"
    )]
    #[staticmethod]
    #[pyo3(
        text_signature = "(parameters, max_eta, max_iter_cross_assoc, tol_cross_assoc, dq_variant)"
    )]
    pub fn pcsaft(
        parameters: PyPcSaftParameters,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
        dq_variant: DQVariants,
    ) -> Self {
        let options = PcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            dq_variant,
        };
        Self(Arc::new(EosVariant::PcSaft(PcSaft::with_options(
            parameters.0,
            options,
        ))))
    }

    /// Initialize the (heterosegmented) group contribution PC-SAFT equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : GcPcSaftEosParameters
    ///     The parameters of the PC-Saft equation of state to use.
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
    ///     The gc-PC-SAFT equation of state that can be used to compute thermodynamic
    ///     states.
    #[cfg(feature = "gc_pcsaft")]
    #[args(
        max_eta = "0.5",
        max_iter_cross_assoc = "50",
        tol_cross_assoc = "1e-10"
    )]
    #[staticmethod]
    #[pyo3(text_signature = "(parameters, max_eta, max_iter_cross_assoc, tol_cross_assoc)")]
    pub fn gc_pcsaft(
        parameters: PyGcPcSaftEosParameters,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
    ) -> Self {
        let options = GcPcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
        };
        Self(Arc::new(EosVariant::GcPcSaft(GcPcSaft::with_options(
            parameters.0,
            options,
        ))))
    }

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
    #[pyo3(text_signature = "(parameters)")]
    pub fn peng_robinson(parameters: PyPengRobinsonParameters) -> Self {
        Self(Arc::new(EosVariant::PengRobinson(PengRobinson::new(
            parameters.0,
        ))))
    }

    /// Equation of state from a Python class.
    ///
    /// Parameters
    /// ----------
    /// obj : Class
    ///     A python class implementing the necessary methods
    ///     to be used as equation of state.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    #[staticmethod]
    #[pyo3(text_signature = "(obj)")]
    fn python(obj: Py<PyAny>) -> PyResult<Self> {
        Ok(Self(Arc::new(EosVariant::Python(PyEoSObj::new(obj)?))))
    }

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
    #[args(max_eta = "0.5")]
    #[staticmethod]
    #[pyo3(text_signature = "(parameters, max_eta)")]
    fn pets(parameters: PyPetsParameters, max_eta: f64) -> Self {
        let options = PetsOptions { max_eta };
        Self(Arc::new(EosVariant::Pets(Pets::with_options(
            parameters.0,
            options,
        ))))
    }

    /// UV-Theory equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : UVParameters
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
    #[args(max_eta = "0.5", perturbation = "Perturbation::WeeksChandlerAndersen")]
    #[staticmethod]
    #[pyo3(text_signature = "(parameters, max_eta, perturbation)")]
    fn uvtheory(parameters: PyUVParameters, max_eta: f64, perturbation: Perturbation) -> Self {
        let options = UVTheoryOptions {
            max_eta,
            perturbation,
        };
        Self(Arc::new(EosVariant::UVTheory(UVTheory::with_options(
            parameters.0,
            options,
        ))))
    }
}

impl_equation_of_state!(PyEosVariant);
impl_virial_coefficients!(PyEosVariant);
impl_state!(EosVariant, PyEosVariant);
impl_state_molarweight!(EosVariant, PyEosVariant);
#[cfg(feature = "pcsaft")]
impl_state_entropy_scaling!(EosVariant, PyEosVariant);
impl_phase_equilibrium!(EosVariant, PyEosVariant);

#[cfg(feature = "estimator")]
impl_estimator!(EosVariant, PyEosVariant);
#[cfg(all(feature = "estimator", feature = "pcsaft"))]
impl_estimator_entropy_scaling!(EosVariant, PyEosVariant);

#[pymodule]
pub fn eos(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Contributions>()?;
    m.add_class::<Verbosity>()?;

    m.add_class::<PyEosVariant>()?;
    m.add_class::<PyState>()?;
    m.add_class::<PyPhaseDiagram>()?;
    m.add_class::<PyPhaseEquilibrium>()?;

    #[cfg(feature = "estimator")]
    m.add_wrapped(wrap_pymodule!(estimator_eos))?;

    Ok(())
}

#[cfg(feature = "estimator")]
#[pymodule]
pub fn estimator_eos(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDataSet>()?;
    m.add_class::<PyEstimator>()?;
    m.add_class::<PyLoss>()?;
    m.add_class::<Phase>()
}
