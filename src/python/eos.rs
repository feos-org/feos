#[cfg(feature = "cubic")]
use crate::cubic::python::PyCubicParameters;
use crate::cubic::python::{PyAlpha, PyMixingRule};
#[cfg(feature = "cubic")]
use crate::cubic::Cubic;
use crate::cubic::CubicOptions;
use crate::eos::ResidualModel;
#[cfg(feature = "epcsaft")]
use crate::epcsaft::python::PyElectrolytePcSaftParameters;
#[cfg(feature = "epcsaft")]
use crate::epcsaft::{ElectrolytePcSaft, ElectrolytePcSaftOptions, ElectrolytePcSaftVariants};
#[cfg(feature = "estimator")]
use crate::estimator::*;
#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::python::PyGcPcSaftEosParameters;
#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::{GcPcSaft, GcPcSaftOptions};
use crate::ideal_gas::IdealGasModel;
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
#[cfg(feature = "saftvrmie")]
use crate::saftvrmie::python::PySaftVRMieParameters;
#[cfg(feature = "saftvrmie")]
use crate::saftvrmie::{SaftVRMie, SaftVRMieOptions};
#[cfg(feature = "saftvrqmie")]
use crate::saftvrqmie::python::PySaftVRQMieParameters;
#[cfg(feature = "saftvrqmie")]
use crate::saftvrqmie::{SaftVRQMie, SaftVRQMieOptions};
#[cfg(feature = "uvtheory")]
use crate::uvtheory::python::PyUVTheoryParameters;
#[cfg(feature = "uvtheory")]
use crate::uvtheory::{Perturbation, UVTheory, UVTheoryOptions};

use super::dippr::PyDippr;
use super::joback::PyJoback;
use feos_core::cubic::PengRobinson;
use feos_core::python::cubic::PyPengRobinsonParameters;
use feos_core::python::user_defined::{PyIdealGas, PyResidual};
use feos_core::si::*;
use feos_core::*;
use numpy::prelude::*;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
#[cfg(feature = "estimator")]
use pyo3::wrap_pymodule;
use quantity::python::{PySIArray1, PySIArray2, PySINumber};
use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;
use typenum::P3;

/// Collection of equations of state.
#[pyclass(name = "EquationOfState")]
#[derive(Clone)]
pub struct PyEquationOfState(pub Arc<EquationOfState<IdealGasModel, ResidualModel>>);

#[pymethods]
impl PyEquationOfState {
    /// PC-SAFT equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : PcSaftParameters
    ///     The parameters of the PC-SAFT equation of state to use.
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
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant=DQVariants::DQ35),
        text_signature = "(parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant)"
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
        let residual = Arc::new(ResidualModel::PcSaft(PcSaft::with_options(
            parameters.0,
            options,
        )));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Self(Arc::new(EquationOfState::new(ideal_gas, residual)))
    }

    /// SAFT-VR Mie equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : SaftVRMieParameters
    ///     The parameters of the PC-SAFT equation of state to use.
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
    ///     The SAFT-VR Mie equation of state that can be used to compute thermodynamic
    ///     states.
    #[cfg(feature = "saftvrmie")]
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10),
        text_signature = "(parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10)"
    )]
    pub fn saftvrmie(
        parameters: PySaftVRMieParameters,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
    ) -> Self {
        let options = SaftVRMieOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
        };
        let residual = Arc::new(ResidualModel::SaftVRMie(SaftVRMie::with_options(
            parameters.0,
            options,
        )));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Self(Arc::new(EquationOfState::new(ideal_gas, residual)))
    }

    /// (heterosegmented) group contribution PC-SAFT equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : GcPcSaftEosParameters
    ///     The parameters of the PC-SAFT equation of state to use.
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
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10),
        text_signature = "(parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10)"
    )]
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
        let residual = Arc::new(ResidualModel::GcPcSaft(GcPcSaft::with_options(
            parameters.0,
            options,
        )));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Self(Arc::new(EquationOfState::new(ideal_gas, residual)))
    }

    /// ePC-SAFT equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : ElectrolytePcSaftParameters
    ///     The parameters of the PC-SAFT equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// max_iter_cross_assoc : unsigned integer, optional
    ///     Maximum number of iterations for cross association. Defaults to 50.
    /// tol_cross_assoc : float
    ///     Tolerance for convergence of cross association. Defaults to 1e-10.
    /// epcsaft_variant : ElectrolytePcSaftVariants, optional
    ///     Variant of the ePC-SAFT equation of state. Defaults to 'advanced'
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The ePC-SAFT equation of state that can be used to compute thermodynamic
    ///     states.
    #[cfg(feature = "epcsaft")]
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, epcsaft_variant=ElectrolytePcSaftVariants::Advanced),
        text_signature = "(parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, epcsaft_variant)",
    )]
    pub fn epcsaft(
        parameters: PyElectrolytePcSaftParameters,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
        epcsaft_variant: ElectrolytePcSaftVariants,
    ) -> Self {
        let options = ElectrolytePcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            epcsaft_variant,
        };
        let residual = Arc::new(ResidualModel::ElectrolytePcSaft(
            ElectrolytePcSaft::with_options(parameters.0, options),
        ));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Self(Arc::new(EquationOfState::new(ideal_gas, residual)))
    }

    /// Generalized cubic equations of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : CubicParameters
    ///     The parameters of the equation of state to use.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The equation of state that can be used to compute thermodynamic
    ///     states.
    #[staticmethod]
    pub fn cubic(
        parameters: PyCubicParameters,
        alpha: PyAlpha,
        mixing_rule: PyMixingRule,
        delta: (f64, f64),
    ) -> PyResult<Self> {
        let options = CubicOptions {
            alpha: alpha.0,
            mixing: mixing_rule.0,
            delta: delta.into(),
        };
        let residual = Arc::new(ResidualModel::Cubic(Cubic::new(parameters.0, options)?));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }

    /// Peng-Robinson equations of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : CubicParameters
    ///     The parameters of the equation of state to use.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The equation of state that can be used to compute thermodynamic
    ///     states.
    #[staticmethod]
    pub fn peng_robinson(
        parameters: PyCubicParameters,
        alpha: Option<PyAlpha>,
        mixing_rule: Option<PyMixingRule>,
    ) -> PyResult<Self> {
        let residual = Arc::new(ResidualModel::Cubic(Cubic::peng_robinson(
            parameters.0,
            alpha.map(|a| a.0),
            mixing_rule.map(|mr| mr.0),
        )?));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }

    /// Redlich-Kwong equations of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : CubicParameters
    ///     The parameters of the equation of state to use.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The equation of state that can be used to compute thermodynamic
    ///     states.
    #[staticmethod]
    pub fn redlich_kwong(
        parameters: PyCubicParameters,
        alpha: Option<PyAlpha>,
        mixing_rule: Option<PyMixingRule>,
    ) -> PyResult<Self> {
        let residual = Arc::new(ResidualModel::Cubic(Cubic::redlich_kwong(
            parameters.0,
            alpha.map(|a| a.0),
            mixing_rule.map(|mr| mr.0),
        )?));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
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
    pub fn peng_robinson_old(parameters: PyPengRobinsonParameters) -> Self {
        let residual = Arc::new(ResidualModel::PengRobinson(PengRobinson::new(parameters.0)));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Self(Arc::new(EquationOfState::new(ideal_gas, residual)))
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
    fn pets(parameters: PyPetsParameters, max_eta: f64) -> Self {
        let options = PetsOptions { max_eta };
        let residual = Arc::new(ResidualModel::Pets(Pets::with_options(
            parameters.0,
            options,
        )));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Self(Arc::new(EquationOfState::new(ideal_gas, residual)))
    }

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
        signature = (parameters, max_eta=0.5, perturbation=Perturbation::WeeksChandlerAndersen),
        text_signature = "(parameters, max_eta=0.5, perturbation)"
    )]
    fn uvtheory(
        parameters: PyUVTheoryParameters,
        max_eta: f64,
        perturbation: Perturbation,
    ) -> PyResult<Self> {
        let options = UVTheoryOptions {
            max_eta,
            perturbation,
        };
        let residual = Arc::new(ResidualModel::UVTheory(UVTheory::with_options(
            parameters.0,
            options,
        )));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }

    /// SAFT-VRQ Mie equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : SaftVRQMieParameters
    ///     The parameters of the SAFT-VRQ Mie equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// inc_nonadd_term : bool, optional
    ///     Include non-additive correction to the hard-sphere reference. Defaults to True.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The SAFT-VRQ Mie equation of state that can be used to compute thermodynamic
    ///     states.
    #[cfg(feature = "saftvrqmie")]
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, inc_nonadd_term=true),
        text_signature = "(parameters, max_eta=0.5, inc_nonadd_term=True)"
    )]
    fn saftvrqmie(parameters: PySaftVRQMieParameters, max_eta: f64, inc_nonadd_term: bool) -> Self {
        let options = SaftVRQMieOptions {
            max_eta,
            inc_nonadd_term,
        };
        let residual = Arc::new(ResidualModel::SaftVRQMie(SaftVRQMie::with_options(
            parameters.0,
            options,
        )));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Self(Arc::new(EquationOfState::new(ideal_gas, residual)))
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
    fn joback(&self, joback: PyJoback) -> Self {
        self.add_ideal_gas(IdealGasModel::Joback(joback.0))
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
    fn dippr(&self, dippr: PyDippr) -> Self {
        self.add_ideal_gas(IdealGasModel::Dippr(dippr.0))
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

impl_equation_of_state!(PyEquationOfState);
impl_virial_coefficients!(PyEquationOfState);
impl_state!(EquationOfState<IdealGasModel, ResidualModel>, PyEquationOfState);
impl_state_entropy_scaling!(EquationOfState<IdealGasModel, ResidualModel>, PyEquationOfState);
impl_phase_equilibrium!(EquationOfState<IdealGasModel, ResidualModel>, PyEquationOfState);

#[cfg(feature = "estimator")]
impl_estimator!(EquationOfState<IdealGasModel, ResidualModel>, PyEquationOfState);
#[cfg(all(feature = "estimator", feature = "pcsaft"))]
impl_estimator_entropy_scaling!(EquationOfState<IdealGasModel, ResidualModel>, PyEquationOfState);

#[pymodule]
pub fn eos(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Contributions>()?;
    m.add_class::<Verbosity>()?;

    m.add_class::<PyEquationOfState>()?;
    m.add_class::<PyState>()?;
    m.add_class::<PyStateVec>()?;
    m.add_class::<PyPhaseDiagram>()?;
    m.add_class::<PyPhaseEquilibrium>()?;

    #[cfg(feature = "estimator")]
    m.add_wrapped(wrap_pymodule!(estimator_eos))?;

    Ok(())
}

#[cfg(feature = "estimator")]
#[pymodule]
pub fn estimator_eos(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataSet>()?;
    m.add_class::<PyEstimator>()?;
    m.add_class::<PyLoss>()?;
    m.add_class::<Phase>()
}
