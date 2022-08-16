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
use ndarray::Array1;
use numpy::convert::ToPyArray;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
#[cfg(feature = "estimator")]
use pyo3::wrap_pymodule;
use quantity::python::*;
use quantity::si::*;
use std::collections::HashMap;
use std::rc::Rc;

pub enum EosVariant {
    #[cfg(feature = "pcsaft")]
    PcSaft(PcSaft),
    #[cfg(feature = "gc_pcsaft")]
    GcPcSaft(GcPcSaft),
    PengRobinson(PengRobinson),
    Python(PyEoSObj),
    #[cfg(feature = "pets")]
    Pets(Pets),
    #[cfg(feature = "uvtheory")]
    UVTheory(UVTheory),
}

impl EquationOfState for EosVariant {
    fn components(&self) -> usize {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.components(),
            #[cfg(feature = "gc_pcsaft")]
            EosVariant::GcPcSaft(eos) => eos.components(),
            EosVariant::PengRobinson(eos) => eos.components(),
            EosVariant::Python(eos) => eos.components(),
            #[cfg(feature = "pets")]
            EosVariant::Pets(eos) => eos.components(),
            #[cfg(feature = "uvtheory")]
            EosVariant::UVTheory(eos) => eos.components(),
        }
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.compute_max_density(moles),
            #[cfg(feature = "gc_pcsaft")]
            EosVariant::GcPcSaft(eos) => eos.compute_max_density(moles),
            EosVariant::PengRobinson(eos) => eos.compute_max_density(moles),
            EosVariant::Python(eos) => eos.compute_max_density(moles),
            #[cfg(feature = "pets")]
            EosVariant::Pets(eos) => eos.compute_max_density(moles),
            #[cfg(feature = "uvtheory")]
            EosVariant::UVTheory(eos) => eos.compute_max_density(moles),
        }
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => Self::PcSaft(eos.subset(component_list)),
            #[cfg(feature = "gc_pcsaft")]
            EosVariant::GcPcSaft(eos) => Self::GcPcSaft(eos.subset(component_list)),
            EosVariant::PengRobinson(eos) => Self::PengRobinson(eos.subset(component_list)),
            EosVariant::Python(eos) => Self::Python(eos.subset(component_list)),
            #[cfg(feature = "pets")]
            EosVariant::Pets(eos) => Self::Pets(eos.subset(component_list)),
            #[cfg(feature = "uvtheory")]
            EosVariant::UVTheory(eos) => Self::UVTheory(eos.subset(component_list)),
        }
    }

    fn residual(&self) -> &[Box<dyn HelmholtzEnergy>] {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.residual(),
            #[cfg(feature = "gc_pcsaft")]
            EosVariant::GcPcSaft(eos) => eos.residual(),
            EosVariant::PengRobinson(eos) => eos.residual(),
            EosVariant::Python(eos) => eos.residual(),
            #[cfg(feature = "pets")]
            EosVariant::Pets(eos) => eos.residual(),
            #[cfg(feature = "uvtheory")]
            EosVariant::UVTheory(eos) => eos.residual(),
        }
    }

    fn ideal_gas(&self) -> &dyn IdealGasContribution {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.ideal_gas(),
            #[cfg(feature = "gc_pcsaft")]
            EosVariant::GcPcSaft(eos) => eos.ideal_gas(),
            EosVariant::PengRobinson(eos) => eos.ideal_gas(),
            EosVariant::Python(eos) => eos.ideal_gas(),
            #[cfg(feature = "pets")]
            EosVariant::Pets(eos) => eos.ideal_gas(),
            #[cfg(feature = "uvtheory")]
            EosVariant::UVTheory(eos) => eos.ideal_gas(),
        }
    }
}

impl MolarWeight<SIUnit> for EosVariant {
    fn molar_weight(&self) -> SIArray1 {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.molar_weight(),
            #[cfg(feature = "gc_pcsaft")]
            EosVariant::GcPcSaft(eos) => eos.molar_weight(),
            EosVariant::PengRobinson(eos) => eos.molar_weight(),
            EosVariant::Python(eos) => eos.molar_weight(),
            #[cfg(feature = "pets")]
            EosVariant::Pets(eos) => eos.molar_weight(),
            #[cfg(feature = "uvtheory")]
            EosVariant::UVTheory(_) => unimplemented!(),
        }
    }
}

#[cfg(feature = "pcsaft")]
impl EntropyScaling<SIUnit> for EosVariant {
    fn viscosity_reference(
        &self,
        temperature: SINumber,
        volume: SINumber,
        moles: &SIArray1,
    ) -> EosResult<SINumber> {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.viscosity_reference(temperature, volume, moles),
            _ => unimplemented!(),
        }
    }

    fn viscosity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.viscosity_correlation(s_res, x),
            _ => unimplemented!(),
        }
    }

    fn diffusion_reference(
        &self,
        temperature: SINumber,
        volume: SINumber,
        moles: &SIArray1,
    ) -> EosResult<SINumber> {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.diffusion_reference(temperature, volume, moles),
            _ => unimplemented!(),
        }
    }

    fn diffusion_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.diffusion_correlation(s_res, x),
            _ => unimplemented!(),
        }
    }

    fn thermal_conductivity_reference(
        &self,
        temperature: SINumber,
        volume: SINumber,
        moles: &SIArray1,
    ) -> EosResult<SINumber> {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => {
                eos.thermal_conductivity_reference(temperature, volume, moles)
            }
            _ => unimplemented!(),
        }
    }

    fn thermal_conductivity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.thermal_conductivity_correlation(s_res, x),
            _ => unimplemented!(),
        }
    }
}

#[pyclass(name = "EquationOfState", unsendable)]
#[derive(Clone)]
pub struct PyEosVariant(pub Rc<EosVariant>);

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
        Self(Rc::new(EosVariant::PcSaft(PcSaft::with_options(
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
        Self(Rc::new(EosVariant::GcPcSaft(GcPcSaft::with_options(
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
        Self(Rc::new(EosVariant::PengRobinson(PengRobinson::new(
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
        Ok(Self(Rc::new(EosVariant::Python(PyEoSObj::new(obj)?))))
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
        Self(Rc::new(EosVariant::Pets(Pets::with_options(
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
        Self(Rc::new(EosVariant::UVTheory(UVTheory::with_options(
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
