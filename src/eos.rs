use feos_core::cubic::PengRobinson;
use feos_core::python::cubic::PyPengRobinsonParameters;
use feos_core::python::user_defined::PyEoSObj;
use feos_core::*;
use feos_gc_pcsaft::python::PyGcPcSaftEosParameters;
use feos_gc_pcsaft::{GcPcSaft, GcPcSaftOptions};
use feos_pcsaft::python::PyPcSaftParameters;
use feos_pcsaft::{PcSaft, PcSaftOptions};
use feos_pets::python::PyPetsParameters;
use feos_pets::{Pets, PetsOptions};
use feos_uvtheory::python::PyUVParameters;
use feos_uvtheory::{Perturbation, UVTheory, UVTheoryOptions};
use ndarray::Array1;
use numpy::convert::ToPyArray;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use quantity::python::*;
use quantity::si::*;
use std::collections::HashMap;
use std::rc::Rc;

pub enum EosVariant {
    PcSaft(PcSaft),
    GcPcSaft(GcPcSaft),
    PengRobinson(PengRobinson),
    Python(PyEoSObj),
    Pets(Pets),
    UVTheory(UVTheory),
}

impl EquationOfState for EosVariant {
    fn components(&self) -> usize {
        match self {
            EosVariant::PcSaft(eos) => eos.components(),
            EosVariant::GcPcSaft(eos) => eos.components(),
            EosVariant::PengRobinson(eos) => eos.components(),
            EosVariant::Python(eos) => eos.components(),
            EosVariant::Pets(eos) => eos.components(),
            EosVariant::UVTheory(eos) => eos.components(),
        }
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        match self {
            EosVariant::PcSaft(eos) => eos.compute_max_density(moles),
            EosVariant::GcPcSaft(eos) => eos.compute_max_density(moles),
            EosVariant::PengRobinson(eos) => eos.compute_max_density(moles),
            EosVariant::Python(eos) => eos.compute_max_density(moles),
            EosVariant::Pets(eos) => eos.compute_max_density(moles),
            EosVariant::UVTheory(eos) => eos.compute_max_density(moles),
        }
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        match self {
            EosVariant::PcSaft(eos) => Self::PcSaft(eos.subset(component_list)),
            EosVariant::GcPcSaft(eos) => Self::GcPcSaft(eos.subset(component_list)),
            EosVariant::PengRobinson(eos) => Self::PengRobinson(eos.subset(component_list)),
            EosVariant::Python(eos) => Self::Python(eos.subset(component_list)),
            EosVariant::Pets(eos) => Self::Pets(eos.subset(component_list)),
            EosVariant::UVTheory(eos) => Self::UVTheory(eos.subset(component_list)),
        }
    }

    fn residual(&self) -> &[Box<dyn HelmholtzEnergy>] {
        match self {
            EosVariant::PcSaft(eos) => eos.residual(),
            EosVariant::GcPcSaft(eos) => eos.residual(),
            EosVariant::PengRobinson(eos) => eos.residual(),
            EosVariant::Python(eos) => eos.residual(),
            EosVariant::Pets(eos) => eos.residual(),
            EosVariant::UVTheory(eos) => eos.residual(),
        }
    }
}

impl MolarWeight<SIUnit> for EosVariant {
    fn molar_weight(&self) -> SIArray1 {
        match self {
            EosVariant::PcSaft(eos) => eos.molar_weight(),
            EosVariant::GcPcSaft(eos) => eos.molar_weight(),
            EosVariant::PengRobinson(eos) => eos.molar_weight(),
            EosVariant::Python(eos) => eos.molar_weight(),
            EosVariant::Pets(eos) => eos.molar_weight(),
            _ => unimplemented!(),
        }
    }
}

impl EntropyScaling<SIUnit> for EosVariant {
    fn viscosity_reference(
        &self,
        temperature: SINumber,
        volume: SINumber,
        moles: &SIArray1,
    ) -> EosResult<SINumber> {
        match self {
            EosVariant::PcSaft(eos) => eos.viscosity_reference(temperature, volume, moles),
            _ => unimplemented!(),
        }
    }

    fn viscosity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        match self {
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
            EosVariant::PcSaft(eos) => eos.diffusion_reference(temperature, volume, moles),
            _ => unimplemented!(),
        }
    }

    fn diffusion_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        match self {
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
            EosVariant::PcSaft(eos) => {
                eos.thermal_conductivity_reference(temperature, volume, moles)
            }
            _ => unimplemented!(),
        }
    }

    fn thermal_conductivity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        match self {
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
    /// dq_variant : {'dq35', 'dq44'}, optional
    ///     Combination rule used in the dipole/quadrupole term. Defaults to 'dq35'
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The PC-SAFT equation of state that can be used to compute thermodynamic
    ///     states.
    #[args(
        max_eta = "0.5",
        max_iter_cross_assoc = "50",
        tol_cross_assoc = "1e-10",
        dq_variant = "\"dq35\""
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
        dq_variant: &str,
    ) -> Self {
        let options = PcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            dq_variant: dq_variant.into(),
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
    /// parameters : PetsParameters
    ///     The parameters of the PeTS equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// perturbation : Perturbation, optional
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The UV-Theory equation of state that can be used to compute thermodynamic
    ///     states.
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
impl_state_entropy_scaling!(EosVariant, PyEosVariant);
impl_phase_equilibrium!(EosVariant, PyEosVariant);

#[pymodule]
pub fn eos(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Contributions>()?;
    m.add_class::<Verbosity>()?;

    m.add_class::<PyEosVariant>()?;
    m.add_class::<PyState>()?;
    m.add_class::<PyPhaseDiagram>()?;
    m.add_class::<PyPhaseEquilibrium>()
}
