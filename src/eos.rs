use feos_core::cubic::PengRobinson;
use feos_core::python::cubic::PyPengRobinsonParameters;
use feos_core::python::user_defined::PyEoSObj;
use feos_core::*;
use feos_pcsaft::python::PyPcSaftParameters;
use feos_pcsaft::{PcSaft, PcSaftOptions};
use feos_pets::python::PyPetsParameters;
use feos_pets::{Pets, PetsOptions};
use ndarray::Array1;
use numpy::convert::ToPyArray;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use quantity::python::*;
use quantity::si::*;
use std::collections::HashMap;
use std::rc::Rc;

pub enum Eos {
    PcSaft(PcSaft),
    PengRobinson(PengRobinson),
    Python(PyEoSObj),
    Pets(Pets),
}

impl EquationOfState for Eos {
    fn components(&self) -> usize {
        match self {
            Eos::PcSaft(eos) => eos.components(),
            Eos::PengRobinson(eos) => eos.components(),
            Eos::Python(eos) => eos.components(),
            Eos::Pets(eos) => eos.components(),
        }
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        match self {
            Eos::PcSaft(eos) => eos.compute_max_density(moles),
            Eos::PengRobinson(eos) => eos.compute_max_density(moles),
            Eos::Python(eos) => eos.compute_max_density(moles),
            Eos::Pets(eos) => eos.compute_max_density(moles),
        }
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        match self {
            Eos::PcSaft(eos) => Self::PcSaft(eos.subset(component_list)),
            Eos::PengRobinson(eos) => Self::PengRobinson(eos.subset(component_list)),
            Eos::Python(eos) => Self::Python(eos.subset(component_list)),
            Eos::Pets(eos) => Self::Pets(eos.subset(component_list)),
        }
    }

    fn residual(&self) -> &[Box<dyn HelmholtzEnergy>] {
        match self {
            Eos::PcSaft(eos) => eos.residual(),
            Eos::PengRobinson(eos) => eos.residual(),
            Eos::Python(eos) => eos.residual(),
            Eos::Pets(eos) => eos.residual(),
        }
    }
}

impl MolarWeight<SIUnit> for Eos {
    fn molar_weight(&self) -> SIArray1 {
        match self {
            Eos::PcSaft(eos) => eos.molar_weight(),
            Eos::PengRobinson(eos) => eos.molar_weight(),
            Eos::Python(eos) => eos.molar_weight(),
            Eos::Pets(eos) => eos.molar_weight(),
        }
    }
}

impl EntropyScaling<SIUnit> for Eos {
    fn viscosity_reference(
        &self,
        temperature: SINumber,
        volume: SINumber,
        moles: &SIArray1,
    ) -> EosResult<SINumber> {
        match self {
            Eos::PcSaft(eos) => eos.viscosity_reference(temperature, volume, moles),
            _ => unimplemented!(),
        }
    }

    fn viscosity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        match self {
            Eos::PcSaft(eos) => eos.viscosity_correlation(s_res, x),
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
            Eos::PcSaft(eos) => eos.diffusion_reference(temperature, volume, moles),
            _ => unimplemented!(),
        }
    }

    fn diffusion_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        match self {
            Eos::PcSaft(eos) => eos.diffusion_correlation(s_res, x),
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
            Eos::PcSaft(eos) => eos.thermal_conductivity_reference(temperature, volume, moles),
            _ => unimplemented!(),
        }
    }

    fn thermal_conductivity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        match self {
            Eos::PcSaft(eos) => eos.thermal_conductivity_correlation(s_res, x),
            _ => unimplemented!(),
        }
    }
}

#[pyclass(name = "EquationOfState", unsendable)]
#[derive(Clone)]
pub struct PyEos(pub Rc<Eos>);

#[pymethods]
impl PyEos {
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
    /// PcSaft
    ///     The PC-SAFT equation of state that can be used to compute thermodynamic
    ///     states.
    #[args(
        max_eta = "0.5",
        max_iter_cross_assoc = "50",
        tol_cross_assoc = "1e-10",
        dq_variant = "\"dq35\""
    )]
    #[staticmethod]
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
        Self(Rc::new(Eos::PcSaft(PcSaft::with_options(
            parameters.0.clone(),
            options,
        ))))
    }

    /// Peng-Robinson equation of state.
    #[staticmethod]
    pub fn peng_robinson(parameters: PyPengRobinsonParameters) -> Self {
        Self(Rc::new(Eos::PengRobinson(PengRobinson::new(
            parameters.0.clone(),
        ))))
    }

    /// Generate equation of state from Python class.
    #[staticmethod]
    fn python(obj: Py<PyAny>) -> PyResult<Self> {
        Ok(Self(Rc::new(Eos::Python(PyEoSObj::new(obj)?))))
    }

    /// Initialize PeTS equation of state.
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
    /// Pets
    ///     The PeTS equation of state that can be used to compute thermodynamic
    ///     states.
    #[args(max_eta = "0.5")]
    #[staticmethod]
    fn pets(parameters: PyPetsParameters, max_eta: f64) -> Self {
        let options = PetsOptions { max_eta };
        Self(Rc::new(Eos::Pets(Pets::with_options(parameters.0.clone(), options))))
    }
}

impl_equation_of_state!(PyEos);
impl_virial_coefficients!(PyEos);
impl_state!(Eos, PyEos);
impl_state_molarweight!(Eos, PyEos);
impl_state_entropy_scaling!(Eos, PyEos);
impl_vle_state!(Eos, PyEos);

#[pymodule]
pub fn eos(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Contributions>()?;
    m.add_class::<Verbosity>()?;

    m.add_class::<PyEos>()?;
    m.add_class::<PyPengRobinsonParameters>()?;
    m.add_class::<PyState>()?;
    m.add_class::<PyPhaseDiagramPure>()?;
    m.add_class::<PyPhaseDiagramBinary>()?;
    m.add_class::<PyPhaseDiagramHetero>()?;
    m.add_class::<PyPhaseEquilibrium>()?;
    Ok(())
}
