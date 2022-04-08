use feos_core::*;
use feos_dft::adsorption::*;
use feos_dft::fundamental_measure_theory::FMTVersion;
use feos_dft::interface::*;
use feos_dft::python::*;
use feos_dft::solvation::*;
use feos_dft::*;
use feos_pcsaft::python::PyPcSaftParameters;
use feos_pcsaft::{PcSaftFunctional, PcSaftOptions};
use feos_pets::python::PyPetsParameters;
use feos_pets::{PetsFunctional, PetsOptions};
use ndarray::{Array1, Array2};
use numpy::convert::ToPyArray;
use numpy::{PyArray1, PyArray2, PyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use quantity::python::*;
use quantity::si::*;
use std::collections::HashMap;
use std::rc::Rc;

pub enum FunctionalVariant {
    PcSaftFunctional(PcSaftFunctional),
    PetsFunctional(PetsFunctional),
}

impl From<PcSaftFunctional> for FunctionalVariant {
    fn from(f: PcSaftFunctional) -> Self {
        Self::PcSaftFunctional(f)
    }
}

impl From<PetsFunctional> for FunctionalVariant {
    fn from(f: PetsFunctional) -> Self {
        Self::PetsFunctional(f)
    }
}

impl HelmholtzEnergyFunctional for FunctionalVariant {
    fn subset(&self, component_list: &[usize]) -> DFT<Self> {
        match self {
            FunctionalVariant::PcSaftFunctional(functional) => {
                functional.subset(component_list).into()
            }
            FunctionalVariant::PetsFunctional(functional) => {
                functional.subset(component_list).into()
            }
        }
    }

    fn molecule_shape(&self) -> MoleculeShape {
        match self {
            FunctionalVariant::PcSaftFunctional(functional) => functional.molecule_shape(),
            FunctionalVariant::PetsFunctional(functional) => functional.molecule_shape(),
        }
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        match self {
            FunctionalVariant::PcSaftFunctional(functional) => {
                functional.compute_max_density(moles)
            }
            FunctionalVariant::PetsFunctional(functional) => functional.compute_max_density(moles),
        }
    }

    fn contributions(&self) -> &[Box<dyn FunctionalContribution>] {
        match self {
            FunctionalVariant::PcSaftFunctional(functional) => functional.contributions(),
            FunctionalVariant::PetsFunctional(functional) => functional.contributions(),
        }
    }

    fn ideal_gas(&self) -> &dyn IdealGasContribution {
        match self {
            FunctionalVariant::PcSaftFunctional(functional) => functional.ideal_gas(),
            FunctionalVariant::PetsFunctional(functional) => functional.ideal_gas(),
        }
    }
}

impl MolarWeight<SIUnit> for FunctionalVariant {
    fn molar_weight(&self) -> SIArray1 {
        match self {
            FunctionalVariant::PcSaftFunctional(functional) => functional.molar_weight(),
            FunctionalVariant::PetsFunctional(functional) => functional.molar_weight(),
        }
    }
}

impl FluidParameters for FunctionalVariant {
    fn epsilon_k_ff(&self) -> Array1<f64> {
        match self {
            FunctionalVariant::PcSaftFunctional(functional) => functional.epsilon_k_ff(),
            FunctionalVariant::PetsFunctional(functional) => functional.epsilon_k_ff(),
        }
    }

    fn sigma_ff(&self) -> &Array1<f64> {
        match self {
            FunctionalVariant::PcSaftFunctional(functional) => functional.sigma_ff(),
            FunctionalVariant::PetsFunctional(functional) => functional.sigma_ff(),
        }
    }
}

impl PairPotential for FunctionalVariant {
    fn pair_potential(&self, r: &Array1<f64>) -> Array2<f64> {
        match self {
            FunctionalVariant::PcSaftFunctional(functional) => functional.pair_potential(r),
            FunctionalVariant::PetsFunctional(functional) => functional.pair_potential(r),
        }
    }
}

#[pyclass(name = "Functional", unsendable)]
#[derive(Clone)]
pub struct PyFunctional(pub Rc<DFT<FunctionalVariant>>);

#[pymethods]
impl PyFunctional {
    /// PC-SAFT Helmholtz energy functional.
    ///
    /// Parameters
    /// ----------
    /// parameters: PcSaftParameters
    ///     The set of PC-SAFT parameters.
    /// fmt_version: FMTVersion, optional
    ///     The specific variant of the FMT term. Defaults to FMTVersion.WhiteBear
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
    /// PcSaftFunctional
    #[args(
        fmt_version = "FMTVersion::WhiteBear",
        max_eta = "0.5",
        max_iter_cross_assoc = "50",
        tol_cross_assoc = "1e-10",
        dq_variant = "\"dq35\""
    )]
    #[staticmethod]
    #[pyo3(
        text_signature = "(parameters, fmt_version, max_eta, max_iter_cross_assoc, tol_cross_assoc, dq_variant)"
    )]
    fn pcsaft(
        parameters: PyPcSaftParameters,
        fmt_version: FMTVersion,
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
        Self(Rc::new(
            PcSaftFunctional::with_options(parameters.0, fmt_version, options).into(),
        ))
    }

    /// PeTS Helmholtz energy functional without simplifications
    /// for pure components.
    ///
    /// Parameters
    /// ----------
    /// parameters: PetsParameters
    ///     The set of PeTS parameters.
    /// fmt_version: FMTVersion, optional
    ///     The specific variant of the FMT term. Defaults to FMTVersion.WhiteBear
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    ///
    /// Returns
    /// -------
    /// PetsFunctional
    #[args(fmt_version = "FMTVersion::WhiteBear", max_eta = "0.5")]
    #[staticmethod]
    #[pyo3(text_signature = "(parameters, fmt_version, max_eta)")]
    fn pets(parameters: PyPetsParameters, fmt_version: FMTVersion, max_eta: f64) -> Self {
        let options = PetsOptions { max_eta };
        Self(Rc::new(
            PetsFunctional::with_options(parameters.0, fmt_version, options).into(),
        ))
    }
}

impl_equation_of_state!(PyFunctional);

impl_state!(DFT<FunctionalVariant>, PyFunctional);
impl_state_molarweight!(DFT<FunctionalVariant>, PyFunctional);
impl_phase_equilibrium!(DFT<FunctionalVariant>, PyFunctional);

impl_planar_interface!(FunctionalVariant);
impl_surface_tension_diagram!(FunctionalVariant);

impl_pore!(FunctionalVariant, PyFunctional);
impl_adsorption!(FunctionalVariant, PyFunctional);

impl_pair_correlation!(FunctionalVariant);
impl_solvation_profile!(FunctionalVariant);

#[pymodule]
pub fn dft(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Contributions>()?;
    m.add_class::<Verbosity>()?;

    m.add_class::<PyFunctional>()?;
    m.add_class::<PyState>()?;
    m.add_class::<PyPhaseDiagram>()?;
    m.add_class::<PyPhaseEquilibrium>()?;
    m.add_class::<FMTVersion>()?;

    m.add_class::<PyPlanarInterface>()?;
    m.add_class::<Geometry>()?;
    m.add_class::<PyPore1D>()?;
    m.add_class::<PyPore3D>()?;
    m.add_class::<PyPairCorrelation>()?;
    m.add_class::<PyExternalPotential>()?;
    m.add_class::<PyAdsorption1D>()?;
    m.add_class::<PyAdsorption3D>()?;
    m.add_class::<PySurfaceTensionDiagram>()?;
    m.add_class::<PyDFTSolver>()?;
    m.add_class::<PySolvationProfile>()
}
