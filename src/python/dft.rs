#[cfg(feature = "fit")]
use crate::fit::*;
#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::python::PyGcPcSaftFunctionalParameters;
#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::{GcPcSaftFunctional, GcPcSaftOptions};
#[cfg(feature = "fit")]
use crate::impl_estimator;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::python::PyPcSaftParameters;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::{PcSaftFunctional, PcSaftOptions};
use feos_core::*;
use feos_dft::adsorption::*;
use feos_dft::fundamental_measure_theory::{FMTFunctional, FMTVersion};
use feos_dft::interface::*;
use feos_dft::python::*;
use feos_dft::solvation::*;
use feos_dft::*;
// use feos_pets::python::PyPetsParameters;
// use feos_pets::{PetsFunctional, PetsOptions};
use ndarray::{Array1, Array2};
use numpy::convert::ToPyArray;
use numpy::{PyArray1, PyArray2, PyArray4};
#[cfg(feature = "gc_pcsaft")]
use petgraph::graph::UnGraph;
#[cfg(feature = "gc_pcsaft")]
use petgraph::Graph;
// use petgraph::graph::{Graph, UnGraph};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
#[cfg(feature = "fit")]
use pyo3::wrap_pymodule;
use quantity::python::*;
use quantity::si::*;
use std::collections::HashMap;
use std::rc::Rc;

pub enum FunctionalVariant {
    #[cfg(feature = "pcsaft")]
    PcSaft(PcSaftFunctional),
    #[cfg(feature = "gc_pcsaft")]
    GcPcSaft(GcPcSaftFunctional),
    // Pets(PetsFunctional),
    Fmt(FMTFunctional),
}

#[cfg(feature = "pcsaft")]
impl From<PcSaftFunctional> for FunctionalVariant {
    fn from(f: PcSaftFunctional) -> Self {
        Self::PcSaft(f)
    }
}

#[cfg(feature = "gc_pcsaft")]
impl From<GcPcSaftFunctional> for FunctionalVariant {
    fn from(f: GcPcSaftFunctional) -> Self {
        Self::GcPcSaft(f)
    }
}

// impl From<PetsFunctional> for FunctionalVariant {
//     fn from(f: PetsFunctional) -> Self {
//         Self::Pets(f)
//     }
// }

impl From<FMTFunctional> for FunctionalVariant {
    fn from(f: FMTFunctional) -> Self {
        Self::Fmt(f)
    }
}

impl HelmholtzEnergyFunctional for FunctionalVariant {
    fn subset(&self, component_list: &[usize]) -> DFT<Self> {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.subset(component_list).into(),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(functional) => functional.subset(component_list).into(),
            // FunctionalVariant::Pets(functional) => functional.subset(component_list).into(),
            FunctionalVariant::Fmt(functional) => functional.subset(component_list).into(),
        }
    }

    fn molecule_shape(&self) -> MoleculeShape {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.molecule_shape(),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(functional) => functional.molecule_shape(),
            // FunctionalVariant::Pets(functional) => functional.molecule_shape(),
            FunctionalVariant::Fmt(functional) => functional.molecule_shape(),
        }
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.compute_max_density(moles),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(functional) => functional.compute_max_density(moles),
            // FunctionalVariant::Pets(functional) => functional.compute_max_density(moles),
            FunctionalVariant::Fmt(functional) => functional.compute_max_density(moles),
        }
    }

    fn contributions(&self) -> &[Box<dyn FunctionalContribution>] {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.contributions(),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(functional) => functional.contributions(),
            // FunctionalVariant::Pets(functional) => functional.contributions(),
            FunctionalVariant::Fmt(functional) => functional.contributions(),
        }
    }

    fn ideal_gas(&self) -> &dyn IdealGasContribution {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.ideal_gas(),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(functional) => functional.ideal_gas(),
            // FunctionalVariant::Pets(functional) => functional.ideal_gas(),
            FunctionalVariant::Fmt(functional) => functional.ideal_gas(),
        }
    }

    #[cfg(feature = "gc_pcsaft")]
    fn bond_lengths(&self, temperature: f64) -> UnGraph<(), f64> {
        match self {
            FunctionalVariant::GcPcSaft(functional) => functional.bond_lengths(temperature),
            _ => Graph::with_capacity(0, 0),
        }
    }
}

impl MolarWeight<SIUnit> for FunctionalVariant {
    fn molar_weight(&self) -> SIArray1 {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.molar_weight(),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(functional) => functional.molar_weight(),
            // FunctionalVariant::Pets(functional) => functional.molar_weight(),
            _ => unimplemented!(),
        }
    }
}

impl FluidParameters for FunctionalVariant {
    fn epsilon_k_ff(&self) -> Array1<f64> {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.epsilon_k_ff(),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(functional) => functional.epsilon_k_ff(),
            // FunctionalVariant::Pets(functional) => functional.epsilon_k_ff(),
            FunctionalVariant::Fmt(functional) => functional.epsilon_k_ff(),
        }
    }

    fn sigma_ff(&self) -> &Array1<f64> {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.sigma_ff(),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(functional) => functional.sigma_ff(),
            // FunctionalVariant::Pets(functional) => functional.sigma_ff(),
            FunctionalVariant::Fmt(functional) => functional.sigma_ff(),
        }
    }
}

impl PairPotential for FunctionalVariant {
    fn pair_potential(&self, i: usize, r: &Array1<f64>, temperature: f64) -> Array2<f64> {
        match self {
            #[cfg(feature = "pcsaft")]
            FunctionalVariant::PcSaft(functional) => functional.pair_potential(i, r, temperature),
            #[cfg(feature = "gc_pcsaft")]
            FunctionalVariant::GcPcSaft(_) => unimplemented!(),
            // FunctionalVariant::Pets(functional) => functional.pair_potential(i, r, temperature),
            FunctionalVariant::Fmt(functional) => functional.pair_potential(i, r, temperature),
        }
    }
}

#[pyclass(name = "HelmholtzEnergyFunctional", unsendable)]
#[derive(Clone)]
pub struct PyFunctionalVariant(pub Rc<DFT<FunctionalVariant>>);

#[pymethods]
impl PyFunctionalVariant {
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
    /// Functional
    #[cfg(feature = "pcsaft")]
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

    /// (heterosegmented) group contribution PC-SAFT Helmholtz energy functional.
    ///
    /// Parameters
    /// ----------
    /// parameters: GcPcSaftFunctionalParameters
    ///     The set of PC-SAFT parameters.
    /// fmt_version: FMTVersion, optional
    ///     The specific variant of the FMT term. Defaults to FMTVersion.WhiteBear
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// max_iter_cross_assoc : unsigned integer, optional
    ///     Maximum number of iterations for cross association. Defaults to 50.
    /// tol_cross_assoc : float
    ///     Tolerance for convergence of cross association. Defaults to 1e-10.
    ///
    /// Returns
    /// -------
    /// Functional
    #[cfg(feature = "gc_pcsaft")]
    #[args(
        fmt_version = "FMTVersion::WhiteBear",
        max_eta = "0.5",
        max_iter_cross_assoc = "50",
        tol_cross_assoc = "1e-10"
    )]
    #[staticmethod]
    #[pyo3(
        text_signature = "(parameters, fmt_version, max_eta, max_iter_cross_assoc, tol_cross_assoc)"
    )]
    fn gc_pcsaft(
        parameters: PyGcPcSaftFunctionalParameters,
        fmt_version: FMTVersion,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
    ) -> Self {
        let options = GcPcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
        };
        Self(Rc::new(
            GcPcSaftFunctional::with_options(parameters.0, fmt_version, options).into(),
        ))
    }

    // /// PeTS Helmholtz energy functional without simplifications
    // /// for pure components.
    // ///
    // /// Parameters
    // /// ----------
    // /// parameters: PetsParameters
    // ///     The set of PeTS parameters.
    // /// fmt_version: FMTVersion, optional
    // ///     The specific variant of the FMT term. Defaults to FMTVersion.WhiteBear
    // /// max_eta : float, optional
    // ///     Maximum packing fraction. Defaults to 0.5.
    // ///
    // /// Returns
    // /// -------
    // /// Functional
    // #[args(fmt_version = "FMTVersion::WhiteBear", max_eta = "0.5")]
    // #[staticmethod]
    // #[pyo3(text_signature = "(parameters, fmt_version, max_eta)")]
    // fn pets(parameters: PyPetsParameters, fmt_version: FMTVersion, max_eta: f64) -> Self {
    //     let options = PetsOptions { max_eta };
    //     Self(Rc::new(
    //         PetsFunctional::with_options(parameters.0, fmt_version, options).into(),
    //     ))
    // }

    /// Helmholtz energy functional for hard sphere systems.
    ///
    /// Parameters
    /// ----------
    /// sigma : numpy.ndarray[float]
    ///     The diameters of the hard spheres in Angstrom.
    /// fmt_version : FMTVersion
    ///     The specific variant of the FMT term.
    ///
    /// Returns
    /// -------
    /// Functional
    #[staticmethod]
    #[pyo3(text_signature = "(sigma, version)")]
    fn fmt(sigma: &PyArray1<f64>, fmt_version: FMTVersion) -> Self {
        Self(Rc::new(
            FMTFunctional::new(&sigma.to_owned_array(), fmt_version).into(),
        ))
    }
}

impl_equation_of_state!(PyFunctionalVariant);

impl_state!(DFT<FunctionalVariant>, PyFunctionalVariant);
impl_state_molarweight!(DFT<FunctionalVariant>, PyFunctionalVariant);
impl_phase_equilibrium!(DFT<FunctionalVariant>, PyFunctionalVariant);

impl_planar_interface!(FunctionalVariant);
impl_surface_tension_diagram!(FunctionalVariant);

impl_pore!(FunctionalVariant, PyFunctionalVariant);
impl_adsorption!(FunctionalVariant, PyFunctionalVariant);

impl_pair_correlation!(FunctionalVariant);
impl_solvation_profile!(FunctionalVariant);

#[cfg(feature = "fit")]
impl_estimator!(DFT<FunctionalVariant>, PyFunctionalVariant);

#[pymodule]
pub fn dft(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Contributions>()?;
    m.add_class::<Verbosity>()?;

    m.add_class::<PyFunctionalVariant>()?;
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
    m.add_class::<PySolvationProfile>()?;

    #[cfg(feature = "fit")]
    m.add_wrapped(wrap_pymodule!(estimator_dft))?;

    Ok(())
}

#[cfg(feature = "fit")]
#[pymodule]
pub fn estimator_dft(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDataSet>()?;
    m.add_class::<PyEstimator>()?;
    m.add_class::<PyLoss>()
}
