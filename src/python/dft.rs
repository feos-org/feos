use crate::dft::FunctionalVariant;
#[cfg(feature = "estimator")]
use crate::estimator::*;
#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::python::PyGcPcSaftFunctionalParameters;
#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::{GcPcSaftFunctional, GcPcSaftOptions};
use crate::hard_sphere::{FMTFunctional, FMTVersion};
#[cfg(feature = "estimator")]
use crate::impl_estimator;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::python::PyPcSaftParameters;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::{DQVariants, PcSaftFunctional, PcSaftOptions};
#[cfg(feature = "pets")]
use crate::pets::python::PyPetsParameters;
#[cfg(feature = "pets")]
use crate::pets::{PetsFunctional, PetsOptions};
#[cfg(feature = "saftvrqmie")]
use crate::saftvrqmie::python::PySaftVRQMieParameters;
#[cfg(feature = "saftvrqmie")]
use crate::saftvrqmie::{FeynmanHibbsOrder, SaftVRQMieFunctional, SaftVRQMieOptions};

use feos_core::*;
use feos_dft::adsorption::*;
use feos_dft::interface::*;
use feos_dft::python::*;
use feos_dft::solvation::*;
use feos_dft::*;
use numpy::convert::ToPyArray;
use numpy::{PyArray1, PyArray2, PyArray4};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
#[cfg(feature = "estimator")]
use pyo3::wrap_pymodule;
use quantity::python::{PySINumber, PySIArray1, PySIArray2, PySIArray3, PySIArray4};
use quantity::si::*;
use std::collections::HashMap;
use std::sync::Arc;

#[pyclass(name = "HelmholtzEnergyFunctional")]
#[derive(Clone)]
pub struct PyFunctionalVariant(pub Arc<DFT<FunctionalVariant>>);

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
    /// dq_variant : DQVariants, optional
    ///     Combination rule used in the dipole/quadrupole term. Defaults to 'DQVariants.DQ35'
    ///
    /// Returns
    /// -------
    /// HelmholtzEnergyFunctional
    #[cfg(feature = "pcsaft")]
    #[staticmethod]
    #[pyo3(
        signature = (parameters, fmt_version=FMTVersion::WhiteBear, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant=DQVariants::DQ35),
        text_signature = "(parameters, fmt_version, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant)"
    )]
    fn pcsaft(
        parameters: PyPcSaftParameters,
        fmt_version: FMTVersion,
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
        Self(Arc::new(
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
    /// HelmholtzEnergyFunctional
    #[cfg(feature = "gc_pcsaft")]
    #[staticmethod]
    #[pyo3(
        signature = (parameters, fmt_version=FMTVersion::WhiteBear, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10),
        text_signature = "(parameters, fmt_version, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10)"
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
        Self(Arc::new(
            GcPcSaftFunctional::with_options(parameters.0, fmt_version, options).into(),
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
    /// HelmholtzEnergyFunctional
    #[cfg(feature = "pets")]
    #[staticmethod]
    #[pyo3(
        signature = (parameters, fmt_version=FMTVersion::WhiteBear, max_eta=0.5),
        text_signature = "(parameters, fmt_version, max_eta=0.5)"
    )]
    fn pets(parameters: PyPetsParameters, fmt_version: FMTVersion, max_eta: f64) -> Self {
        let options = PetsOptions { max_eta };
        Self(Arc::new(
            PetsFunctional::with_options(parameters.0, fmt_version, options).into(),
        ))
    }

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
    /// HelmholtzEnergyFunctional
    #[staticmethod]
    fn fmt(sigma: &PyArray1<f64>, fmt_version: FMTVersion) -> Self {
        Self(Arc::new(
            FMTFunctional::new(&sigma.to_owned_array(), fmt_version).into(),
        ))
    }

    /// SAFT-VRQ Mie Helmholtz energy functional.
    ///
    /// Parameters
    /// ----------
    /// parameters : SaftVRQMieParameters
    ///     The parameters of the SAFT-VRQ Mie Helmholtz energy functional to use.
    /// fmt_version: FMTVersion, optional
    ///     The specific variant of the FMT term. Defaults to FMTVersion.WhiteBear
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// fh_order : FeynmanHibbsOrder, optional
    ///     Which Feyman-Hibbs correction order to use. Defaults to FeynmanHibbsOrder.FH1.
    ///     Currently, only the first order is implemented.
    /// inc_nonadd_term : bool, optional
    ///     Include non-additive correction to the hard-sphere reference. Defaults to True.
    ///
    /// Returns
    /// -------
    /// HelmholtzEnergyFunctional
    #[cfg(feature = "saftvrqmie")]
    #[staticmethod]
    #[pyo3(
        signature = (parameters, fmt_version=FMTVersion::WhiteBear, max_eta=0.5, fh_order=FeynmanHibbsOrder::FH1, inc_nonadd_term=true),
        text_signature = "(parameters, fmt_version, max_eta=0.5, fh_order, inc_nonadd_term=True)"
    )]
    fn saftvrqmie(
        parameters: PySaftVRQMieParameters,
        fmt_version: FMTVersion,
        max_eta: f64,
        fh_order: FeynmanHibbsOrder,
        inc_nonadd_term: bool,
    ) -> Self {
        let options = SaftVRQMieOptions {
            max_eta,
            fh_order,
            inc_nonadd_term,
        };
        Self(Arc::new(
            SaftVRQMieFunctional::with_options(parameters.0, fmt_version, options).into(),
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

#[cfg(feature = "estimator")]
impl_estimator!(DFT<FunctionalVariant>, PyFunctionalVariant);

#[pymodule]
pub fn dft(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Contributions>()?;
    m.add_class::<Verbosity>()?;

    m.add_class::<PyFunctionalVariant>()?;
    m.add_class::<PyState>()?;
    m.add_class::<PyStateVec>()?;
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

    #[cfg(feature = "estimator")]
    m.add_wrapped(wrap_pymodule!(estimator_dft))?;

    Ok(())
}

#[cfg(feature = "estimator")]
#[pymodule]
pub fn estimator_dft(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDataSet>()?;
    m.add_class::<PyEstimator>()?;
    m.add_class::<PyLoss>()
}
