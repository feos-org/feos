#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::{GcPcSaftFunctional, GcPcSaftOptions};
use crate::hard_sphere::{FMTFunctional, FMTVersion};
use crate::ideal_gas::IdealGasModel;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::{DQVariants, PcSaftFunctional, PcSaftOptions};
#[cfg(feature = "pets")]
use crate::pets::{PetsFunctional, PetsOptions};
#[cfg(feature = "saftvrqmie")]
use crate::saftvrqmie::{SaftVRQMieFunctional, SaftVRQMieOptions};
use crate::ResidualModel;

use super::eos::{PyEquationOfState, PyPhaseEquilibrium, PyState, PyStateVec};
use feos_core::parameter::ParameterError;
#[cfg(feature = "gc_pcsaft")]
use feos_core::python::parameter::PyGcParameters;
#[cfg(any(feature = "pcsaft", feature = "pets", feature = "saftvrqmie"))]
use feos_core::python::parameter::PyParameters;
use feos_core::*;
use feos_dft::adsorption::*;
use feos_dft::interface::*;
use feos_dft::python::*;
use feos_dft::solvation::*;
use feos_dft::*;
use ndarray::{Array1, Array2, Array3, Array4};
use numpy::prelude::*;
use numpy::{PyArray1, PyArray2, PyArray3, PyArray4};
use pyo3::prelude::*;
use quantity::*;
use std::convert::TryInto;
use std::sync::Arc;
use typenum::Quot;

#[pyclass(name = "HelmholtzEnergyFunctional")]
#[derive(Clone)]
pub struct PyHelmholtzEnergyFunctional;

#[pymethods]
impl PyHelmholtzEnergyFunctional {
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
        parameters: PyParameters,
        fmt_version: FMTVersion,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
        dq_variant: DQVariants,
    ) -> Result<PyEquationOfState, ParameterError> {
        let options = PcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            dq_variant,
        };
        let func = Arc::new(ResidualModel::PcSaftFunctional(
            PcSaftFunctional::with_options(
                Arc::new(parameters.try_convert()?),
                fmt_version,
                options,
            ),
        ));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(func.components()));
        Ok(PyEquationOfState(Arc::new(EquationOfState::new(
            ideal_gas, func,
        ))))
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
        parameters: PyGcParameters,
        fmt_version: FMTVersion,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
    ) -> Result<PyEquationOfState, ParameterError> {
        let options = GcPcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
        };
        let func = Arc::new(ResidualModel::GcPcSaftFunctional(
            GcPcSaftFunctional::with_options(
                Arc::new(parameters.try_convert_heterosegmented()?),
                fmt_version,
                options,
            ),
        ));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(func.components()));
        Ok(PyEquationOfState(Arc::new(EquationOfState::new(
            ideal_gas, func,
        ))))
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
    fn pets(
        parameters: PyParameters,
        fmt_version: FMTVersion,
        max_eta: f64,
    ) -> Result<PyEquationOfState, ParameterError> {
        let options = PetsOptions { max_eta };
        let func = Arc::new(ResidualModel::PetsFunctional(PetsFunctional::with_options(
            Arc::new(parameters.try_convert()?),
            fmt_version,
            options,
        )));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(func.components()));
        Ok(PyEquationOfState(Arc::new(EquationOfState::new(
            ideal_gas, func,
        ))))
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
    fn fmt(
        sigma: &Bound<'_, PyArray1<f64>>,
        fmt_version: FMTVersion,
    ) -> Result<PyEquationOfState, ParameterError> {
        let func = Arc::new(ResidualModel::FmtFunctional(FMTFunctional::new(
            &sigma.to_owned_array(),
            fmt_version,
        )));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(func.components()));
        Ok(PyEquationOfState(Arc::new(EquationOfState::new(
            ideal_gas, func,
        ))))
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
    /// inc_nonadd_term : bool, optional
    ///     Include non-additive correction to the hard-sphere reference. Defaults to True.
    ///
    /// Returns
    /// -------
    /// HelmholtzEnergyFunctional
    #[cfg(feature = "saftvrqmie")]
    #[staticmethod]
    #[pyo3(
        signature = (parameters, fmt_version=FMTVersion::WhiteBear, max_eta=0.5, inc_nonadd_term=true),
        text_signature = "(parameters, fmt_version, max_eta=0.5, inc_nonadd_term=True)"
    )]
    fn saftvrqmie(
        parameters: PyParameters,
        fmt_version: FMTVersion,
        max_eta: f64,
        inc_nonadd_term: bool,
    ) -> Result<PyEquationOfState, ParameterError> {
        let options = SaftVRQMieOptions {
            max_eta,
            inc_nonadd_term,
        };
        let func = Arc::new(ResidualModel::SaftVRQMieFunctional(
            SaftVRQMieFunctional::with_options(
                Arc::new(parameters.try_convert()?),
                fmt_version,
                options,
            ),
        ));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(func.components()));
        Ok(PyEquationOfState(Arc::new(EquationOfState::new(
            ideal_gas, func,
        ))))
    }
}

impl_planar_interface!(EquationOfState<IdealGasModel, ResidualModel>);
impl_surface_tension_diagram!(EquationOfState<IdealGasModel, ResidualModel>);

impl_pore!(EquationOfState<IdealGasModel, ResidualModel>, PyEquationOfState);
impl_adsorption!(EquationOfState<IdealGasModel, ResidualModel>, PyEquationOfState);

impl_pair_correlation!(EquationOfState<IdealGasModel, ResidualModel>);
impl_solvation_profile!(EquationOfState<IdealGasModel, ResidualModel>);

#[pymodule]
pub fn dft(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FMTVersion>()?;
    m.add_class::<PyHelmholtzEnergyFunctional>()?;

    m.add_class::<PyPlanarInterface>()?;
    m.add_class::<Geometry>()?;
    m.add_class::<PyPore1D>()?;
    m.add_class::<PyPore2D>()?;
    m.add_class::<PyPore3D>()?;
    m.add_class::<PyPairCorrelation>()?;
    m.add_class::<PyExternalPotential>()?;
    m.add_class::<PyAdsorption1D>()?;
    m.add_class::<PyAdsorption3D>()?;
    m.add_class::<PySurfaceTensionDiagram>()?;
    m.add_class::<PyDFTSolver>()?;
    m.add_class::<PySolvationProfile>()?;

    Ok(())
}
