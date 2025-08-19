use super::PyEquationOfState;
#[cfg(feature = "dft")]
use crate::dft::{PyFMTVersion, PyHelmholtzEnergyFunctional};
use crate::{ideal_gas::IdealGasModel, parameter::PyGcParameters, residual::ResidualModel};
#[cfg(feature = "dft")]
use feos::gc_pcsaft::GcPcSaftFunctional;
use feos::gc_pcsaft::{GcPcSaft, GcPcSaftOptions};
use feos_core::{EquationOfState, ResidualDyn};
use pyo3::prelude::*;
use std::sync::Arc;

#[pymethods]
impl PyEquationOfState {
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
        parameters: PyGcParameters,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
    ) -> PyResult<Self> {
        let options = GcPcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
        };
        let residual = ResidualModel::GcPcSaft(GcPcSaft::with_options(
            parameters.try_convert_heterosegmented()?,
            options,
        ));
        let ideal_gas = vec![IdealGasModel::NoModel; residual.components()];
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }
}

#[cfg(feature = "dft")]
#[pymethods]
impl PyHelmholtzEnergyFunctional {
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
        signature = (parameters, fmt_version=PyFMTVersion::WhiteBear, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10),
        text_signature = "(parameters, fmt_version, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10)"
    )]
    fn gc_pcsaft(
        parameters: PyGcParameters,
        fmt_version: PyFMTVersion,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
    ) -> PyResult<PyEquationOfState> {
        let options = GcPcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
        };
        let func = Arc::new(ResidualModel::GcPcSaftFunctional(
            GcPcSaftFunctional::with_options(
                parameters.try_convert_heterosegmented()?,
                fmt_version.into(),
                options,
            ),
        ));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(func.components()));
        Ok(PyEquationOfState(Arc::new(EquationOfState::new(
            ideal_gas, func,
        ))))
    }
}
