use super::PyEquationOfState;
#[cfg(feature = "dft")]
use crate::dft::{PyFMTVersion, PyHelmholtzEnergyFunctional};
use crate::{ideal_gas::IdealGasModel, parameter::PyGcParameters, residual::ResidualModel};
#[cfg(feature = "dft")]
use feos::fcsaft::FcSaftFunctional;
use feos::fcsaft::{FcSaft, FcSaftOptions};
use feos_core::{EquationOfState, ResidualDyn};
use pyo3::prelude::*;
use std::sync::Arc;

#[pymethods]
impl PyEquationOfState {
    /// (heterosegmented) fused-chain SAFT (FC-SAFT) equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : GcParameters
    ///     The parameters of the FC-SAFT equation of state to use. Requires
    ///     bond records that specify the bond lengths between segments.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// max_iter_cross_assoc : unsigned integer, optional
    ///     Maximum number of iterations for cross association. Defaults to 50.
    /// tol_cross_assoc : float
    ///     Tolerance for convergence of cross association. Defaults to 1e-10.
    /// model_params : [[float]], optional
    ///     Model constants [a1, a2, b1, b2] (each with 7 entries) of the
    ///     dispersion contribution. Defaults to the published FC-SAFT constants.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The FC-SAFT equation of state that can be used to compute thermodynamic
    ///     states.
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, model_params=None),
        text_signature = "(parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, model_params=None)"
    )]
    pub fn fcsaft(
        parameters: PyGcParameters,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
        model_params: Option<[[f64; 7]; 4]>,
    ) -> PyResult<Self> {
        let options = FcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
        };
        let residual = ResidualModel::FcSaft(FcSaft::with_options(
            parameters.try_convert_heterosegmented_with_bonds()?,
            options,
            model_params,
        ));
        let ideal_gas = vec![IdealGasModel::NoModel; residual.components()];
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }
}

#[cfg(feature = "dft")]
#[pymethods]
impl PyHelmholtzEnergyFunctional {
    /// (heterosegmented) fused-chain SAFT (FC-SAFT) Helmholtz energy functional.
    ///
    /// Parameters
    /// ----------
    /// parameters: GcParameters
    ///     The set of FC-SAFT parameters. Requires bond records that specify
    ///     the bond lengths between segments.
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
    #[staticmethod]
    #[pyo3(
        signature = (parameters, fmt_version=PyFMTVersion::WhiteBear, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10),
        text_signature = "(parameters, fmt_version, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10)"
    )]
    fn fcsaft(
        parameters: PyGcParameters,
        fmt_version: PyFMTVersion,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
    ) -> PyResult<PyEquationOfState> {
        let options = FcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
        };
        let func = ResidualModel::FcSaftFunctional(FcSaftFunctional::with_options(
            parameters.try_convert_heterosegmented_with_bonds()?,
            fmt_version.into(),
            options,
        ));
        let ideal_gas = vec![IdealGasModel::NoModel; func.components()];
        Ok(PyEquationOfState(Arc::new(EquationOfState::new(
            ideal_gas, func,
        ))))
    }
}
