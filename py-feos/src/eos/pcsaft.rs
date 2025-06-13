use super::PyEquationOfState;
#[cfg(feature = "dft")]
use crate::dft::{PyFMTVersion, PyHelmholtzEnergyFunctional};
use crate::ideal_gas::IdealGasModel;
use crate::parameter::{PyGcParameters, PyParameters};
use crate::residual::ResidualModel;
#[cfg(feature = "dft")]
use feos::pcsaft::PcSaftFunctional;
use feos::pcsaft::{DQVariants, PcSaft, PcSaftOptions};
use feos_core::{Components, EquationOfState};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

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
    /// tol_cross_assoc : float, optional
    ///     Tolerance for convergence of cross association. Defaults to 1e-10.
    /// dq_variant : "dq35" | "dq44", optional
    ///     Combination rule used in the dipole/quadrupole term. Defaults to "dq35"
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The PC-SAFT equation of state that can be used to compute thermodynamic
    ///     states.
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant="dq35"),
        text_signature = r#"(parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant="dq35")"#
    )]
    pub fn pcsaft(
        parameters: &Bound<'_, PyAny>,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
        dq_variant: &str,
    ) -> PyResult<Self> {
        let dq_variant = match dq_variant {
            "dq35" => DQVariants::DQ35,
            "dq44" => DQVariants::DQ44,
            _ => {
                return Err(PyErr::new::<PyValueError, _>(
                    r#"dq_variant must be "dq35" or "dq44""#.to_string(),
                ))
            }
        };
        let options = PcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            dq_variant,
        };
        let parameters = if let Ok(parameters) = parameters.extract::<PyParameters>() {
            parameters.try_convert()
        } else if let Ok(parameters) = parameters.extract::<PyGcParameters>() {
            parameters.try_convert_homosegmented()
        } else {
            todo!()
        }?;
        let residual = Arc::new(ResidualModel::PcSaft(PcSaft::with_options(
            parameters, options,
        )));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }
}

#[cfg(feature = "dft")]
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
        signature = (parameters, fmt_version=PyFMTVersion::WhiteBear, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant="dq35"),
        text_signature = r#"(parameters, fmt_version, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant="dq35")"#
    )]
    fn pcsaft(
        parameters: crate::parameter::PyParameters,
        fmt_version: PyFMTVersion,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
        dq_variant: &str,
    ) -> PyResult<PyEquationOfState> {
        let dq_variant = match dq_variant {
            "dq35" => DQVariants::DQ35,
            "dq44" => DQVariants::DQ44,
            _ => {
                return Err(PyErr::new::<PyValueError, _>(
                    r#"dq_variant must be "dq35" or "dq44""#.to_string(),
                ))
            }
        };
        let options = PcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            dq_variant,
        };
        let func = Arc::new(ResidualModel::PcSaftFunctional(
            PcSaftFunctional::with_options(parameters.try_convert()?, fmt_version.into(), options),
        ));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(func.components()));
        Ok(PyEquationOfState(Arc::new(EquationOfState::new(
            ideal_gas, func,
        ))))
    }
}
