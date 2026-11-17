use super::PyEquationOfState;
#[cfg(feature = "dft")]
use crate::dft::{PyAssociationVersion, PyFMTVersion, PyHelmholtzEnergyFunctional};
use crate::ideal_gas::IdealGasModel;
use crate::parameter::PyParametersOrGcParameters;
use crate::residual::ResidualModel;
#[cfg(feature = "dft")]
use feos::pcsaft::PcSaftFunctional;
use feos::pcsaft::{DQVariants, PcSaft, PcSaftOptions};
use feos_core::{EquationOfState, ResidualDyn};
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
        signature = (parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant="dq35")
    )]
    pub fn pcsaft(
        parameters: PyParametersOrGcParameters,
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
                    r#"dq_variant must be "dq35" or "dq44""#,
                ));
            }
        };
        let options = PcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            dq_variant,
        };
        let parameters = match parameters {
            PyParametersOrGcParameters::Parameters(parameters) => parameters.try_convert(),
            PyParametersOrGcParameters::GcParameters(parameters) => {
                parameters.try_convert_homosegmented()
            }
        }?;
        let residual = ResidualModel::PcSaft(PcSaft::with_options(parameters, options));
        let ideal_gas = vec![IdealGasModel::NoModel; residual.components()];
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
    /// association_version: AssociationVersion, optional
    ///     The version of the association functional. Defaults to AssociationVersion.ISaft
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
        signature = (parameters, fmt_version=PyFMTVersion::WhiteBear, association_version=PyAssociationVersion::ISaft, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant="dq35")
    )]
    fn pcsaft(
        parameters: crate::parameter::PyParameters,
        fmt_version: PyFMTVersion,
        association_version: PyAssociationVersion,
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
                ));
            }
        };
        let options = PcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            dq_variant,
        };
        let func = ResidualModel::PcSaftFunctional(PcSaftFunctional::with_options(
            parameters.try_convert()?,
            fmt_version.into(),
            association_version.into(),
            options,
        ));
        let ideal_gas = vec![IdealGasModel::NoModel; func.components()];
        Ok(PyEquationOfState(Arc::new(EquationOfState::new(
            ideal_gas, func,
        ))))
    }
}
