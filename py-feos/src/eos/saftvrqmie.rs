use super::PyEquationOfState;
#[cfg(feature = "dft")]
use crate::dft::{PyFMTVersion, PyHelmholtzEnergyFunctional};
use crate::error::PyFeosError;
use crate::ideal_gas::IdealGasModel;
use crate::parameter::PyParameters;
use crate::residual::ResidualModel;
use feos::saftvrqmie::{SaftVRQMie, SaftVRQMieOptions};
use feos_core::{EquationOfState, ResidualDyn};
use pyo3::prelude::*;
use std::sync::Arc;

#[pymethods]
impl PyEquationOfState {
    /// SAFT-VRQ Mie equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : SaftVRQMieParameters
    ///     The parameters of the SAFT-VRQ Mie equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// inc_nonadd_term : bool, optional
    ///     Include non-additive correction to the hard-sphere reference. Defaults to True.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The SAFT-VRQ Mie equation of state that can be used to compute thermodynamic
    ///     states.
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, inc_nonadd_term=true),
        text_signature = "(parameters, max_eta=0.5, inc_nonadd_term=True)"
    )]
    fn saftvrqmie(parameters: PyParameters, max_eta: f64, inc_nonadd_term: bool) -> PyResult<Self> {
        let options = SaftVRQMieOptions {
            max_eta,
            inc_nonadd_term,
            ..Default::default()
        };
        let residual = ResidualModel::SaftVRQMie(
            SaftVRQMie::with_options(parameters.try_convert()?, options)
                .map_err(PyFeosError::from)?,
        );
        let ideal_gas = vec![IdealGasModel::NoModel; residual.components()];
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }
}

#[cfg(feature = "dft")]
#[pymethods]
impl PyHelmholtzEnergyFunctional {
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
    #[staticmethod]
    #[pyo3(
        signature = (parameters, fmt_version=PyFMTVersion::WhiteBear, max_eta=0.5, inc_nonadd_term=true),
        text_signature = "(parameters, fmt_version, max_eta=0.5, inc_nonadd_term=True)"
    )]
    fn saftvrqmie(
        parameters: PyParameters,
        fmt_version: PyFMTVersion,
        max_eta: f64,
        inc_nonadd_term: bool,
    ) -> PyResult<PyEquationOfState> {
        use crate::error::PyFeosError;

        let options = SaftVRQMieOptions {
            max_eta,
            inc_nonadd_term,
            fmt_version: fmt_version.into(),
        };
        let func = ResidualModel::SaftVRQMieFunctional(
            SaftVRQMie::with_options(parameters.try_convert()?, options)
                .map_err(PyFeosError::from)?,
        );
        let ideal_gas = vec![IdealGasModel::NoModel; func.components()];
        Ok(PyEquationOfState(Arc::new(EquationOfState::new(
            ideal_gas, func,
        ))))
    }
}
