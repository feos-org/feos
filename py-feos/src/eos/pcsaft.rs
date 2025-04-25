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
use pyo3::prelude::*;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq)]
#[pyclass(name = "DQVariants", eq, eq_int)]
pub enum PyDQVariants {
    DQ35,
    DQ44,
}

impl From<DQVariants> for PyDQVariants {
    fn from(value: DQVariants) -> Self {
        use DQVariants::*;
        match value {
            DQ35 => Self::DQ35,
            DQ44 => Self::DQ44,
        }
    }
}

impl From<PyDQVariants> for DQVariants {
    fn from(value: PyDQVariants) -> Self {
        use PyDQVariants::*;
        match value {
            DQ35 => Self::DQ35,
            DQ44 => Self::DQ44,
        }
    }
}

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
    /// dq_variant : DQVariants, optional
    ///     Combination rule used in the dipole/quadrupole term. Defaults to 'DQVariants.DQ35'
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The PC-SAFT equation of state that can be used to compute thermodynamic
    ///     states.
    #[cfg(feature = "pcsaft")]
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant=PyDQVariants::DQ35),
        text_signature = "(parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant)"
    )]
    pub fn pcsaft(
        parameters: &Bound<'_, PyAny>,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
        dq_variant: PyDQVariants,
    ) -> PyResult<Self> {
        let options = PcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            dq_variant: dq_variant.into(),
        };
        let parameters = if let Ok(parameters) = parameters.extract::<PyParameters>() {
            parameters.try_convert()
        } else if let Ok(parameters) = parameters.extract::<PyGcParameters>() {
            parameters.try_convert_homosegmented()
        } else {
            todo!()
        }?;
        let residual = Arc::new(ResidualModel::PcSaft(PcSaft::with_options(
            Arc::new(parameters),
            options,
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
        signature = (parameters, fmt_version=PyFMTVersion::WhiteBear, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant=PyDQVariants::DQ35),
        text_signature = "(parameters, fmt_version, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant)"
    )]
    fn pcsaft(
        parameters: crate::parameter::PyParameters,
        fmt_version: PyFMTVersion,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
        dq_variant: PyDQVariants,
    ) -> PyResult<PyEquationOfState> {
        let options = PcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            dq_variant: dq_variant.into(),
        };
        let func = Arc::new(ResidualModel::PcSaftFunctional(
            PcSaftFunctional::with_options(
                Arc::new(parameters.try_convert()?),
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
