use super::PyEquationOfState;
use crate::parameter::PyParameters;
use crate::{ideal_gas::IdealGasModel, residual::ResidualModel};
use feos::epcsaft::{ElectrolytePcSaft, ElectrolytePcSaftOptions, ElectrolytePcSaftVariants};
use feos_core::{Components, EquationOfState};
use pyo3::prelude::*;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq)]
#[pyclass(name = "ElectrolytePcSaftVariants", eq, eq_int)]
pub enum PyElectrolytePcSaftVariants {
    Advanced,
    Revised,
}

impl From<ElectrolytePcSaftVariants> for PyElectrolytePcSaftVariants {
    fn from(value: ElectrolytePcSaftVariants) -> Self {
        use ElectrolytePcSaftVariants::*;
        match value {
            Advanced => Self::Advanced,
            Revised => Self::Revised,
        }
    }
}

impl From<PyElectrolytePcSaftVariants> for ElectrolytePcSaftVariants {
    fn from(value: PyElectrolytePcSaftVariants) -> Self {
        use PyElectrolytePcSaftVariants::*;
        match value {
            Advanced => Self::Advanced,
            Revised => Self::Revised,
        }
    }
}

#[pymethods]
impl PyEquationOfState {
    /// ePC-SAFT equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : ElectrolytePcSaftParameters
    ///     The parameters of the PC-SAFT equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// max_iter_cross_assoc : unsigned integer, optional
    ///     Maximum number of iterations for cross association. Defaults to 50.
    /// tol_cross_assoc : float
    ///     Tolerance for convergence of cross association. Defaults to 1e-10.
    /// epcsaft_variant : ElectrolytePcSaftVariants, optional
    ///     Variant of the ePC-SAFT equation of state. Defaults to 'advanced'
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The ePC-SAFT equation of state that can be used to compute thermodynamic
    ///     states.
    #[cfg(feature = "epcsaft")]
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, epcsaft_variant=PyElectrolytePcSaftVariants::Advanced),
        text_signature = "(parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, epcsaft_variant)",
    )]
    pub fn epcsaft(
        parameters: PyParameters,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
        epcsaft_variant: PyElectrolytePcSaftVariants,
    ) -> PyResult<Self> {
        let options = ElectrolytePcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            epcsaft_variant: epcsaft_variant.into(),
        };
        let residual = Arc::new(ResidualModel::ElectrolytePcSaft(
            ElectrolytePcSaft::with_options(Arc::new(parameters.try_convert()?), options),
        ));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }
}
