use super::PyEquationOfState;
use crate::parameter::PyParameters;
use crate::{ideal_gas::IdealGasModel, residual::ResidualModel};
use feos::epcsaft::{ElectrolytePcSaft, ElectrolytePcSaftOptions, ElectrolytePcSaftVariants};
use feos_core::{Components, EquationOfState};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

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
    /// tol_cross_assoc : float, optional
    ///     Tolerance for convergence of cross association. Defaults to 1e-10.
    /// epcsaft_variant : "advanced" | "revised", optional
    ///     Variant of the ePC-SAFT equation of state. Defaults to "advanced"
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The ePC-SAFT equation of state that can be used to compute thermodynamic
    ///     states.
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, epcsaft_variant="advanced"),
        text_signature = r#"(parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, epcsaft_variant="advanced")"#,
    )]
    pub fn epcsaft(
        parameters: PyParameters,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
        epcsaft_variant: &str,
    ) -> PyResult<Self> {
        let epcsaft_variant = match epcsaft_variant {
            "advanced" => ElectrolytePcSaftVariants::Advanced,
            "revised" => ElectrolytePcSaftVariants::Revised,
            _ => {
                return Err(PyErr::new::<PyValueError, _>(
                    r#"epcsaft_variant must be "advanced" or "revised""#.to_string(),
                ))
            }
        };
        let options = ElectrolytePcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            epcsaft_variant,
        };
        let residual = Arc::new(ResidualModel::ElectrolytePcSaft(
            ElectrolytePcSaft::with_options(Arc::new(parameters.try_convert()?), options),
        ));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(residual.components()));
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }
}
