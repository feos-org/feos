use super::AlphaFunction;
use crate::cubic::parameters::CubicParameters;
use feos_core::{FeosError, FeosResult};
use ndarray::{Array1, Zip};
use num_dual::DualNum;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MathiasCopeman(pub Vec<[f64; 3]>);

impl AlphaFunction for MathiasCopeman {
    #[inline]
    fn alpha<D: DualNum<f64> + Copy>(
        &self,
        _: &Array1<f64>,
        reduced_temperature: &Array1<D>,
    ) -> Array1<D> {
        Zip::from(reduced_temperature)
            .and(&self.0)
            .map_collect(|tr, c| {
                let trsq = -tr.sqrt() + 1.0;
                let a1 = trsq + c[0] + 1.0;
                let a2 = match tr {
                    tr if tr.re() < 1.0 => trsq * (c[1] + c[2]),
                    _ => D::zero(),
                };
                (a1 + a2).powi(2)
            })
    }

    fn validate(&self, parameters: &Arc<CubicParameters>) -> FeosResult<()> {
        if self.0.len() == parameters.tc.len() {
            Ok(())
        } else {
            Err(FeosError::IncompatibleParameters(
                format!(
                    "Mathias Copeman alpha function was initialized for {} components, but the equation of state contains {}.",
                    self.0.len(), parameters.tc.len()
                )
            ).into())
        }
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        let mi = component_list.iter().map(|&i| self.0[i]).collect();
        Self(mi)
    }
}
