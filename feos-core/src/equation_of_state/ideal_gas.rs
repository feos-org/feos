use super::Components;
use crate::StateHD;
use ndarray::Array1;
use num_dual::DualNum;
use std::fmt::Display;

/// Ideal gas Helmholtz energy contribution.
pub trait IdealGas: Components + Sync + Send {
    /// Implementation of an ideal gas model in terms of the
    /// logarithm of the cubic thermal de Broglie wavelength
    /// in units ln(A³) for each component in the system.
    fn ln_lambda3<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D>;

    /// Short description (usually the name) of the model.
    fn ideal_gas_model(&self) -> String;

    /// Evaluate the ideal gas Helmholtz energy contribution for a given state.
    fn ideal_gas_helmholtz_energy<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> D {
        let ln_lambda3 = self.ln_lambda3(state.temperature);
        ((ln_lambda3
            + state.partial_density.mapv(|x| {
                if x.re() == 0.0 {
                    D::from(0.0)
                } else {
                    x.ln() - 1.0
                }
            }))
            * &state.moles)
            .sum()
    }
}
