use super::Components;
use crate::StateHD;
use ndarray::Array1;
use num_dual::DualNum;

/// Ideal gas Helmholtz energy contribution that can
/// be evaluated using generalized (hyper) dual numbers.
///
/// This trait needs to be implemented generically or for
/// the specific types in the supertraits of [IdealGasContribution]
/// so that the implementor can be used as an ideal gas
/// contribution in the equation of state.
pub trait IdealGas: Components + Sync + Send {
    fn de_broglie_wavelength<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D>;

    fn ideal_gas_model(&self) -> String;

    /// Evaluate the ideal gas contribution for a given state.
    ///
    /// In some cases it could be advantageous to overwrite this
    /// implementation instead of implementing the de Broglie
    /// wavelength.
    fn evaluate_ideal_gas<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> D {
        let lambda = self.de_broglie_wavelength(state.temperature);
        ((lambda
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
