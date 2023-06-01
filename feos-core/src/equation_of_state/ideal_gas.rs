use crate::StateHD;
use ndarray::Array1;
use num_dual::DualNum;
use std::fmt;

use super::debroglie::{DeBroglieWavelength, DeBroglieWavelengthDual};

/// Ideal gas Helmholtz energy contribution that can
/// be evaluated using generalized (hyper) dual numbers.
///
/// This trait needs to be implemented generically or for
/// the specific types in the supertraits of [IdealGasContribution]
/// so that the implementor can be used as an ideal gas
/// contribution in the equation of state.
pub trait IdealGas: Sync + Send + fmt::Display {
    // /// Return the number of components
    // fn components(&self) -> usize;

    // /// Return an equation of state consisting of the components
    // /// contained in component_list.
    // fn subset(&self, component_list: &[usize]) -> Self;

    fn de_broglie_wavelength(&self) -> &Box<dyn DeBroglieWavelength>;

    /// Evaluate the ideal gas contribution for a given state.
    ///
    /// In some cases it could be advantageous to overwrite this
    /// implementation instead of implementing the de Broglie
    /// wavelength.
    fn evaluate_ideal_gas<D: DualNum<f64>>(&self, state: &StateHD<D>) -> D
    where
        dyn DeBroglieWavelength: DeBroglieWavelengthDual<D>,
    {
        let lambda = self
            .de_broglie_wavelength()
            .de_broglie_wavelength(state.temperature);
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
