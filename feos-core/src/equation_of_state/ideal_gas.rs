use super::Components;
use crate::StateHD;
use ndarray::Array1;
use num_dual::DualNum;
use num_dual::*;
use std::fmt;

/// Ideal gas Helmholtz energy contribution.
pub trait IdealGas: Components + Sync + Send {
    // Return a reference to the implementation of the de Broglie wavelength.
    fn ideal_gas_model(&self) -> &dyn DeBroglieWavelength;

    /// Evaluate the ideal gas Helmholtz energy contribution for a given state.
    ///
    /// In some cases it could be advantageous to overwrite this
    /// implementation instead of implementing the de Broglie
    /// wavelength.
    fn evaluate_ideal_gas<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> D
    where
        for<'a> dyn DeBroglieWavelength + 'a: DeBroglieWavelengthDual<D>,
    {
        let ln_lambda3 = self.ideal_gas_model().ln_lambda3(state.temperature);
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

/// Implementation of an ideal gas model in terms of the
/// logarithm of the cubic thermal de Broglie wavelength
/// in units ln(A³).
///
/// This trait needs to be implemented generically or for
/// the specific types in the supertraits of [DeBroglieWavelength]
/// so that the implementor can be used as an ideal gas
/// contribution in the equation of state.
pub trait DeBroglieWavelengthDual<D: DualNum<f64>> {
    fn ln_lambda3(&self, temperature: D) -> Array1<D>;
}

/// Object safe version of the [DeBroglieWavelengthDual] trait.
///
/// The trait is implemented automatically for every struct that implements
/// the supertraits.
pub trait DeBroglieWavelength:
    DeBroglieWavelengthDual<f64>
    + DeBroglieWavelengthDual<Dual64>
    + DeBroglieWavelengthDual<Dual<DualSVec64<3>, f64>>
    + DeBroglieWavelengthDual<HyperDual64>
    + DeBroglieWavelengthDual<Dual2_64>
    + DeBroglieWavelengthDual<Dual3_64>
    + DeBroglieWavelengthDual<HyperDual<Dual64, f64>>
    + DeBroglieWavelengthDual<HyperDual<DualSVec64<2>, f64>>
    + DeBroglieWavelengthDual<HyperDual<DualSVec64<3>, f64>>
    + DeBroglieWavelengthDual<Dual2<Dual64, f64>>
    + DeBroglieWavelengthDual<Dual3<Dual64, f64>>
    + DeBroglieWavelengthDual<Dual3<DualSVec64<2>, f64>>
    + DeBroglieWavelengthDual<Dual3<DualSVec64<3>, f64>>
    + fmt::Display
    + Send
    + Sync
{
}

impl<T> DeBroglieWavelength for T where
    T: DeBroglieWavelengthDual<f64>
        + DeBroglieWavelengthDual<Dual64>
        + DeBroglieWavelengthDual<Dual<DualSVec64<3>, f64>>
        + DeBroglieWavelengthDual<HyperDual64>
        + DeBroglieWavelengthDual<Dual2_64>
        + DeBroglieWavelengthDual<Dual3_64>
        + DeBroglieWavelengthDual<HyperDual<Dual64, f64>>
        + DeBroglieWavelengthDual<HyperDual<DualSVec64<2>, f64>>
        + DeBroglieWavelengthDual<HyperDual<DualSVec64<3>, f64>>
        + DeBroglieWavelengthDual<Dual2<Dual64, f64>>
        + DeBroglieWavelengthDual<Dual3<Dual64, f64>>
        + DeBroglieWavelengthDual<Dual3<DualSVec64<2>, f64>>
        + DeBroglieWavelengthDual<Dual3<DualSVec64<3>, f64>>
        + fmt::Display
        + Send
        + Sync
{
}
