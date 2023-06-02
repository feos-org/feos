use crate::StateHD;
use ndarray::Array1;
use num_dual::DualNum;
use num_dual::*;
use std::fmt;

/// Ideal gas Helmholtz energy contribution that can
/// be evaluated using generalized (hyper) dual numbers.
///
/// This trait needs to be implemented generically or for
/// the specific types in the supertraits of [IdealGasContribution]
/// so that the implementor can be used as an ideal gas
/// contribution in the equation of state.
pub trait IdealGasDual<D: DualNum<f64>> {
    /// Return the number of components
    fn components(&self) -> usize;

    /// Return an equation of state consisting of the components
    /// contained in component_list.
    fn subset(&self, component_list: &[usize]) -> Self;

    fn de_broglie_wavelength(&self, temperature: D) -> Array1<D>;

    /// Evaluate the ideal gas contribution for a given state.
    ///
    /// In some cases it could be advantageous to overwrite this
    /// implementation instead of implementing the de Broglie
    /// wavelength.
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
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

pub struct DefaultIdealGas(pub usize);

impl<D: DualNum<f64>> IdealGasDual<D> for DefaultIdealGas {
    fn components(&self) -> usize {
        self.0
    }
    fn subset(&self, component_list: &[usize]) -> Self {
        Self(component_list.len())
    }
    fn de_broglie_wavelength(&self, temperature: D) -> Array1<D> {
        Array1::zeros(self.0)
    }
}

impl fmt::Display for DefaultIdealGas {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ideal gas (default)")
    }
}

pub trait IdealGas:
    IdealGasDual<f64>
    + IdealGasDual<Dual64>
    + IdealGasDual<Dual<DualVec64<3>, f64>>
    + IdealGasDual<HyperDual64>
    + IdealGasDual<Dual2_64>
    + IdealGasDual<Dual3_64>
    + IdealGasDual<HyperDual<Dual64, f64>>
    + IdealGasDual<HyperDual<DualVec64<2>, f64>>
    + IdealGasDual<HyperDual<DualVec64<3>, f64>>
    + IdealGasDual<Dual3<Dual64, f64>>
    + IdealGasDual<Dual3<DualVec64<2>, f64>>
    + IdealGasDual<Dual3<DualVec64<3>, f64>>
    + fmt::Display
    + Send
    + Sync
{
}

impl<T> IdealGas for T where
    T: IdealGasDual<f64>
        + IdealGasDual<Dual64>
        + IdealGasDual<Dual<DualVec64<3>, f64>>
        + IdealGasDual<HyperDual64>
        + IdealGasDual<Dual2_64>
        + IdealGasDual<Dual3_64>
        + IdealGasDual<HyperDual<Dual64, f64>>
        + IdealGasDual<HyperDual<DualVec64<2>, f64>>
        + IdealGasDual<HyperDual<DualVec64<3>, f64>>
        + IdealGasDual<Dual3<Dual64, f64>>
        + IdealGasDual<Dual3<DualVec64<2>, f64>>
        + IdealGasDual<Dual3<DualVec64<3>, f64>>
        + fmt::Display
        + Send
        + Sync
{
}
