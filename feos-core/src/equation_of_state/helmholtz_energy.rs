use crate::StateHD;
use num_dual::*;
use std::fmt;

/// Individual Helmholtz energy contribution that can
/// be evaluated using generalized (hyper) dual numbers.
///
/// This trait needs to be implemented generically or for
/// the specific types in the supertraits of [HelmholtzEnergy]
/// so that the implementor can be used as a Helmholtz energy
/// contribution in the equation of state.
pub trait HelmholtzEnergyDual<D: DualNum<f64>> {
    /// The Helmholtz energy contribution $\beta A$ of a given state in reduced units.
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D;
}

/// Object safe version of the [HelmholtzEnergyDual] trait.
///
/// The trait is implemented automatically for every struct that implements
/// the supertraits.
pub trait HelmholtzEnergy:
    HelmholtzEnergyDual<f64>
    + HelmholtzEnergyDual<Dual64>
    + HelmholtzEnergyDual<Dual<DualSVec64<3>, f64>>
    + HelmholtzEnergyDual<HyperDual64>
    + HelmholtzEnergyDual<Dual2_64>
    + HelmholtzEnergyDual<Dual3_64>
    + HelmholtzEnergyDual<HyperDual<Dual64, f64>>
    + HelmholtzEnergyDual<HyperDual<DualSVec64<2>, f64>>
    + HelmholtzEnergyDual<HyperDual<DualSVec64<3>, f64>>
    + HelmholtzEnergyDual<Dual2<Dual64, f64>>
    + HelmholtzEnergyDual<Dual3<Dual64, f64>>
    + HelmholtzEnergyDual<Dual3<DualSVec64<2>, f64>>
    + HelmholtzEnergyDual<Dual3<DualSVec64<3>, f64>>
    + fmt::Display
    + Send
    + Sync
{
}

impl<T> HelmholtzEnergy for T where
    T: HelmholtzEnergyDual<f64>
        + HelmholtzEnergyDual<Dual64>
        + HelmholtzEnergyDual<Dual<DualSVec64<3>, f64>>
        + HelmholtzEnergyDual<HyperDual64>
        + HelmholtzEnergyDual<Dual2_64>
        + HelmholtzEnergyDual<Dual3_64>
        + HelmholtzEnergyDual<HyperDual<Dual64, f64>>
        + HelmholtzEnergyDual<HyperDual<DualSVec64<2>, f64>>
        + HelmholtzEnergyDual<HyperDual<DualSVec64<3>, f64>>
        + HelmholtzEnergyDual<Dual2<Dual64, f64>>
        + HelmholtzEnergyDual<Dual3<Dual64, f64>>
        + HelmholtzEnergyDual<Dual3<DualSVec64<2>, f64>>
        + HelmholtzEnergyDual<Dual3<DualSVec64<3>, f64>>
        + fmt::Display
        + Send
        + Sync
{
}
