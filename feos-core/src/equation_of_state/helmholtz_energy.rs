use crate::{Residual, StateHD};
use num_dual::*;
use std::fmt;

/// Individual Helmholtz energy contribution that can
/// be evaluated using generalized (hyper) dual numbers.
///
/// This trait needs to be implemented generically or for
/// the specific types in the supertraits of [HelmholtzEnergy]
/// so that the implementor can be used as a Helmholtz energy
/// contribution in the equation of state.
pub trait HelmholtzEnergyDual<P, D: DualNum<f64>> {
    /// The Helmholtz energy contribution $\beta A$ of a given state in reduced units.
    fn helmholtz_energy(&self, state: &StateHD<D>, properties: &P) -> D;
}

/// Object safe version of the [HelmholtzEnergyDual] trait.
///
/// The trait is implemented automatically for every struct that implements
/// the supertraits.
pub trait HelmholtzEnergy<P: Residual>: HelmholtzEnergyDual<P::Properties<f64>, f64>
    + HelmholtzEnergyDual<P::Properties<Dual64>, Dual64>
    + HelmholtzEnergyDual<P::Properties<Dual<DualSVec64<3>, f64>>, Dual<DualSVec64<3>, f64>>
    + HelmholtzEnergyDual<P::Properties<HyperDual64>, HyperDual64>
    + HelmholtzEnergyDual<P::Properties<Dual2_64>, Dual2_64>
    + HelmholtzEnergyDual<P::Properties<Dual3_64>, Dual3_64>
    + HelmholtzEnergyDual<P::Properties<HyperDual<Dual64, f64>>, HyperDual<Dual64, f64>>
    + HelmholtzEnergyDual<P::Properties<HyperDual<DualSVec64<2>, f64>>, HyperDual<DualSVec64<2>, f64>>
    + HelmholtzEnergyDual<P::Properties<HyperDual<DualSVec64<3>, f64>>, HyperDual<DualSVec64<3>, f64>>
    + HelmholtzEnergyDual<P::Properties<Dual2<Dual64, f64>>, Dual2<Dual64, f64>>
    + HelmholtzEnergyDual<P::Properties<Dual3<Dual64, f64>>, Dual3<Dual64, f64>>
    + HelmholtzEnergyDual<P::Properties<Dual3<DualSVec64<2>, f64>>, Dual3<DualSVec64<2>, f64>>
    + HelmholtzEnergyDual<P::Properties<Dual3<DualSVec64<3>, f64>>, Dual3<DualSVec64<3>, f64>>
    + fmt::Display
    + Send
    + Sync
{
}

impl<T, P: Residual> HelmholtzEnergy<P> for T where
    T: HelmholtzEnergyDual<P::Properties<f64>, f64>
        + HelmholtzEnergyDual<P::Properties<Dual64>, Dual64>
        + HelmholtzEnergyDual<P::Properties<Dual<DualSVec64<3>, f64>>, Dual<DualSVec64<3>, f64>>
        + HelmholtzEnergyDual<P::Properties<HyperDual64>, HyperDual64>
        + HelmholtzEnergyDual<P::Properties<Dual2_64>, Dual2_64>
        + HelmholtzEnergyDual<P::Properties<Dual3_64>, Dual3_64>
        + HelmholtzEnergyDual<P::Properties<HyperDual<Dual64, f64>>, HyperDual<Dual64, f64>>
        + HelmholtzEnergyDual<
            P::Properties<HyperDual<DualSVec64<2>, f64>>,
            HyperDual<DualSVec64<2>, f64>,
        > + HelmholtzEnergyDual<
            P::Properties<HyperDual<DualSVec64<3>, f64>>,
            HyperDual<DualSVec64<3>, f64>,
        > + HelmholtzEnergyDual<P::Properties<Dual2<Dual64, f64>>, Dual2<Dual64, f64>>
        + HelmholtzEnergyDual<P::Properties<Dual3<Dual64, f64>>, Dual3<Dual64, f64>>
        + HelmholtzEnergyDual<P::Properties<Dual3<DualSVec64<2>, f64>>, Dual3<DualSVec64<2>, f64>>
        + HelmholtzEnergyDual<P::Properties<Dual3<DualSVec64<3>, f64>>, Dual3<DualSVec64<3>, f64>>
        + fmt::Display
        + Send
        + Sync
{
}
