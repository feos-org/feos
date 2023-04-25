use feos_core::StateHD;
use num_dual::{
    Dual, Dual2_64, Dual3, Dual3_64, Dual64, DualNum, DualVec64, HyperDual, HyperDual64,
};

pub trait UFractionDual<D: DualNum<f64>> {
    fn ufraction(&self, state: &StateHD<D>) -> D;
}

pub trait UFraction:
    UFractionDual<f64>
    + UFractionDual<Dual64>
    + UFractionDual<Dual<DualVec64<3>, f64>>
    + UFractionDual<HyperDual64>
    + UFractionDual<Dual2_64>
    + UFractionDual<Dual3_64>
    + UFractionDual<HyperDual<Dual64, f64>>
    + UFractionDual<HyperDual<DualVec64<2>, f64>>
    + UFractionDual<HyperDual<DualVec64<3>, f64>>
    + UFractionDual<Dual3<Dual64, f64>>
    + UFractionDual<Dual3<DualVec64<2>, f64>>
    + UFractionDual<Dual3<DualVec64<3>, f64>>
    + Send
    + Sync
{
}

impl<T> UFraction for T where
    T: UFractionDual<f64>
        + UFractionDual<Dual64>
        + UFractionDual<Dual<DualVec64<3>, f64>>
        + UFractionDual<HyperDual64>
        + UFractionDual<Dual2_64>
        + UFractionDual<Dual3_64>
        + UFractionDual<HyperDual<Dual64, f64>>
        + UFractionDual<HyperDual<DualVec64<2>, f64>>
        + UFractionDual<HyperDual<DualVec64<3>, f64>>
        + UFractionDual<Dual3<Dual64, f64>>
        + UFractionDual<Dual3<DualVec64<2>, f64>>
        + UFractionDual<Dual3<DualVec64<3>, f64>>
        + Send
        + Sync
{
}