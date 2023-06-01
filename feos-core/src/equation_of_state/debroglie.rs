use ndarray::Array1;
use num_dual::*;
use std::fmt;

pub trait DeBroglieWavelengthDual<D: DualNum<f64>> {
    fn de_broglie_wavelength(&self, temperature: D) -> Array1<D>;
}

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
