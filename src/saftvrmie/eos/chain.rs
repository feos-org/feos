use std::f64::consts::{FRAC_PI_6, PI};

use feos_core::StateHD;
use itertools::Itertools;
use ndarray::{Array1, Array2, ScalarOperand};
use num_dual::DualNum;

use crate::hard_sphere::HardSphereProperties;

use super::SaftVRMieParameters;

pub(super) fn a_chain<D: DualNum<f64> + Copy>() -> D {
    unimplemented!()
}
