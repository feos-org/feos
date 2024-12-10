use feos_core::*;
use feos_derive::{Components, Residual};
use ndarray::{Array1, ScalarOperand};
use num_dual::DualNum;
use quantity::*;

#[cfg(feature = "dft")]
mod dft;
mod heterosegmented;
pub mod homosegmented;
#[cfg(feature = "dft")]
pub use dft::{FcSaftFunctional, FcSaftFunctionalContribution};
pub use heterosegmented::{FcSaft, FcSaftParameters, FcSaftRecord};
use homosegmented::FcSaftHomo;

#[cfg(feature = "python")]
pub mod python;

#[derive(Components, Residual)]
pub enum ResidualModel {
    FcSaft(FcSaft),
    FcSaftHomo(FcSaftHomo),
}

#[derive(Copy, Clone)]
pub struct FcSaftOptions {
    pub max_eta: f64,
    pub max_iter_cross_assoc: usize,
    pub tol_cross_assoc: f64,
}

impl Default for FcSaftOptions {
    fn default() -> Self {
        Self {
            max_eta: 0.5,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
        }
    }
}
