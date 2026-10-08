//! Fused-chain SAFT (FC-SAFT)
//!
//! Equation of state and Helmholtz energy functional for (heterosegmented)
//! chains of fused hard spheres with explicit bond lengths.

#[cfg(feature = "dft")]
mod dft;
mod eos;
pub mod homosegmented;
mod parameters;
mod record;
mod reference;

#[cfg(feature = "dft")]
pub use dft::{FcSaftFunctional, FcSaftFunctionalContribution};
pub use eos::{DispersionConstants, FcSaft};
pub use homosegmented::{FcSaftBinary, FcSaftHomo, FcSaftPure};
pub use parameters::FcSaftPars;
pub use record::{
    FcSaftAssociationRecord, FcSaftBinaryRecord, FcSaftBondRecord, FcSaftParameters, FcSaftRecord,
    new_pure_homosegmented,
};

/// Customization options for the FC-SAFT equation of state and functional.
#[derive(Copy, Clone)]
pub struct FcSaftOptions {
    /// maximum packing fraction
    pub max_eta: f64,
    /// maximum number of iterations for cross association calculation
    pub max_iter_cross_assoc: usize,
    /// tolerance for cross association calculation
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
