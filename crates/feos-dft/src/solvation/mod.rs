//! Solvation free energies and pair correlaion functions.
mod pair_correlation;
pub use pair_correlation::{PairCorrelation, PairPotential};

#[cfg(feature = "rayon")]
mod solvation_profile;
#[cfg(feature = "rayon")]
pub use solvation_profile::SolvationProfile;
