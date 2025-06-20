//! Collection of ideal gas models.
mod dippr;
mod joback;
pub use dippr::{Dippr, DipprParameters, DipprRecord};
pub use joback::{Joback, JobackParameters, JobackRecord};
