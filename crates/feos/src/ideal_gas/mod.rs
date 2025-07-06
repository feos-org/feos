//! Collection of ideal gas models.
mod dippr;
mod joback;
pub use dippr::{Dippr, DipprRecord};
pub use joback::{Joback, JobackRecord};
