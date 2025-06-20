//! A collection of equation of state models.

mod gc_pcsaft;
pub(crate) mod ideal_gas;
pub(crate) mod pcsaft;
pub use gc_pcsaft::{GcPcSaft, GcPcSaftParameters};
pub use ideal_gas::Joback;
pub use pcsaft::{PcSaftBinary, PcSaftPure};
