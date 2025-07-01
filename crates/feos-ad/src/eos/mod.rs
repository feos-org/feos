//! A collection of equation of state models.

pub(crate) mod ideal_gas;
pub use ideal_gas::Joback;

#[cfg(feature = "gc_pcsaft")]
mod gc_pcsaft;
#[cfg(feature = "gc_pcsaft")]
pub use gc_pcsaft::{GcPcSaft, GcPcSaftParameters};

#[cfg(feature = "pcsaft")]
pub(crate) mod pcsaft;
#[cfg(feature = "pcsaft")]
pub use pcsaft::{PcSaftBinary, PcSaftPure};
