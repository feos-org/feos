//! A collection of equation of state models.

pub(crate) mod ideal_gas;
pub use ideal_gas::Joback;

#[cfg(any(feature = "pcsaft", feature = "gc_pcsaft"))]
pub(crate) mod pcsaft;
#[cfg(feature = "pcsaft")]
pub use pcsaft::{PcSaftBinary, PcSaftPure};

#[cfg(feature = "gc_pcsaft")]
mod gc_pcsaft;
#[cfg(feature = "gc_pcsaft")]
pub use gc_pcsaft::{GcPcSaft, GcPcSaftParameters};
