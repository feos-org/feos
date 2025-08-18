pub mod eos;

mod core;
pub use core::{HelmholtzEnergyWrapper, NamedParameters, ParametersAD};

#[cfg(feature = "parameter_fit")]
mod parameter_fit;
#[cfg(feature = "parameter_fit")]
pub use parameter_fit::{BinaryModel, PureModel};
