pub mod eos;

mod core;
pub use core::{
    Eigen, EquationOfStateAD, HelmholtzEnergyWrapper, IdealGasAD, NamedParameters, ParametersAD,
    PhaseEquilibriumAD, ResidualHelmholtzEnergy, StateAD, TotalHelmholtzEnergy,
};

#[cfg(feature = "parameter_fit")]
mod parameter_fit;
#[cfg(feature = "parameter_fit")]
pub use parameter_fit::{
    bubble_point_pressure, dew_point_pressure, equilibrium_liquid_density, liquid_density,
    vapor_pressure,
};
