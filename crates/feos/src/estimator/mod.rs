//! Utilities for working with experimental data.
use feos_core::{DensityInitialization, FeosError};
mod dataset;
pub use dataset::DataSet;
#[expect(clippy::module_inception)]
mod estimator;
pub use estimator::Estimator;
mod loss;
pub use loss::Loss;

// Properties
mod vapor_pressure;
pub use vapor_pressure::VaporPressure;
mod liquid_density;
pub use liquid_density::{EquilibriumLiquidDensity, LiquidDensity};
mod binary_vle;
pub use binary_vle::{BinaryPhaseDiagram, BinaryVleChemicalPotential, BinaryVlePressure};
mod viscosity;
pub use viscosity::Viscosity;
mod thermal_conductivity;
pub use thermal_conductivity::ThermalConductivity;
mod diffusion;
pub use diffusion::Diffusion;

/// Different phases of experimental data points.
#[derive(Clone, Copy, PartialEq)]
pub enum Phase {
    Vapor,
    Liquid,
}

impl From<Phase> for DensityInitialization {
    fn from(value: Phase) -> Self {
        match value {
            Phase::Liquid => DensityInitialization::Liquid,
            Phase::Vapor => DensityInitialization::Vapor,
        }
    }
}
