//! Utilities for working with experimental data.
use feos_core::EosError;
use quantity::QuantityError;
use std::num::ParseFloatError;
use thiserror::Error;

mod dataset;
pub use dataset::DataSet;
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
pub use binary_vle::{BinaryPhaseDiagram, BinaryVleChemicalPotential, BinaryVlePressure, Phase};
mod viscosity;
pub use viscosity::Viscosity;
mod thermal_conductivity;
pub use thermal_conductivity::ThermalConductivity;
mod diffusion;
pub use diffusion::Diffusion;

#[cfg(feature = "python")]
pub mod python;

#[derive(Debug, Error)]
pub enum EstimatorError {
    #[error("Input has not the same amount of data as the target.")]
    IncompatibleInput,
    #[error(transparent)]
    ShapeError(#[from] ndarray::ShapeError),
    #[error(transparent)]
    ParseError(#[from] ParseFloatError),
    #[error(transparent)]
    QuantityError(#[from] QuantityError),
    #[error(transparent)]
    EosError(#[from] EosError),
}
