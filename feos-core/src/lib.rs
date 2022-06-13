#![warn(clippy::all)]
#![allow(clippy::reversed_empty_ranges)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]

use quantity::si::*;
use quantity::*;

/// Print messages with level `Verbosity::Iter` or higher.
#[macro_export]
macro_rules! log_iter {
    ($verbosity:expr, $($arg:tt)*) => {
        if $verbosity >= Verbosity::Iter {
            println!($($arg)*);
        }
    }
}

/// Print messages with level `Verbosity::Result` or higher.
#[macro_export]
macro_rules! log_result {
    ($verbosity:expr, $($arg:tt)*) => {
        if $verbosity >= Verbosity::Result {
            println!($($arg)*);
        }
    }
}

pub mod cubic;
mod density_iteration;
mod equation_of_state;
mod errors;
pub mod joback;
pub mod parameter;
mod phase_equilibria;
mod state;
pub use equation_of_state::{
    EntropyScaling, EquationOfState, HelmholtzEnergy, HelmholtzEnergyDual, IdealGasContribution,
    IdealGasContributionDual, MolarWeight,
};
pub use errors::{EosError, EosResult};
pub use phase_equilibria::{
    PhaseDiagram, PhaseDiagramHetero, PhaseEquilibrium, SolverOptions, Verbosity,
};
pub use state::{Contributions, DensityInitialization, State, StateBuilder, StateHD, StateVec};

#[cfg(feature = "python")]
pub mod python;

/// Consistent conversions between quantities and reduced properties.
pub trait EosUnit: Unit {
    fn reference_temperature() -> QuantityScalar<Self>;
    fn reference_length() -> QuantityScalar<Self>;
    fn reference_density() -> QuantityScalar<Self>;
    fn reference_time() -> QuantityScalar<Self>;
    fn gas_constant() -> QuantityScalar<Self>;
    fn reference_volume() -> QuantityScalar<Self> {
        Self::reference_length().powi(3)
    }
    fn reference_velocity() -> QuantityScalar<Self> {
        Self::reference_length() / Self::reference_time()
    }
    fn reference_moles() -> QuantityScalar<Self> {
        Self::reference_density() * Self::reference_volume()
    }
    fn reference_mass() -> QuantityScalar<Self> {
        Self::reference_energy() * Self::reference_velocity().powi(-2)
    }
    fn reference_energy() -> QuantityScalar<Self> {
        Self::gas_constant() * Self::reference_temperature() * Self::reference_moles()
    }
    fn reference_pressure() -> QuantityScalar<Self> {
        Self::reference_energy() / Self::reference_volume()
    }
    fn reference_entropy() -> QuantityScalar<Self> {
        Self::reference_energy() / Self::reference_temperature()
    }
    fn reference_molar_energy() -> QuantityScalar<Self> {
        Self::reference_energy() / Self::reference_moles()
    }
    fn reference_molar_entropy() -> QuantityScalar<Self> {
        Self::reference_entropy() / Self::reference_moles()
    }
    fn reference_surface_tension() -> QuantityScalar<Self> {
        Self::reference_pressure() * Self::reference_length()
    }
    fn reference_influence_parameter() -> QuantityScalar<Self> {
        Self::reference_temperature() * Self::gas_constant() * Self::reference_length().powi(2)
            / Self::reference_density()
    }
    fn reference_molar_mass() -> QuantityScalar<Self> {
        Self::reference_mass() / Self::reference_moles()
    }
    fn reference_viscosity() -> QuantityScalar<Self> {
        Self::reference_pressure() * Self::reference_time()
    }
    fn reference_diffusion() -> QuantityScalar<Self> {
        Self::reference_length().powi(2) / Self::reference_time()
    }
    fn reference_momentum() -> QuantityScalar<Self> {
        Self::reference_molar_mass() * Self::reference_density() * Self::reference_velocity()
    }
}

impl EosUnit for SIUnit {
    fn reference_temperature() -> SINumber {
        KELVIN
    }
    fn reference_length() -> SINumber {
        ANGSTROM
    }
    fn reference_density() -> SINumber {
        ANGSTROM.powi(-3) / NAV
    }
    fn reference_time() -> SINumber {
        PICO * SECOND
    }
    fn gas_constant() -> SINumber {
        RGAS
    }
}
