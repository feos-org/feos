use super::{DensityInitialization, State};
use crate::equation_of_state::EquationOfState;
use crate::errors::EosResult;
use crate::EosUnit;
use ndarray::Array1;
use quantity::{QuantityArray1, QuantityScalar};
use std::rc::Rc;

/// A simple tool to construct [State]s with arbitrary input parameters.
///
/// # Examples
/// ```
/// # use feos_core::{EosResult, StateBuilder};
/// # use feos_core::cubic::{PengRobinson, PengRobinsonParameters};
/// # use quantity::si::*;
/// # use std::rc::Rc;
/// # use ndarray::arr1;
/// # use approx::assert_relative_eq;
/// # fn main() -> EosResult<()> {
/// // Create a state for given T,V,N
/// let eos = Rc::new(PengRobinson::new(Rc::new(PengRobinsonParameters::new_simple(&[369.8], &[41.9 * 1e5], &[0.15], &[15.0])?)));
/// let state = StateBuilder::new(&eos)
///                 .temperature(300.0 * KELVIN)
///                 .volume(12.5 * METER.powi(3))
///                 .moles(&(arr1(&[2.5]) * MOL))
///                 .build()?;
/// assert_eq!(state.density, 0.2 * MOL / METER.powi(3));
///
/// // For a pure component, the composition does not need to be specified.
/// let eos = Rc::new(PengRobinson::new(Rc::new(PengRobinsonParameters::new_simple(&[369.8], &[41.9 * 1e5], &[0.15], &[15.0])?)));
/// let state = StateBuilder::new(&eos)
///                 .temperature(300.0 * KELVIN)
///                 .volume(12.5 * METER.powi(3))
///                 .total_moles(2.5 * MOL)
///                 .build()?;
/// assert_eq!(state.density, 0.2 * MOL / METER.powi(3));
///
/// // The state can be constructed without providing any extensive property.
/// let eos = Rc::new(PengRobinson::new(
///     Rc::new(PengRobinsonParameters::new_simple(
///         &[369.8, 305.4],
///         &[41.9 * 1e5, 48.2 * 1e5],
///         &[0.15, 0.10],
///         &[15.0, 30.0]
///     )?)
/// ));
/// let state = StateBuilder::new(&eos)
///                 .temperature(300.0 * KELVIN)
///                 .partial_density(&(arr1(&[0.2, 0.6]) * MOL / METER.powi(3)))
///                 .build()?;
/// assert_relative_eq!(state.molefracs, arr1(&[0.25, 0.75]));
/// assert_relative_eq!(state.density, 0.8 * MOL / METER.powi(3));
/// # Ok(())
/// # }
/// ```
pub struct StateBuilder<'a, U: EosUnit, E: EquationOfState> {
    eos: Rc<E>,
    temperature: Option<QuantityScalar<U>>,
    volume: Option<QuantityScalar<U>>,
    density: Option<QuantityScalar<U>>,
    partial_density: Option<&'a QuantityArray1<U>>,
    total_moles: Option<QuantityScalar<U>>,
    moles: Option<&'a QuantityArray1<U>>,
    molefracs: Option<&'a Array1<f64>>,
    pressure: Option<QuantityScalar<U>>,
    molar_enthalpy: Option<QuantityScalar<U>>,
    molar_entropy: Option<QuantityScalar<U>>,
    molar_internal_energy: Option<QuantityScalar<U>>,
    density_initialization: DensityInitialization<U>,
    initial_temperature: Option<QuantityScalar<U>>,
}

impl<'a, U: EosUnit, E: EquationOfState> StateBuilder<'a, U, E> {
    /// Create a new `StateBuilder` for the given equation of state.
    pub fn new(eos: &Rc<E>) -> Self {
        StateBuilder {
            eos: eos.clone(),
            temperature: None,
            volume: None,
            density: None,
            partial_density: None,
            total_moles: None,
            moles: None,
            molefracs: None,
            pressure: None,
            molar_enthalpy: None,
            molar_entropy: None,
            molar_internal_energy: None,
            density_initialization: DensityInitialization::None,
            initial_temperature: None,
        }
    }

    /// Provide the temperature for the new state.
    pub fn temperature(mut self, temperature: QuantityScalar<U>) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Provide the volume for the new state.
    pub fn volume(mut self, volume: QuantityScalar<U>) -> Self {
        self.volume = Some(volume);
        self
    }

    /// Provide the density for the new state.
    pub fn density(mut self, density: QuantityScalar<U>) -> Self {
        self.density = Some(density);
        self
    }

    /// Provide partial densities for the new state.
    pub fn partial_density(mut self, partial_density: &'a QuantityArray1<U>) -> Self {
        self.partial_density = Some(partial_density);
        self
    }

    /// Provide the total moles for the new state.
    pub fn total_moles(mut self, total_moles: QuantityScalar<U>) -> Self {
        self.total_moles = Some(total_moles);
        self
    }

    /// Provide the moles for the new state.
    pub fn moles(mut self, moles: &'a QuantityArray1<U>) -> Self {
        self.moles = Some(moles);
        self
    }

    /// Provide the molefracs for the new state.
    pub fn molefracs(mut self, molefracs: &'a Array1<f64>) -> Self {
        self.molefracs = Some(molefracs);
        self
    }

    /// Provide the pressure for the new state.
    pub fn pressure(mut self, pressure: QuantityScalar<U>) -> Self {
        self.pressure = Some(pressure);
        self
    }

    /// Provide the molar enthalpy for the new state.
    pub fn molar_enthalpy(mut self, molar_enthalpy: QuantityScalar<U>) -> Self {
        self.molar_enthalpy = Some(molar_enthalpy);
        self
    }

    /// Provide the molar entropy for the new state.
    pub fn molar_entropy(mut self, molar_entropy: QuantityScalar<U>) -> Self {
        self.molar_entropy = Some(molar_entropy);
        self
    }

    /// Provide the molar internal energy for the new state.
    pub fn molar_internal_energy(mut self, molar_internal_energy: QuantityScalar<U>) -> Self {
        self.molar_internal_energy = Some(molar_internal_energy);
        self
    }

    /// Specify a vapor state.
    pub fn vapor(mut self) -> Self {
        self.density_initialization = DensityInitialization::Vapor;
        self
    }

    /// Specify a liquid state.
    pub fn liquid(mut self) -> Self {
        self.density_initialization = DensityInitialization::Liquid;
        self
    }

    /// Provide an initial density used in density iterations.
    pub fn initial_density(mut self, initial_density: QuantityScalar<U>) -> Self {
        self.density_initialization = DensityInitialization::InitialDensity(initial_density);
        self
    }

    /// Provide an initial temperature used in the Newton solver.
    pub fn initial_temperature(mut self, initial_temperature: QuantityScalar<U>) -> Self {
        self.initial_temperature = Some(initial_temperature);
        self
    }

    /// Try to build the state with the given inputs.
    pub fn build(self) -> EosResult<State<U, E>> {
        State::new(
            &self.eos,
            self.temperature,
            self.volume,
            self.density,
            self.partial_density,
            self.total_moles,
            self.moles,
            self.molefracs,
            self.pressure,
            self.molar_enthalpy,
            self.molar_entropy,
            self.molar_internal_energy,
            self.density_initialization,
            self.initial_temperature,
        )
    }
}

impl<'a, U: EosUnit, E: EquationOfState> Clone for StateBuilder<'a, U, E> {
    fn clone(&self) -> Self {
        Self {
            eos: self.eos.clone(),
            temperature: self.temperature,
            volume: self.volume,
            density: self.density,
            partial_density: self.partial_density,
            total_moles: self.total_moles,
            moles: self.moles,
            molefracs: self.molefracs,
            pressure: self.pressure,
            molar_enthalpy: self.molar_enthalpy,
            molar_entropy: self.molar_entropy,
            molar_internal_energy: self.molar_internal_energy,
            density_initialization: self.density_initialization,
            initial_temperature: self.initial_temperature,
        }
    }
}
