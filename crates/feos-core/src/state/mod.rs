//! Description of a thermodynamic state.
//!
//! A thermodynamic state in SAFT is defined by
//! * a temperature
//! * an array of mole numbers
//! * the volume
//!
//! Internally, all properties are computed using such states as input.
use crate::density_iteration::density_iteration;
use crate::equation_of_state::Residual;
use crate::errors::{FeosError, FeosResult};
use crate::{ReferenceSystem, Total};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, Dyn, OVector, U1};
use num_dual::*;
use quantity::*;
use std::fmt;
use std::ops::Sub;
use std::sync::Mutex;

mod builder;
mod cache;
mod properties;
mod residual_properties;
mod statevec;
pub use builder::StateBuilder;
pub(crate) use cache::Cache;
pub use statevec::StateVec;

/// Possible contributions that can be computed.
#[derive(Clone, Copy, PartialEq)]
pub enum Contributions {
    /// Only compute the ideal gas contribution
    IdealGas,
    /// Only compute the difference between the total and the ideal gas contribution
    Residual,
    // /// Compute the differnce between the total and the ideal gas contribution for a (N,p,T) reference state
    // ResidualNpt,
    /// Compute ideal gas and residual contributions
    Total,
}

/// Initial values in a density iteration.
#[derive(Clone, Copy)]
pub enum DensityInitialization<D = Density> {
    /// Calculate a vapor phase by initializing using the ideal gas.
    Vapor,
    /// Calculate a liquid phase by using the `max_density`.
    Liquid,
    /// Use the given density as initial value.
    InitialDensity(D),
}

impl DensityInitialization {
    pub fn into_reduced(self) -> DensityInitialization<f64> {
        match self {
            Self::Vapor => DensityInitialization::Vapor,
            Self::Liquid => DensityInitialization::Liquid,
            Self::InitialDensity(d) => DensityInitialization::InitialDensity(d.into_reduced()),
        }
    }
}

/// Thermodynamic state of the system in reduced variables
/// including their derivatives.
///
/// Properties are stored as generalized (hyper) dual numbers which allows
/// for automatic differentiation.
#[derive(Clone, Debug)]
pub struct StateHD<D: DualNum<f64> + Copy, N: Dim = Dyn>
where
    DefaultAllocator: Allocator<N>,
{
    /// temperature in Kelvin
    pub temperature: D,
    // /// volume in Angstrom^3
    // pub molar_volume: D,
    /// mole fractions
    pub molefracs: OVector<D, N>,
    /// partial number densities in Angstrom^-3
    pub partial_density: OVector<D, N>,
}

impl<N: Dim, D: DualNum<f64> + Copy> StateHD<D, N>
where
    DefaultAllocator: Allocator<N>,
{
    /// Create a new `StateHD` for given temperature, molar volume and composition.
    pub fn new(temperature: D, molar_volume: D, molefracs: &OVector<D, N>) -> Self {
        let partial_density = molefracs / molar_volume;

        Self {
            temperature,
            // molar_volume,
            molefracs: molefracs.clone(),
            partial_density,
        }
    }

    /// Create a new `StateHD` for given temperature and partial densities
    pub fn new_density(temperature: D, partial_density: &OVector<D, N>) -> Self {
        let molefracs = partial_density / partial_density.sum();

        Self {
            temperature,
            // molar_volume,
            molefracs,
            partial_density: partial_density.clone(),
        }
    }

    // Since the molefracs can not be reproduced from moles if the density is zero,
    // this constructor exists specifically for these cases.
    pub(crate) fn new_virial(temperature: D, density: D, molefracs: &OVector<D, N>) -> Self {
        // let volume = D::one();
        let partial_density = molefracs * density;
        // let moles = partial_density.map(|pd| pd * volume);
        // let molefracs = molefracs.map(D::from);
        Self {
            temperature,
            // volume,
            // moles,
            molefracs: molefracs.clone(),
            partial_density,
        }
    }
}

/// Thermodynamic state of the system.
///
/// The state is always specified by the variables of the Helmholtz energy: volume $V$,
/// temperature $T$ and mole numbers $N_i$. Additional to these variables, the state saves
/// properties like the density, that can be calculated directly from the basic variables.
/// The state also contains a reference to the equation of state used to create the state.
/// Therefore, it can be used directly to calculate all state properties.
///
/// Calculated partial derivatives are cached in the state. Therefore, the second evaluation
/// of a property like the pressure, does not require a recalculation of the equation of state.
/// This can be used in situations where both lower and higher order derivatives are required, as
/// in a calculation of a derivative all lower derivatives have to be calculated internally as well.
/// Since they are cached it is more efficient to calculate the highest derivatives first.
/// For example during the calculation of the isochoric heat capacity $c_v$, the entropy and the
/// Helmholtz energy are calculated as well.
///
/// `State` objects are meant to be immutable. If individual fields like `volume` are changed, the
/// calculations are wrong as the internal fields of the state are not updated.
///
/// ## Contents
///
/// + [State properties](#state-properties)
/// + [Mass specific state properties](#mass-specific-state-properties)
/// + [Transport properties](#transport-properties)
/// + [Critical points](#critical-points)
/// + [State constructors](#state-constructors)
/// + [Stability analysis](#stability-analysis)
/// + [Flash calculations](#flash-calculations)
#[derive(Debug)]
pub struct State<E, N: Dim = Dyn, D: DualNum<f64> + Copy = f64>
where
    DefaultAllocator: Allocator<N>,
{
    /// Equation of state
    pub eos: E,
    /// Temperature $T$
    pub temperature: Temperature<D>,
    /// Volume $V$
    pub volume: Volume<D>,
    /// Mole numbers $N_i$
    pub moles: Moles<OVector<D, N>>,
    /// Total number of moles $N=\sum_iN_i$
    pub total_moles: Moles<D>,
    /// Partial densities $\rho_i=\frac{N_i}{V}$
    pub partial_density: Density<OVector<D, N>>,
    /// Total density $\rho=\frac{N}{V}=\sum_i\rho_i$
    pub density: Density<D>,
    /// Mole fractions $x_i=\frac{N_i}{N}=\frac{\rho_i}{\rho}$
    pub molefracs: OVector<D, N>,
    /// Cache
    cache: Mutex<Cache<D, N>>,
}

impl<E: Clone, N: Dim, D: DualNum<f64> + Copy> Clone for State<E, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    fn clone(&self) -> Self {
        Self {
            eos: self.eos.clone(),
            temperature: self.temperature,
            volume: self.volume,
            moles: self.moles.clone(),
            total_moles: self.total_moles,
            partial_density: self.partial_density.clone(),
            density: self.density,
            molefracs: self.molefracs.clone(),
            cache: Mutex::new(self.cache.lock().unwrap().clone()),
        }
    }
}

impl<E: Residual, N: Dim, D: DualNum<f64> + Copy> fmt::Display for State<E, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.eos.components() == 1 {
            write!(
                f,
                "T = {:.5}, ρ = {:.5}",
                self.temperature.re(),
                self.density.re()
            )
        } else {
            write!(
                f,
                "T = {:.5}, ρ = {:.5}, x = {:.5}",
                self.temperature.re(),
                self.density.re(),
                self.molefracs.map(|x| x.re())
            )
        }
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq, Debug, PartialOrd, Ord)]
pub(crate) enum Derivative {
    DV,
    DT,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq, Debug, PartialOrd, Ord)]
pub(crate) enum PartialDerivative {
    Zeroth,
    First(Derivative),
    Second(Derivative),
    SecondMixed,
    Third(Derivative),
}

#[derive(Clone, Copy, Eq, Hash, PartialEq, Debug, PartialOrd, Ord)]
pub(crate) enum VectorPartialDerivative {
    First,
    SecondMixed(Derivative),
}

// /// Derivatives of the helmholtz energy.
// #[derive(Clone, Copy, Eq, Hash, PartialEq, Debug, PartialOrd, Ord)]
// pub enum Derivative {
//     /// Derivative with respect to system volume.
//     DV,
//     /// Derivative with respect to temperature.
//     DT,
//     /// Derivative with respect to component `i`.
//     DN(usize),
// }

// #[derive(Clone, Copy, Eq, Hash, PartialEq, Debug)]
// pub enum PartialDerivative {
//     Zeroth,
//     First(Derivative),
//     Second(Derivative),
//     SecondMixed(Derivative, Derivative),
//     Third(Derivative),
// }

impl<E: Residual<N, D>, N: Dim, D: DualNum<f64> + Copy> State<E, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    /// Return a new `State` given a temperature, an array of mole numbers and a volume.
    ///
    /// This function will perform a validation of the given properties, i.e. test for signs
    /// and if values are finite. It will **not** validate physics, i.e. if the resulting
    /// densities are below the maximum packing fraction.
    pub fn new_nvt(
        eos: &E,
        temperature: Temperature<D>,
        volume: Volume<D>,
        moles: &Moles<OVector<D, N>>,
    ) -> FeosResult<Self> {
        let total_moles = moles.sum();
        let molefracs = (moles / total_moles).into_value();
        let density = total_moles / volume;
        validate(temperature, density, &molefracs)?;

        Ok(Self::new_unchecked(
            eos,
            temperature,
            density,
            total_moles,
            &molefracs,
        ))
    }

    /// Return a new `State` for which the total amount of substance is unspecified.
    ///
    /// Internally the total number of moles will be set to 1 mol.
    ///
    /// This function will perform a validation of the given properties, i.e. test for signs
    /// and if values are finite. It will **not** validate physics, i.e. if the resulting
    /// densities are below the maximum packing fraction.
    pub fn new_intensive(
        eos: &E,
        temperature: Temperature<D>,
        density: Density<D>,
        molefracs: &OVector<D, N>,
    ) -> FeosResult<Self> {
        validate(temperature, density, molefracs)?;
        let total_moles = Moles::new(D::one());
        Ok(Self::new_unchecked(
            eos,
            temperature,
            density,
            total_moles,
            molefracs,
        ))
    }

    fn new_unchecked(
        eos: &E,
        temperature: Temperature<D>,
        density: Density<D>,
        total_moles: Moles<D>,
        molefracs: &OVector<D, N>,
    ) -> Self {
        let volume = total_moles / density;
        let moles = Dimensionless::new(molefracs.clone()) * total_moles;
        let partial_density = moles.clone() / volume;

        State {
            eos: eos.clone(),
            temperature,
            volume,
            moles,
            total_moles,
            partial_density,
            density,
            molefracs: molefracs.clone(),
            cache: Mutex::new(Cache::new()),
        }
    }

    /// Return a new `State` for a pure component given a temperature and a density. The moles
    /// are set to the reference value for each component.
    ///
    /// This function will perform a validation of the given properties, i.e. test for signs
    /// and if values are finite. It will **not** validate physics, i.e. if the resulting
    /// densities are below the maximum packing fraction.
    pub fn new_pure(eos: &E, temperature: Temperature<D>, density: Density<D>) -> FeosResult<Self> {
        let molefracs = OVector::from_element_generic(N::from_usize(1), U1, D::one());
        Self::new_intensive(eos, temperature, density, &molefracs)
    }
}

impl<E: Residual<N>, N: Dim> State<E, N>
where
    DefaultAllocator: Allocator<N>,
{
    /// Return a new `State` for the combination of inputs.
    ///
    /// The function attempts to create a new state using the given input values. If the state
    /// is overdetermined, it will choose a method based on the following hierarchy.
    /// 1. Create a state non-iteratively from the set of $T$, $V$, $\rho$, $\rho_i$, $N$, $N_i$ and $x_i$.
    /// 2. Use a density iteration for a given pressure.
    ///
    /// The [StateBuilder] provides a convenient way of calling this function without the need to provide
    /// all the optional input values.
    ///
    /// # Errors
    ///
    /// When the state cannot be created using the combination of inputs.
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        eos: &E,
        temperature: Option<Temperature>,
        volume: Option<Volume>,
        density: Option<Density>,
        partial_density: Option<&Density<OVector<f64, N>>>,
        total_moles: Option<Moles>,
        moles: Option<&Moles<OVector<f64, N>>>,
        molefracs: Option<&OVector<f64, N>>,
        pressure: Option<Pressure>,
        density_initialization: Option<DensityInitialization>,
    ) -> FeosResult<Self> {
        Self::_new(
            eos,
            temperature,
            volume,
            density,
            partial_density,
            total_moles,
            moles,
            molefracs,
            pressure,
            density_initialization,
        )?
        .map_err(|_| FeosError::UndeterminedState(String::from("Missing input parameters.")))
    }

    #[expect(clippy::too_many_arguments)]
    #[expect(clippy::type_complexity)]
    fn _new(
        eos: &E,
        temperature: Option<Temperature>,
        volume: Option<Volume>,
        density: Option<Density>,
        partial_density: Option<&Density<OVector<f64, N>>>,
        total_moles: Option<Moles>,
        moles: Option<&Moles<OVector<f64, N>>>,
        molefracs: Option<&OVector<f64, N>>,
        pressure: Option<Pressure>,
        density_initialization: Option<DensityInitialization>,
    ) -> FeosResult<Result<Self, Option<Moles<OVector<f64, N>>>>> {
        // check for density
        if density.and(partial_density).is_some() {
            return Err(FeosError::UndeterminedState(String::from(
                "Both density and partial density given.",
            )));
        }
        let rho = density.or_else(|| partial_density.map(|pd| pd.sum()));

        // check for total moles
        if moles.and(total_moles).is_some() {
            return Err(FeosError::UndeterminedState(String::from(
                "Both moles and total moles given.",
            )));
        }
        let mut n = total_moles.or_else(|| moles.map(|m| m.sum()));

        // check if total moles can be inferred from volume
        if rho.and(n).and(volume).is_some() {
            return Err(FeosError::UndeterminedState(String::from(
                "Density is overdetermined.",
            )));
        }
        n = n.or_else(|| rho.and_then(|d| volume.map(|v| v * d)));

        // check for composition
        if partial_density.and(moles).is_some() {
            return Err(FeosError::UndeterminedState(String::from(
                "Composition is overdetermined.",
            )));
        }
        let x = partial_density
            .map(|pd| pd / pd.sum())
            .or_else(|| moles.map(|ms| ms / ms.sum()))
            .map(Quantity::into_value);
        let x_u = match (x, molefracs, eos.components()) {
            (Some(_), Some(_), _) => {
                return Err(FeosError::UndeterminedState(String::from(
                    "Composition is overdetermined.",
                )));
            }
            (Some(x), None, _) => x,
            (None, Some(x), _) => x.clone(),
            (None, None, 1) => OVector::from_element_generic(N::from_usize(1), U1, 1.0),
            _ => {
                return Err(FeosError::UndeterminedState(String::from(
                    "Missing composition.",
                )));
            }
        };

        // If no extensive property is given, moles is set to the reference value.
        if let (None, None) = (volume, n) {
            n = Some(Moles::from_reduced(1.0))
        }
        let n_i = n.map(|n| &x_u * n / x_u.sum());
        let v = volume.or_else(|| rho.and_then(|d| n.map(|n| n / d)));

        // check if new state can be created using default constructor
        if let (Some(v), Some(t), Some(n_i)) = (v, temperature, &n_i) {
            return Ok(Ok(State::new_nvt(eos, t, v, n_i)?));
        }

        // Check if new state can be created using density iteration
        if let (Some(p), Some(t), Some(n_i)) = (pressure, temperature, &n_i) {
            return Ok(Ok(State::new_npt(eos, t, p, n_i, density_initialization)?));
        }
        if let (Some(p), Some(t), Some(v)) = (pressure, temperature, v) {
            return Ok(Ok(State::new_npvx(
                eos,
                t,
                p,
                v,
                &x_u,
                density_initialization,
            )?));
        }
        Ok(Err(n_i.to_owned()))
    }
}

impl<E: Residual<N, D>, N: Dim, D: DualNum<f64> + Copy> State<E, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    /// Return a new `State` using a density iteration. [DensityInitialization] is used to
    /// influence the calculation with respect to the possible solutions.
    pub fn new_npt(
        eos: &E,
        temperature: Temperature<D>,
        pressure: Pressure<D>,
        moles: &Moles<OVector<D, N>>,
        density_initialization: Option<DensityInitialization>,
    ) -> FeosResult<Self> {
        let total_moles = moles.sum();
        let molefracs = (moles / total_moles).into_value();
        let density = Self::new_xpt(
            eos,
            temperature,
            pressure,
            &molefracs,
            density_initialization,
        )?
        .density;
        Ok(Self::new_unchecked(
            eos,
            temperature,
            density,
            total_moles,
            &molefracs,
        ))
    }

    /// Return a new `State` using a density iteration. [DensityInitialization] is used to
    /// influence the calculation with respect to the possible solutions.
    pub fn new_xpt(
        eos: &E,
        temperature: Temperature<D>,
        pressure: Pressure<D>,
        molefracs: &OVector<D, N>,
        density_initialization: Option<DensityInitialization>,
    ) -> FeosResult<Self> {
        density_iteration(
            eos,
            temperature,
            pressure,
            molefracs,
            density_initialization,
        )
        .and_then(|density| Self::new_intensive(eos, temperature, density, molefracs))
    }

    /// Return a new `State` for given pressure $p$, volume $V$, temperature $T$ and composition $x_i$.
    pub fn new_npvx(
        eos: &E,
        temperature: Temperature<D>,
        pressure: Pressure<D>,
        volume: Volume<D>,
        molefracs: &OVector<D, N>,
        density_initialization: Option<DensityInitialization>,
    ) -> FeosResult<Self> {
        let density = Self::new_xpt(
            eos,
            temperature,
            pressure,
            molefracs,
            density_initialization,
        )?
        .density;
        let total_moles = density * volume;
        Ok(Self::new_unchecked(
            eos,
            temperature,
            density,
            total_moles,
            molefracs,
        ))
    }
}

impl<E: Total<N>, N: Gradients> State<E, N>
where
    DefaultAllocator: Allocator<N>,
{
    /// Return a new `State` for the combination of inputs.
    ///
    /// The function attempts to create a new state using the given input values. If the state
    /// is overdetermined, it will choose a method based on the following hierarchy.
    /// 1. Create a state non-iteratively from the set of $T$, $V$, $\rho$, $\rho_i$, $N$, $N_i$ and $x_i$.
    /// 2. Use a density iteration for a given pressure.
    /// 3. Determine the state using a Newton iteration from (in this order): $(p, h)$, $(p, s)$, $(T, h)$, $(T, s)$, $(V, u)$
    ///
    /// The [StateBuilder] provides a convenient way of calling this function without the need to provide
    /// all the optional input values.
    ///
    /// # Errors
    ///
    /// When the state cannot be created using the combination of inputs.
    #[expect(clippy::too_many_arguments)]
    pub fn new_full(
        eos: &E,
        temperature: Option<Temperature>,
        volume: Option<Volume>,
        density: Option<Density>,
        partial_density: Option<&Density<OVector<f64, N>>>,
        total_moles: Option<Moles>,
        moles: Option<&Moles<OVector<f64, N>>>,
        molefracs: Option<&OVector<f64, N>>,
        pressure: Option<Pressure>,
        molar_enthalpy: Option<MolarEnergy>,
        molar_entropy: Option<MolarEntropy>,
        molar_internal_energy: Option<MolarEnergy>,
        density_initialization: Option<DensityInitialization>,
        initial_temperature: Option<Temperature>,
    ) -> FeosResult<Self> {
        let state = Self::_new(
            eos,
            temperature,
            volume,
            density,
            partial_density,
            total_moles,
            moles,
            molefracs,
            pressure,
            density_initialization,
        )?;

        let ti = initial_temperature;
        match state {
            Ok(state) => Ok(state),
            Err(n_i) => {
                // Check if new state can be created using molar_enthalpy and temperature
                if let (Some(p), Some(h), Some(n_i)) = (pressure, molar_enthalpy, &n_i) {
                    return State::new_nph(eos, p, h, n_i, density_initialization, ti);
                }
                if let (Some(p), Some(s), Some(n_i)) = (pressure, molar_entropy, &n_i) {
                    return State::new_nps(eos, p, s, n_i, density_initialization, ti);
                }
                if let (Some(t), Some(h), Some(n_i)) = (temperature, molar_enthalpy, &n_i) {
                    return State::new_nth(eos, t, h, n_i, density_initialization);
                }
                if let (Some(t), Some(s), Some(n_i)) = (temperature, molar_entropy, &n_i) {
                    return State::new_nts(eos, t, s, n_i, density_initialization);
                }
                if let (Some(u), Some(v), Some(n_i)) = (molar_internal_energy, volume, &n_i) {
                    return State::new_nvu(eos, v, u, n_i, ti);
                }
                Err(FeosError::UndeterminedState(String::from(
                    "Missing input parameters.",
                )))
            }
        }
    }

    /// Return a new `State` for given pressure $p$ and molar enthalpy $h$.
    pub fn new_nph(
        eos: &E,
        pressure: Pressure,
        molar_enthalpy: MolarEnergy,
        moles: &Moles<OVector<f64, N>>,
        density_initialization: Option<DensityInitialization>,
        initial_temperature: Option<Temperature>,
    ) -> FeosResult<Self> {
        let t0 = initial_temperature.unwrap_or(Temperature::from_reduced(298.15));
        let mut density = density_initialization;
        let f = |x0| {
            let s = State::new_npt(eos, x0, pressure, moles, density)?;
            let dfx = s.molar_isobaric_heat_capacity(Contributions::Total);
            let fx = s.molar_enthalpy(Contributions::Total) - molar_enthalpy;
            density = Some(DensityInitialization::InitialDensity(s.density));
            Ok((fx, dfx, s))
        };
        newton(t0, f, Temperature::from_reduced(1.0e-8))
    }

    /// Return a new `State` for given temperature $T$ and molar enthalpy $h$.
    pub fn new_nth(
        eos: &E,
        temperature: Temperature,
        molar_enthalpy: MolarEnergy,
        moles: &Moles<OVector<f64, N>>,
        density_initialization: Option<DensityInitialization>,
    ) -> FeosResult<Self> {
        let x = moles.convert_to(moles.sum());
        let rho0 = match density_initialization {
            Some(DensityInitialization::InitialDensity(r)) => r,
            Some(DensityInitialization::Liquid) => eos.max_density(&Some(x))?,
            Some(DensityInitialization::Vapor) => 1.0e-5 * eos.max_density(&Some(x))?,
            None => 0.01 * eos.max_density(&Some(x))?,
        };
        let n_inv = 1.0 / moles.sum();
        let f = |x0| {
            let s = State::new_nvt(eos, temperature, moles.sum() / x0, moles)?;
            let dfx = -s.volume / s.density
                * n_inv
                * (s.volume * s.dp_dv(Contributions::Total)
                    + temperature * s.dp_dt(Contributions::Total));
            let fx = s.molar_enthalpy(Contributions::Total) - molar_enthalpy;
            Ok((fx, dfx, s))
        };
        newton(rho0, f, Density::from_reduced(1.0e-12))
    }

    /// Return a new `State` for given temperature $T$ and molar entropy $s$.
    pub fn new_nts(
        eos: &E,
        temperature: Temperature,
        molar_entropy: MolarEntropy,
        moles: &Moles<OVector<f64, N>>,
        density_initialization: Option<DensityInitialization>,
    ) -> FeosResult<Self> {
        let x = moles.convert_to(moles.sum());
        let rho0 = match density_initialization {
            Some(DensityInitialization::InitialDensity(r)) => r,
            Some(DensityInitialization::Liquid) => eos.max_density(&Some(x))?,
            Some(DensityInitialization::Vapor) => 1.0e-5 * eos.max_density(&Some(x))?,
            None => 0.01 * eos.max_density(&Some(x))?,
        };
        let n_inv = 1.0 / moles.sum();
        let f = |x0| {
            let s = State::new_nvt(eos, temperature, moles.sum() / x0, moles)?;
            let dfx = -n_inv * s.volume / s.density * s.dp_dt(Contributions::Total);
            let fx = s.molar_entropy(Contributions::Total) - molar_entropy;
            Ok((fx, dfx, s))
        };
        newton(rho0, f, Density::from_reduced(1.0e-12))
    }

    /// Return a new `State` for given pressure $p$ and molar entropy $s$.
    pub fn new_nps(
        eos: &E,
        pressure: Pressure,
        molar_entropy: MolarEntropy,
        moles: &Moles<OVector<f64, N>>,
        density_initialization: Option<DensityInitialization>,
        initial_temperature: Option<Temperature>,
    ) -> FeosResult<Self> {
        let t0 = initial_temperature.unwrap_or(Temperature::from_reduced(298.15));
        let mut density = density_initialization;
        let f = |x0| {
            let s = State::new_npt(eos, x0, pressure, moles, density)?;
            let dfx = s.molar_isobaric_heat_capacity(Contributions::Total) / s.temperature;
            let fx = s.molar_entropy(Contributions::Total) - molar_entropy;
            density = Some(DensityInitialization::InitialDensity(s.density));
            Ok((fx, dfx, s))
        };
        newton(t0, f, Temperature::from_reduced(1.0e-8))
    }

    /// Return a new `State` for given volume $V$ and molar internal energy $u$.
    pub fn new_nvu(
        eos: &E,
        volume: Volume,
        molar_internal_energy: MolarEnergy,
        moles: &Moles<OVector<f64, N>>,
        initial_temperature: Option<Temperature>,
    ) -> FeosResult<Self> {
        let t0 = initial_temperature.unwrap_or(Temperature::from_reduced(298.15));
        let f = |x0| {
            let s = State::new_nvt(eos, x0, volume, moles)?;
            let fx = s.molar_internal_energy(Contributions::Total) - molar_internal_energy;
            let dfx = s.molar_isochoric_heat_capacity(Contributions::Total);
            Ok((fx, dfx, s))
        };
        newton(t0, f, Temperature::from_reduced(1.0e-8))
    }
}

fn is_close<U: Copy>(
    x: Quantity<f64, U>,
    y: Quantity<f64, U>,
    atol: Quantity<f64, U>,
    rtol: f64,
) -> bool {
    (x - y).abs() <= atol + rtol * y.abs()
}

fn newton<E: Residual<N>, N: Dim, F, X: Copy, Y>(
    mut x0: Quantity<f64, X>,
    mut f: F,
    atol: Quantity<f64, X>,
) -> FeosResult<State<E, N>>
where
    DefaultAllocator: Allocator<N>,
    Y: Sub<X> + Sub<<Y as Sub<X>>::Output, Output = X>,
    F: FnMut(
        Quantity<f64, X>,
    ) -> FeosResult<(
        Quantity<f64, Y>,
        Quantity<f64, <Y as Sub<X>>::Output>,
        State<E, N>,
    )>,
{
    let rtol = 1e-10;
    let maxiter = 50;

    for _ in 0..maxiter {
        let (fx, dfx, state) = f(x0)?;
        let x = x0 - fx / dfx;
        if is_close(x, x0, atol, rtol) {
            return Ok(state);
        }
        x0 = x;
    }
    Err(FeosError::NotConverged("newton".to_owned()))
}

/// Validate the given temperature, mole numbers and volume.
///
/// Properties are valid if
/// * they are finite
/// * they have a positive sign
///
/// There is no validation of the physical state, e.g.
/// if resulting densities are below maximum packing fraction.
fn validate<N: Dim, D: DualNum<f64>>(
    temperature: Temperature<D>,
    density: Density<D>,
    molefracs: &OVector<D, N>,
) -> FeosResult<()>
where
    DefaultAllocator: Allocator<N>,
{
    let t = temperature.re().to_reduced();
    let rho = density.re().to_reduced();
    if !t.is_finite() || t.is_sign_negative() {
        return Err(FeosError::InvalidState(
            String::from("validate"),
            String::from("temperature"),
            t,
        ));
    }
    if !rho.is_finite() || rho.is_sign_negative() {
        return Err(FeosError::InvalidState(
            String::from("validate"),
            String::from("density"),
            rho,
        ));
    }
    for n in molefracs.iter() {
        if !n.re().is_finite() || n.re().is_sign_negative() {
            return Err(FeosError::InvalidState(
                String::from("validate"),
                String::from("molefracs"),
                n.re(),
            ));
        }
    }
    Ok(())
}

mod critical_point;

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dvector;
    use typenum::P3;

    #[test]
    fn test_validate() {
        let temperature = 298.15 * KELVIN;
        let density = 3000.0 * MOL / METER.powi::<P3>();
        let molefracs = dvector![0.03, 0.02, 0.05];
        assert!(validate(temperature, density, &molefracs).is_ok());
    }

    #[test]
    fn test_negative_temperature() {
        let temperature = -298.15 * KELVIN;
        let density = 3000.0 * MOL / METER.powi::<P3>();
        let molefracs = dvector![0.03, 0.02, 0.05];
        assert!(validate(temperature, density, &molefracs).is_err());
    }

    #[test]
    fn test_nan_temperature() {
        let temperature = f64::NAN * KELVIN;
        let density = 3000.0 * MOL / METER.powi::<P3>();
        let molefracs = dvector![0.03, 0.02, 0.05];
        assert!(validate(temperature, density, &molefracs).is_err());
    }

    #[test]
    fn test_negative_mole_number() {
        let temperature = 298.15 * KELVIN;
        let density = 3000.0 * MOL / METER.powi::<P3>();
        let molefracs = dvector![-0.03, 0.02, 0.05];
        assert!(validate(temperature, density, &molefracs).is_err());
    }

    #[test]
    fn test_nan_mole_number() {
        let temperature = 298.15 * KELVIN;
        let density = 3000.0 * MOL / METER.powi::<P3>();
        let molefracs = dvector![f64::NAN, 0.02, 0.05];
        assert!(validate(temperature, density, &molefracs).is_err());
    }

    #[test]
    fn test_negative_density() {
        let temperature = 298.15 * KELVIN;
        let density = -3000.0 * MOL / METER.powi::<P3>();
        let molefracs = dvector![0.01, 0.02, 0.05];
        assert!(validate(temperature, density, &molefracs).is_err());
    }
}
