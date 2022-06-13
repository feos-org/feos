//! Description of a thermodynamic state.
//!
//! A thermodynamic state in SAFT is defined by
//! * a temperature
//! * an array of mole numbers
//! * the volume
//!
//! Internally, all properties are computed using such states as input.
use crate::density_iteration::density_iteration;
use crate::equation_of_state::EquationOfState;
use crate::errors::{EosError, EosResult};
use crate::EosUnit;
use cache::Cache;
use ndarray::prelude::*;
use num_dual::linalg::{norm, LU};
use num_dual::*;
use quantity::{QuantityArray1, QuantityScalar};
use std::cell::RefCell;
use std::convert::TryFrom;
use std::fmt;
use std::rc::Rc;

mod builder;
mod cache;
mod properties;
pub use builder::StateBuilder;
pub use properties::{Contributions, StateVec};

/// Initial values in a density iteration.
#[derive(Clone, Copy)]
pub enum DensityInitialization<U: EosUnit> {
    /// Calculate a vapor phase by initializing using the ideal gas.
    Vapor,
    /// Calculate a liquid phase by using the `max_density`.
    Liquid,
    /// Use the given density as initial value.
    InitialDensity(QuantityScalar<U>),
    /// Calculate the most stable phase by calculating both a vapor and a liquid
    /// and return the one with the lower molar Gibbs energy.
    None,
}

/// Thermodynamic state of the system in reduced variables
/// including their derivatives.
///
/// Properties are stored as generalized (hyper) dual numbers which allows
/// for automatic differentiation.
#[derive(Clone, Debug)]
pub struct StateHD<D: DualNum<f64>> {
    /// temperature in Kelvin
    pub temperature: D,
    /// volume in Angstrom^3
    pub volume: D,
    /// number of particles
    pub moles: Array1<D>,
    /// mole fractions
    pub molefracs: Array1<D>,
    /// partial number densities in Angstrom^-3
    pub partial_density: Array1<D>,
}

impl<D: DualNum<f64>> StateHD<D> {
    /// Create a new `StateHD` for given temperature volume and moles.
    pub fn new(temperature: D, volume: D, moles: Array1<D>) -> Self {
        let total_moles = moles.sum();
        let partial_density = moles.mapv(|n| n / volume);
        let molefracs = moles.mapv(|n| n / total_moles);

        Self {
            temperature,
            volume,
            moles,
            molefracs,
            partial_density,
        }
    }

    // Since the molefracs can not be reproduced from moles if the density is zero,
    // this constructor exists specifically for these cases.
    pub(crate) fn new_virial(temperature: D, density: D, molefracs: Array1<f64>) -> Self {
        let volume = D::one();
        let partial_density = molefracs.mapv(|x| density * x);
        let moles = partial_density.mapv(|pd| pd * volume);
        let molefracs = molefracs.mapv(D::from);
        Self {
            temperature,
            volume,
            moles,
            molefracs,
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
pub struct State<U, E> {
    /// Equation of state
    pub eos: Rc<E>,
    /// Temperature $T$
    pub temperature: QuantityScalar<U>,
    /// Volume $V$
    pub volume: QuantityScalar<U>,
    /// Mole numbers $N_i$
    pub moles: QuantityArray1<U>,
    /// Total number of moles $N=\sum_iN_i$
    pub total_moles: QuantityScalar<U>,
    /// Partial densities $\rho_i=\frac{N_i}{V}$
    pub partial_density: QuantityArray1<U>,
    /// Total density $\rho=\frac{N}{V}=\sum_i\rho_i$
    pub density: QuantityScalar<U>,
    /// Mole fractions $x_i=\frac{N_i}{N}=\frac{\rho_i}{\rho}$
    pub molefracs: Array1<f64>,
    /// Reduced temperature
    reduced_temperature: f64,
    /// Reduced volume,
    reduced_volume: f64,
    /// Reduced moles
    reduced_moles: Array1<f64>,
    /// Cache
    cache: RefCell<Cache>,
}

impl<U: Clone, E> Clone for State<U, E> {
    fn clone(&self) -> Self {
        Self {
            eos: self.eos.clone(),
            total_moles: self.total_moles.clone(),
            temperature: self.temperature.clone(),
            volume: self.volume.clone(),
            moles: self.moles.clone(),
            partial_density: self.partial_density.clone(),
            density: self.density.clone(),
            molefracs: self.molefracs.clone(),
            reduced_temperature: self.reduced_temperature,
            reduced_volume: self.reduced_volume,
            reduced_moles: self.reduced_moles.clone(),
            cache: self.cache.clone(),
        }
    }
}

impl<U, E> fmt::Display for State<U, E>
where
    QuantityScalar<U>: fmt::Display,
    QuantityArray1<U>: fmt::Display,
    E: EquationOfState,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.eos.components() == 1 {
            write!(f, "T = {:.5}, ρ = {:.5}", self.temperature, self.density)
        } else {
            write!(
                f,
                "T = {:.5}, ρ = {:.5}, x = {:.5}",
                self.temperature, self.density, self.molefracs
            )
        }
    }
}

/// Derivatives of the helmholtz energy.
#[derive(Clone, Copy, Eq, Hash, PartialEq, Debug, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
pub(crate) enum Derivative {
    /// Derivative with respect to system volume.
    DV,
    /// Derivative with respect to temperature.
    DT,
    /// Derivative with respect to component `i`.
    DN(usize),
}

impl Derivative {
    pub fn reference<U: EosUnit>(&self) -> QuantityScalar<U> {
        match self {
            Derivative::DV => U::reference_volume(),
            Derivative::DT => U::reference_temperature(),
            Derivative::DN(_) => U::reference_moles(),
        }
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq, Debug)]
pub(crate) enum PartialDerivative {
    Zeroth,
    First(Derivative),
    Second(Derivative, Derivative),
    Third(Derivative),
}

/// # State constructors
impl<U: EosUnit, E: EquationOfState> State<U, E> {
    /// Return a new `State` given a temperature, an array of mole numbers and a volume.
    ///
    /// This function will perform a validation of the given properties, i.e. test for signs
    /// and if values are finite. It will **not** validate physics, i.e. if the resulting
    /// densities are below the maximum packing fraction.
    pub fn new_nvt(
        eos: &Rc<E>,
        temperature: QuantityScalar<U>,
        volume: QuantityScalar<U>,
        moles: &QuantityArray1<U>,
    ) -> EosResult<Self> {
        eos.validate_moles(Some(moles))?;
        validate(temperature, volume, moles)?;

        let t = temperature.to_reduced(U::reference_temperature())?;
        let v = volume.to_reduced(U::reference_volume())?;
        let m = moles.to_reduced(U::reference_moles())?;

        let total_moles = moles.sum();
        let partial_density = moles / volume;
        let density = total_moles / volume;
        let molefracs = &m / total_moles.to_reduced(U::reference_moles())?;

        Ok(State {
            eos: eos.clone(),
            total_moles,
            temperature,
            volume,
            moles: moles.to_owned(),
            partial_density,
            density,
            molefracs,
            reduced_temperature: t,
            reduced_volume: v,
            reduced_moles: m,
            cache: RefCell::new(Cache::with_capacity(eos.components())),
        })
    }

    /// Return a new `State` for a pure component given a temperature and a density. The moles
    /// are set to the reference value for each component.
    ///
    /// This function will perform a validation of the given properties, i.e. test for signs
    /// and if values are finite. It will **not** validate physics, i.e. if the resulting
    /// densities are below the maximum packing fraction.
    pub fn new_pure(
        eos: &Rc<E>,
        temperature: QuantityScalar<U>,
        density: QuantityScalar<U>,
    ) -> EosResult<Self> {
        let moles = arr1(&[1.0]) * U::reference_moles();
        Self::new_nvt(eos, temperature, U::reference_moles() / density, &moles)
    }

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
    pub fn new(
        eos: &Rc<E>,
        temperature: Option<QuantityScalar<U>>,
        volume: Option<QuantityScalar<U>>,
        density: Option<QuantityScalar<U>>,
        partial_density: Option<&QuantityArray1<U>>,
        total_moles: Option<QuantityScalar<U>>,
        moles: Option<&QuantityArray1<U>>,
        molefracs: Option<&Array1<f64>>,
        pressure: Option<QuantityScalar<U>>,
        molar_enthalpy: Option<QuantityScalar<U>>,
        molar_entropy: Option<QuantityScalar<U>>,
        molar_internal_energy: Option<QuantityScalar<U>>,
        density_initialization: DensityInitialization<U>,
        initial_temperature: Option<QuantityScalar<U>>,
    ) -> EosResult<Self> {
        // Check if the provided densities have correct units.
        if let DensityInitialization::InitialDensity(rho0) = density_initialization {
            if !rho0.has_unit(&U::reference_density()) {
                return Err(EosError::UndeterminedState(String::from(
                    "The provided initial density has to be the molar density",
                )));
            }
        }
        if let Some(rho) = density {
            if !rho.has_unit(&U::reference_density()) {
                return Err(EosError::UndeterminedState(String::from(
                    "The provided density has to be the molar density",
                )));
            }
        }
        if let Some(rho) = partial_density {
            if !rho.has_unit(&U::reference_density()) {
                return Err(EosError::UndeterminedState(String::from(
                    "The provided partial density has to be the molar density",
                )));
            }
        }

        // check for density
        if density.and(partial_density).is_some() {
            return Err(EosError::UndeterminedState(String::from(
                "Both density and partial density given.",
            )));
        }
        let rho = density.or_else(|| partial_density.map(|pd| pd.sum()));

        // check for total moles
        if moles.and(total_moles).is_some() {
            return Err(EosError::UndeterminedState(String::from(
                "Both moles and total moles given.",
            )));
        }
        let mut n = total_moles.or_else(|| moles.map(|m| m.sum()));

        // check if total moles can be inferred from volume
        if rho.and(n).and(volume).is_some() {
            return Err(EosError::UndeterminedState(String::from(
                "Density is overdetermined.",
            )));
        }
        n = n.or_else(|| rho.and_then(|d| volume.map(|v| v * d)));

        // check for composition
        if partial_density.and(moles).is_some() {
            return Err(EosError::UndeterminedState(String::from(
                "Composition is overdetermined.",
            )));
        }
        let x = partial_density
            .map(|pd| pd.to_reduced(pd.sum()))
            .or_else(|| moles.map(|ms| ms.to_reduced(ms.sum())))
            .transpose()?;
        let x_u = match (x, molefracs, eos.components()) {
            (Some(_), Some(_), _) => {
                return Err(EosError::UndeterminedState(String::from(
                    "Composition is overdetermined.",
                )))
            }
            (Some(x), None, _) => x,
            (None, Some(x), _) => x.clone(),
            (None, None, 1) => arr1(&[1.0]),
            _ => {
                return Err(EosError::UndeterminedState(String::from(
                    "Missing composition.",
                )))
            }
        };

        // If no extensive property is given, moles is set to the reference value.
        if let (None, None) = (volume, n) {
            n = Some(U::reference_moles())
        }
        let n_i = n.map(|n| &x_u * n);
        let v = volume.or_else(|| rho.and_then(|d| n.map(|n| n / d)));

        // check if new state can be created using default constructor
        if let (Some(v), Some(t), Some(n_i)) = (v, temperature, &n_i) {
            return State::new_nvt(eos, t, v, n_i);
        }

        // Check if new state can be created using density iteration
        if let (Some(p), Some(t), Some(n_i)) = (pressure, temperature, &n_i) {
            return State::new_npt(eos, t, p, n_i, density_initialization);
        }
        if let (Some(p), Some(t), Some(v)) = (pressure, temperature, v) {
            return State::new_npvx(eos, t, p, v, &x_u, density_initialization);
        }

        // Check if new state can be created using molar_enthalpy and temperature
        if let (Some(p), Some(h), Some(n_i)) = (pressure, molar_enthalpy, &n_i) {
            return State::new_nph(eos, p, h, n_i, density_initialization, initial_temperature);
        }
        if let (Some(p), Some(s), Some(n_i)) = (pressure, molar_entropy, &n_i) {
            return State::new_nps(eos, p, s, n_i, density_initialization, initial_temperature);
        }
        if let (Some(t), Some(h), Some(n_i)) = (temperature, molar_enthalpy, &n_i) {
            return State::new_nth(eos, t, h, n_i, density_initialization);
        }
        if let (Some(t), Some(s), Some(n_i)) = (temperature, molar_entropy, &n_i) {
            return State::new_nts(eos, t, s, n_i, density_initialization);
        }
        if let (Some(u), Some(v), Some(n_i)) = (molar_internal_energy, volume, &n_i) {
            return State::new_nvu(eos, v, u, n_i, initial_temperature);
        }
        Err(EosError::UndeterminedState(String::from(
            "Missing input parameters.",
        )))
    }

    /// Return a new `State` using a density iteration. [DensityInitialization] is used to
    /// influence the calculation with respect to the possible solutions.
    pub fn new_npt(
        eos: &Rc<E>,
        temperature: QuantityScalar<U>,
        pressure: QuantityScalar<U>,
        moles: &QuantityArray1<U>,
        density_initialization: DensityInitialization<U>,
    ) -> EosResult<Self> {
        // calculate state from initial density or given phase
        match density_initialization {
            DensityInitialization::InitialDensity(rho0) => {
                return density_iteration(eos, temperature, pressure, moles, rho0)
            }
            DensityInitialization::Vapor => {
                return density_iteration(
                    eos,
                    temperature,
                    pressure,
                    moles,
                    pressure / temperature / U::gas_constant(),
                )
            }
            DensityInitialization::Liquid => {
                return density_iteration(
                    eos,
                    temperature,
                    pressure,
                    moles,
                    eos.max_density(Some(moles))?,
                )
            }
            DensityInitialization::None => (),
        }

        // calculate stable phase
        let max_density = eos.max_density(Some(moles))?;
        let liquid = density_iteration(eos, temperature, pressure, moles, max_density);

        if pressure < max_density * temperature * U::gas_constant() {
            let vapor = density_iteration(
                eos,
                temperature,
                pressure,
                moles,
                pressure / temperature / U::gas_constant(),
            );
            match (&liquid, &vapor) {
                (Ok(_), Err(_)) => liquid,
                (Err(_), Ok(_)) => vapor,
                (Ok(l), Ok(v)) => {
                    if l.molar_gibbs_energy(Contributions::Total)
                        > v.molar_gibbs_energy(Contributions::Total)
                    {
                        vapor
                    } else {
                        liquid
                    }
                }
                _ => Err(EosError::UndeterminedState(String::from(
                    "Density iteration did not find a solution.",
                ))),
            }
        } else {
            liquid
        }
    }

    /// Return a new `State` for given pressure $p$, volume $V$, temperature $T$ and composition $x_i$.
    pub fn new_npvx(
        eos: &Rc<E>,
        temperature: QuantityScalar<U>,
        pressure: QuantityScalar<U>,
        volume: QuantityScalar<U>,
        molefracs: &Array1<f64>,
        density_initialization: DensityInitialization<U>,
    ) -> EosResult<Self> {
        let moles = molefracs * U::reference_moles();
        let state = Self::new_npt(eos, temperature, pressure, &moles, density_initialization)?;
        let moles = state.partial_density * volume;
        Self::new_nvt(eos, temperature, volume, &moles)
    }

    /// Return a new `State` for given pressure $p$ and molar enthalpy $h$.
    pub fn new_nph(
        eos: &Rc<E>,
        pressure: QuantityScalar<U>,
        molar_enthalpy: QuantityScalar<U>,
        moles: &QuantityArray1<U>,
        density_initialization: DensityInitialization<U>,
        initial_temperature: Option<QuantityScalar<U>>,
    ) -> EosResult<Self> {
        let t0 = initial_temperature.unwrap_or(298.15 * U::reference_temperature());
        let mut density = density_initialization;
        let f = |x0| {
            let s = State::new_npt(eos, x0, pressure, moles, density)?;
            let dfx = s.c_p(Contributions::Total);
            let fx = s.molar_enthalpy(Contributions::Total) - molar_enthalpy;
            density = DensityInitialization::InitialDensity(s.density);
            Ok((fx, dfx, s))
        };
        newton(t0, f, 1.0e-8 * U::reference_temperature())
    }

    /// Return a new `State` for given temperature $T$ and molar enthalpy $h$.
    pub fn new_nth(
        eos: &Rc<E>,
        temperature: QuantityScalar<U>,
        molar_enthalpy: QuantityScalar<U>,
        moles: &QuantityArray1<U>,
        density_initialization: DensityInitialization<U>,
    ) -> EosResult<Self> {
        let rho0 = match density_initialization {
            DensityInitialization::InitialDensity(r) => r,
            DensityInitialization::Liquid => eos.max_density(Some(moles))?,
            DensityInitialization::Vapor => 1.0e-5 * eos.max_density(Some(moles))?,
            DensityInitialization::None => 0.01 * eos.max_density(Some(moles))?,
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
        newton(rho0, f, 1.0e-12 * U::reference_density())
    }

    /// Return a new `State` for given temperature $T$ and molar entropy $s$.
    pub fn new_nts(
        eos: &Rc<E>,
        temperature: QuantityScalar<U>,
        molar_entropy: QuantityScalar<U>,
        moles: &QuantityArray1<U>,
        density_initialization: DensityInitialization<U>,
    ) -> EosResult<Self> {
        let rho0 = match density_initialization {
            DensityInitialization::InitialDensity(r) => r,
            DensityInitialization::Liquid => eos.max_density(Some(moles))?,
            DensityInitialization::Vapor => 1.0e-5 * eos.max_density(Some(moles))?,
            DensityInitialization::None => 0.01 * eos.max_density(Some(moles))?,
        };
        let n_inv = 1.0 / moles.sum();
        let f = |x0| {
            let s = State::new_nvt(eos, temperature, moles.sum() / x0, moles)?;
            let dfx = -n_inv * s.volume / s.density * s.dp_dt(Contributions::Total);
            let fx = s.molar_entropy(Contributions::Total) - molar_entropy;
            Ok((fx, dfx, s))
        };
        newton(rho0, f, 1.0e-12 * U::reference_density())
    }

    /// Return a new `State` for given pressure $p$ and molar entropy $s$.
    pub fn new_nps(
        eos: &Rc<E>,
        pressure: QuantityScalar<U>,
        molar_entropy: QuantityScalar<U>,
        moles: &QuantityArray1<U>,
        density_initialization: DensityInitialization<U>,
        initial_temperature: Option<QuantityScalar<U>>,
    ) -> EosResult<Self> {
        let t0 = initial_temperature.unwrap_or(298.15 * U::reference_temperature());
        let mut density = density_initialization;
        let f = |x0| {
            let s = State::new_npt(eos, x0, pressure, moles, density)?;
            let dfx = s.c_p(Contributions::Total) / s.temperature;
            let fx = s.molar_entropy(Contributions::Total) - molar_entropy;
            density = DensityInitialization::InitialDensity(s.density);
            Ok((fx, dfx, s))
        };
        newton(t0, f, 1.0e-8 * U::reference_temperature())
    }

    /// Return a new `State` for given volume $V$ and molar internal energy $u$.
    pub fn new_nvu(
        eos: &Rc<E>,
        volume: QuantityScalar<U>,
        molar_internal_energy: QuantityScalar<U>,
        moles: &QuantityArray1<U>,
        initial_temperature: Option<QuantityScalar<U>>,
    ) -> EosResult<Self> {
        let t0 = initial_temperature.unwrap_or(298.15 * U::reference_temperature());
        let f = |x0| {
            let s = State::new_nvt(eos, x0, volume, moles)?;
            let fx = s.molar_internal_energy(Contributions::Total) - molar_internal_energy;
            let dfx = s.c_v(Contributions::Total);
            Ok((fx, dfx, s))
        };
        newton(t0, f, 1.0e-8 * U::reference_temperature())
    }

    /// Update the state with the given temperature
    pub fn update_temperature(&self, temperature: QuantityScalar<U>) -> EosResult<Self> {
        Self::new_nvt(&self.eos, temperature, self.volume, &self.moles)
    }

    /// Update the state with the given chemical potential.
    pub fn update_chemical_potential(
        &mut self,
        chemical_potential: &QuantityArray1<U>,
    ) -> EosResult<()> {
        for _ in 0..50 {
            let dmu_drho = self.dmu_dni(Contributions::Total) * self.volume;
            let f = self.chemical_potential(Contributions::Total) - chemical_potential;
            let dmu_drho_r =
                dmu_drho.to_reduced(U::reference_molar_energy() / U::reference_density())?;
            let f_r = f.to_reduced(U::reference_molar_energy())?;
            let rho = &self.partial_density
                - &(LU::new(dmu_drho_r)?.solve(&f_r) * U::reference_density());
            *self = State::new_nvt(
                &self.eos,
                self.temperature,
                self.volume,
                &(rho * self.volume),
            )?;
            if norm(&f.to_reduced(U::reference_molar_energy())?) < 1e-8 {
                return Ok(());
            }
        }
        Err(EosError::NotConverged(
            "State::update_chemical_potential".into(),
        ))
    }

    /// Update the state with the given molar Gibbs energy.
    pub fn update_gibbs_energy(mut self, molar_gibbs_energy: QuantityScalar<U>) -> EosResult<Self> {
        for _ in 0..50 {
            let df = self.volume / self.density * self.dp_dv(Contributions::Total);
            let f = self.molar_gibbs_energy(Contributions::Total) - molar_gibbs_energy;
            let rho = self.density * (f.to_reduced(df)?).exp();
            self = State::new_nvt(
                &self.eos,
                self.temperature,
                self.total_moles / rho,
                &self.moles,
            )?;
            if f.to_reduced(U::reference_molar_energy())?.abs() < 1e-8 {
                return Ok(self);
            }
        }
        Err(EosError::NotConverged("State::update_gibbs_energy".into()))
    }

    fn derive0(&self) -> StateHD<f64> {
        StateHD::new(
            self.reduced_temperature,
            self.reduced_volume,
            self.reduced_moles.clone(),
        )
    }

    fn derive1(&self, derivative: Derivative) -> StateHD<Dual64> {
        let mut t = Dual64::from(self.reduced_temperature);
        let mut v = Dual64::from(self.reduced_volume);
        let mut n = self.reduced_moles.mapv(Dual64::from);
        match derivative {
            Derivative::DT => t.eps[0] = 1.0,
            Derivative::DV => v.eps[0] = 1.0,
            Derivative::DN(i) => n[i].eps[0] = 1.0,
        }
        StateHD::new(t, v, n)
    }

    fn derive2(&self, derivative1: Derivative, derivative2: Derivative) -> StateHD<HyperDual64> {
        let mut t = HyperDual64::from(self.reduced_temperature);
        let mut v = HyperDual64::from(self.reduced_volume);
        let mut n = self.reduced_moles.mapv(HyperDual64::from);
        match derivative1 {
            Derivative::DT => t.eps1[0] = 1.0,
            Derivative::DV => v.eps1[0] = 1.0,
            Derivative::DN(i) => n[i].eps1[0] = 1.0,
        }
        match derivative2 {
            Derivative::DT => t.eps2[0] = 1.0,
            Derivative::DV => v.eps2[0] = 1.0,
            Derivative::DN(i) => n[i].eps2[0] = 1.0,
        }
        StateHD::new(t, v, n)
    }

    fn derive3(&self, derivative: Derivative) -> StateHD<Dual3_64> {
        let mut t = Dual3_64::from(self.reduced_temperature);
        let mut v = Dual3_64::from(self.reduced_volume);
        let mut n = self.reduced_moles.mapv(Dual3_64::from);
        match derivative {
            Derivative::DT => t = t.derive(),
            Derivative::DV => v = v.derive(),
            Derivative::DN(i) => n[i] = n[i].derive(),
        };
        StateHD::new(t, v, n)
    }
}

fn is_close<U: EosUnit>(
    x: QuantityScalar<U>,
    y: QuantityScalar<U>,
    atol: QuantityScalar<U>,
    rtol: f64,
) -> bool {
    (x - y).abs() <= atol + rtol * y.abs()
}

fn newton<U: EosUnit, E: EquationOfState, F>(
    mut x0: QuantityScalar<U>,
    mut f: F,
    atol: QuantityScalar<U>,
) -> EosResult<State<U, E>>
where
    F: FnMut(QuantityScalar<U>) -> EosResult<(QuantityScalar<U>, QuantityScalar<U>, State<U, E>)>,
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
    Err(EosError::NotConverged("newton".to_owned()))
}

/// Validate the given temperature, mole numbers and volume.
///
/// Properties are valid if
/// * they are finite
/// * they have a positive sign
///
/// There is no validation of the physical state, e.g.
/// if resulting densities are below maximum packing fraction.
fn validate<U: EosUnit>(
    temperature: QuantityScalar<U>,
    volume: QuantityScalar<U>,
    moles: &QuantityArray1<U>,
) -> EosResult<()> {
    let t = temperature.to_reduced(U::reference_temperature())?;
    let v = volume.to_reduced(U::reference_volume())?;
    let m = moles.to_reduced(U::reference_moles())?;
    if !t.is_finite() || t.is_sign_negative() {
        return Err(EosError::InvalidState(
            String::from("validate"),
            String::from("temperature"),
            t,
        ));
    }
    if !v.is_finite() || v.is_sign_negative() {
        return Err(EosError::InvalidState(
            String::from("validate"),
            String::from("volume"),
            v,
        ));
    }
    for &n in m.iter() {
        if !n.is_finite() || n.is_sign_negative() {
            return Err(EosError::InvalidState(
                String::from("validate"),
                String::from("moles"),
                n,
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub enum TPSpec<U> {
    Temperature(QuantityScalar<U>),
    Pressure(QuantityScalar<U>),
}

impl<U: EosUnit> TryFrom<QuantityScalar<U>> for TPSpec<U>
where
    QuantityScalar<U>: std::fmt::Display,
{
    type Error = EosError;
    fn try_from(quantity: QuantityScalar<U>) -> EosResult<Self> {
        if quantity.has_unit(&U::reference_temperature()) {
            Ok(Self::Temperature(quantity))
        } else if quantity.has_unit(&U::reference_pressure()) {
            Ok(Self::Pressure(quantity))
        } else {
            Err(EosError::WrongUnits(
                "temperature or pressure".into(),
                format!("{}", quantity),
            ))
        }
    }
}

mod critical_point;

#[cfg(test)]
mod tests {
    use super::*;
    use quantity::si::*;
    use std::f64::NAN;

    #[test]
    fn test_validate() {
        let temperature = 298.15 * KELVIN;
        let volume = 3000.0 * METER.powi(3);
        let moles = arr1(&[0.03, 0.02, 0.05]) * MOL;
        assert!(validate(temperature, volume, &moles).is_ok());
    }

    #[test]
    fn test_negative_temperature() {
        let temperature = -298.15 * KELVIN;
        let volume = 3000.0 * METER.powi(3);
        let moles = arr1(&[0.03, 0.02, 0.05]) * MOL;
        assert!(validate(temperature, volume, &moles).is_err());
    }

    #[test]
    fn test_nan_temperature() {
        let temperature = NAN * KELVIN;
        let volume = 3000.0 * METER.powi(3);
        let moles = arr1(&[0.03, 0.02, 0.05]) * MOL;
        assert!(validate(temperature, volume, &moles).is_err());
    }

    #[test]
    fn test_negative_mole_number() {
        let temperature = 298.15 * KELVIN;
        let volume = 3000.0 * METER.powi(3);
        let moles = arr1(&[-0.03, 0.02, 0.05]) * MOL;
        assert!(validate(temperature, volume, &moles).is_err());
    }

    #[test]
    fn test_nan_mole_number() {
        let temperature = 298.15 * KELVIN;
        let volume = 3000.0 * METER.powi(3);
        let moles = arr1(&[NAN, 0.02, 0.05]) * MOL;
        assert!(validate(temperature, volume, &moles).is_err());
    }

    #[test]
    fn test_negative_volume() {
        let temperature = 298.15 * KELVIN;
        let volume = -3000.0 * METER.powi(3);
        let moles = arr1(&[0.01, 0.02, 0.05]) * MOL;
        assert!(validate(temperature, volume, &moles).is_err());
    }
}
