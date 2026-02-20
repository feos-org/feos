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
use nalgebra::{DefaultAllocator, Dim, Dyn, OVector};
use num_dual::*;
use quantity::*;
use std::fmt;
use std::ops::Sub;

mod cache;
mod composition;
mod properties;
mod residual_properties;
mod statevec;
pub(crate) use cache::Cache;
pub use composition::Composition;
pub use statevec::StateVec;

/// Possible contributions that can be computed.
#[derive(Clone, Copy, PartialEq)]
pub enum Contributions {
    /// Only compute the ideal gas contribution
    IdealGas,
    /// Only compute the difference between the total and the ideal gas contribution
    Residual,
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
    /// mole fractions
    pub molefracs: OVector<D, N>,
    /// partial number densities in Angstrom^-3
    pub partial_density: OVector<D, N>,
}

impl<N: Dim, D: DualNum<f64> + Copy> StateHD<D, N>
where
    DefaultAllocator: Allocator<N>,
{
    /// Create a new `StateHD` for given temperature, volume and composition.
    pub fn new(temperature: D, volume: D, moles: &OVector<D, N>) -> Self {
        Self::new_density(temperature, &(moles / volume))
    }

    /// Create a new `StateHD` for given temperature and partial densities
    pub fn new_density(temperature: D, partial_density: &OVector<D, N>) -> Self {
        let molefracs = partial_density / partial_density.sum();

        Self {
            temperature,
            molefracs,
            partial_density: partial_density.clone(),
        }
    }

    // Since the molefracs can not be reproduced from moles if the density is zero,
    // this constructor exists specifically for these cases.
    pub(crate) fn new_virial(temperature: D, density: D, molefracs: &OVector<D, N>) -> Self {
        let partial_density = molefracs * density;
        Self {
            temperature,
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
#[derive(Debug, Clone)]
pub struct State<E, N: Dim = Dyn, D: DualNum<f64> + Copy = f64>
where
    DefaultAllocator: Allocator<N>,
{
    /// Equation of state
    pub eos: E,
    /// Temperature $T$
    pub temperature: Temperature<D>,
    /// Molar volume $v=\frac{V}{N}$
    pub molar_volume: MolarVolume<D>,
    /// Total number of moles $N=\sum_iN_i$
    pub total_moles: Option<Moles<D>>,
    /// Total density $\rho=\frac{N}{V}=\sum_i\rho_i$
    pub density: Density<D>,
    /// Mole fractions $x_i=\frac{N_i}{N}=\frac{\rho_i}{\rho}$
    pub molefracs: OVector<D, N>,
    /// Cache
    cache: Cache<D, N>,
}

impl<E, N: Dim, D: DualNum<f64> + Copy> State<E, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    /// Set the total amount of substance to the given value.
    ///
    /// This method does not introduce inconsistencies, because the
    /// total moles are the only field that stores information about
    /// the size of the state.
    pub fn set_total_moles(mut self, total_moles: Moles<D>) -> State<E, N, D> {
        self.total_moles = Some(total_moles);
        self
    }

    /// Partial densities $\rho_i=\frac{N_i}{V}$
    pub fn partial_density(&self) -> Density<OVector<D, N>> {
        Dimensionless::new(&self.molefracs) * self.density
    }

    /// Mole numbers $N_i$
    pub fn moles(&self) -> FeosResult<Moles<OVector<D, N>>> {
        Ok(Dimensionless::new(&self.molefracs) * self.total_moles()?)
    }

    /// Total moles $N=\sum_iN_i$
    pub fn total_moles(&self) -> FeosResult<Moles<D>> {
        self.total_moles.ok_or(FeosError::IntensiveState)
    }

    /// Volume $V$
    pub fn volume(&self) -> FeosResult<Volume<D>> {
        Ok(self.molar_volume * self.total_moles()?)
    }
}

impl<E: Residual<N, D>, N: Dim, D: DualNum<f64> + Copy> fmt::Display for State<E, N, D>
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
                "T = {:.5}, ρ = {:.5}, x = {:.5?}",
                self.temperature.re(),
                self.density.re(),
                self.molefracs.map(|x| x.re()).as_slice()
            )
        }
    }
}

impl<E: Residual<N, D>, N: Dim, D: DualNum<f64> + Copy> State<E, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    /// Return a new `State` given a temperature, an array of mole numbers and a volume.
    ///
    /// This function will perform a validation of the given properties, i.e. test for signs
    /// and if values are finite. It will **not** validate physics, i.e. if the resulting
    /// densities are below the maximum packing fraction.
    pub fn new_nvt<X: Composition<D, N>>(
        eos: &E,
        temperature: Temperature<D>,
        volume: Volume<D>,
        composition: X,
    ) -> FeosResult<Self> {
        let (molefracs, total_moles) = composition.into_molefracs(eos)?;
        let Some(total_moles) = total_moles else {
            return Err(FeosError::UndeterminedState(
                "Missing total mole number in the specification!".into(),
            ));
        };

        let density = total_moles / volume;
        Self::new(eos, temperature, density, (molefracs, total_moles))
    }

    /// Return a new `State` given a temperature and the partial density of all components.
    ///
    /// This function will perform a validation of the given properties, i.e. test for signs
    /// and if values are finite. It will **not** validate physics, i.e. if the resulting
    /// densities are below the maximum packing fraction.
    pub fn new_density(
        eos: &E,
        temperature: Temperature<D>,
        partial_density: Density<OVector<D, N>>,
    ) -> FeosResult<Self> {
        let density = partial_density.sum();
        let molefracs = partial_density.convert_into(density);
        Self::new(eos, temperature, density, molefracs)
    }

    /// Return a new `State` for a pure component given a temperature and a density.
    ///
    /// This function will perform a validation of the given properties, i.e. test for signs
    /// and if values are finite. It will **not** validate physics, i.e. if the resulting
    /// densities are below the maximum packing fraction.
    pub fn new_pure(eos: &E, temperature: Temperature<D>, density: Density<D>) -> FeosResult<Self>
    where
        (): Composition<D, N>,
    {
        Self::new(eos, temperature, density, ())
    }

    /// Return a new `State` given a temperature, a density and the composition.
    ///
    /// This function will perform a validation of the given properties, i.e. test for signs
    /// and if values are finite. It will **not** validate physics, i.e. if the resulting
    /// densities are below the maximum packing fraction.
    pub fn new<X: Composition<D, N>>(
        eos: &E,
        temperature: Temperature<D>,
        density: Density<D>,
        composition: X,
    ) -> FeosResult<Self> {
        let (molefracs, total_moles) = composition.into_molefracs(eos)?;
        Self::_new(eos, temperature, density, molefracs, total_moles)
    }

    fn _new(
        eos: &E,
        temperature: Temperature<D>,
        density: Density<D>,
        molefracs: OVector<D, N>,
        total_moles: Option<Moles<D>>,
    ) -> FeosResult<Self> {
        let molar_volume = density.inv();
        validate(temperature, density, &molefracs)?;
        Ok(State {
            eos: eos.clone(),
            temperature,
            molar_volume,
            density,
            molefracs,
            total_moles,
            cache: Cache::new(),
        })
    }

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
    pub fn build<X: Composition<D, N>>(
        eos: &E,
        temperature: Temperature<D>,
        volume: Option<Volume<D>>,
        density: Option<Density<D>>,
        composition: X,
        pressure: Option<Pressure<D>>,
        density_initialization: Option<DensityInitialization>,
    ) -> FeosResult<Self> {
        Self::_build(
            eos,
            temperature,
            volume,
            density,
            composition,
            pressure,
            density_initialization,
        )?
        .ok_or_else(|| FeosError::UndeterminedState(String::from("Missing input parameters.")))
    }

    fn _build<X: Composition<D, N>>(
        eos: &E,
        temperature: Temperature<D>,
        volume: Option<Volume<D>>,
        density: Option<Density<D>>,
        composition: X,
        pressure: Option<Pressure<D>>,
        density_initialization: Option<DensityInitialization>,
    ) -> FeosResult<Option<Self>> {
        // unwrap composition
        let (x, n) = composition.into_molefracs(eos)?;

        let t = temperature;
        let di = density_initialization;
        // find the appropriate state constructor
        match (volume, density, n, pressure) {
            (None, None, None, None) => Ok(None),
            (None, None, Some(_), None) => Ok(None),
            (Some(_), None, None, None) => Ok(None),
            (None, None, _, Some(p)) => State::new_npt(eos, t, p, (x, n), di).map(Some),
            (None, Some(d), _, None) => State::new(eos, t, d, (x, n)).map(Some),
            (Some(v), None, None, Some(p)) => State::new_tpvx(eos, t, p, v, x, di).map(Some),
            (Some(v), None, Some(n), None) => State::new_nvt(eos, t, v, (x, n)).map(Some),
            (Some(v), Some(d), None, None) => State::new_nvt(eos, t, v, (x, d * v)).map(Some),
            (Some(_), Some(_), Some(_), _) => Err(FeosError::UndeterminedState(String::from(
                "Density is overdetermined.",
            ))),
            (_, _, _, Some(_)) => Err(FeosError::UndeterminedState(String::from(
                "Pressure is overdetermined.",
            ))),
        }
    }

    /// Return a new `State` using a density iteration. [DensityInitialization] is used to
    /// influence the calculation with respect to the possible solutions.
    pub fn new_npt<X: Composition<D, N>>(
        eos: &E,
        temperature: Temperature<D>,
        pressure: Pressure<D>,
        composition: X,
        density_initialization: Option<DensityInitialization>,
    ) -> FeosResult<Self> {
        let (molefracs, total_moles) = composition.into_molefracs(eos)?;
        density_iteration(
            eos,
            temperature,
            pressure,
            &molefracs,
            density_initialization,
        )
        .and_then(|density| Self::_new(eos, temperature, density, molefracs, total_moles))
    }

    /// Return a new `State` for given pressure $p$, volume $V$, temperature $T$ and composition $x_i$.
    pub fn new_tpvx(
        eos: &E,
        temperature: Temperature<D>,
        pressure: Pressure<D>,
        volume: Volume<D>,
        molefracs: OVector<D, N>,
        density_initialization: Option<DensityInitialization>,
    ) -> FeosResult<Self> {
        let density = density_iteration(
            eos,
            temperature,
            pressure,
            &molefracs,
            density_initialization,
        )?;
        Self::new_nvt(eos, temperature, volume, (molefracs, density * volume))
    }
}

impl<E: Total<N, D>, N: Gradients, D: DualNum<f64> + Copy> State<E, N, D>
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
    pub fn build_full<X: Composition<D, N> + Clone>(
        eos: &E,
        temperature: Option<Temperature<D>>,
        volume: Option<Volume<D>>,
        density: Option<Density<D>>,
        composition: X,
        pressure: Option<Pressure<D>>,
        molar_enthalpy: Option<MolarEnergy<D>>,
        molar_entropy: Option<MolarEntropy<D>>,
        molar_internal_energy: Option<MolarEnergy<D>>,
        density_initialization: Option<DensityInitialization>,
        initial_temperature: Option<Temperature<D>>,
    ) -> FeosResult<Self> {
        let state = if let Some(temperature) = temperature {
            Self::_build(
                eos,
                temperature,
                volume,
                density,
                composition.clone(),
                pressure,
                density_initialization,
            )?
        } else {
            None
        };

        let ti = initial_temperature;
        match state {
            Some(state) => Ok(state),
            None => {
                // Check if new state can be created using molar_enthalpy and temperature
                if let (Some(p), Some(h)) = (pressure, molar_enthalpy) {
                    return State::new_nph(eos, p, h, composition, density_initialization, ti);
                }
                if let (Some(p), Some(s)) = (pressure, molar_entropy) {
                    return State::new_nps(eos, p, s, composition, density_initialization, ti);
                }
                if let (Some(t), Some(h)) = (temperature, molar_enthalpy) {
                    return State::new_nth(eos, t, h, composition, density_initialization);
                }
                if let (Some(t), Some(s)) = (temperature, molar_entropy) {
                    return State::new_nts(eos, t, s, composition, density_initialization);
                }
                if let (Some(u), Some(v)) = (molar_internal_energy, volume) {
                    let (molefracs, total_moles) = composition.into_molefracs(eos)?;
                    if let Some(n) = total_moles {
                        return State::new_nvu(eos, v, u, (molefracs, n), ti);
                    }
                }
                Err(FeosError::UndeterminedState(String::from(
                    "Missing input parameters.",
                )))
            }
        }
    }

    /// Return a new `State` for given pressure $p$ and molar enthalpy $h$.
    pub fn new_nph<X: Composition<D, N> + Clone>(
        eos: &E,
        pressure: Pressure<D>,
        molar_enthalpy: MolarEnergy<D>,
        composition: X,
        density_initialization: Option<DensityInitialization>,
        initial_temperature: Option<Temperature<D>>,
    ) -> FeosResult<Self> {
        let t0 = initial_temperature.unwrap_or(Temperature::from_reduced(D::from(298.15)));
        let mut density = density_initialization;
        let f = |x0| {
            let s = State::new_npt(eos, x0, pressure, composition.clone(), density)?;
            let dfx = s.molar_isobaric_heat_capacity(Contributions::Total);
            let fx = s.molar_enthalpy(Contributions::Total) - molar_enthalpy;
            density = Some(DensityInitialization::InitialDensity(s.density.re()));
            Ok((fx, dfx, s))
        };
        newton(t0, f, Temperature::from_reduced(1.0e-8))
    }

    /// Return a new `State` for given temperature $T$ and molar enthalpy $h$.
    pub fn new_nth<X: Composition<D, N> + Clone>(
        eos: &E,
        temperature: Temperature<D>,
        molar_enthalpy: MolarEnergy<D>,
        composition: X,
        density_initialization: Option<DensityInitialization>,
    ) -> FeosResult<Self> {
        let (x, _) = composition.clone().into_molefracs(eos)?;
        let rho0 = match density_initialization {
            Some(DensityInitialization::InitialDensity(r)) => {
                Density::from_reduced(D::from(r.into_reduced()))
            }
            Some(DensityInitialization::Liquid) => eos.max_density(&x)?,
            Some(DensityInitialization::Vapor) => eos.max_density(&x)? * 1.0e-5,
            None => eos.max_density(&x)? * 0.01,
        };
        let f = |rho| {
            let s = State::new(eos, temperature, rho, composition.clone())?;
            let dfx = -s.molar_volume
                * s.molar_volume
                * (s.molar_volume * s.dp_dv(Contributions::Total)
                    + temperature * s.dp_dt(Contributions::Total));
            let fx = s.molar_enthalpy(Contributions::Total) - molar_enthalpy;
            Ok((fx, dfx, s))
        };
        newton(rho0, f, Density::from_reduced(1.0e-12))
    }

    /// Return a new `State` for given temperature $T$ and molar entropy $s$.
    pub fn new_nts<X: Composition<D, N> + Clone>(
        eos: &E,
        temperature: Temperature<D>,
        molar_entropy: MolarEntropy<D>,
        composition: X,
        density_initialization: Option<DensityInitialization>,
    ) -> FeosResult<Self> {
        let (x, _) = composition.clone().into_molefracs(eos)?;
        let rho0 = match density_initialization {
            Some(DensityInitialization::InitialDensity(r)) => {
                Density::from_reduced(D::from(r.into_reduced()))
            }
            Some(DensityInitialization::Liquid) => eos.max_density(&x)?,
            Some(DensityInitialization::Vapor) => eos.max_density(&x)? * 1.0e-5,
            None => eos.max_density(&x)? * 0.01,
        };
        let f = |rho| {
            let s = State::new(eos, temperature, rho, composition.clone())?;
            let dfx = -s.molar_volume * s.molar_volume * s.dp_dt(Contributions::Total);
            let fx = s.molar_entropy(Contributions::Total) - molar_entropy;
            Ok((fx, dfx, s))
        };
        newton(rho0, f, Density::from_reduced(1.0e-12))
    }

    /// Return a new `State` for given pressure $p$ and molar entropy $s$.
    pub fn new_nps<X: Composition<D, N> + Clone>(
        eos: &E,
        pressure: Pressure<D>,
        molar_entropy: MolarEntropy<D>,
        composition: X,
        density_initialization: Option<DensityInitialization>,
        initial_temperature: Option<Temperature<D>>,
    ) -> FeosResult<Self> {
        let t0 = initial_temperature.unwrap_or(Temperature::from_reduced(D::from(298.15)));
        let mut density = density_initialization;
        let f = |x0| {
            let s = State::new_npt(eos, x0, pressure, composition.clone(), density)?;
            let dfx = s.molar_isobaric_heat_capacity(Contributions::Total) / s.temperature;
            let fx = s.molar_entropy(Contributions::Total) - molar_entropy;
            density = Some(DensityInitialization::InitialDensity(s.density.re()));
            Ok((fx, dfx, s))
        };
        newton(t0, f, Temperature::from_reduced(1.0e-8))
    }

    /// Return a new `State` for given volume $V$ and molar internal energy $u$.
    pub fn new_nvu<X: Composition<D, N> + Clone>(
        eos: &E,
        volume: Volume<D>,
        molar_internal_energy: MolarEnergy<D>,
        composition: X,
        initial_temperature: Option<Temperature<D>>,
    ) -> FeosResult<Self> {
        let t0 = initial_temperature.unwrap_or(Temperature::from_reduced(D::from(298.15)));
        let f = |x0| {
            let s = State::new_nvt(eos, x0, volume, composition.clone())?;
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

fn newton<E: Residual<N, D>, N: Dim, D: DualNum<f64> + Copy, F, X: Copy, Y>(
    mut x0: Quantity<D, X>,
    mut f: F,
    atol: Quantity<f64, X>,
) -> FeosResult<State<E, N, D>>
where
    DefaultAllocator: Allocator<N>,
    Y: Sub<X> + Sub<<Y as Sub<X>>::Output, Output = X>,
    F: FnMut(
        Quantity<D, X>,
    ) -> FeosResult<(
        Quantity<D, Y>,
        Quantity<D, <Y as Sub<X>>::Output>,
        State<E, N, D>,
    )>,
{
    let rtol = 1e-10;
    let maxiter = 50;

    for _ in 0..maxiter {
        let (fx, dfx, mut state) = f(x0)?;
        let x = x0 - fx / dfx;
        if is_close(x.re(), x0.re(), atol, rtol) {
            // Ensure that at least NDERIV iterations are performed (for implicit AD)
            for _ in 0..D::NDERIV {
                let (fx, dfx, s) = f(x0)?;
                x0 -= fx / dfx;
                state = s;
            }
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

    #[test]
    fn test_validate() {
        let temperature = 298.15 * KELVIN;
        let density = 3000.0 * MOL / METER.powi::<3>();
        let molefracs = dvector![0.03, 0.02, 0.05];
        assert!(validate(temperature, density, &molefracs).is_ok());
    }

    #[test]
    fn test_negative_temperature() {
        let temperature = -298.15 * KELVIN;
        let density = 3000.0 * MOL / METER.powi::<3>();
        let molefracs = dvector![0.03, 0.02, 0.05];
        assert!(validate(temperature, density, &molefracs).is_err());
    }

    #[test]
    fn test_nan_temperature() {
        let temperature = f64::NAN * KELVIN;
        let density = 3000.0 * MOL / METER.powi::<3>();
        let molefracs = dvector![0.03, 0.02, 0.05];
        assert!(validate(temperature, density, &molefracs).is_err());
    }

    #[test]
    fn test_negative_mole_number() {
        let temperature = 298.15 * KELVIN;
        let density = 3000.0 * MOL / METER.powi::<3>();
        let molefracs = dvector![-0.03, 0.02, 0.05];
        assert!(validate(temperature, density, &molefracs).is_err());
    }

    #[test]
    fn test_nan_mole_number() {
        let temperature = 298.15 * KELVIN;
        let density = 3000.0 * MOL / METER.powi::<3>();
        let molefracs = dvector![f64::NAN, 0.02, 0.05];
        assert!(validate(temperature, density, &molefracs).is_err());
    }

    #[test]
    fn test_negative_density() {
        let temperature = 298.15 * KELVIN;
        let density = -3000.0 * MOL / METER.powi::<3>();
        let molefracs = dvector![0.01, 0.02, 0.05];
        assert!(validate(temperature, density, &molefracs).is_err());
    }
}
