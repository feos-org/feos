//! Physical quantities with compile-time checked units.

#![allow(clippy::type_complexity)]
use ang::{Angle, Degrees, Radians};
use ndarray::{Array, ArrayBase, Data, Dimension};
use num_traits::Zero;
use std::marker::PhantomData;
use std::ops::{Div, Mul, Sub};
use typenum::{ATerm, Diff, Integer, Negate, Quot, Sum, TArr, N1, N2, P1, P3, Z0};

mod array;
mod fmt;
mod ops;
#[cfg(feature = "python")]
mod python;

pub type SIUnit<T, L, M, I, THETA, N, J> =
    TArr<T, TArr<L, TArr<M, TArr<I, TArr<THETA, TArr<N, TArr<J, ATerm>>>>>>>;

/// Physical quantity with compile-time checked unit.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Quantity<T, U>(T, PhantomData<U>);

pub type _Dimensionless = SIUnit<Z0, Z0, Z0, Z0, Z0, Z0, Z0>;
pub type _Time = SIUnit<P1, Z0, Z0, Z0, Z0, Z0, Z0>;
pub type _Length = SIUnit<Z0, P1, Z0, Z0, Z0, Z0, Z0>;
pub type _Mass = SIUnit<Z0, Z0, P1, Z0, Z0, Z0, Z0>;
pub type _Current = SIUnit<Z0, Z0, Z0, P1, Z0, Z0, Z0>;
pub type _Temperature = SIUnit<Z0, Z0, Z0, Z0, P1, Z0, Z0>;
pub type _Moles = SIUnit<Z0, Z0, Z0, Z0, Z0, P1, Z0>;
pub type _LuminousIntensity = SIUnit<Z0, Z0, Z0, Z0, Z0, Z0, P1>;

pub type Dimensionless<T = f64> = Quantity<T, _Dimensionless>;
pub type Time<T = f64> = Quantity<T, _Time>;
pub type Length<T = f64> = Quantity<T, _Length>;
pub type Mass<T = f64> = Quantity<T, _Mass>;
pub type Current<T = f64> = Quantity<T, _Current>;
pub type Temperature<T = f64> = Quantity<T, _Temperature>;
pub type Moles<T = f64> = Quantity<T, _Moles>;
pub type LuminousIntensity<T = f64> = Quantity<T, _LuminousIntensity>;

pub type _Frequency = Negate<_Time>;
pub type Frequency<T = f64> = Quantity<T, _Frequency>;
pub type _Velocity = Diff<_Length, _Time>;
pub type Velocity<T = f64> = Quantity<T, _Velocity>;
pub type _Acceleration = Diff<_Velocity, _Time>;
pub type Acceleration<T = f64> = Quantity<T, _Acceleration>;
pub type _Force = Sum<_Mass, _Acceleration>;
pub type Force<T = f64> = Quantity<T, _Force>;
pub type _Area = Sum<_Length, _Length>;
pub type Area<T = f64> = Quantity<T, _Area>;
pub type _Volume = Sum<_Area, _Length>;
pub type Volume<T = f64> = Quantity<T, _Volume>;
pub type _Energy = Sum<_Force, _Length>;
pub type Energy<T = f64> = Quantity<T, _Energy>;
pub type _Pressure = Diff<_Energy, _Volume>;
pub type Pressure<T = f64> = Quantity<T, _Pressure>;
pub type _Power = Diff<_Energy, _Time>;
pub type Power<T = f64> = Quantity<T, _Power>;
pub type _Charge = Sum<_Current, _Time>;
pub type Charge<T = f64> = Quantity<T, _Charge>;
pub type _ElectricPotential = Diff<_Power, _Current>;
pub type ElectricPotential<T = f64> = Quantity<T, _ElectricPotential>;
pub type _Capacitance = Diff<_Charge, _ElectricPotential>;
pub type Capacitance<T = f64> = Quantity<T, _Capacitance>;
pub type _Resistance = Diff<_ElectricPotential, _Current>;
pub type Resistance<T = f64> = Quantity<T, _Resistance>;
pub type _ElectricalConductance = Negate<_Resistance>;
pub type ElectricalConductance<T = f64> = Quantity<T, _ElectricalConductance>;
pub type _MagneticFlux = Sum<_ElectricPotential, _Time>;
pub type MagneticFlux<T = f64> = Quantity<T, _MagneticFlux>;
pub type _MagneticFluxDensity = Diff<_MagneticFlux, _Area>;
pub type MagneticFluxDensity<T = f64> = Quantity<T, _MagneticFluxDensity>;
pub type _Inductance = Diff<_MagneticFlux, _Current>;
pub type Inductance<T = f64> = Quantity<T, _Inductance>;

pub type _Entropy = Diff<_Energy, _Temperature>;
pub type Entropy<T = f64> = Quantity<T, _Entropy>;
pub type _EntropyPerTemperature = Diff<_Entropy, _Temperature>;
pub type EntropyPerTemperature<T = f64> = Quantity<T, _EntropyPerTemperature>;
pub type _MolarEntropy = Diff<_Entropy, _Moles>;
pub type MolarEntropy<T = f64> = Quantity<T, _MolarEntropy>;
pub type _MolarEnergy = Diff<_Energy, _Moles>;
pub type MolarEnergy<T = f64> = Quantity<T, _MolarEnergy>;
pub type _SpecificEntropy = Diff<_Entropy, _Mass>;
pub type SpecificEntropy<T = f64> = Quantity<T, _SpecificEntropy>;
pub type _SpecificEnergy = Diff<_Energy, _Mass>;
pub type SpecificEnergy<T = f64> = Quantity<T, _SpecificEnergy>;
pub type _MolarWeight = Diff<_Mass, _Moles>;
pub type MolarWeight<T = f64> = Quantity<T, _MolarWeight>;
pub type _Density = Diff<_Moles, _Volume>;
pub type Density<T = f64> = Quantity<T, _Density>;
pub type _MassDensity = Diff<_Mass, _Volume>;
pub type MassDensity<T = f64> = Quantity<T, _MassDensity>;
pub type _PressurePerVolume = Diff<_Pressure, _Volume>;
pub type PressurePerVolume<T = f64> = Quantity<T, _PressurePerVolume>;
pub type _PressurePerTemperature = Diff<_Pressure, _Temperature>;
pub type PressurePerTemperature<T = f64> = Quantity<T, _PressurePerTemperature>;
pub type _Compressibility = Negate<_Pressure>;
pub type Compressibility<T = f64> = Quantity<T, _Compressibility>;
pub type _MolarVolume = Diff<_Volume, _Moles>;
pub type MolarVolume<T = f64> = Quantity<T, _MolarVolume>;
pub type _EntropyDensity = Diff<_Entropy, _Volume>;
pub type EntropyDensity<T = f64> = Quantity<T, _EntropyDensity>;
pub type _Action = Sum<_Energy, _Time>;
pub type Action<T=f64> = Quantity<T, _Action>;
pub type _HeatCapacityRate = Diff<_Power, _Temperature>;
pub type HeatCapacityRate<T=f64> = Quantity<T, _HeatCapacityRate>;
pub type _MassFlowRate = Diff<_Mass, _Time>;
pub type MassFlowRate<T=f64> = Quantity<T, _MassFlowRate>;
pub type _MoleFlowRate = Diff<_Moles, _Time>;
pub type MoleFlowRate<T=f64> = Quantity<T, _MoleFlowRate>;

pub type _Viscosity = Sum<_Pressure, _Time>;
pub type Viscosity<T = f64> = Quantity<T, _Viscosity>;
pub type _Diffusivity = Sum<_Velocity, _Length>;
pub type Diffusivity<T = f64> = Quantity<T, _Diffusivity>;
pub type _ThermalConductivity = Diff<_Power, Sum<_Length, _Temperature>>;
pub type ThermalConductivity<T = f64> = Quantity<T, _ThermalConductivity>;
pub type _SurfaceTension = Diff<_Force, _Length>;
pub type SurfaceTension<T = f64> = Quantity<T, _SurfaceTension>;

/// SI base unit second $\\left(\text{s}\\right)$
pub const SECOND: Time = Quantity(1.0, PhantomData);
/// SI base unit meter $\\left(\text{m}\\right)$
pub const METER: Length = Quantity(1.0, PhantomData);
/// SI base unit kilogram $\\left(\text{kg}\\right)$
pub const KILOGRAM: Mass = Quantity(1.0, PhantomData);
/// SI base unit Ampere $\\left(\text{A}\\right)$
pub const AMPERE: Current = Quantity(1.0, PhantomData);
/// SI base unit Kelvin $\\left(\text{K}\\right)$
pub const KELVIN: Temperature = Quantity(1.0, PhantomData);
/// SI base unit mol $\\left(\text{mol}\\right)$
pub const MOL: Moles = Quantity(1.0, PhantomData);
/// SI base unit candela $\\left(\text{cd}\\right)$
pub const CANDELA: LuminousIntensity = Quantity(1.0, PhantomData);

/// Derived unit Hertz $\\left(1\\,\text{Hz}=1\\,\text{s}^{-1}\\right)$
pub const HERTZ: Frequency = Quantity(1.0, PhantomData);
/// Derived unit Newton $\\left(1\\,\text{N}=1\\,\text{kg}\\frac{\text{m}}{\text{s}^2}\\right)$
pub const NEWTON: Force = Quantity(1.0, PhantomData);
/// Derived unit Pascal $\\left(1\\,\text{Pa}=1\\,\\frac{\text{kg}}{\text{m}\\cdot\text{s}^2}\\right)$
pub const PASCAL: Pressure = Quantity(1.0, PhantomData);
/// Derived unit Joule $\\left(1\\,\text{J}=1\\,\text{kg}\\frac{\text{m}^2}{\text{s}^2}\\right)$
pub const JOULE: Energy = Quantity(1.0, PhantomData);
/// Derived unit Watt $\\left(1\\,\text{J}=1\\,\text{kg}\\frac{\text{m}^2}{\text{s}^3}\\right)$
pub const WATT: Power = Quantity(1.0, PhantomData);
/// Derived unit Coulomb $\\left(1\\,\text{C}=1\\,\text{A}\cdot\text{s}\\right)$
pub const COULOMB: Charge = Quantity(1.0, PhantomData);
/// Derived unit Volt $\\left(1\\,\text{V}=1\\,\\frac{\text{W}}{\text{A}}\\right)$
pub const VOLT: ElectricPotential = Quantity(1.0, PhantomData);
/// Derived unit Farad $\\left(1\\,\text{F}=1\\,\\frac{\text{C}}{\text{V}}\\right)$
pub const FARAD: Capacitance = Quantity(1.0, PhantomData);
/// Derived unit Ohm $\\left(1\\,\text{Ω}=1\\,\\frac{\text{V}}{\text{A}}\\right)$
pub const OHM: Resistance = Quantity(1.0, PhantomData);
/// Derived unit Siemens $\\left(1\\,\text{S}=1\\,\text{Ω}^{-1}\\right)$
pub const SIEMENS: ElectricalConductance = Quantity(1.0, PhantomData);
/// Derived unit Weber $\\left(1\\,\text{Wb}=1\\,\text{V}\\cdot\text{s}\\right)$
pub const WEBER: MagneticFlux = Quantity(1.0, PhantomData);
/// Derived unit Tesla $\\left(1\\,\text{T}=1\\,\\frac{\text{Wb}}{\text{m}^2}\\right)$
pub const TESLA: MagneticFluxDensity = Quantity(1.0, PhantomData);
/// Derived unit Henry $\\left(1\\,\text{T}=1\\,\\frac{\text{Wb}}{\text{A}}\\right)$
pub const HENRY: Inductance = Quantity(1.0, PhantomData);

/// Additional unit Ångstrom $\\left(1\\,\text{\\AA}=10^{-10}\\,\text{m}\\right)$
pub const ANGSTROM: Length = Quantity(1e-10, PhantomData);
/// Additional unit unified atomic mass $\\left(1\\,\text{u}\\approx 1.660539\\times 10^{-27}\\,\text{kg}\\right)$
pub const AMU: Mass = Quantity(1.6605390671738466e-27, PhantomData);
/// Additional unit astronomical unit $\\left(1\\,\text{au}=149597870700\\,\text{m}\\right)$
pub const AU: Length = Quantity(149597870700.0, PhantomData);
/// Additional unit bar $\\left(1\\,\text{bar}=10^5\\,\text{Pa}\\right)$
pub const BAR: Pressure = Quantity(1e5, PhantomData);
/// Additional unit calorie $\\left(1\\,\text{cal}=4.184\\,\text{J}\\right)$
pub const CALORIE: Energy = Quantity(4.184, PhantomData);
/// Additional unit day $\\left(1\\,\text{d}=86400,\text{s}\\right)$
pub const DAY: Time = Quantity(86400.0, PhantomData);
/// Additional unit gram $\\left(1\\,\text{g}=10^{-3}\\,\text{kg}\\right)$
pub const GRAM: Mass = Quantity(1e-3, PhantomData);
/// Additional unit hour $\\left(1\\,\text{h}=3600,\text{s}\\right)$
pub const HOUR: Time = Quantity(3600.0, PhantomData);
/// Additional unit liter $\\left(1\\,\text{l}=10^{-3}\\,\text{m}^3\\right)$
pub const LITER: Volume = Quantity(1e-3, PhantomData);
/// Additional unit minute $\\left(1\\,\text{min}=60,\text{s}\\right)$
pub const MINUTE: Time = Quantity(60.0, PhantomData);

/// Angle unit radian $\\left(\text{rad}\\right)$
pub const RADIANS: Angle = Radians(1.0);
/// Angle unit degree $\\left(1^\\circ=\frac{\pi}{180}\\,\text{rad}\\approx 0.0174532925\\,\text{rad}\\right)$
pub const DEGREES: Angle = Degrees(1.0);

/// Boltzmann constant $\\left(k_\text{B}=1.380649\times 10^{-23}\\,\\frac{\text{J}}{\text{K}}\\right)$
pub const KB: Entropy = Quantity(1.380649e-23, PhantomData);
/// Avogadro constant $\\left(N_\text{A}=6.02214076\times 10^{23}\\,\text{mol}^{-1}\\right)$
pub const NAV: Quantity<f64, Negate<_Moles>> = Quantity(6.02214076e23, PhantomData);
/// Planck constant $\\left(h=6.62607015\times 10^{-34}\\,\text{J}\\cdot\text{s}\\right)$
pub const PLANCK: Action = Quantity(6.62607015e-34, PhantomData);
/// Ideal gas constant $\\left(R=8.31446261815324\\,\\frac{\text{J}}{\text{molK}}\\right)$
pub const RGAS: MolarEntropy = Quantity(8.31446261815324, PhantomData);
/// Hyperfine transition frequency of Cs $\\left(\Delta\\nu_\text{Cs}=9192631770\\,\text{Hz}\\right)$
pub const DVCS: Frequency = Quantity(9192631770.0, PhantomData);
/// Elementary charge $\\left(e=1.602176634\\times 10^{-19}\\,\text{C}\\right)$
pub const QE: Charge = Quantity(1.602176634e-19, PhantomData);
/// Speed of light $\\left(c=299792458\\,\\frac{\text{m}}{\text{s}}\\right)$
pub const CLIGHT: Velocity = Quantity(299792458.0, PhantomData);
/// Luminous efficacy of $540\\,\text{THz}$ radiation $\\left(K_\text{cd}=683\\,\\frac{\text{lm}}{\text{W}}\\right)$
pub const KCD: Quantity<f64, SIUnit<N2, N1, P3, Z0, Z0, Z0, P1>> = Quantity(683.0, PhantomData);
/// Gravitational constant $\\left(G=6.6743\\times 10^{-11}\\,\\frac{\text{m}^3}{\text{kg}\cdot\text{s}^2}\\right)$
pub const G: Quantity<f64, SIUnit<P3, N1, N2, Z0, Z0, Z0, Z0>> = Quantity(6.6743e-11, PhantomData);

/// Prefix quecto $\\left(\text{q}=10^{-30}\\right)$
pub const QUECTO: f64 = 1e-30;
/// Prefix ronto $\\left(\text{r}=10^{-27}\\right)$
pub const RONTO: f64 = 1e-27;
/// Prefix yocto $\\left(\text{y}=10^{-24}\\right)$
pub const YOCTO: f64 = 1e-24;
/// Prefix zepto $\\left(\text{z}=10^{-21}\\right)$
pub const ZEPTO: f64 = 1e-21;
/// Prefix atto $\\left(\text{a}=10^{-18}\\right)$
pub const ATTO: f64 = 1e-18;
/// Prefix femto $\\left(\text{f}=10^{-15}\\right)$
pub const FEMTO: f64 = 1e-15;
/// Prefix pico $\\left(\text{p}=10^{-12}\\right)$
pub const PICO: f64 = 1e-12;
/// Prefix nano $\\left(\text{n}=10^{-9}\\right)$
pub const NANO: f64 = 1e-9;
/// Prefix micro $\\left(\text{µ}=10^{-6}\\right)$
pub const MICRO: f64 = 1e-6;
/// Prefix milli $\\left(\text{m}=10^{-3}\\right)$
pub const MILLI: f64 = 1e-3;
/// Prefix centi $\\left(\text{c}=10^{-2}\\right)$
pub const CENTI: f64 = 1e-2;
/// Prefix deci $\\left(\text{d}=10^{-1}\\right)$
pub const DECI: f64 = 1e-1;
/// Prefix deca $\\left(\text{da}=10^{1}\\right)$
pub const DECA: f64 = 1e1;
/// Prefix hecto $\\left(\text{h}=10^{2}\\right)$
pub const HECTO: f64 = 1e2;
/// Prefix kilo $\\left(\text{k}=10^{3}\\right)$
pub const KILO: f64 = 1e3;
/// Prefix mega $\\left(\text{M}=10^{6}\\right)$
pub const MEGA: f64 = 1e6;
/// Prefix giga $\\left(\text{G}=10^{9}\\right)$
pub const GIGA: f64 = 1e9;
/// Prefix tera $\\left(\text{T}=10^{12}\\right)$
pub const TERA: f64 = 1e12;
/// Prefix peta $\\left(\text{P}=10^{15}\\right)$
pub const PETA: f64 = 1e15;
/// Prefix exa $\\left(\text{E}=10^{18}\\right)$
pub const EXA: f64 = 1e18;
/// Prefix zetta $\\left(\text{Z}=10^{21}\\right)$
pub const ZETTA: f64 = 1e21;
/// Prefix yotta $\\left(\text{Y}=10^{24}\\right)$
pub const YOTTA: f64 = 1e24;
/// Prefix ronna $\\left(\text{R}=10^{27}\\right)$
pub const RONNA: f64 = 1e27;
/// Prefix quetta $\\left(\text{Q}=10^{30}\\right)$
pub const QUETTA: f64 = 1e30;

/// Additional unit degrees Celsius
pub struct CELSIUS;

impl Mul<CELSIUS> for f64 {
    type Output = Temperature<f64>;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, _: CELSIUS) -> Temperature<f64> {
        Quantity(self + 273.15, PhantomData)
    }
}

impl<S: Data<Elem = f64>, D: Dimension> Mul<CELSIUS> for ArrayBase<S, D> {
    type Output = Temperature<Array<f64, D>>;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, _: CELSIUS) -> Temperature<Array<f64, D>> {
        Quantity(&self + 273.15, PhantomData)
    }
}

impl Div<CELSIUS> for Temperature<f64> {
    type Output = f64;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, _: CELSIUS) -> Self::Output {
        self.0 - 273.15
    }
}

impl<D: Dimension> Div<CELSIUS> for Temperature<Array<f64, D>> {
    type Output = Array<f64, D>;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, _: CELSIUS) -> Self::Output {
        self.0 - 273.15
    }
}

impl<T> Dimensionless<T> {
    /// Return the value of a dimensionless quantity.
    pub fn into_value(self) -> T {
        self.0
    }
}

impl<T, U> Quantity<T, U> {
    /// Convert a quantity into the given unit and return it
    /// as a float or array.
    pub fn convert_into<T2>(self, unit: Quantity<T2, U>) -> Quot<T, T2>
    where
        T: Div<T2>,
        U: Sub<U, Output = _Dimensionless>,
    {
        (self / unit).into_value()
    }
}

impl<T> From<T> for Dimensionless<T> {
    fn from(value: T) -> Self {
        Quantity(value, PhantomData)
    }
}

impl<U> Zero for Quantity<f64, U> {
    fn zero() -> Self {
        Quantity(0.0, PhantomData)
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

/// Reference values used for reduced properties in feos
const REFERENCE_VALUES: [f64; 7] = [
    1e-12,               // 1 ps
    1e-10,               // 1 Å
    1.380649e-27,        // Fixed through k_B
    1.0,                 // 1 A
    1.0,                 // 1 K
    1.0 / 6.02214076e23, // 1/N_AV
    1.0,                 // 1 Cd
];

/// Conversion to and from reduced units
impl<
        Inner,
        T: Integer,
        L: Integer,
        M: Integer,
        I: Integer,
        THETA: Integer,
        N: Integer,
        J: Integer,
    > Quantity<Inner, SIUnit<T, L, M, I, THETA, N, J>>
{
    pub fn from_reduced(value: Inner) -> Self
    where
        Inner: Mul<f64, Output = Inner>,
    {
        Self(
            value
                * (REFERENCE_VALUES[0].powi(T::I32)
                    * REFERENCE_VALUES[1].powi(L::I32)
                    * REFERENCE_VALUES[2].powi(M::I32)
                    * REFERENCE_VALUES[3].powi(I::I32)
                    * REFERENCE_VALUES[4].powi(THETA::I32)
                    * REFERENCE_VALUES[5].powi(N::I32)
                    * REFERENCE_VALUES[6].powi(J::I32)),
            PhantomData,
        )
    }

    pub fn to_reduced<'a>(&'a self) -> Inner
    where
        &'a Inner: Div<f64, Output = Inner>,
    {
        &self.0
            / (REFERENCE_VALUES[0].powi(T::I32)
                * REFERENCE_VALUES[1].powi(L::I32)
                * REFERENCE_VALUES[2].powi(M::I32)
                * REFERENCE_VALUES[3].powi(I::I32)
                * REFERENCE_VALUES[4].powi(THETA::I32)
                * REFERENCE_VALUES[5].powi(N::I32)
                * REFERENCE_VALUES[6].powi(J::I32))
    }
}
