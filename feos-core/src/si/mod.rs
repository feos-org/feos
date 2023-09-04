use ang::{Angle, Degrees, Radians};
use num_traits::Zero;
use std::marker::PhantomData;
use std::ops::{Div, Mul};
use typenum::{ATerm, Diff, Integer, Negate, Sum, TArr, P1, Z0};

mod array;
mod fmt;
mod ops;
#[cfg(feature = "python")]
mod python;

pub type SIUnit<T, L, M, I, THETA, N, J> =
    TArr<T, TArr<L, TArr<M, TArr<I, TArr<THETA, TArr<N, TArr<J, ATerm>>>>>>>;

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

pub type _Viscosity = Sum<_Pressure, _Time>;
pub type Viscosity<T = f64> = Quantity<T, _Viscosity>;
pub type _Diffusivity = Sum<_Velocity, _Length>;
pub type Diffusivity<T = f64> = Quantity<T, _Diffusivity>;
pub type _ThermalConductivity = Diff<_Power, Sum<_Length, _Temperature>>;
pub type ThermalConductivity<T = f64> = Quantity<T, _ThermalConductivity>;
pub type _SurfaceTension = Diff<_Force, _Length>;
pub type SurfaceTension<T = f64> = Quantity<T, _SurfaceTension>;

pub const SECOND: Time = Quantity(1.0, PhantomData);
pub const METER: Length = Quantity(1.0, PhantomData);
pub const KILOGRAM: Mass = Quantity(1.0, PhantomData);
pub const AMPERE: Current = Quantity(1.0, PhantomData);
pub const KELVIN: Temperature = Quantity(1.0, PhantomData);
pub const MOL: Moles = Quantity(1.0, PhantomData);
pub const CANDELA: LuminousIntensity = Quantity(1.0, PhantomData);

pub const HERTZ: Frequency = Quantity(1.0, PhantomData);
pub const NEWTON: Force = Quantity(1.0, PhantomData);
pub const PASCAL: Pressure = Quantity(1.0, PhantomData);
pub const JOULE: Energy = Quantity(1.0, PhantomData);
pub const WATT: Power = Quantity(1.0, PhantomData);
pub const COULOMB: Charge = Quantity(1.0, PhantomData);
pub const VOLT: ElectricPotential = Quantity(1.0, PhantomData);
pub const FARAD: Capacitance = Quantity(1.0, PhantomData);
pub const OHM: Resistance = Quantity(1.0, PhantomData);
pub const SIEMENS: ElectricalConductance = Quantity(1.0, PhantomData);
pub const WEBER: MagneticFlux = Quantity(1.0, PhantomData);
pub const TESLA: MagneticFluxDensity = Quantity(1.0, PhantomData);
pub const HENRY: Inductance = Quantity(1.0, PhantomData);

pub const ANGSTROM: Length = Quantity(1e-10, PhantomData);
pub const AMU: Mass = Quantity(1.6605390671738466e-27, PhantomData);
pub const AU: Length = Quantity(149597870700.0, PhantomData);
pub const BAR: Pressure = Quantity(1e5, PhantomData);
pub const CALORIE: Energy = Quantity(4.184, PhantomData);
pub const DAY: Time = Quantity(86400.0, PhantomData);
pub const GRAM: Mass = Quantity(1e-3, PhantomData);
pub const HOUR: Time = Quantity(3600.0, PhantomData);
pub const LITER: Volume = Quantity(1e-3, PhantomData);
pub const MINUTE: Time = Quantity(60.0, PhantomData);

/// Angle unit radian $\\left(\text{rad}\\right)$
pub const RADIANS: Angle = Radians(1.0);
/// Angle unit degree $\\left(1^\\circ=\frac{\pi}{180}\\,\text{rad}\\approx 0.0174532925\\,\text{rad}\\right)$
pub const DEGREES: Angle = Degrees(1.0);

/// Boltzmann constant $\\left(k_\text{B}=1.380649\times 10^{-23}\\,\\frac{\text{J}}{\text{K}}\\right)$
pub const KB: Entropy = Quantity(1.380649e-23, PhantomData);
/// Avogadro constant $\\left(N_\text{A}=6.02214076\times 10^{23}\\,\text{mol}^{-1}\\right)$
pub const NAV: Quantity<f64, Negate<_Moles>> = Quantity(6.02214076e23, PhantomData);
/// Ideal gas constant $\\left(R=8.31446261815324\\,\\frac{\text{J}}{\text{molK}}\\right)$
pub const RGAS: MolarEntropy = Quantity(8.31446261815324, PhantomData);

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

/// Basic conversions and constructors
impl<T> Dimensionless<T> {
    pub fn into_value(self) -> T {
        self.0
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
