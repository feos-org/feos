use approx::{AbsDiffEq, RelativeEq};
use ndarray::{
    Array, Array1, ArrayBase, ArrayView, Axis, Data, DataMut, Dimension, NdIndex, RemoveAxis,
    ShapeBuilder,
};
use num_traits::Zero;
use std::convert::TryInto;
use std::fmt;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use typenum::{ATerm, Diff, Integer, Negate, Sum, TArr, P1, P2, P3, Z0};

pub type SIUnit<T, L, M, I, THETA, N, J> =
    TArr<T, TArr<L, TArr<M, TArr<I, TArr<THETA, TArr<N, TArr<J, ATerm>>>>>>>;

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Quantity<T, U>(T, PhantomData<U>);

type SINumber<T, L, M, I, THETA, N, J> = Quantity<f64, SIUnit<T, L, M, I, THETA, N, J>>;

impl<T> Quantity<T, _Dimensionless> {
    pub fn into_value(self) -> T {
        self.0
    }
}

impl<T1, T2, U1, U2> Mul<Quantity<T2, U2>> for Quantity<T1, U1>
where
    T1: Mul<T2>,
    U1: Add<U2>,
{
    type Output = Quantity<<T1 as Mul<T2>>::Output, <U1 as Add<U2>>::Output>;
    fn mul(self, other: Quantity<T2, U2>) -> Self::Output {
        Quantity(self.0 * other.0, PhantomData)
    }
}

impl<'a, T1, T2, U1, U2> Mul<Quantity<T2, U2>> for &'a Quantity<T1, U1>
where
    &'a T1: Mul<T2>,
    U1: Add<U2>,
{
    type Output = Quantity<<&'a T1 as Mul<T2>>::Output, <U1 as Add<U2>>::Output>;
    fn mul(self, other: Quantity<T2, U2>) -> Self::Output {
        Quantity(&self.0 * other.0, PhantomData)
    }
}

impl<'a, 'b, T1, T2, U1, U2> Mul<&'b Quantity<T2, U2>> for &'a Quantity<T1, U1>
where
    &'a T1: Mul<&'b T2>,
    U1: Add<U2>,
{
    type Output = Quantity<<&'a T1 as Mul<&'b T2>>::Output, <U1 as Add<U2>>::Output>;
    fn mul(self, other: &'b Quantity<T2, U2>) -> Self::Output {
        Quantity(&self.0 * &other.0, PhantomData)
    }
}

impl<U> Mul<Quantity<f64, U>> for f64 {
    type Output = Quantity<f64, U>;
    fn mul(self, other: Quantity<f64, U>) -> Self::Output {
        Quantity(self * other.0, PhantomData)
    }
}

impl<U> Mul<Quantity<f64, U>> for Array1<f64> {
    type Output = Quantity<Array1<f64>, U>;
    fn mul(self, other: Quantity<f64, U>) -> Self::Output {
        Quantity(self * other.0, PhantomData)
    }
}

impl<U> Mul<Quantity<f64, U>> for &Array1<f64> {
    type Output = Quantity<Array1<f64>, U>;
    fn mul(self, other: Quantity<f64, U>) -> Self::Output {
        Quantity(self * other.0, PhantomData)
    }
}

impl<T: Mul<f64>, U> Mul<f64> for Quantity<T, U> {
    type Output = Quantity<<T as Mul<f64>>::Output, U>;
    fn mul(self, other: f64) -> Self::Output {
        Quantity(self.0 * other, PhantomData)
    }
}

impl<'a, T, U> Mul<f64> for &'a Quantity<T, U>
where
    &'a T: Mul<f64>,
{
    type Output = Quantity<<&'a T as Mul<f64>>::Output, U>;
    fn mul(self, other: f64) -> Self::Output {
        Quantity(&self.0 * other, PhantomData)
    }
}

impl<U> Mul<&Array1<f64>> for Quantity<Array1<f64>, U> {
    type Output = Quantity<Array1<f64>, U>;
    fn mul(self, other: &Array1<f64>) -> Self::Output {
        Quantity(self.0 * other, PhantomData)
    }
}

impl<T1, T2, U1, U2> Div<Quantity<T2, U2>> for Quantity<T1, U1>
where
    T1: Div<T2>,
    U1: Sub<U2>,
{
    type Output = Quantity<<T1 as Div<T2>>::Output, <U1 as Sub<U2>>::Output>;
    fn div(self, other: Quantity<T2, U2>) -> Self::Output {
        Quantity(self.0 / other.0, PhantomData)
    }
}

impl<'a, T1, T2, U1, U2> Div<Quantity<T2, U2>> for &'a Quantity<T1, U1>
where
    &'a T1: Div<T2>,
    U1: Sub<U2>,
{
    type Output = Quantity<<&'a T1 as Div<T2>>::Output, <U1 as Sub<U2>>::Output>;
    fn div(self, other: Quantity<T2, U2>) -> Self::Output {
        Quantity(&self.0 / other.0, PhantomData)
    }
}

impl<'a, 'b, T1, T2, U1, U2> Div<&'b Quantity<T2, U2>> for &'a Quantity<T1, U1>
where
    &'a T1: Div<&'b T2>,
    U1: Sub<U2>,
{
    type Output = Quantity<<&'a T1 as Div<&'b T2>>::Output, <U1 as Sub<U2>>::Output>;
    fn div(self, other: &'b Quantity<T2, U2>) -> Self::Output {
        Quantity(&self.0 / &other.0, PhantomData)
    }
}

impl<T, U> Div<Quantity<T, U>> for f64
where
    U: Neg,
    f64: Div<T>,
{
    type Output = Quantity<<f64 as Div<T>>::Output, <U as Neg>::Output>;
    fn div(self, other: Quantity<T, U>) -> Self::Output {
        Quantity(self / other.0, PhantomData)
    }
}

impl<T: Div<f64>, U> Div<f64> for Quantity<T, U> {
    type Output = Quantity<<T as Div<f64>>::Output, U>;
    fn div(self, other: f64) -> Self::Output {
        Quantity(self.0 / other, PhantomData)
    }
}

impl<T1, T2, U> Add<Quantity<T2, U>> for Quantity<T1, U>
where
    T1: Add<T2>,
{
    type Output = Quantity<<T1 as std::ops::Add<T2>>::Output, U>;
    fn add(self, other: Quantity<T2, U>) -> Self::Output {
        Quantity(self.0 + other.0, PhantomData)
    }
}

impl<'a, 'b, T1, T2, U> Add<&'b Quantity<T2, U>> for &'a Quantity<T1, U>
where
    &'a T1: Add<&'b T2>,
{
    type Output = Quantity<<&'a T1 as std::ops::Add<&'b T2>>::Output, U>;
    fn add(self, other: &'b Quantity<T2, U>) -> Self::Output {
        Quantity(&self.0 + &other.0, PhantomData)
    }
}

impl<T1, T2, U> AddAssign<Quantity<T2, U>> for Quantity<T1, U>
where
    T1: AddAssign<T2>,
{
    fn add_assign(&mut self, rhs: Quantity<T2, U>) {
        self.0 += rhs.0;
    }
}

impl<'a, T1, T2, U> AddAssign<&'a Quantity<T2, U>> for Quantity<T1, U>
where
    T1: AddAssign<&'a T2>,
{
    fn add_assign(&mut self, rhs: &'a Quantity<T2, U>) {
        self.0 += &rhs.0;
    }
}

impl<T1, T2, U> Sub<Quantity<T2, U>> for Quantity<T1, U>
where
    T1: Sub<T2>,
{
    type Output = Quantity<<T1 as std::ops::Sub<T2>>::Output, U>;
    fn sub(self, other: Quantity<T2, U>) -> Self::Output {
        Quantity(self.0 - other.0, PhantomData)
    }
}

impl<'a, T1, T2, U> Sub<Quantity<T2, U>> for &'a Quantity<T1, U>
where
    &'a T1: Sub<T2>,
{
    type Output = Quantity<<&'a T1 as std::ops::Sub<T2>>::Output, U>;
    fn sub(self, other: Quantity<T2, U>) -> Self::Output {
        Quantity(&self.0 - other.0, PhantomData)
    }
}

impl<'a, 'b, T1, T2, U> Sub<&'b Quantity<T2, U>> for &'a Quantity<T1, U>
where
    &'a T1: Sub<&'b T2>,
{
    type Output = Quantity<<&'a T1 as std::ops::Sub<&'b T2>>::Output, U>;
    fn sub(self, other: &'b Quantity<T2, U>) -> Self::Output {
        Quantity(&self.0 - &other.0, PhantomData)
    }
}

impl<T1, T2, U> SubAssign<Quantity<T2, U>> for Quantity<T1, U>
where
    T1: SubAssign<T2>,
{
    fn sub_assign(&mut self, rhs: Quantity<T2, U>) {
        self.0 -= rhs.0;
    }
}

impl<T, U> Neg for Quantity<T, U>
where
    T: Neg,
{
    type Output = Quantity<<T as Neg>::Output, U>;
    fn neg(self) -> Self::Output {
        Quantity(-self.0, PhantomData)
    }
}

impl From<Dimensionless<f64>> for f64 {
    fn from(si: Dimensionless<f64>) -> f64 {
        si.0
    }
}

impl<U> Quantity<f64, U> {
    pub fn powi<E: Integer>(self) -> Quantity<f64, <U as Mul<E>>::Output>
    where
        U: Mul<E>,
    {
        Quantity(self.0.powi(E::to_i32()), PhantomData)
    }

    pub fn sqrt(self) -> Quantity<f64, <U as Div<P2>>::Output>
    where
        U: Div<P2>,
    {
        Quantity(self.0.sqrt(), PhantomData)
    }

    pub fn cbrt(self) -> Quantity<f64, <U as Div<P3>>::Output>
    where
        U: Div<P3>,
    {
        Quantity(self.0.cbrt(), PhantomData)
    }

    pub fn abs(self) -> Self {
        Self(self.0.abs(), PhantomData)
    }

    pub fn signum(self) -> f64 {
        self.0.signum()
    }

    pub fn is_sign_negative(&self) -> bool {
        self.0.is_sign_negative()
    }

    pub fn is_sign_positive(&self) -> bool {
        self.0.is_sign_positive()
    }

    pub fn is_nan(&self) -> bool {
        self.0.is_nan()
    }

    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0), PhantomData)
    }

    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0), PhantomData)
    }
}

impl<T: PartialEq, U> PartialEq for Quantity<T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: PartialOrd, U> PartialOrd for Quantity<T, U> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T: AbsDiffEq, U: Eq> AbsDiffEq for Quantity<T, U> {
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.0.abs_diff_eq(&other.0, epsilon)
    }
}

impl<T: RelativeEq, U: Eq> RelativeEq for Quantity<T, U> {
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.0.relative_eq(&other.0, epsilon, max_relative)
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

impl<U> Quantity<Array1<f64>, U> {
    pub fn from_vec(v: Vec<Quantity<f64, U>>) -> Self {
        Self(v.iter().map(|e| e.0).collect(), PhantomData)
    }

    pub fn linspace(start: Quantity<f64, U>, end: Quantity<f64, U>, n: usize) -> Self {
        Self(Array1::linspace(start.0, end.0, n), PhantomData)
    }
}

impl<U, D: Dimension> Quantity<Array<f64, D>, U> {
    pub fn zeros<Sh: ShapeBuilder<Dim = D>>(shape: Sh) -> Self {
        Quantity(Array::zeros(shape), PhantomData)
    }

    pub fn from_shape_fn<Sh, F>(shape: Sh, mut f: F) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
        F: FnMut(D::Pattern) -> Quantity<f64, U>,
    {
        Quantity(Array::from_shape_fn(shape, |x| f(x).0), PhantomData)
    }
}

impl<S: Data<Elem = f64>, U, D: Dimension> Quantity<ArrayBase<S, D>, U> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn sum(&self) -> Quantity<f64, U> {
        Quantity(self.0.sum(), PhantomData)
    }

    pub fn to_owned(&self) -> Quantity<Array<f64, D>, U> {
        Quantity(self.0.to_owned(), PhantomData)
    }

    pub fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    pub fn raw_dim(&self) -> D {
        self.0.raw_dim()
    }

    pub fn mapv<F, U2>(&self, mut f: F) -> Quantity<Array<f64, D>, U2>
    where
        S: DataMut,
        F: FnMut(Quantity<f64, U>) -> Quantity<f64, U2>,
    {
        Quantity(self.0.mapv(|x| f(Quantity(x, PhantomData)).0), PhantomData)
    }

    pub fn index_axis(
        &self,
        axis: Axis,
        index: usize,
    ) -> Quantity<ArrayView<'_, f64, D::Smaller>, U>
    where
        D: RemoveAxis,
    {
        Quantity(self.0.index_axis(axis, index), PhantomData)
    }

    pub fn sum_axis(&self, axis: Axis) -> Quantity<Array<f64, D::Smaller>, U>
    where
        D: RemoveAxis,
    {
        Quantity(self.0.sum_axis(axis), PhantomData)
    }

    pub fn insert_axis(self, axis: Axis) -> Quantity<ArrayBase<S, D::Larger>, U> {
        Quantity(self.0.insert_axis(axis), PhantomData)
    }

    pub fn get<I: NdIndex<D>>(&self, index: I) -> Quantity<f64, U> {
        Quantity(self.0[index], PhantomData)
    }
}

impl<U, D: Dimension> Quantity<Array<f64, D>, U> {
    pub fn set<I: NdIndex<D>>(&mut self, index: I, value: Quantity<f64, U>) {
        self.0[index] = value.0;
    }
}

impl<S: Data<Elem = f64>, U, D: Dimension> Quantity<&ArrayBase<S, D>, U> {
    pub fn get<I: NdIndex<D>>(&self, index: I) -> Quantity<f64, U> {
        Quantity(self.0[index], PhantomData)
    }
}

pub struct QuantityIter<I, U> {
    inner: I,
    unit: PhantomData<U>,
}

impl<'a, I: Iterator<Item = &'a f64>, U: Copy> Iterator for QuantityIter<I, U> {
    type Item = Quantity<f64, U>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|value| Quantity(*value, PhantomData))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, I: Iterator<Item = &'a f64> + ExactSizeIterator, U: Copy> ExactSizeIterator
    for QuantityIter<I, U>
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, I: Iterator<Item = &'a f64> + DoubleEndedIterator, U: Copy> DoubleEndedIterator
    for QuantityIter<I, U>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner
            .next_back()
            .map(|value| Quantity(*value, PhantomData))
    }
}

impl<'a, F, U: Copy> IntoIterator for &'a Quantity<F, U>
where
    &'a F: IntoIterator<Item = &'a f64>,
{
    type Item = Quantity<f64, U>;
    type IntoIter = QuantityIter<<&'a F as IntoIterator>::IntoIter, U>;

    fn into_iter(self) -> Self::IntoIter {
        QuantityIter {
            inner: (&self.0).into_iter(),
            unit: PhantomData,
        }
    }
}

impl<U> FromIterator<Quantity<f64, U>> for Quantity<Array1<f64>, U> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Quantity<f64, U>>,
    {
        Self(iter.into_iter().map(|v| v.0).collect(), PhantomData)
    }
}

impl<S: Data<Elem = f64>, D: Dimension, U> Quantity<ArrayBase<S, D>, U> {
    #[allow(clippy::type_complexity)]
    pub fn integrate<U2: Add<U>>(&self, weights: &[&Array1<f64>]) -> Quantity<f64, Sum<U2, U>> {
        let mut value = self.0.to_owned();
        for (i, &w) in weights.iter().enumerate() {
            for mut l in value.lanes_mut(Axis(i)) {
                l.assign(&(&l * w));
            }
        }
        Quantity(value.sum(), PhantomData)
    }
}

pub type _Dimensionless = SIUnit<Z0, Z0, Z0, Z0, Z0, Z0, Z0>;
pub type _Time = SIUnit<P1, Z0, Z0, Z0, Z0, Z0, Z0>;
pub type _Length = SIUnit<Z0, P1, Z0, Z0, Z0, Z0, Z0>;
pub type _Mass = SIUnit<Z0, Z0, P1, Z0, Z0, Z0, Z0>;
pub type _Current = SIUnit<Z0, Z0, Z0, P1, Z0, Z0, Z0>;
pub type _Temperature = SIUnit<Z0, Z0, Z0, Z0, P1, Z0, Z0>;
pub type _Moles = SIUnit<Z0, Z0, Z0, Z0, Z0, P1, Z0>;
pub type _LuminousIntensity = SIUnit<Z0, Z0, Z0, Z0, Z0, Z0, P1>;

pub type Dimensionless<T> = Quantity<T, _Dimensionless>;
pub type Time<T> = Quantity<T, _Time>;
pub type Length<T> = Quantity<T, _Length>;
pub type Mass<T> = Quantity<T, _Mass>;
pub type Current<T> = Quantity<T, _Current>;
pub type Temperature<T> = Quantity<T, _Temperature>;
pub type Moles<T> = Quantity<T, _Moles>;
pub type LuminousIntensity<T> = Quantity<T, _LuminousIntensity>;

pub type _Frequency = Negate<_Time>;
pub type Frequency<T> = Quantity<T, _Frequency>;
pub type _Velocity = Diff<_Length, _Time>;
pub type Velocity<T> = Quantity<T, _Velocity>;
pub type _Acceleration = Diff<_Velocity, _Time>;
pub type Acceleration<T> = Quantity<T, _Acceleration>;
pub type _Force = Sum<_Mass, _Acceleration>;
pub type Force<T> = Quantity<T, _Force>;
pub type _Area = Sum<_Length, _Length>;
pub type Area<T> = Quantity<T, _Area>;
pub type _Volume = Sum<_Area, _Length>;
pub type Volume<T> = Quantity<T, _Volume>;
pub type _Energy = Sum<_Force, _Length>;
pub type Energy<T> = Quantity<T, _Energy>;
pub type _Pressure = Diff<_Energy, _Volume>;
pub type Pressure<T> = Quantity<T, _Pressure>;
pub type _Power = Diff<_Energy, _Time>;
pub type Power<T> = Quantity<T, _Power>;
pub type _Charge = Sum<_Current, _Time>;
pub type Charge<T> = Quantity<T, _Charge>;
pub type _ElectricPotential = Diff<_Power, _Current>;
pub type ElectricPotential<T> = Quantity<T, _ElectricPotential>;
pub type _Capacitance = Diff<_Charge, _ElectricPotential>;
pub type Capacitance<T> = Quantity<T, _Capacitance>;
pub type _Resistance = Diff<_ElectricPotential, _Current>;
pub type Resistance<T> = Quantity<T, _Resistance>;
pub type _ElectricalConductance = Negate<_Resistance>;
pub type ElectricalConductance<T> = Quantity<T, _ElectricalConductance>;
pub type _MagneticFlux = Sum<_ElectricPotential, _Time>;
pub type MagneticFlux<T> = Quantity<T, _MagneticFlux>;
pub type _MagneticFluxDensity = Diff<_MagneticFlux, _Area>;
pub type MagneticFluxDensity<T> = Quantity<T, _MagneticFluxDensity>;
pub type _Inductance = Diff<_MagneticFlux, _Current>;
pub type Inductance<T> = Quantity<T, _Inductance>;

pub type _Entropy = Diff<_Energy, _Temperature>;
pub type Entropy<T> = Quantity<T, _Entropy>;
pub type _EntropyPerTemperature = Diff<_Entropy, _Temperature>;
pub type EntropyPerTemperature<T> = Quantity<T, _EntropyPerTemperature>;
pub type _MolarEntropy = Diff<_Entropy, _Moles>;
pub type MolarEntropy<T> = Quantity<T, _MolarEntropy>;
pub type _MolarEnergy = Diff<_Energy, _Moles>;
pub type MolarEnergy<T> = Quantity<T, _MolarEnergy>;
pub type _SpecificEntropy = Diff<_Entropy, _Mass>;
pub type SpecificEntropy<T> = Quantity<T, _SpecificEntropy>;
pub type _SpecificEnergy = Diff<_Energy, _Mass>;
pub type SpecificEnergy<T> = Quantity<T, _SpecificEnergy>;
pub type _MolarWeight = Diff<_Mass, _Moles>;
pub type MolarWeight<T> = Quantity<T, _MolarWeight>;
pub type _Density = Diff<_Moles, _Volume>;
pub type Density<T> = Quantity<T, _Density>;
pub type _MassDensity = Diff<_Mass, _Volume>;
pub type MassDensity<T> = Quantity<T, _MassDensity>;
pub type _PressurePerVolume = Diff<_Pressure, _Volume>;
pub type PressurePerVolume<T> = Quantity<T, _PressurePerVolume>;
pub type _PressurePerTemperature = Diff<_Pressure, _Temperature>;
pub type PressurePerTemperature<T> = Quantity<T, _PressurePerTemperature>;
pub type _Compressibility = Negate<_Pressure>;
pub type Compressibility<T> = Quantity<T, _Compressibility>;
pub type _MolarVolume = Diff<_Volume, _Moles>;
pub type MolarVolume<T> = Quantity<T, _MolarVolume>;
pub type _EntropyDensity = Diff<_Entropy, _Volume>;
pub type EntropyDensity<T> = Quantity<T, _EntropyDensity>;

pub type _Viscosity = Sum<_Pressure, _Time>;
pub type Viscosity<T> = Quantity<T, _Viscosity>;
pub type _Diffusivity = Sum<_Velocity, _Length>;
pub type Diffusivity<T> = Quantity<T, _Diffusivity>;
pub type _ThermalConductivity = Diff<_Power, Sum<_Length, _Temperature>>;
pub type ThermalConductivity<T> = Quantity<T, _ThermalConductivity>;
pub type _SurfaceTension = Diff<_Force, _Length>;
pub type SurfaceTension<T> = Quantity<T, _SurfaceTension>;

pub const SECOND: Time<f64> = Quantity(1.0, PhantomData);
pub const METER: Length<f64> = Quantity(1.0, PhantomData);
pub const KILOGRAM: Mass<f64> = Quantity(1.0, PhantomData);
pub const AMPERE: Current<f64> = Quantity(1.0, PhantomData);
pub const KELVIN: Temperature<f64> = Quantity(1.0, PhantomData);
pub const MOL: Moles<f64> = Quantity(1.0, PhantomData);
pub const CANDELA: LuminousIntensity<f64> = Quantity(1.0, PhantomData);

pub const HERTZ: Frequency<f64> = Quantity(1.0, PhantomData);
pub const NEWTON: Force<f64> = Quantity(1.0, PhantomData);
pub const PASCAL: Pressure<f64> = Quantity(1.0, PhantomData);
pub const JOULE: Energy<f64> = Quantity(1.0, PhantomData);
pub const WATT: Power<f64> = Quantity(1.0, PhantomData);
pub const COULOMB: Charge<f64> = Quantity(1.0, PhantomData);
pub const VOLT: ElectricPotential<f64> = Quantity(1.0, PhantomData);
pub const FARAD: Capacitance<f64> = Quantity(1.0, PhantomData);
pub const OHM: Resistance<f64> = Quantity(1.0, PhantomData);
pub const SIEMENS: ElectricalConductance<f64> = Quantity(1.0, PhantomData);
pub const WEBER: MagneticFlux<f64> = Quantity(1.0, PhantomData);
pub const TESLA: MagneticFluxDensity<f64> = Quantity(1.0, PhantomData);
pub const HENRY: Inductance<f64> = Quantity(1.0, PhantomData);

pub const ANGSTROM: Length<f64> = Quantity(1e-10, PhantomData);
pub const AMU: Mass<f64> = Quantity(1.6605390671738466e-27, PhantomData);
pub const AU: Length<f64> = Quantity(149597870700.0, PhantomData);
pub const BAR: Pressure<f64> = Quantity(1e5, PhantomData);
pub const CALORIE: Energy<f64> = Quantity(4.184, PhantomData);
pub const DAY: Time<f64> = Quantity(86400.0, PhantomData);
pub const GRAM: Mass<f64> = Quantity(1e-3, PhantomData);
pub const HOUR: Time<f64> = Quantity(3600.0, PhantomData);
pub const LITER: Volume<f64> = Quantity(1e-3, PhantomData);
pub const MINUTE: Time<f64> = Quantity(60.0, PhantomData);

/// Boltzmann constant $\\left(k_\text{B}=1.380649\times 10^{-23}\\,\\frac{\text{J}}{\text{K}}\\right)$
pub const KB: Entropy<f64> = Quantity(1.380649e-23, PhantomData);
/// Avogadro constant $\\left(N_\text{A}=6.02214076\times 10^{23}\\,\text{mol}^{-1}\\right)$
pub const NAV: Quantity<f64, Negate<_Moles>> = Quantity(6.02214076e23, PhantomData);
pub const RGAS: MolarEntropy<f64> = Quantity(8.31446261815324, PhantomData);

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

// impl<T: Mul<f64>> Mul<Temperature<f64>> for quantity::Quantity<T, quantity::si::SIUnit> {
//     type Output =
//         <quantity::Quantity<T, quantity::si::SIUnit> as Mul<quantity::si::SINumber>>::Output;

//     fn mul(self, rhs: Temperature<f64>) -> Self::Output {
//         self * (rhs.0 * quantity::si::KELVIN)
//     }
// }

// impl<T> Mul<quantity::Quantity<T, quantity::si::SIUnit>> for Temperature<f64>
// where
//     f64: Mul<T>
//         + Mul<
//             quantity::Quantity<f64, quantity::si::SIUnit>,
//             Output = quantity::Quantity<f64, quantity::si::SIUnit>,
//         >,
// {
//     type Output =
//         <quantity::si::SINumber as Mul<quantity::Quantity<T, quantity::si::SIUnit>>>::Output;

//     fn mul(self, rhs: quantity::Quantity<T, quantity::si::SIUnit>) -> Self::Output {
//         (self.0 * quantity::si::KELVIN) * rhs
//     }
// }

// impl<T: Div<f64>> Div<Temperature<f64>> for quantity::Quantity<T, quantity::si::SIUnit> {
//     type Output =
//         <quantity::Quantity<T, quantity::si::SIUnit> as Div<quantity::si::SINumber>>::Output;

//     fn div(self, rhs: Temperature<f64>) -> Self::Output {
//         self / (rhs.0 * quantity::si::KELVIN)
//     }
// }

// impl<T> Div<quantity::Quantity<T, quantity::si::SIUnit>> for Temperature<f64>
// where
//     f64: Div<T>,
// {
//     type Output =
//         <quantity::si::SINumber as Div<quantity::Quantity<T, quantity::si::SIUnit>>>::Output;

//     fn div(self, rhs: quantity::Quantity<T, quantity::si::SIUnit>) -> Self::Output {
//         self.0 * quantity::si::KELVIN / rhs
//     }
// }

impl<T: fmt::Display, U> fmt::Display for Quantity<T, U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

impl<T: fmt::LowerExp, U> fmt::LowerExp for Quantity<T, U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

// impl<T: Add<f64>> Add<Temperature<f64>> for quantity::Quantity<T, quantity::si::SIUnit> {
//     type Output =
//         <quantity::Quantity<T, quantity::si::SIUnit> as Add<quantity::si::SINumber>>::Output;

//     fn add(self, rhs: Temperature<f64>) -> Self::Output {
//         self + (rhs.0 * quantity::si::KELVIN)
//     }
// }

// impl Add<quantity::si::SINumber> for Temperature<f64> {
//     type Output = Self;

//     fn add(self, rhs: quantity::si::SINumber) -> Self::Output {
//         self + rhs.to_reduced(quantity::si::KELVIN).unwrap() * KELVIN
//     }
// }

#[derive(Copy, Clone, Debug)]
struct SIPython {
    value: f64,
    unit: [i8; 7],
}

impl<T: Integer, L: Integer, M: Integer, I: Integer, THETA: Integer, N: Integer, J: Integer>
    TryInto<SINumber<T, L, M, I, THETA, N, J>> for SIPython
{
    type Error = String;
    fn try_into(self) -> Result<SINumber<T, L, M, I, THETA, N, J>, String> {
        if self.unit[0] == T::to_i8()
            && self.unit[1] == L::to_i8()
            && self.unit[2] == M::to_i8()
            && self.unit[3] == I::to_i8()
            && self.unit[4] == THETA::to_i8()
            && self.unit[5] == N::to_i8()
            && self.unit[6] == J::to_i8()
        {
            Ok(Quantity(self.value, PhantomData))
        } else {
            Err("Wrong unit mate!".into())
        }
    }
}

impl<T: Integer, L: Integer, M: Integer, I: Integer, THETA: Integer, N: Integer, J: Integer>
    From<SINumber<T, L, M, I, THETA, N, J>> for SIPython
{
    fn from(si: SINumber<T, L, M, I, THETA, N, J>) -> Self {
        Self {
            unit: [
                T::to_i8(),
                L::to_i8(),
                M::to_i8(),
                I::to_i8(),
                THETA::to_i8(),
                N::to_i8(),
                J::to_i8(),
            ],
            value: si.0,
        }
    }
}

const REFERENCE_VALUES: [f64; 7] = [
    1e-12,               // 1 ps
    1e-10,               // 1 A
    1.380649e-27,        // Fixed through k_B
    1.0,                 // 1 A
    1.0,                 // 1 K
    1.0 / 6.02214076e23, // 1/N_AV
    1.0,                 // 1 Cd
];

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
                * (REFERENCE_VALUES[0].powi(T::to_i32())
                    * REFERENCE_VALUES[1].powi(L::to_i32())
                    * REFERENCE_VALUES[2].powi(M::to_i32())
                    * REFERENCE_VALUES[3].powi(I::to_i32())
                    * REFERENCE_VALUES[4].powi(THETA::to_i32())
                    * REFERENCE_VALUES[5].powi(N::to_i32())
                    * REFERENCE_VALUES[6].powi(J::to_i32())),
            PhantomData,
        )
    }

    pub fn to_reduced<'a>(&'a self) -> Inner
    where
        &'a Inner: Div<f64, Output = Inner>,
    {
        &self.0
            / (REFERENCE_VALUES[0].powi(T::to_i32())
                * REFERENCE_VALUES[1].powi(L::to_i32())
                * REFERENCE_VALUES[2].powi(M::to_i32())
                * REFERENCE_VALUES[3].powi(I::to_i32())
                * REFERENCE_VALUES[4].powi(THETA::to_i32())
                * REFERENCE_VALUES[5].powi(N::to_i32())
                * REFERENCE_VALUES[6].powi(J::to_i32()))
    }
}

// impl<
//         Inner,
//         T: Integer,
//         L: Integer,
//         M: Integer,
//         I: Integer,
//         THETA: Integer,
//         N: Integer,
//         J: Integer,
//     > Quantity<&Inner, SIUnit<T, L, M, I, THETA, N, J>>
// {
//     pub fn to_reduced(self) -> Inner
//     where
//         for<'a> &'a Inner: Div<f64, Output = Inner>,
//     {
//         self.0
//             / (REFERENCE_VALUES[0].powi(T::to_i32())
//                 * REFERENCE_VALUES[1].powi(L::to_i32())
//                 * REFERENCE_VALUES[2].powi(M::to_i32())
//                 * REFERENCE_VALUES[3].powi(I::to_i32())
//                 * REFERENCE_VALUES[4].powi(THETA::to_i32())
//                 * REFERENCE_VALUES[5].powi(N::to_i32())
//                 * REFERENCE_VALUES[6].powi(J::to_i32()))
//     }
// }

// #[cfg(test)]
// mod tests {
//     use ndarray::Array1;

//     use super::*;

//     #[test]
//     fn test_basics() {
//         let x1 = 3.0 * KELVIN;
//         let x2 = 5.0 * METER;
//         let y = x1 / x2;
//     }

//     //     #[test]
//     //     fn test_unit() {
//     //         println!("{}", mul_unit(1, 4));
//     //         assert!(false)
//     //     }

//     fn ideal_gas_pressure(t: Temperature<f64>, v: Volume<f64>, n: Moles<f64>) -> Pressure<f64> {
//         n * RGAS * t / v
//     }

//     fn ideal_gas_pressure_python(
//         t: SIPython,
//         v: SIPython,
//         n: SIPython,
//     ) -> Result<SIPython, String> {
//         Ok(ideal_gas_pressure(t.try_into()?, v.try_into()?, n.try_into()?).into())
//     }

//     #[test]
//     fn test_si() {
//         let n = 20.0 * MOL;
//         let t = 300.0 * KELVIN;
//         let v = METER.powi::<P3>();
//         let p = BAR;
//         let z: f64 = ((p * v) / (n * RGAS * t)).into();
//         println!("{}", z);
//         let t_py = SIPython {
//             value: 273.15,
//             unit: [1, 0, 0, 0, 0, 0, 0],
//         };
//         let rho = 0.5 * MOL / METER.powi::<P3>();
//         let v: Volume<_> = Moles::from_reduced(1.0) / rho;
//         println!("{:?}", ideal_gas_pressure_python(t_py, t_py, t_py).unwrap());
//     }

//     // #[test]
//     // fn test_ndarray() {
//     //     let t0 = 0.5 * KELVIN;
//     //     let p0 = 5.0 * PASCAL;
//     //     let x0 = t0 * p0;
//     //     let t = Array1::from_vec(vec![0.5 * KELVIN, 120. * KELVIN]);
//     //     let t = Array1::from_vec(vec![0.5, 12.]) * KELVIN;
//     //     let p = Array1::from_vec(vec![5.0 * PASCAL, 4. * BAR]);
//     //     let x = t * p;
//     // }
// }
