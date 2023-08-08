use super::Quantity;
use approx::{AbsDiffEq, RelativeEq};
use ndarray::{Array, ArrayBase, Data, DataMut, DataOwned, Dimension};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use typenum::{Diff, Integer, Negate, Prod, Quot, Sum, P2, P3};

// Multiplication
impl<T1, T2, U1, U2> Mul<Quantity<T2, U2>> for Quantity<T1, U1>
where
    T1: Mul<T2>,
    U1: Add<U2>,
{
    type Output = Quantity<Prod<T1, T2>, Sum<U1, U2>>;
    fn mul(self, other: Quantity<T2, U2>) -> Self::Output {
        Quantity(self.0 * other.0, PhantomData)
    }
}

impl<'a, T1, T2, U1, U2> Mul<Quantity<T2, U2>> for &'a Quantity<T1, U1>
where
    &'a T1: Mul<T2>,
    U1: Add<U2>,
{
    type Output = Quantity<Prod<&'a T1, T2>, Sum<U1, U2>>;
    fn mul(self, other: Quantity<T2, U2>) -> Self::Output {
        Quantity(&self.0 * other.0, PhantomData)
    }
}

impl<'b, T1, T2, U1, U2> Mul<&'b Quantity<T2, U2>> for Quantity<T1, U1>
where
    T1: Mul<&'b T2>,
    U1: Add<U2>,
{
    type Output = Quantity<Prod<T1, &'b T2>, Sum<U1, U2>>;
    fn mul(self, other: &'b Quantity<T2, U2>) -> Self::Output {
        Quantity(self.0 * &other.0, PhantomData)
    }
}

impl<'a, 'b, T1, T2, U1, U2> Mul<&'b Quantity<T2, U2>> for &'a Quantity<T1, U1>
where
    &'a T1: Mul<&'b T2>,
    U1: Add<U2>,
{
    type Output = Quantity<Prod<&'a T1, &'b T2>, Sum<U1, U2>>;
    fn mul(self, other: &'b Quantity<T2, U2>) -> Self::Output {
        Quantity(&self.0 * &other.0, PhantomData)
    }
}

impl<T, U> Mul<Quantity<T, U>> for f64
where
    f64: Mul<T>,
{
    type Output = Quantity<Prod<f64, T>, U>;
    fn mul(self, other: Quantity<T, U>) -> Self::Output {
        Quantity(self * other.0, PhantomData)
    }
}

impl<T: Mul<f64>, U> Mul<f64> for Quantity<T, U> {
    type Output = Quantity<Prod<T, f64>, U>;
    fn mul(self, other: f64) -> Self::Output {
        Quantity(self.0 * other, PhantomData)
    }
}

impl<'a, T, U> Mul<f64> for &'a Quantity<T, U>
where
    &'a T: Mul<f64>,
{
    type Output = Quantity<Prod<&'a T, f64>, U>;
    fn mul(self, other: f64) -> Self::Output {
        Quantity(&self.0 * other, PhantomData)
    }
}

impl<U, S: Data<Elem = f64>, D: Dimension> Mul<Quantity<f64, U>> for &ArrayBase<S, D> {
    type Output = Quantity<Array<f64, D>, U>;
    fn mul(self, other: Quantity<f64, U>) -> Self::Output {
        Quantity(self * other.0, PhantomData)
    }
}

impl<U, S: DataOwned<Elem = f64> + DataMut, D: Dimension> Mul<Quantity<f64, U>>
    for ArrayBase<S, D>
{
    type Output = Quantity<ArrayBase<S, D>, U>;
    fn mul(self, other: Quantity<f64, U>) -> Self::Output {
        Quantity(self * other.0, PhantomData)
    }
}

impl<U, T1, T2> MulAssign<T2> for Quantity<T1, U>
where
    T1: MulAssign<T2>,
{
    fn mul_assign(&mut self, other: T2) {
        self.0 *= other;
    }
}

// Division
impl<T1, T2, U1, U2> Div<Quantity<T2, U2>> for Quantity<T1, U1>
where
    T1: Div<T2>,
    U1: Sub<U2>,
{
    type Output = Quantity<Quot<T1, T2>, Diff<U1, U2>>;
    fn div(self, other: Quantity<T2, U2>) -> Self::Output {
        Quantity(self.0 / other.0, PhantomData)
    }
}

impl<'a, T1, T2, U1, U2> Div<Quantity<T2, U2>> for &'a Quantity<T1, U1>
where
    &'a T1: Div<T2>,
    U1: Sub<U2>,
{
    type Output = Quantity<Quot<&'a T1, T2>, Diff<U1, U2>>;
    fn div(self, other: Quantity<T2, U2>) -> Self::Output {
        Quantity(&self.0 / other.0, PhantomData)
    }
}

impl<'b, T1, T2, U1, U2> Div<&'b Quantity<T2, U2>> for Quantity<T1, U1>
where
    T1: Div<&'b T2>,
    U1: Sub<U2>,
{
    type Output = Quantity<Quot<T1, &'b T2>, Diff<U1, U2>>;
    fn div(self, other: &'b Quantity<T2, U2>) -> Self::Output {
        Quantity(self.0 / &other.0, PhantomData)
    }
}

impl<'a, 'b, T1, T2, U1, U2> Div<&'b Quantity<T2, U2>> for &'a Quantity<T1, U1>
where
    &'a T1: Div<&'b T2>,
    U1: Sub<U2>,
{
    type Output = Quantity<Quot<&'a T1, &'b T2>, Diff<U1, U2>>;
    fn div(self, other: &'b Quantity<T2, U2>) -> Self::Output {
        Quantity(&self.0 / &other.0, PhantomData)
    }
}

impl<T, U> Div<Quantity<T, U>> for f64
where
    U: Neg,
    f64: Div<T>,
{
    type Output = Quantity<Quot<f64, T>, Negate<U>>;
    fn div(self, other: Quantity<T, U>) -> Self::Output {
        Quantity(self / other.0, PhantomData)
    }
}

impl<T: Div<f64>, U> Div<f64> for Quantity<T, U> {
    type Output = Quantity<Quot<T, f64>, U>;
    fn div(self, other: f64) -> Self::Output {
        Quantity(self.0 / other, PhantomData)
    }
}

impl<'a, T, U> Div<f64> for &'a Quantity<T, U>
where
    &'a T: Div<f64>,
{
    type Output = Quantity<Quot<&'a T, f64>, U>;
    fn div(self, other: f64) -> Self::Output {
        Quantity(&self.0 / other, PhantomData)
    }
}

impl<U: Neg, S: Data<Elem = f64>, D: Dimension> Div<Quantity<f64, U>> for &ArrayBase<S, D> {
    type Output = Quantity<Array<f64, D>, Negate<U>>;
    fn div(self, other: Quantity<f64, U>) -> Self::Output {
        Quantity(self / other.0, PhantomData)
    }
}

impl<U: Neg, S: DataOwned<Elem = f64> + DataMut, D: Dimension> Div<Quantity<f64, U>>
    for ArrayBase<S, D>
{
    type Output = Quantity<ArrayBase<S, D>, Negate<U>>;
    fn div(self, other: Quantity<f64, U>) -> Self::Output {
        Quantity(self / other.0, PhantomData)
    }
}

impl<U, T1, T2> DivAssign<T2> for Quantity<T1, U>
where
    T1: DivAssign<T2>,
{
    fn div_assign(&mut self, other: T2) {
        self.0 /= other;
    }
}

// Addition
impl<T1, T2, U> Add<Quantity<T2, U>> for Quantity<T1, U>
where
    T1: Add<T2>,
{
    type Output = Quantity<Sum<T1, T2>, U>;
    fn add(self, other: Quantity<T2, U>) -> Self::Output {
        Quantity(self.0 + other.0, PhantomData)
    }
}

impl<'a, T1, T2, U> Add<Quantity<T2, U>> for &'a Quantity<T1, U>
where
    &'a T1: Add<T2>,
{
    type Output = Quantity<Sum<&'a T1, T2>, U>;
    fn add(self, other: Quantity<T2, U>) -> Self::Output {
        Quantity(&self.0 + other.0, PhantomData)
    }
}

impl<'b, T1, T2, U> Add<&'b Quantity<T2, U>> for Quantity<T1, U>
where
    T1: Add<&'b T2>,
{
    type Output = Quantity<Sum<T1, &'b T2>, U>;
    fn add(self, other: &'b Quantity<T2, U>) -> Self::Output {
        Quantity(self.0 + &other.0, PhantomData)
    }
}

impl<'a, 'b, T1, T2, U> Add<&'b Quantity<T2, U>> for &'a Quantity<T1, U>
where
    &'a T1: Add<&'b T2>,
{
    type Output = Quantity<Sum<&'a T1, &'b T2>, U>;
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

// Subtraction
impl<T1, T2, U> Sub<Quantity<T2, U>> for Quantity<T1, U>
where
    T1: Sub<T2>,
{
    type Output = Quantity<Diff<T1, T2>, U>;
    fn sub(self, other: Quantity<T2, U>) -> Self::Output {
        Quantity(self.0 - other.0, PhantomData)
    }
}

impl<'a, T1, T2, U> Sub<Quantity<T2, U>> for &'a Quantity<T1, U>
where
    &'a T1: Sub<T2>,
{
    type Output = Quantity<Diff<&'a T1, T2>, U>;
    fn sub(self, other: Quantity<T2, U>) -> Self::Output {
        Quantity(&self.0 - other.0, PhantomData)
    }
}

impl<'b, T1, T2, U> Sub<&'b Quantity<T2, U>> for Quantity<T1, U>
where
    T1: Sub<&'b T2>,
{
    type Output = Quantity<Diff<T1, &'b T2>, U>;
    fn sub(self, other: &'b Quantity<T2, U>) -> Self::Output {
        Quantity(self.0 - &other.0, PhantomData)
    }
}

impl<'a, 'b, T1, T2, U> Sub<&'b Quantity<T2, U>> for &'a Quantity<T1, U>
where
    &'a T1: Sub<&'b T2>,
{
    type Output = Quantity<Diff<&'a T1, &'b T2>, U>;
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

impl<'a, T1, T2, U> SubAssign<&'a Quantity<T2, U>> for Quantity<T1, U>
where
    T1: SubAssign<&'a T2>,
{
    fn sub_assign(&mut self, rhs: &'a Quantity<T2, U>) {
        self.0 -= &rhs.0;
    }
}

// Negation
impl<T, U> Neg for Quantity<T, U>
where
    T: Neg,
{
    type Output = Quantity<Negate<T>, U>;
    fn neg(self) -> Self::Output {
        Quantity(-self.0, PhantomData)
    }
}

impl<U> Quantity<f64, U> {
    /// Calculate the integer power of self.
    ///
    /// # Example
    /// ```
    /// # use feos_core::si::METER;
    /// # use ndarray::arr1;
    /// # use approx::assert_relative_eq;
    /// # use typenum::P2;
    /// let x = 3.0 * METER;
    /// assert_relative_eq!(x.powi::<P2>(), 9.0 * METER * METER);
    /// ```
    pub fn powi<E: Integer>(self) -> Quantity<f64, Prod<U, E>>
    where
        U: Mul<E>,
    {
        Quantity(self.0.powi(E::I32), PhantomData)
    }

    /// Calculate the square root of self.
    ///
    /// # Example
    /// ```
    /// # use feos_core::si::METER;
    /// # use ndarray::arr1;
    /// # use approx::assert_relative_eq;
    /// let x = 9.0 * METER * METER;
    /// assert_relative_eq!(x.sqrt(), 3.0 * METER);
    /// ```
    pub fn sqrt(self) -> Quantity<f64, Quot<U, P2>>
    where
        U: Div<P2>,
    {
        Quantity(self.0.sqrt(), PhantomData)
    }

    /// Calculate the cubic root of self.
    ///
    /// # Example
    /// ```
    /// # use feos_core::si::METER;
    /// # use ndarray::arr1;
    /// # use approx::assert_relative_eq;
    /// let x = 27.0 * METER * METER * METER;
    /// assert_relative_eq!(x.cbrt(), 3.0 * METER);
    /// ```
    pub fn cbrt(self) -> Quantity<f64, Quot<U, P3>>
    where
        U: Div<P3>,
    {
        Quantity(self.0.cbrt(), PhantomData)
    }

    /// Calculate the integer root of self.
    ///
    /// # Example
    /// ```
    /// # use feos_core::si::METER;
    /// # use ndarray::arr1;
    /// # use approx::assert_relative_eq;
    /// # use typenum::P4;
    /// let x = 81.0 * METER * METER * METER * METER;
    /// assert_relative_eq!(x.root::<P4>(), 3.0 * METER);
    /// ```
    pub fn root<R: Integer>(self) -> Quantity<f64, Quot<U, R>>
    where
        U: Div<R>,
    {
        Quantity(self.0.powf(1.0 / R::I32 as f64), PhantomData)
    }

    /// Return the absolute value of `self`.
    ///
    /// # Example
    /// ```
    /// # use feos_core::si::KELVIN;
    /// # use approx::assert_relative_eq;
    /// let t = -50.0 * KELVIN;
    /// assert_relative_eq!(t.abs(), &(50.0 * KELVIN));
    pub fn abs(self) -> Self {
        Self(self.0.abs(), PhantomData)
    }

    /// Returns a number that represents the sign of `self`.
    ///
    /// - `1.0` if the number is positive, `+0.0` or `INFINITY`
    /// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
    /// - `NAN` if the number is `NAN`
    pub fn signum(self) -> f64 {
        self.0.signum()
    }

    /// Returns `true` if `self` has a negative sign, including `-0.0`, `NaN`s with
    /// negative sign bit and negative infinity.
    pub fn is_sign_negative(&self) -> bool {
        self.0.is_sign_negative()
    }

    /// Returns `true` if `self` has a positive sign, including `+0.0`, `NaN`s with
    /// positive sign bit and positive infinity.
    pub fn is_sign_positive(&self) -> bool {
        self.0.is_sign_positive()
    }

    /// Returns true if this value is NaN.
    pub fn is_nan(&self) -> bool {
        self.0.is_nan()
    }

    /// Return the minimum of `self` and `other`.
    ///
    /// # Example
    /// ```
    /// # use feos_core::si::{KILO, PASCAL, BAR};
    /// # use approx::assert_relative_eq;
    /// let p1 = 110.0 * KILO * PASCAL;
    /// let p2 = BAR;
    /// assert_relative_eq!(p1.min(p2), &p2);
    /// ```
    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0), PhantomData)
    }

    /// Return the maximum of `self` and `other`.
    ///
    /// # Example
    /// ```
    /// # use feos_core::si::{KILO, PASCAL, BAR};
    /// # use approx::assert_relative_eq;
    /// let p1 = 110.0 * KILO * PASCAL;
    /// let p2 = BAR;
    /// assert_relative_eq!(p1.max(p2), &p1);
    /// ```
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

impl<T: AbsDiffEq, U> AbsDiffEq for Quantity<T, U> {
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.0.abs_diff_eq(&other.0, epsilon)
    }
}

impl<T: RelativeEq, U> RelativeEq for Quantity<T, U> {
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
