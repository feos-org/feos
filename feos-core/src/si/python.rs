use super::{Pressure, Quantity, SIUnit, Temperature};
use crate::{EosError, EosResult, TPSpec};
use ndarray::{Array1, Array2, Array3, Array4};
use quantity::python::{PySIArray1, PySIArray2, PySIArray3, PySIArray4, PySINumber};
use quantity::si;
use quantity::Quantity as PyQuantity;
use std::convert::TryFrom;
use std::marker::PhantomData;
use std::ops::{Div, Mul};
use typenum::Integer;

impl<
        Inner,
        T: Integer,
        L: Integer,
        M: Integer,
        I: Integer,
        THETA: Integer,
        N: Integer,
        J: Integer,
    > TryFrom<PyQuantity<Inner, si::SIUnit>> for Quantity<Inner, SIUnit<T, L, M, I, THETA, N, J>>
where
    for<'a> &'a Inner: Div<f64, Output = Inner>,
{
    type Error = EosError;
    fn try_from(quantity: PyQuantity<Inner, si::SIUnit>) -> EosResult<Self> {
        let unit = si::SECOND.powi(T::to_i32())
            * si::METER.powi(L::to_i32())
            * si::KILOGRAM.powi(M::to_i32())
            * si::AMPERE.powi(I::to_i32())
            * si::KELVIN.powi(THETA::to_i32())
            * si::MOL.powi(N::to_i32())
            * si::CANDELA.powi(J::to_i32());
        if quantity.has_unit(&unit) {
            let value = quantity.to_reduced(unit).unwrap();
            Ok(Quantity(value, PhantomData))
        } else {
            todo!()
        }
    }
}

impl<T: Integer, L: Integer, M: Integer, I: Integer, THETA: Integer, N: Integer, J: Integer>
    TryFrom<PySINumber> for Quantity<f64, SIUnit<T, L, M, I, THETA, N, J>>
{
    type Error = <Self as TryFrom<PyQuantity<f64, si::SIUnit>>>::Error;

    fn try_from(value: PySINumber) -> Result<Self, Self::Error> {
        Self::try_from(PyQuantity::from(value))
    }
}

impl<T: Integer, L: Integer, M: Integer, I: Integer, THETA: Integer, N: Integer, J: Integer>
    TryFrom<PySIArray1> for Quantity<Array1<f64>, SIUnit<T, L, M, I, THETA, N, J>>
{
    type Error = <Self as TryFrom<PyQuantity<Array1<f64>, si::SIUnit>>>::Error;

    fn try_from(value: PySIArray1) -> Result<Self, Self::Error> {
        Self::try_from(PyQuantity::from(value))
    }
}

impl<
        Inner,
        T: Integer,
        L: Integer,
        M: Integer,
        I: Integer,
        THETA: Integer,
        N: Integer,
        J: Integer,
    > From<Quantity<Inner, SIUnit<T, L, M, I, THETA, N, J>>> for PyQuantity<Inner, si::SIUnit>
where
    Inner: Mul<si::SINumber, Output = PyQuantity<Inner, si::SIUnit>>,
{
    fn from(quantity: Quantity<Inner, SIUnit<T, L, M, I, THETA, N, J>>) -> Self {
        quantity.0
            * (si::SECOND.powi(T::to_i32())
                * si::METER.powi(L::to_i32())
                * si::KILOGRAM.powi(M::to_i32())
                * si::AMPERE.powi(I::to_i32())
                * si::KELVIN.powi(THETA::to_i32())
                * si::MOL.powi(N::to_i32())
                * si::CANDELA.powi(J::to_i32()))
    }
}

impl<T: Integer, L: Integer, M: Integer, I: Integer, THETA: Integer, N: Integer, J: Integer>
    From<Quantity<f64, SIUnit<T, L, M, I, THETA, N, J>>> for PySINumber
{
    fn from(quantity: Quantity<f64, SIUnit<T, L, M, I, THETA, N, J>>) -> Self {
        Self::from(PyQuantity::from(quantity))
    }
}

impl<T: Integer, L: Integer, M: Integer, I: Integer, THETA: Integer, N: Integer, J: Integer>
    From<Quantity<Array1<f64>, SIUnit<T, L, M, I, THETA, N, J>>> for PySIArray1
{
    fn from(quantity: Quantity<Array1<f64>, SIUnit<T, L, M, I, THETA, N, J>>) -> Self {
        Self::from(PyQuantity::from(quantity))
    }
}

impl<T: Integer, L: Integer, M: Integer, I: Integer, THETA: Integer, N: Integer, J: Integer>
    From<Quantity<Array2<f64>, SIUnit<T, L, M, I, THETA, N, J>>> for PySIArray2
{
    fn from(quantity: Quantity<Array2<f64>, SIUnit<T, L, M, I, THETA, N, J>>) -> Self {
        Self::from(PyQuantity::from(quantity))
    }
}

impl<T: Integer, L: Integer, M: Integer, I: Integer, THETA: Integer, N: Integer, J: Integer>
    From<Quantity<Array3<f64>, SIUnit<T, L, M, I, THETA, N, J>>> for PySIArray3
{
    fn from(quantity: Quantity<Array3<f64>, SIUnit<T, L, M, I, THETA, N, J>>) -> Self {
        Self::from(PyQuantity::from(quantity))
    }
}

impl<T: Integer, L: Integer, M: Integer, I: Integer, THETA: Integer, N: Integer, J: Integer>
    From<Quantity<Array4<f64>, SIUnit<T, L, M, I, THETA, N, J>>> for PySIArray4
{
    fn from(quantity: Quantity<Array4<f64>, SIUnit<T, L, M, I, THETA, N, J>>) -> Self {
        Self::from(PyQuantity::from(quantity))
    }
}

impl TryFrom<PySINumber> for TPSpec {
    type Error = EosError;

    fn try_from(quantity: PySINumber) -> EosResult<Self> {
        let quantity = PyQuantity::from(quantity);
        if let Ok(t) = Temperature::<f64>::try_from(quantity) {
            return Ok(TPSpec::Temperature(t));
        }
        if let Ok(p) = Pressure::<f64>::try_from(quantity) {
            return Ok(TPSpec::Pressure(p));
        }
        Err(EosError::Error("TODO".into()))
    }
}
