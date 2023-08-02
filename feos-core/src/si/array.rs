use super::Quantity;
use ndarray::iter::LanesMut;
use ndarray::{
    Array, Array1, ArrayBase, ArrayView, Axis, Data, DataMut, Dimension, NdIndex, RemoveAxis,
    ShapeBuilder,
};
use std::iter::FromIterator;
use std::marker::PhantomData;

/// Constructors for 1D arrays
impl<U> Quantity<Array1<f64>, U> {
    pub fn from_vec(v: Vec<Quantity<f64, U>>) -> Self {
        Self(v.iter().map(|e| e.0).collect(), PhantomData)
    }

    pub fn linspace(start: Quantity<f64, U>, end: Quantity<f64, U>, n: usize) -> Self {
        Self(Array1::linspace(start.0, end.0, n), PhantomData)
    }

    pub fn logspace(start: Quantity<f64, U>, end: Quantity<f64, U>, n: usize) -> Self {
        Self(
            Array1::logspace(10.0, start.0.log10(), end.0.log10(), n),
            PhantomData,
        )
    }
}

/// Constructors for nD arrays
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

    pub fn lanes_mut(&mut self, axis: Axis) -> LanesMut<f64, D::Smaller>
    where
        S: DataMut,
    {
        self.0.lanes_mut(axis)
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

    pub fn set<I: NdIndex<D>>(&mut self, index: I, value: Quantity<f64, U>)
    where
        S: DataMut,
    {
        self.0[index] = value.0;
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
            inner: self.0.into_iter(),
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
