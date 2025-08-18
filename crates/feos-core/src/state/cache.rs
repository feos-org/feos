use crate::state::VectorPartialDerivative;

use super::{Derivative, PartialDerivative};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, OMatrix, OVector, Scalar};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Cache<D: Scalar, N: Dim>
where
    DefaultAllocator: Allocator<N>,
{
    pub map: HashMap<PartialDerivative, D>,
    pub vec_map: HashMap<VectorPartialDerivative, OVector<D, N>>,
    pub hit: u64,
    pub miss: u64,
}

impl<D: Scalar + Copy, N: Dim> Cache<D, N>
where
    DefaultAllocator: Allocator<N>,
{
    pub fn new() -> Self {
        Self {
            map: HashMap::with_capacity(8),
            vec_map: HashMap::with_capacity(3),
            hit: 0,
            miss: 0,
        }
    }

    pub fn get_or_insert_with_zeroth<F: FnOnce() -> D>(&mut self, f: F) -> D {
        if let Some(&value) = self.map.get(&PartialDerivative::Zeroth) {
            self.hit += 1;
            value
        } else {
            self.miss += 1;
            let value = f();
            self.map.insert(PartialDerivative::Zeroth, value);
            value
        }
    }

    pub fn get_or_insert_with_first_scalar<F: FnOnce() -> (D, D)>(
        &mut self,
        derivative: Derivative,
        f: F,
    ) -> D {
        if let Some(&value) = self.map.get(&PartialDerivative::First(derivative)) {
            self.hit += 1;
            value
        } else {
            self.miss += 1;
            let (f, df) = f();
            self.map.insert(PartialDerivative::Zeroth, f);
            self.map.insert(PartialDerivative::First(derivative), df);
            df
        }
    }

    pub fn get_or_insert_with_second<F: FnOnce() -> (D, D, D)>(
        &mut self,
        derivative: Derivative,
        f: F,
    ) -> D {
        if let Some(&value) = self.map.get(&PartialDerivative::Second(derivative)) {
            self.hit += 1;
            value
        } else {
            self.miss += 1;
            let (f, df, d2f) = f();
            self.map.insert(PartialDerivative::Zeroth, f);
            self.map.insert(PartialDerivative::First(derivative), df);
            self.map.insert(PartialDerivative::Second(derivative), d2f);
            d2f
        }
    }

    pub fn get_or_insert_with_second_mixed_scalar<F: FnOnce() -> (D, D, D, D)>(
        &mut self,
        derivative1: Derivative,
        derivative2: Derivative,
        f: F,
    ) -> D {
        if let Some(&value) = self.map.get(&PartialDerivative::SecondMixed) {
            self.hit += 1;
            value
        } else {
            self.miss += 1;
            let (f, df1, df2, d2f) = f();
            self.map.insert(PartialDerivative::Zeroth, f);
            self.map.insert(PartialDerivative::First(derivative1), df1);
            self.map.insert(PartialDerivative::First(derivative2), df2);
            self.map.insert(PartialDerivative::SecondMixed, d2f);
            d2f
        }
    }

    pub fn get_or_insert_with_third<F: FnOnce() -> (D, D, D, D)>(
        &mut self,
        derivative: Derivative,
        f: F,
    ) -> D {
        if let Some(&value) = self.map.get(&PartialDerivative::Third(derivative)) {
            self.hit += 1;
            value
        } else {
            self.miss += 1;
            let (f, df, d2f, d3f) = f();
            self.map.insert(PartialDerivative::Zeroth, f);
            self.map.insert(PartialDerivative::First(derivative), df);
            self.map.insert(PartialDerivative::Second(derivative), d2f);
            self.map.insert(PartialDerivative::Third(derivative), d3f);
            d3f
        }
    }

    pub fn get_or_insert_with_first_vector<F: FnOnce() -> (D, OVector<D, N>)>(
        &mut self,
        f: F,
    ) -> OVector<D, N> {
        if let Some(value) = self.vec_map.get(&VectorPartialDerivative::First) {
            self.hit += 1;
            value.clone()
        } else {
            self.miss += 1;
            let (f, df) = f();
            self.map.insert(PartialDerivative::Zeroth, f);
            self.vec_map
                .insert(VectorPartialDerivative::First, df.clone());
            df
        }
    }

    pub fn get_or_insert_with_second_vector<F: FnOnce() -> (D, OVector<D, N>, OMatrix<D, N, N>)>(
        &mut self,
        f: F,
    ) -> OMatrix<D, N, N>
    where
        DefaultAllocator: Allocator<N, N>,
    {
        self.miss += 1;
        let (f, df, d2f) = f();
        self.map.insert(PartialDerivative::Zeroth, f);
        self.vec_map.insert(VectorPartialDerivative::First, df);
        d2f
    }

    pub fn get_or_insert_with_second_mixed_vector<
        F: FnOnce() -> (D, OVector<D, N>, D, OVector<D, N>),
    >(
        &mut self,
        derivative: Derivative,
        f: F,
    ) -> OVector<D, N> {
        if let Some(value) = self
            .vec_map
            .get(&VectorPartialDerivative::SecondMixed(derivative))
        {
            self.hit += 1;
            value.clone()
        } else {
            self.miss += 1;
            let (f, df1, df2, d2f) = f();
            self.map.insert(PartialDerivative::Zeroth, f);
            self.vec_map.insert(VectorPartialDerivative::First, df1);
            self.map.insert(PartialDerivative::First(derivative), df2);
            self.vec_map.insert(
                VectorPartialDerivative::SecondMixed(derivative),
                d2f.clone(),
            );
            d2f
        }
    }
}
