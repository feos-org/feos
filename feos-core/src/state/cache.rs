use super::{Derivative, PartialDerivative};
use num_dual::*;
use std::cmp::{max, min};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub(crate) struct Cache {
    pub map: HashMap<PartialDerivative, f64>,
    pub hit: u64,
    pub miss: u64,
}

impl Cache {
    pub fn with_capacity(components: usize) -> Cache {
        let capacity = 6 + 3 * components + components * (components + 1) / 2;
        Cache {
            map: HashMap::with_capacity(capacity),
            hit: 0,
            miss: 0,
        }
    }

    pub fn get_or_insert_with_f64<F: FnOnce() -> f64>(&mut self, f: F) -> f64 {
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

    pub fn get_or_insert_with_d64<F: FnOnce() -> Dual64>(
        &mut self,
        derivative: Derivative,
        f: F,
    ) -> f64 {
        if let Some(&value) = self.map.get(&PartialDerivative::First(derivative)) {
            self.hit += 1;
            value
        } else {
            self.miss += 1;
            let value = f();
            self.map.insert(PartialDerivative::Zeroth, value.re);
            self.map
                .insert(PartialDerivative::First(derivative), value.eps);
            value.eps
        }
    }

    pub fn get_or_insert_with_d2_64<F: FnOnce() -> Dual2_64>(
        &mut self,
        derivative: Derivative,
        f: F,
    ) -> f64 {
        if let Some(&value) = self
            .map
            .get(&PartialDerivative::SecondMixed(derivative, derivative))
        {
            self.hit += 1;
            value
        } else {
            self.miss += 1;
            let value = f();
            self.map.insert(PartialDerivative::Zeroth, value.re);
            self.map
                .insert(PartialDerivative::First(derivative), value.v1);
            self.map.insert(
                PartialDerivative::SecondMixed(derivative, derivative),
                value.v2,
            );
            value.v2
        }
    }

    pub fn get_or_insert_with_hd64<F: FnOnce() -> HyperDual64>(
        &mut self,
        derivative1: Derivative,
        derivative2: Derivative,
        f: F,
    ) -> f64 {
        let d1 = min(derivative1, derivative2);
        let d2 = max(derivative1, derivative2);
        if let Some(&value) = self.map.get(&PartialDerivative::SecondMixed(d1, d2)) {
            self.hit += 1;
            value
        } else {
            self.miss += 1;
            let value = f();
            self.map.insert(PartialDerivative::Zeroth, value.re);
            self.map
                .insert(PartialDerivative::First(derivative1), value.eps1);
            self.map
                .insert(PartialDerivative::First(derivative2), value.eps2);
            self.map
                .insert(PartialDerivative::SecondMixed(d1, d2), value.eps1eps2);
            value.eps1eps2
        }
    }

    pub fn get_or_insert_with_hd364<F: FnOnce() -> Dual3_64>(
        &mut self,
        derivative: Derivative,
        f: F,
    ) -> f64 {
        if let Some(&value) = self.map.get(&PartialDerivative::Third(derivative)) {
            self.hit += 1;
            value
        } else {
            self.miss += 1;
            let value = f();
            self.map.insert(PartialDerivative::Zeroth, value.re);
            self.map
                .insert(PartialDerivative::First(derivative), value.v1);
            self.map.insert(
                PartialDerivative::SecondMixed(derivative, derivative),
                value.v2,
            );
            self.map
                .insert(PartialDerivative::Third(derivative), value.v3);
            value.v3
        }
    }
}
