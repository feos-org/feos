use super::{Contributions, State};
use crate::equation_of_state::{IdealGas, Residual};
use ndarray::{Array1, Array2};
use quantity::si::{SIArray1, SIArray2};
use std::iter::FromIterator;
use std::ops::Deref;

/// A list of states for a simple access to properties
/// of multiple states.
pub struct StateVec<'a, E>(pub Vec<&'a State<E>>);

impl<'a, E> FromIterator<&'a State<E>> for StateVec<'a, E> {
    fn from_iter<I: IntoIterator<Item = &'a State<E>>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<'a, E> IntoIterator for StateVec<'a, E> {
    type Item = &'a State<E>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, E> Deref for StateVec<'a, E> {
    type Target = Vec<&'a State<E>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, E: Residual> StateVec<'a, E> {
    pub fn temperature(&self) -> SIArray1 {
        SIArray1::from_shape_fn(self.0.len(), |i| self.0[i].temperature)
    }

    pub fn pressure(&self) -> SIArray1 {
        SIArray1::from_shape_fn(self.0.len(), |i| self.0[i].pressure(Contributions::Total))
    }

    pub fn compressibility(&self) -> Array1<f64> {
        Array1::from_shape_fn(self.0.len(), |i| {
            self.0[i].compressibility(Contributions::Total)
        })
    }

    pub fn density(&self) -> SIArray1 {
        SIArray1::from_shape_fn(self.0.len(), |i| self.0[i].density)
    }

    pub fn moles(&self) -> SIArray2 {
        SIArray2::from_shape_fn((self.0.len(), self.0[0].eos.components()), |(i, j)| {
            self.0[i].moles.get(j)
        })
    }

    pub fn molefracs(&self) -> Array2<f64> {
        Array2::from_shape_fn((self.0.len(), self.0[0].eos.components()), |(i, j)| {
            self.0[i].molefracs[j]
        })
    }
}

impl<'a, E: Residual + IdealGas> StateVec<'a, E> {
    pub fn molar_enthalpy(&self) -> SIArray1 {
        SIArray1::from_shape_fn(self.0.len(), |i| {
            self.0[i].molar_enthalpy(Contributions::Total)
        })
    }

    pub fn molar_entropy(&self) -> SIArray1 {
        SIArray1::from_shape_fn(self.0.len(), |i| {
            self.0[i].molar_entropy(Contributions::Total)
        })
    }

    pub fn mass_density(&self) -> SIArray1 {
        SIArray1::from_shape_fn(self.0.len(), |i| self.0[i].mass_density())
    }

    pub fn massfracs(&self) -> Array2<f64> {
        Array2::from_shape_fn((self.0.len(), self.0[0].eos.components()), |(i, j)| {
            self.0[i].massfracs()[j]
        })
    }

    pub fn specific_enthalpy(&self) -> SIArray1 {
        SIArray1::from_shape_fn(self.0.len(), |i| {
            self.0[i].specific_enthalpy(Contributions::Total)
        })
    }

    pub fn specific_entropy(&self) -> SIArray1 {
        SIArray1::from_shape_fn(self.0.len(), |i| {
            self.0[i].specific_entropy(Contributions::Total)
        })
    }
}
