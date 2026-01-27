#[cfg(feature = "ndarray")]
use super::Contributions;
use super::State;
#[cfg(feature = "ndarray")]
use crate::equation_of_state::{Molarweight, Residual, Total};
#[cfg(feature = "ndarray")]
use ndarray::{Array1, Array2};

#[cfg(feature = "ndarray")]
use quantity::{
    Density, MassDensity, MolarEnergy, MolarEntropy, Moles, Pressure, SpecificEnergy,
    SpecificEntropy, Temperature,
};
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

#[cfg(feature = "ndarray")]
impl<E: Residual> StateVec<'_, E> {
    pub fn temperature(&self) -> Temperature<Array1<f64>> {
        Temperature::from_shape_fn(self.0.len(), |i| self.0[i].temperature)
    }

    pub fn pressure(&self) -> Pressure<Array1<f64>> {
        Pressure::from_shape_fn(self.0.len(), |i| self.0[i].pressure(Contributions::Total))
    }

    pub fn compressibility(&self) -> Array1<f64> {
        Array1::from_shape_fn(self.0.len(), |i| {
            self.0[i].compressibility(Contributions::Total)
        })
    }

    pub fn density(&self) -> Density<Array1<f64>> {
        Density::from_shape_fn(self.0.len(), |i| self.0[i].density)
    }

    pub fn moles(&self) -> Moles<Array2<f64>> {
        Moles::from_shape_fn((self.0.len(), self.0[0].eos.components()), |(i, j)| {
            self.0[i].moles().get(j)
        })
    }

    pub fn molefracs(&self) -> Array2<f64> {
        Array2::from_shape_fn((self.0.len(), self.0[0].eos.components()), |(i, j)| {
            self.0[i].molefracs[j]
        })
    }
}

#[cfg(feature = "ndarray")]
impl<E: Residual + Molarweight> StateVec<'_, E> {
    pub fn mass_density(&self) -> MassDensity<Array1<f64>> {
        MassDensity::from_shape_fn(self.0.len(), |i| self.0[i].mass_density())
    }

    pub fn massfracs(&self) -> Array2<f64> {
        Array2::from_shape_fn((self.0.len(), self.0[0].eos.components()), |(i, j)| {
            self.0[i].massfracs()[j]
        })
    }
}

#[cfg(feature = "ndarray")]
impl<E: Total> StateVec<'_, E> {
    pub fn molar_enthalpy(&self, contributions: Contributions) -> MolarEnergy<Array1<f64>> {
        MolarEnergy::from_shape_fn(self.0.len(), |i| self.0[i].molar_enthalpy(contributions))
    }

    pub fn molar_entropy(&self, contributions: Contributions) -> MolarEntropy<Array1<f64>> {
        MolarEntropy::from_shape_fn(self.0.len(), |i| self.0[i].molar_entropy(contributions))
    }
}

#[cfg(feature = "ndarray")]
impl<E: Total + Molarweight> StateVec<'_, E> {
    pub fn specific_enthalpy(&self, contributions: Contributions) -> SpecificEnergy<Array1<f64>> {
        SpecificEnergy::from_shape_fn(self.0.len(), |i| self.0[i].specific_enthalpy(contributions))
    }

    pub fn specific_entropy(&self, contributions: Contributions) -> SpecificEntropy<Array1<f64>> {
        SpecificEntropy::from_shape_fn(self.0.len(), |i| self.0[i].specific_entropy(contributions))
    }
}
