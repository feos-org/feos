use super::{DataSet, EstimatorError};
use feos_core::{DensityInitialization, EntropyScaling, EosUnit, EquationOfState, State};
use ndarray::arr1;
use quantity::{QuantityArray1, QuantityScalar};
use std::collections::HashMap;
use std::sync::Arc;

/// Store experimental viscosity data.
#[derive(Clone)]
pub struct Viscosity<U: EosUnit> {
    pub target: QuantityArray1<U>,
    temperature: QuantityArray1<U>,
    pressure: QuantityArray1<U>,
}

impl<U: EosUnit> Viscosity<U> {
    /// Create a new data set for experimental viscosity data.
    pub fn new(
        target: QuantityArray1<U>,
        temperature: QuantityArray1<U>,
        pressure: QuantityArray1<U>,
    ) -> Result<Self, EstimatorError> {
        Ok(Self {
            target,
            temperature,
            pressure,
        })
    }

    /// Return temperature.
    pub fn temperature(&self) -> QuantityArray1<U> {
        self.temperature.clone()
    }

    /// Return pressure.
    pub fn pressure(&self) -> QuantityArray1<U> {
        self.pressure.clone()
    }
}

impl<U: EosUnit, E: EquationOfState + EntropyScaling<U>> DataSet<U, E> for Viscosity<U> {
    fn target(&self) -> &QuantityArray1<U> {
        &self.target
    }

    fn target_str(&self) -> &str {
        "viscosity"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature", "pressure"]
    }

    fn predict(&self, eos: &Arc<E>) -> Result<QuantityArray1<U>, EstimatorError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let moles = arr1(&[1.0]) * U::reference_moles();
        self.temperature
            .into_iter()
            .zip(self.pressure.into_iter())
            .map(|(t, p)| {
                State::new_npt(eos, t, p, &moles, DensityInitialization::None)?
                    .viscosity()
                    .map_err(EstimatorError::from)
            })
            .collect()
    }

    fn get_input(&self) -> HashMap<String, QuantityArray1<U>> {
        let mut m = HashMap::with_capacity(1);
        m.insert("temperature".to_owned(), self.temperature());
        m.insert("pressure".to_owned(), self.pressure());
        m
    }
}
