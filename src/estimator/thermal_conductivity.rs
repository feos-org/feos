use super::{DataSet, EstimatorError};
use feos_core::{DensityInitialization, EntropyScaling, EosUnit, EquationOfState, State};
use ndarray::{arr1, Array1};
use quantity::{QuantityArray1, QuantityScalar};
#[cfg(feature = "rayon")]
use rayon_::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Store experimental thermal conductivity data.
#[derive(Clone)]
pub struct ThermalConductivity<U: EosUnit> {
    pub target: QuantityArray1<U>,
    temperature: QuantityArray1<U>,
    pressure: QuantityArray1<U>,
}

impl<U: EosUnit> ThermalConductivity<U> {
    /// Create a new data set for experimental thermal conductivity data.
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

impl<U: EosUnit, E: EquationOfState + EntropyScaling<U>> DataSet<U, E> for ThermalConductivity<U> {
    fn target(&self) -> &QuantityArray1<U> {
        &self.target
    }

    fn target_str(&self) -> &str {
        "thermal conductivity"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature", "pressure"]
    }

    fn predict(&self, eos: &Arc<E>) -> Result<QuantityArray1<U>, EstimatorError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let moles = arr1(&[1.0]) * U::reference_moles();
        let ts = self
            .temperature
            .to_reduced(U::reference_temperature())
            .unwrap();
        let ps = self.pressure.to_reduced(U::reference_pressure()).unwrap();
        let unit = U::reference_energy()
            / U::reference_time()
            / U::reference_temperature()
            / U::reference_length();

        let res = ts
            .iter()
            .zip(ps.iter())
            .map(|(&t, &p)| {
                State::new_npt(
                    eos,
                    t * U::reference_temperature(),
                    p * U::reference_pressure(),
                    &moles,
                    DensityInitialization::None,
                )?
                .thermal_conductivity()?
                .to_reduced(unit)
                .map_err(EstimatorError::from)
            })
            .collect::<Result<Vec<f64>, EstimatorError>>();
        Ok(Array1::from_vec(res?) * unit)
    }

    fn get_input(&self) -> HashMap<String, QuantityArray1<U>> {
        let mut m = HashMap::with_capacity(1);
        m.insert("temperature".to_owned(), self.temperature());
        m.insert("pressure".to_owned(), self.pressure());
        m
    }
}
