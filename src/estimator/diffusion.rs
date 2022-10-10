use super::{DataSet, EstimatorError};
use feos_core::{DensityInitialization, EntropyScaling, EosUnit, EquationOfState, State};
use ndarray::{arr1, Array1};
use quantity::{QuantityArray1, QuantityScalar};
#[cfg(feature = "rayon")]
use rayon_::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Store experimental diffusion data.
#[derive(Clone)]
pub struct Diffusion<U: EosUnit> {
    pub target: QuantityArray1<U>,
    temperature: QuantityArray1<U>,
    pressure: QuantityArray1<U>,
}

impl<U: EosUnit> Diffusion<U> {
    /// Create a new data set for experimental diffusion data.
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

impl<U: EosUnit, E: EquationOfState + EntropyScaling<U>> DataSet<U, E> for Diffusion<U> {
    fn target(&self) -> &QuantityArray1<U> {
        &self.target
    }

    fn target_str(&self) -> &str {
        "diffusion"
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

        #[cfg(not(feature = "rayon"))]
        let tp_iter = ts.iter().zip(ps.iter());
        #[cfg(feature = "rayon")]
        let tp_iter = (ts.as_slice().unwrap(), ps.as_slice().unwrap())
        .into_par_iter();

        let res = tp_iter
            .map(|(&t, &p)| {
                State::new_npt(
                    eos,
                    t * U::reference_temperature(),
                    p * U::reference_pressure(),
                    &moles,
                    DensityInitialization::None,
                )?
                .diffusion()?
                .to_reduced(U::reference_diffusion())
                .map_err(EstimatorError::from)
            })
            .collect::<Result<Vec<f64>, EstimatorError>>();
        Ok(Array1::from_vec(res?) * U::reference_diffusion())
    }

    fn get_input(&self) -> HashMap<String, QuantityArray1<U>> {
        let mut m = HashMap::with_capacity(1);
        m.insert("temperature".to_owned(), self.temperature());
        m.insert("pressure".to_owned(), self.pressure());
        m
    }
}
