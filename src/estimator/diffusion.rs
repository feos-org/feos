use super::{DataSet, EstimatorError};
use feos_core::{DensityInitialization, EntropyScaling, EosUnit, EquationOfState, State};
use ndarray::{arr1, Array1};
use quantity::si::{SIArray1, SIUnit};
use std::collections::HashMap;
use std::sync::Arc;

/// Store experimental diffusion data.
#[derive(Clone)]
pub struct Diffusion {
    pub target: SIArray1,
    temperature: SIArray1,
    pressure: SIArray1,
}

impl Diffusion {
    /// Create a new data set for experimental diffusion data.
    pub fn new(
        target: SIArray1,
        temperature: SIArray1,
        pressure: SIArray1,
    ) -> Result<Self, EstimatorError> {
        Ok(Self {
            target,
            temperature,
            pressure,
        })
    }

    /// Return temperature.
    pub fn temperature(&self) -> SIArray1 {
        self.temperature.clone()
    }

    /// Return pressure.
    pub fn pressure(&self) -> SIArray1 {
        self.pressure.clone()
    }
}

impl<E: EquationOfState + EntropyScaling> DataSet<E> for Diffusion {
    fn target(&self) -> &SIArray1 {
        &self.target
    }

    fn target_str(&self) -> &str {
        "diffusion"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature", "pressure"]
    }

    fn predict(&self, eos: &Arc<E>) -> Result<SIArray1, EstimatorError> {
        let moles = arr1(&[1.0]) * SIUnit::reference_moles();
        let ts = self
            .temperature
            .to_reduced(SIUnit::reference_temperature())
            .unwrap();
        let ps = self
            .pressure
            .to_reduced(SIUnit::reference_pressure())
            .unwrap();

        let res = ts
            .iter()
            .zip(ps.iter())
            .map(|(&t, &p)| {
                State::new_npt(
                    eos,
                    t * SIUnit::reference_temperature(),
                    p * SIUnit::reference_pressure(),
                    &moles,
                    DensityInitialization::None,
                )?
                .diffusion()?
                .to_reduced(SIUnit::reference_diffusion())
                .map_err(EstimatorError::from)
            })
            .collect::<Result<Vec<f64>, EstimatorError>>();
        Ok(Array1::from_vec(res?) * SIUnit::reference_diffusion())
    }

    fn get_input(&self) -> HashMap<String, SIArray1> {
        let mut m = HashMap::with_capacity(1);
        m.insert("temperature".to_owned(), self.temperature());
        m.insert("pressure".to_owned(), self.pressure());
        m
    }
}
