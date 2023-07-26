use super::{DataSet, EstimatorError, Phase};
use feos_core::{DensityInitialization, EntropyScaling, EosUnit, Residual, State};
use itertools::izip;
use ndarray::arr1;
use quantity::si::{SIArray1, SIUnit};
use std::collections::HashMap;
use std::sync::Arc;

/// Store experimental viscosity data.
#[derive(Clone)]
pub struct Viscosity {
    pub target: SIArray1,
    temperature: SIArray1,
    pressure: SIArray1,
    initial_density: Vec<DensityInitialization>,
}

impl Viscosity {
    /// Create a new data set for experimental viscosity data.
    pub fn new(
        target: SIArray1,
        temperature: SIArray1,
        pressure: SIArray1,
        phase: Option<&Vec<Phase>>,
    ) -> Result<Self, EstimatorError> {
        let n = temperature.len();
        Ok(Self {
            target,
            temperature,
            pressure,
            initial_density: phase.map_or(vec![DensityInitialization::None; n], |phase| {
                phase.iter().map(|&p| p.into()).collect()
            }),
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

impl<E: Residual + EntropyScaling> DataSet<E> for Viscosity {
    fn target(&self) -> &SIArray1 {
        &self.target
    }

    fn target_str(&self) -> &str {
        "viscosity"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature", "pressure"]
    }

    fn predict(&self, eos: &Arc<E>) -> Result<SIArray1, EstimatorError> {
        let moles = arr1(&[1.0]) * SIUnit::reference_moles();
        izip!(&self.temperature, &self.pressure, &self.initial_density)
            .map(|(t, p, &initial_density)| {
                State::new_npt(eos, t, p, &moles, initial_density)?
                    .viscosity()
                    .map_err(EstimatorError::from)
            })
            .collect()
    }

    fn get_input(&self) -> HashMap<String, SIArray1> {
        let mut m = HashMap::with_capacity(1);
        m.insert("temperature".to_owned(), self.temperature());
        m.insert("pressure".to_owned(), self.pressure());
        m
    }
}
