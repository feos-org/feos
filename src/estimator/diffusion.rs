use super::{DataSet, EstimatorError, Phase};
use feos_core::si::{self, Moles, Pressure, Temperature, CENTI, METER, SECOND};
use feos_core::{DensityInitialization, EntropyScaling, Residual, State};
use itertools::izip;
use ndarray::{arr1, Array1};
use std::sync::Arc;
use typenum::P2;

/// Store experimental diffusion data.
#[derive(Clone)]
pub struct Diffusion {
    pub target: Array1<f64>,
    unit: si::Diffusivity,
    temperature: Temperature<Array1<f64>>,
    pressure: Pressure<Array1<f64>>,
    initial_density: Vec<DensityInitialization>,
}

impl Diffusion {
    /// Create a new data set for experimental diffusion data.
    pub fn new(
        target: si::Diffusivity<Array1<f64>>,
        temperature: Temperature<Array1<f64>>,
        pressure: Pressure<Array1<f64>>,
        phase: Option<&Vec<Phase>>,
    ) -> Self {
        let n = temperature.len();
        let unit = (CENTI * METER).powi::<P2>() / SECOND;
        Self {
            target: (target / unit).into_value(),
            unit,
            temperature,
            pressure,
            initial_density: phase.map_or(vec![DensityInitialization::None; n], |phase| {
                phase.iter().map(|&p| p.into()).collect()
            }),
        }
    }

    /// Return temperature.
    pub fn temperature(&self) -> &Temperature<Array1<f64>> {
        &self.temperature
    }

    /// Return pressure.
    pub fn pressure(&self) -> &Pressure<Array1<f64>> {
        &self.pressure
    }
}

impl<E: Residual + EntropyScaling> DataSet<E> for Diffusion {
    fn target(&self) -> &Array1<f64> {
        &self.target
    }

    fn target_str(&self) -> &str {
        "diffusion"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature", "pressure"]
    }

    fn predict(&self, eos: &Arc<E>) -> Result<Array1<f64>, EstimatorError> {
        let moles = Moles::from_reduced(arr1(&[1.0]));

        izip!(&self.temperature, &self.pressure, &self.initial_density)
            .map(|(t, p, &initial_density)| {
                State::new_npt(eos, t, p, &moles, initial_density)?
                    .diffusion()
                    .map(|lambda| (lambda / self.unit).into_value())
                    .map_err(EstimatorError::from)
            })
            .collect()
    }

    // fn get_input(&self) -> HashMap<String, SIArray1> {
    //     let mut m = HashMap::with_capacity(1);
    //     m.insert("temperature".to_owned(), self.temperature());
    //     m.insert("pressure".to_owned(), self.pressure());
    //     m
    // }
}
