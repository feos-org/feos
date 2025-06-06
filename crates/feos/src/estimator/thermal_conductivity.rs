use super::{DataSet, FeosError, Phase};
use feos_core::{DensityInitialization, EntropyScaling, ReferenceSystem, Residual, State};
use itertools::izip;
use ndarray::{Array1, arr1};
use quantity::{self, KELVIN, METER, Moles, Pressure, Temperature, WATT};
use std::sync::Arc;

/// Store experimental thermal conductivity data.
#[derive(Clone)]
pub struct ThermalConductivity {
    pub target: Array1<f64>,
    unit: quantity::ThermalConductivity,
    temperature: Temperature<Array1<f64>>,
    pressure: Pressure<Array1<f64>>,
    initial_density: Vec<DensityInitialization>,
}

impl ThermalConductivity {
    /// Create a new data set for experimental thermal conductivity data.
    pub fn new(
        target: quantity::ThermalConductivity<Array1<f64>>,
        temperature: Temperature<Array1<f64>>,
        pressure: Pressure<Array1<f64>>,
        phase: Option<&Vec<Phase>>,
    ) -> Self {
        let n = temperature.len();
        let unit = WATT / METER / KELVIN;
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

impl<E: Residual + EntropyScaling> DataSet<E> for ThermalConductivity {
    fn target(&self) -> &Array1<f64> {
        &self.target
    }

    fn target_str(&self) -> &str {
        "thermal conductivity"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature", "pressure"]
    }

    fn predict(&self, eos: &Arc<E>) -> Result<Array1<f64>, FeosError> {
        let moles = Moles::from_reduced(arr1(&[1.0]));
        izip!(&self.temperature, &self.pressure, &self.initial_density)
            .map(|(t, p, &initial_density)| {
                State::new_npt(eos, t, p, &moles, initial_density)?
                    .thermal_conductivity()
                    .map(|lambda| lambda.convert_to(self.unit))
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
