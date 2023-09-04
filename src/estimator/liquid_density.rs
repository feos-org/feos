use super::{DataSet, EstimatorError};
use feos_core::si::{MassDensity, Moles, Pressure, Temperature, KILOGRAM, METER};
use feos_core::{DensityInitialization, PhaseEquilibrium, Residual, SolverOptions, State};
use ndarray::{arr1, Array1};
use std::sync::Arc;
use typenum::P3;

/// Liquid mass density data as function of pressure and temperature.
#[derive(Clone)]
pub struct LiquidDensity {
    /// mass density
    pub target: Array1<f64>,
    /// unit of mass density
    unit: MassDensity,
    /// temperature
    temperature: Temperature<Array1<f64>>,
    /// pressure
    pressure: Pressure<Array1<f64>>,
}

impl LiquidDensity {
    /// A new data set for liquid densities with pressures and temperatures as input.
    pub fn new(
        target: MassDensity<Array1<f64>>,
        temperature: Temperature<Array1<f64>>,
        pressure: Pressure<Array1<f64>>,
    ) -> Self {
        let unit = KILOGRAM / METER.powi::<P3>();
        Self {
            target: (target / unit).to_reduced(),
            unit,
            temperature,
            pressure,
        }
    }

    /// Returns temperature of data points.
    pub fn temperature(&self) -> &Temperature<Array1<f64>> {
        &self.temperature
    }

    /// Returns pressure of data points.
    pub fn pressure(&self) -> &Pressure<Array1<f64>> {
        &self.pressure
    }
}

impl<E: Residual> DataSet<E> for LiquidDensity {
    fn target(&self) -> &Array1<f64> {
        &self.target
    }

    fn target_str(&self) -> &str {
        "liquid density"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature", "pressure"]
    }

    fn predict(&self, eos: &Arc<E>) -> Result<Array1<f64>, EstimatorError> {
        let moles = Moles::from_reduced(arr1(&[1.0]));
        Ok(self
            .temperature
            .into_iter()
            .zip(self.pressure.into_iter())
            .map(|(t, p)| {
                let state = State::new_npt(eos, t, p, &moles, DensityInitialization::Liquid);
                if let Ok(s) = state {
                    (s.mass_density() / self.unit).into_value()
                } else {
                    f64::NAN
                }
            })
            .collect())
    }

    // fn get_input(&self) -> HashMap<String, SIArray1> {
    //     let mut m = HashMap::with_capacity(2);
    //     m.insert("temperature".to_owned(), self.temperature());
    //     m.insert("pressure".to_owned(), self.pressure());
    //     m
    // }
}

/// Store experimental data of liquid densities calculated for phase equilibria.
#[derive(Clone)]
pub struct EquilibriumLiquidDensity {
    pub target: Array1<f64>,
    /// unit of mass density
    unit: MassDensity,
    /// temperature
    temperature: Temperature<Array1<f64>>,
    /// options for VLE solver
    solver_options: SolverOptions,
}

impl EquilibriumLiquidDensity {
    /// A new data set for liquid densities with pressures and temperatures as input.
    pub fn new(
        target: MassDensity<Array1<f64>>,
        temperature: Temperature<Array1<f64>>,
        vle_options: Option<SolverOptions>,
    ) -> Self {
        let unit = KILOGRAM / METER.powi::<P3>();
        Self {
            target: (target / unit).to_reduced(),
            unit,
            temperature,
            solver_options: vle_options.unwrap_or_default(),
        }
    }

    /// Returns temperature of data points.
    pub fn temperature(&self) -> &Temperature<Array1<f64>> {
        &self.temperature
    }
}

impl<E: Residual> DataSet<E> for EquilibriumLiquidDensity {
    fn target(&self) -> &Array1<f64> {
        &self.target
    }

    fn target_str(&self) -> &str {
        "equilibrium liquid density"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature"]
    }

    fn predict(&self, eos: &Arc<E>) -> Result<Array1<f64>, EstimatorError> {
        Ok(self
            .temperature
            .into_iter()
            .map(|t| {
                if let Ok(state) = PhaseEquilibrium::pure(eos, t, None, self.solver_options) {
                    (state.liquid().mass_density() / self.unit).into_value()
                } else {
                    f64::NAN
                }
            })
            .collect())
    }

    // fn get_input(&self) -> HashMap<String, SIArray1> {
    //     let mut m = HashMap::with_capacity(2);
    //     m.insert("temperature".to_owned(), self.temperature());
    //     m
    // }
}
