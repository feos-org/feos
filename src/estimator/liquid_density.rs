use super::{DataSet, EstimatorError};
use feos_core::{
    DensityInitialization, EosUnit, MolarWeight, PhaseEquilibrium, Residual, SolverOptions, State,
};
use ndarray::arr1;
use quantity::si::{SIArray1, SIUnit};
use std::collections::HashMap;
use std::sync::Arc;

/// Liquid mass density data as function of pressure and temperature.
#[derive(Clone)]
pub struct LiquidDensity {
    /// mass density
    pub target: SIArray1,
    /// temperature
    temperature: SIArray1,
    /// pressure
    pressure: SIArray1,
}

impl LiquidDensity {
    /// A new data set for liquid densities with pressures and temperatures as input.
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

    /// Returns temperature of data points.
    pub fn temperature(&self) -> SIArray1 {
        self.temperature.clone()
    }

    /// Returns pressure of data points.
    pub fn pressure(&self) -> SIArray1 {
        self.pressure.clone()
    }
}

impl<E: Residual + MolarWeight> DataSet<E> for LiquidDensity {
    fn target(&self) -> &SIArray1 {
        &self.target
    }

    fn target_str(&self) -> &str {
        "liquid density"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature", "pressure"]
    }

    fn predict(&self, eos: &Arc<E>) -> Result<SIArray1, EstimatorError> {
        let moles = arr1(&[1.0]) * SIUnit::reference_moles();
        Ok(self
            .temperature
            .into_iter()
            .zip(self.pressure.into_iter())
            .map(|(t, p)| {
                let state = State::new_npt(eos, t, p, &moles, DensityInitialization::Liquid);
                if let Ok(s) = state {
                    s.mass_density()
                } else {
                    f64::NAN * SIUnit::reference_mass() / SIUnit::reference_volume()
                }
            })
            .collect())
    }

    fn get_input(&self) -> HashMap<String, SIArray1> {
        let mut m = HashMap::with_capacity(2);
        m.insert("temperature".to_owned(), self.temperature());
        m.insert("pressure".to_owned(), self.pressure());
        m
    }
}

/// Store experimental data of liquid densities and compare to the equation of state.
#[derive(Clone)]
pub struct EquilibriumLiquidDensity {
    pub target: SIArray1,
    temperature: SIArray1,
    solver_options: SolverOptions,
}

impl EquilibriumLiquidDensity {
    /// A new data set for liquid densities with pressures and temperatures as input.
    pub fn new(
        target: SIArray1,
        temperature: SIArray1,
        vle_options: Option<SolverOptions>,
    ) -> Result<Self, EstimatorError> {
        Ok(Self {
            target,
            temperature,
            solver_options: vle_options.unwrap_or_default(),
        })
    }

    /// Returns temperature of data points.
    pub fn temperature(&self) -> SIArray1 {
        self.temperature.clone()
    }
}

impl<E: Residual + MolarWeight> DataSet<E> for EquilibriumLiquidDensity {
    fn target(&self) -> &SIArray1 {
        &self.target
    }

    fn target_str(&self) -> &str {
        "equilibrium liquid density"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature"]
    }

    fn predict(&self, eos: &Arc<E>) -> Result<SIArray1, EstimatorError> {
        Ok(self
            .temperature
            .into_iter()
            .map(|t| {
                if let Ok(state) = PhaseEquilibrium::pure(eos, t, None, self.solver_options) {
                    state.liquid().mass_density()
                } else {
                    f64::NAN * SIUnit::reference_mass() / SIUnit::reference_volume()
                }
            })
            .collect())
    }

    fn get_input(&self) -> HashMap<String, SIArray1> {
        let mut m = HashMap::with_capacity(2);
        m.insert("temperature".to_owned(), self.temperature());
        m
    }
}
