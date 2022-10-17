use super::{DataSet, EstimatorError};
use feos_core::{
    DensityInitialization, EosUnit, EquationOfState, MolarWeight, PhaseEquilibrium, SolverOptions,
    State,
};
use ndarray::arr1;
use quantity::{QuantityArray1, QuantityScalar};
#[cfg(feature = "rayon")]
use rayon_::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Liquid mass density data as function of pressure and temperature.
#[derive(Clone)]
pub struct LiquidDensity<U: EosUnit> {
    /// mass density
    pub target: QuantityArray1<U>,
    /// temperature
    temperature: QuantityArray1<U>,
    /// pressure
    pressure: QuantityArray1<U>,
}

impl<U: EosUnit> LiquidDensity<U> {
    /// A new data set for liquid densities with pressures and temperatures as input.
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

    /// Returns temperature of data points.
    pub fn temperature(&self) -> QuantityArray1<U> {
        self.temperature.clone()
    }

    /// Returns pressure of data points.
    pub fn pressure(&self) -> QuantityArray1<U> {
        self.pressure.clone()
    }
}

impl<U: EosUnit, E: EquationOfState + MolarWeight<U>> DataSet<U, E> for LiquidDensity<U> {
    fn target(&self) -> &QuantityArray1<U> {
        &self.target
    }

    fn target_str(&self) -> &str {
        "liquid density"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature", "pressure"]
    }

    fn predict(&self, eos: &Arc<E>) -> Result<QuantityArray1<U>, EstimatorError> {
        let moles = arr1(&[1.0]) * U::reference_moles();
        Ok(self
            .temperature
            .into_iter()
            .zip(self.pressure.into_iter())
            .map(|(t, p)| {
                let state = State::new_npt(eos, t, p, &moles, DensityInitialization::Liquid);
                if let Ok(s) = state {
                    s.mass_density()
                } else {
                    f64::NAN * U::reference_mass() / U::reference_volume()
                }
            })
            .collect())
    }

    fn get_input(&self) -> HashMap<String, QuantityArray1<U>> {
        let mut m = HashMap::with_capacity(2);
        m.insert("temperature".to_owned(), self.temperature());
        m.insert("pressure".to_owned(), self.pressure());
        m
    }
}

/// Store experimental data of liquid densities and compare to the equation of state.
#[derive(Clone)]
pub struct EquilibriumLiquidDensity<U: EosUnit> {
    pub target: QuantityArray1<U>,
    temperature: QuantityArray1<U>,
    solver_options: SolverOptions,
}

impl<U: EosUnit> EquilibriumLiquidDensity<U> {
    /// A new data set for liquid densities with pressures and temperatures as input.
    pub fn new(
        target: QuantityArray1<U>,
        temperature: QuantityArray1<U>,
        vle_options: Option<SolverOptions>,
    ) -> Result<Self, EstimatorError> {
        Ok(Self {
            target,
            temperature,
            solver_options: vle_options.unwrap_or_default(),
        })
    }

    /// Returns temperature of data points.
    pub fn temperature(&self) -> QuantityArray1<U> {
        self.temperature.clone()
    }
}

impl<U: EosUnit, E: EquationOfState + MolarWeight<U>> DataSet<U, E>
    for EquilibriumLiquidDensity<U>
{
    fn target(&self) -> &QuantityArray1<U> {
        &self.target
    }

    fn target_str(&self) -> &str {
        "equilibrium liquid density"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature"]
    }

    fn predict(&self, eos: &Arc<E>) -> Result<QuantityArray1<U>, EstimatorError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        Ok(self
            .temperature
            .into_iter()
            .map(|t| {
                if let Ok(state) = PhaseEquilibrium::pure(eos, t, None, self.solver_options) {
                    state.liquid().mass_density()
                } else {
                    f64::NAN * U::reference_mass() / U::reference_volume()
                }
            })
            .collect())
    }

    fn get_input(&self) -> HashMap<String, QuantityArray1<U>> {
        let mut m = HashMap::with_capacity(2);
        m.insert("temperature".to_owned(), self.temperature());
        m
    }
}
