use super::{DataSet, EstimatorError, Loss};
use feos_core::{DensityInitialization, EntropyScaling, EosUnit, EquationOfState, State};
use ndarray::{arr1, Array1};
use quantity::{QuantityArray1, QuantityScalar};
use std::collections::HashMap;
use std::rc::Rc;

/// Store experimental viscosity data.
#[derive(Clone)]
pub struct Viscosity<U: EosUnit> {
    pub target: QuantityArray1<U>,
    temperature: QuantityArray1<U>,
    pressure: QuantityArray1<U>,
    datapoints: usize,
}

impl<U: EosUnit> Viscosity<U> {
    /// Create a new data set for experimental viscosity data.
    pub fn new(
        target: QuantityArray1<U>,
        temperature: QuantityArray1<U>,
        pressure: QuantityArray1<U>,
    ) -> Result<Self, EstimatorError> {
        let datapoints = target.len();
        Ok(Self {
            target,
            temperature,
            pressure,
            datapoints,
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
    fn target(&self) -> QuantityArray1<U> {
        self.target.clone()
    }

    fn target_str(&self) -> &str {
        "viscosity"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature", "pressure"]
    }

    fn predict(&self, eos: &Rc<E>) -> Result<QuantityArray1<U>, EstimatorError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let unit = self.target.get(0);
        let mut prediction = Array1::zeros(self.datapoints) * unit;
        let moles = arr1(&[1.0]) * U::reference_moles();
        for i in 0..self.datapoints {
            let t = self.temperature.get(i);
            let p = self.pressure.get(i);
            let state = State::new_npt(eos, t, p, &moles, DensityInitialization::None)?;
            prediction.try_set(i, state.viscosity()?)?;
        }
        Ok(prediction)
    }

    fn cost(&self, eos: &Rc<E>, loss: Loss) -> Result<Array1<f64>, EstimatorError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let mut cost = self.relative_difference(eos)?;
        loss.apply(&mut cost.view_mut());
        Ok(cost / self.datapoints as f64)
    }

    fn get_input(&self) -> HashMap<String, QuantityArray1<U>> {
        let mut m = HashMap::with_capacity(1);
        m.insert("temperature".to_owned(), self.temperature());
        m.insert("pressure".to_owned(), self.pressure());
        m
    }
}
