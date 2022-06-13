use super::{DataSet, EstimatorError, Loss};
use feos_core::{Contributions, EosUnit, EquationOfState, PhaseEquilibrium, SolverOptions, State};
use ndarray::Array1;
use quantity::{QuantityArray1, QuantityScalar};
use std::collections::HashMap;
use std::rc::Rc;

/// Store experimental vapor pressure data.
#[derive(Clone)]
pub struct VaporPressure<U: EosUnit> {
    pub target: QuantityArray1<U>,
    temperature: QuantityArray1<U>,
    max_temperature: QuantityScalar<U>,
    datapoints: usize,
    extrapolate: bool,
}

impl<U: EosUnit> VaporPressure<U> {
    /// Create a new data set for vapor pressure.
    ///
    /// If the equation of state fails to compute the vapor pressure
    /// (e.g. when it underestimates the critical point) the vapor
    /// pressure can be estimated.
    /// If `extrapolate` is `true`, the vapor pressure is estimated by
    /// calculating the slope of ln(p) over 1/T.
    /// If `extrapolate` is `false`, it is set to `NAN`.
    pub fn new(
        target: QuantityArray1<U>,
        temperature: QuantityArray1<U>,
        extrapolate: bool,
    ) -> Result<Self, EstimatorError> {
        let datapoints = target.len();
        let max_temperature = temperature
            .to_reduced(U::reference_temperature())?
            .into_iter()
            .reduce(|a, b| a.max(b))
            .unwrap()
            * U::reference_temperature();
        Ok(Self {
            target,
            temperature,
            max_temperature,
            datapoints,
            extrapolate,
        })
    }

    /// Return temperature.
    pub fn temperature(&self) -> QuantityArray1<U> {
        self.temperature.clone()
    }
}

impl<U: EosUnit, E: EquationOfState> DataSet<U, E> for VaporPressure<U> {
    fn target(&self) -> QuantityArray1<U> {
        self.target.clone()
    }

    fn target_str(&self) -> &str {
        "vapor pressure"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature"]
    }

    fn predict(&self, eos: &Rc<E>) -> Result<QuantityArray1<U>, EstimatorError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let critical_point = State::critical_point(
            eos,
            None,
            Some(self.max_temperature),
            SolverOptions::default(),
        )?;
        let tc = critical_point.temperature;
        let pc = critical_point.pressure(Contributions::Total);

        let t0 = 0.9 * tc;
        let p0 = PhaseEquilibrium::pure(eos, t0, None, SolverOptions::default())?
            .vapor()
            .pressure(Contributions::Total);

        let b = pc.to_reduced(p0)?.ln() / (1.0 / tc - 1.0 / t0);
        let a = pc.to_reduced(U::reference_pressure())?.ln() - b.to_reduced(tc)?;

        let unit = self.target.get(0);
        let mut prediction = Array1::zeros(self.datapoints) * unit;
        for i in 0..self.datapoints {
            let t = self.temperature.get(i);
            if let Some(pvap) = PhaseEquilibrium::vapor_pressure(eos, t)[0] {
                prediction.try_set(i, pvap)?;
            } else if self.extrapolate {
                prediction.try_set(i, (a + b.to_reduced(t)?).exp() * U::reference_pressure())?;
            } else {
                prediction.try_set(i, f64::NAN * U::reference_pressure())?
            }
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
        m
    }
}
