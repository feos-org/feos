use super::{DataSet, EstimatorError};
use feos_core::si::{Pressure, Temperature, PASCAL};
use feos_core::{Contributions, PhaseEquilibrium, Residual, SolverOptions, State};
use ndarray::{arr1, Array1};
use std::sync::Arc;

/// Store experimental vapor pressure data.
#[derive(Clone)]
pub struct VaporPressure {
    pub target: Array1<f64>,
    unit: Pressure,
    temperature: Temperature<Array1<f64>>,
    max_temperature: Temperature,
    datapoints: usize,
    extrapolate: bool,
    solver_options: SolverOptions,
}

impl VaporPressure {
    /// Create a new data set for vapor pressure.
    ///
    /// If the equation of state fails to compute the vapor pressure
    /// (e.g. when it underestimates the critical point) the vapor
    /// pressure can be estimated.
    /// If `extrapolate` is `true`, the vapor pressure is estimated by
    /// calculating the slope of ln(p) over 1/T.
    /// If `extrapolate` is `false`, it is set to `NAN`.
    pub fn new(
        target: Pressure<Array1<f64>>,
        temperature: Temperature<Array1<f64>>,
        extrapolate: bool,
        critical_temperature: Option<Temperature>,
        solver_options: Option<SolverOptions>,
    ) -> Self {
        let datapoints = target.len();
        let max_temperature = critical_temperature
            .unwrap_or(temperature.into_iter().reduce(|a, b| a.max(b)).unwrap());
        let target_unit = PASCAL;
        Self {
            target: (target / target_unit).into_value(),
            unit: target_unit,
            temperature,
            max_temperature,
            datapoints,
            extrapolate,
            solver_options: solver_options.unwrap_or_default(),
        }
    }

    /// Return temperature.
    pub fn temperature(&self) -> Temperature<Array1<f64>> {
        self.temperature.clone()
    }
}

impl<E: Residual> DataSet<E> for VaporPressure {
    fn target(&self) -> &Array1<f64> {
        &self.target
    }

    fn target_str(&self) -> &str {
        "vapor pressure"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature"]
    }

    fn predict(&self, eos: &Arc<E>) -> Result<Array1<f64>, EstimatorError> {
        if self.datapoints == 0 {
            return Ok(arr1(&[]));
        }

        let critical_point =
            State::critical_point(eos, None, Some(self.max_temperature), self.solver_options)
                .or_else(|_| State::critical_point(eos, None, None, self.solver_options))?;
        let tc = critical_point.temperature;
        let pc = critical_point.pressure(Contributions::Total);

        let t0 = 0.9 * tc;
        let p0 = PhaseEquilibrium::pure(eos, t0, None, self.solver_options)?
            .vapor()
            .pressure(Contributions::Total);

        let b = (pc / p0).into_value().ln() / (1.0 / tc - 1.0 / t0);
        let a = pc.to_reduced().ln() - (b / tc).into_value();

        Ok((0..self.datapoints)
            .map(|i| {
                let t = self.temperature.get(i);
                if let Some(p) = PhaseEquilibrium::vapor_pressure(eos, t)[0] {
                    (p / self.unit).into_value()
                } else if self.extrapolate {
                    (a + (b / t).into_value()).exp() / self.unit.to_reduced()
                } else {
                    f64::NAN
                }
            })
            .collect())
    }

    // fn get_input(&self) -> HashMap<String, SIArray1> {
    //     let mut m = HashMap::with_capacity(1);
    //     m.insert("temperature".to_owned(), self.temperature());
    //     m
    // }
}
