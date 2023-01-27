use super::{DataSet, EstimatorError};
use feos_core::{Contributions, EosUnit, EquationOfState, PhaseEquilibrium, SolverOptions, State};
use ndarray::{arr1, Array1};
use quantity::si::{SIArray1, SINumber, SIUnit};
use std::collections::HashMap;
use std::sync::Arc;

/// Store experimental vapor pressure data.
#[derive(Clone)]
pub struct VaporPressure {
    pub target: SIArray1,
    temperature: SIArray1,
    max_temperature: SINumber,
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
        target: SIArray1,
        temperature: SIArray1,
        extrapolate: bool,
        critical_temperature: Option<SINumber>,
        solver_options: Option<SolverOptions>,
    ) -> Result<Self, EstimatorError> {
        let datapoints = target.len();
        let max_temperature = critical_temperature.unwrap_or(
            temperature
                .to_reduced(SIUnit::reference_temperature())?
                .into_iter()
                .reduce(|a, b| a.max(b))
                .unwrap()
                * SIUnit::reference_temperature(),
        );
        Ok(Self {
            target,
            temperature,
            max_temperature,
            datapoints,
            extrapolate,
            solver_options: solver_options.unwrap_or_default(),
        })
    }

    /// Return temperature.
    pub fn temperature(&self) -> SIArray1 {
        self.temperature.clone()
    }
}

impl<E: EquationOfState> DataSet<E> for VaporPressure {
    fn target(&self) -> &SIArray1 {
        &self.target
    }

    fn target_str(&self) -> &str {
        "vapor pressure"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature"]
    }

    fn predict(&self, eos: &Arc<E>) -> Result<SIArray1, EstimatorError> {
        if self.datapoints == 0 {
            return Ok(arr1(&[]) * SIUnit::reference_pressure());
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

        let b = pc.to_reduced(p0)?.ln() / (1.0 / tc - 1.0 / t0);
        let a = pc.to_reduced(SIUnit::reference_pressure())?.ln() - b.to_reduced(tc)?;

        let unit = self.target.get(0);
        let mut prediction = Array1::zeros(self.datapoints) * unit;
        for i in 0..self.datapoints {
            let t = self.temperature.get(i);
            if let Some(pvap) = PhaseEquilibrium::vapor_pressure(eos, t)[0] {
                prediction.try_set(i, pvap)?;
            } else if self.extrapolate {
                prediction.try_set(
                    i,
                    (a + b.to_reduced(t)?).exp() * SIUnit::reference_pressure(),
                )?;
            } else {
                prediction.try_set(i, f64::NAN * SIUnit::reference_pressure())?
            }
        }
        Ok(prediction)
    }

    fn get_input(&self) -> HashMap<String, SIArray1> {
        let mut m = HashMap::with_capacity(1);
        m.insert("temperature".to_owned(), self.temperature());
        m
    }
}
