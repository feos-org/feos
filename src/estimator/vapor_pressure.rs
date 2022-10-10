use super::{DataSet, EstimatorError};
use feos_core::{Contributions, EosUnit, EquationOfState, PhaseEquilibrium, SolverOptions, State};
use ndarray::Array1;
use quantity::{Quantity, QuantityArray1, QuantityScalar};
#[cfg(feature = "rayon")]
use rayon_::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Store experimental vapor pressure data.
pub struct VaporPressure<U: EosUnit> {
    pub target: QuantityArray1<U>,
    temperature: QuantityArray1<U>,
    max_temperature: QuantityScalar<U>,
    extrapolate: bool,
    solver_options: SolverOptions,
}

impl<U: EosUnit> VaporPressure<U> {
    /// Create a new data set for vapor pressure data.
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
        solver_options: Option<SolverOptions>,
    ) -> Result<Self, EstimatorError> {
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
            extrapolate,
            solver_options: solver_options.unwrap_or_default(),
        })
    }

    /// Return temperature.
    pub fn temperature(&self) -> QuantityArray1<U> {
        self.temperature.clone()
    }
}

impl<U: EosUnit, E: EquationOfState> DataSet<U, E> for VaporPressure<U> {
    fn target(&self) -> &QuantityArray1<U> {
        &self.target
    }

    fn target_str(&self) -> &str {
        "vapor pressure"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature"]
    }

    fn predict(&self, eos: &Arc<E>) -> Result<QuantityArray1<U>, EstimatorError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let critical_point =
            State::critical_point(eos, None, Some(self.max_temperature), self.solver_options)?;
        let tc = critical_point.temperature;
        let pc = critical_point.pressure(Contributions::Total);

        let t0 = 0.9 * tc;
        let p0 = PhaseEquilibrium::pure(eos, t0, None, self.solver_options)?
            .vapor()
            .pressure(Contributions::Total);

        let b = pc.to_reduced(p0)?.ln() / (1.0 / tc - 1.0 / t0);
        let a = pc.to_reduced(U::reference_pressure())?.ln() - b.to_reduced(tc)?;

        let ts = self
            .temperature
            .to_reduced(U::reference_temperature())
            .unwrap();

        // let res = ts
        //     .iter()
        //     .map(|&t| self.vapor_pressure(eos, a, b, t))
        //     .collect();
        #[cfg(feature = "rayon")]
        let ts_iter = ts.par_iter();
        #[cfg(not(feature = "rayon"))]
        let ts_iter = ts.iter();
        let res = ts_iter
            .map(|&t| {
                if let Some(pvap) =
                    PhaseEquilibrium::vapor_pressure(eos, t * U::reference_temperature())[0]
                {
                    pvap.to_reduced(U::reference_pressure()).unwrap()
                } else if self.extrapolate {
                    (a + b.to_reduced(t * U::reference_temperature()).unwrap()).exp()
                } else {
                    f64::NAN
                }
            })
            .collect();
        Ok(Array1::from_vec(res) * U::reference_pressure())
    }

    // fn par_predict(&self, eos: &Arc<E>) -> Result<QuantityArray1<U>, EstimatorError>
    // where
    //     QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    // {
    //     let critical_point =
    //         State::critical_point(eos, None, Some(self.max_temperature), self.solver_options)?;
    //     let tc = critical_point.temperature;
    //     let pc = critical_point.pressure(Contributions::Total);

    //     let t0 = 0.9 * tc;
    //     let p0 = PhaseEquilibrium::pure(eos, t0, None, self.solver_options)?
    //         .vapor()
    //         .pressure(Contributions::Total);

    //     let b = pc.to_reduced(p0)?.ln() / (1.0 / tc - 1.0 / t0);
    //     let a = pc.to_reduced(U::reference_pressure())?.ln() - b.to_reduced(tc)?;

    //     let ts = self
    //         .temperature
    //         .to_reduced(U::reference_temperature())
    //         .unwrap();

    //     let res = ts
    //         .into_par_iter()
    //         .map(|&t| {
    //             if let Some(pvap) =
    //                 PhaseEquilibrium::vapor_pressure(eos, t * U::reference_temperature())[0]
    //             {
    //                 pvap.to_reduced(U::reference_pressure()).unwrap()
    //             } else if self.extrapolate {
    //                 (a + b.to_reduced(t * U::reference_temperature()).unwrap()).exp()
    //             } else {
    //                 f64::NAN
    //             }
    //         })
    //         .collect();
    //     Ok(Array1::from_vec(res) * U::reference_pressure())
    // }

    fn get_input(&self) -> HashMap<String, QuantityArray1<U>> {
        let mut m = HashMap::with_capacity(1);
        m.insert("temperature".to_owned(), self.temperature());
        m
    }
}
