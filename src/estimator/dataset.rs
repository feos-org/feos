//! The [`DataSet`] trait provides routines that can be used for
//! optimization of parameters of equations of state given
//! a `target` which can be values from experimental data or
//! other models.
use super::{EstimatorError, Loss};
use feos_core::EosUnit;
use feos_core::EquationOfState;
use ndarray::Array1;
use quantity::{QuantityArray1, QuantityScalar};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Utilities for working with experimental data.
///
/// Functionalities in the context of optimizations of
/// parameters of equations of state.
pub trait DataSet<U: EosUnit, E: EquationOfState>: Send + Sync {
    /// Return target quantity.
    fn target(&self) -> &QuantityArray1<U>;

    /// Return the description of the target quantity.
    fn target_str(&self) -> &str;

    /// Return the descritions of the input quantities needed to compute the target.
    fn input_str(&self) -> Vec<&str>;

    /// Evaluation of the equation of state for the target quantity.
    fn predict(&self, eos: &Arc<E>) -> Result<QuantityArray1<U>, EstimatorError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp;

    /// Evaluate the cost function.
    fn cost(&self, eos: &Arc<E>, loss: Loss) -> Result<Array1<f64>, EstimatorError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let mut cost = self.relative_difference(eos)?;
        loss.apply(&mut cost);
        let datapoints = cost.len();
        Ok(cost / datapoints as f64)
    }

    /// Returns the input quantities as HashMap. The keys are the input's descriptions.
    fn get_input(&self) -> HashMap<String, QuantityArray1<U>>;

    /// Returns the number of experimental data points.
    fn datapoints(&self) -> usize {
        self.target().len()
    }

    /// Returns the relative difference between the equation of state and the experimental values.
    fn relative_difference(&self, eos: &Arc<E>) -> Result<Array1<f64>, EstimatorError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let prediction = &self.predict(eos)?;
        let target = self.target();
        Ok(((prediction - target) / target).into_value()?)
    }

    /// Returns the mean of the absolute relative difference between the equation of state and the experimental values.
    fn mean_absolute_relative_difference(&self, eos: &Arc<E>) -> Result<f64, EstimatorError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        Ok(self
            .relative_difference(eos)?
            .into_iter()
            .filter(|&x| x.is_finite())
            .enumerate()
            .fold(0.0, |mean, (i, x)| mean + (x.abs() - mean) / (i + 1) as f64))
    }
}

impl<U: EosUnit, E: EquationOfState> fmt::Display for dyn DataSet<U, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DataSet(target: {}, input: {}, datapoints: {}",
            self.target_str(),
            self.input_str().join(", "),
            self.datapoints()
        )
    }
}
