use super::{DataSet, EstimatorError};
use feos_core::{Contributions, EosUnit, EquationOfState, PhaseEquilibrium, SolverOptions, State};
use feos_dft::{HelmholtzEnergyFunctional, DFT};
use feos_dft::{DFTProfile, DFTSolver, interface::PlanarInterface};
use ndarray::Array1;
use quantity::{QuantityArray1, QuantityScalar};
use std::collections::HashMap;
use std::rc::Rc;

/// Store experimental surface tension data.
#[derive(Clone)]
pub struct SurfaceTension<U: EosUnit> {
    pub target: QuantityArray1<U>,
    temperature: QuantityArray1<U>,
    max_temperature: QuantityScalar<U>,
    datapoints: usize,
    grid: usize,
    width: QuantityScalar<U>,
    solver_options: SolverOptions,
}

impl<U: EosUnit> SurfaceTension<U> {
    /// Create a new data set for surface tension.
    ///
    /// If the equation of state fails to compute the surface tension
    /// (e.g. when it underestimates the critical point) the vapor
    /// pressure can be estimated.
    /// If `extrapolate` is `true`, the surface tension is estimated by
    /// calculating the slope of ln(p) over 1/T.
    /// If `extrapolate` is `false`, it is set to `NAN`.
    pub fn new(
        target: QuantityArray1<U>,
        temperature: QuantityArray1<U>,
        grid: usize,
        width: QuantityScalar<U>,
        solver_options: Option<SolverOptions>,
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
            grid,
            width,
            solver_options: solver_options.unwrap_or_default(),
        })
    }

    /// Return temperature.
    pub fn temperature(&self) -> QuantityArray1<U> {
        self.temperature.clone()
    }
}

impl<U: EosUnit, M: DFT<HelmholtzEnergyFunctional>> DataSet<U, M> for SurfaceTension<U> {
    fn target(&self) -> &QuantityArray1<U> {
        &self.target
    }

    fn target_str(&self) -> &str {
        "surface tension"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature"]
    }

    fn predict(&self, eos: &Rc<M>) -> Result<QuantityArray1<U>, EstimatorError>
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

        let unit = self.target.get(0);
        let mut prediction = Array1::zeros(self.datapoints) * unit;
        for i in 0..self.datapoints {
            let t = self.temperature.get(i);
            let vle = PhaseEquilibrium::pure(&Rc::new(eos.into()), t, None, SolverOptions::default());
            let gamma = vle
                .and_then(|vle| PlanarInterface::from_tanh(&vle, self.grid, self.width, tc))
                .and_then(|interface| interface.solve(None))
                .and_then(|interface| interface.surface_tension());
            
            if let Ok(gamma) = gamma {
                prediction.try_set(i, gamma)?
            } else {
                prediction.try_set(i, f64::NAN * U::reference_surface_tension())?
            }
        }
        Ok(prediction)
    }

    fn get_input(&self) -> HashMap<String, QuantityArray1<U>> {
        let mut m = HashMap::with_capacity(1);
        m.insert("temperature".to_owned(), self.temperature());
        m
    }
}
