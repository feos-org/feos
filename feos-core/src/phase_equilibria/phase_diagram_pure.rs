use super::{PhaseEquilibrium, SolverOptions};
use crate::equation_of_state::EquationOfState;
use crate::errors::EosResult;
use crate::state::{State, StateVec};
use crate::EosUnit;
#[cfg(feature = "rayon")]
use ndarray::{Array1, ArrayView1, Axis};
use quantity::{QuantityArray1, QuantityScalar};
#[cfg(feature = "rayon")]
use rayon_::{prelude::*, ThreadPool};
use std::sync::Arc;

/// Pure component and binary mixture phase diagrams.
pub struct PhaseDiagram<U, E> {
    pub states: Vec<PhaseEquilibrium<U, E, 2>>,
}

impl<U: Clone, E> Clone for PhaseDiagram<U, E> {
    fn clone(&self) -> Self {
        Self {
            states: self.states.clone(),
        }
    }
}

impl<U: EosUnit, E: EquationOfState> PhaseDiagram<U, E> {
    /// Calculate a phase diagram for a pure component.
    pub fn pure(
        eos: &Arc<E>,
        min_temperature: QuantityScalar<U>,
        npoints: usize,
        critical_temperature: Option<QuantityScalar<U>>,
        options: SolverOptions,
    ) -> EosResult<Self>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let mut states = Vec::with_capacity(npoints);

        let sc = State::critical_point(eos, None, critical_temperature, SolverOptions::default())?;

        let max_temperature = min_temperature
            + (sc.temperature - min_temperature) * ((npoints - 2) as f64 / (npoints - 1) as f64);
        let temperatures = QuantityArray1::linspace(min_temperature, max_temperature, npoints - 1)?;

        let mut vle = None;
        for ti in &temperatures {
            vle = PhaseEquilibrium::pure(eos, ti, vle.as_ref(), options).ok();
            if let Some(vle) = vle.as_ref() {
                states.push(vle.clone());
            }
        }
        states.push(PhaseEquilibrium::from_states(sc.clone(), sc));

        Ok(PhaseDiagram { states })
    }

    /// Return the vapor states of the diagram.
    pub fn vapor(&self) -> StateVec<'_, U, E> {
        self.states.iter().map(|s| s.vapor()).collect()
    }

    /// Return the liquid states of the diagram.
    pub fn liquid(&self) -> StateVec<'_, U, E> {
        self.states.iter().map(|s| s.liquid()).collect()
    }
}

#[cfg(feature = "rayon")]
impl<U: EosUnit, E: EquationOfState> PhaseDiagram<U, E> {
    fn solve_temperatures(
        eos: &Arc<E>,
        temperatures: ArrayView1<f64>,
        options: SolverOptions,
    ) -> EosResult<Vec<PhaseEquilibrium<U, E, 2>>>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let mut states = Vec::with_capacity(temperatures.len());
        let mut vle = None;
        for ti in temperatures {
            vle = PhaseEquilibrium::pure(
                eos,
                *ti * U::reference_temperature(),
                vle.as_ref(),
                options,
            )
            .ok();
            if let Some(vle) = vle.as_ref() {
                states.push(vle.clone());
            }
        }
        Ok(states)
    }

    pub fn par_pure(
        eos: &Arc<E>,
        min_temperature: QuantityScalar<U>,
        npoints: usize,
        chunksize: usize,
        thread_pool: ThreadPool,
        critical_temperature: Option<QuantityScalar<U>>,
        options: SolverOptions,
    ) -> EosResult<Self>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let sc = State::critical_point(eos, None, critical_temperature, SolverOptions::default())?;

        let max_temperature = min_temperature
            + (sc.temperature - min_temperature) * ((npoints - 2) as f64 / (npoints - 1) as f64);
        let temperatures = Array1::linspace(
            min_temperature.to_reduced(U::reference_temperature())?,
            max_temperature.to_reduced(U::reference_temperature())?,
            npoints - 1,
        );

        let mut states: Vec<PhaseEquilibrium<U, E, 2>> = thread_pool.install(|| {
            temperatures
                .axis_chunks_iter(Axis(0), chunksize)
                .into_par_iter()
                .filter_map(|t| Self::solve_temperatures(eos, t, options).ok())
                .flatten()
                .collect()
        });

        states.push(PhaseEquilibrium::from_states(sc.clone(), sc));
        Ok(PhaseDiagram { states })
    }
}
