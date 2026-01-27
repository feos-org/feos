use super::PhaseEquilibrium;
#[cfg(feature = "rayon")]
use crate::ReferenceSystem;
use crate::SolverOptions;
use crate::equation_of_state::Residual;
use crate::errors::FeosResult;
use crate::state::{State, StateVec};
#[cfg(feature = "rayon")]
use ndarray::{Array1, ArrayView1, Axis};
use quantity::Temperature;
#[cfg(feature = "rayon")]
use rayon::{ThreadPool, prelude::*};

/// Pure component and binary mixture phase diagrams.
#[derive(Clone)]
pub struct PhaseDiagram<E, const N: usize> {
    pub states: Vec<PhaseEquilibrium<E, N>>,
}

impl<E, const N: usize> PhaseDiagram<E, N> {
    /// Create a phase diagram from a list of phase equilibria.
    pub fn new(states: Vec<PhaseEquilibrium<E, N>>) -> Self {
        Self { states }
    }
}

impl<E: Residual> PhaseDiagram<E, 2> {
    /// Calculate a phase diagram for a pure component.
    pub fn pure(
        eos: &E,
        min_temperature: Temperature,
        npoints: usize,
        critical_temperature: Option<Temperature>,
        options: SolverOptions,
    ) -> FeosResult<Self> {
        let mut states = Vec::with_capacity(npoints);

        let sc = State::critical_point(
            eos,
            (),
            critical_temperature,
            None,
            SolverOptions::default(),
        )?;

        let max_temperature = min_temperature
            + (sc.temperature - min_temperature) * ((npoints - 2) as f64 / (npoints - 1) as f64);
        let temperatures = Temperature::linspace(min_temperature, max_temperature, npoints - 1);

        let mut vle = None;
        for ti in &temperatures {
            vle = PhaseEquilibrium::pure(eos, ti, vle.as_ref(), options).ok();
            if let Some(vle) = vle.as_ref() {
                states.push(vle.clone());
            }
        }
        states.push(PhaseEquilibrium::from_states(sc.clone(), sc));

        Ok(PhaseDiagram::new(states))
    }

    /// Return the vapor states of the diagram.
    pub fn vapor(&self) -> StateVec<'_, E> {
        self.states.iter().map(|s| s.vapor()).collect()
    }

    /// Return the liquid states of the diagram.
    pub fn liquid(&self) -> StateVec<'_, E> {
        self.states.iter().map(|s| s.liquid()).collect()
    }
}

#[cfg(feature = "rayon")]
impl<E: Residual> PhaseDiagram<E, 2> {
    fn solve_temperatures(
        eos: &E,
        temperatures: ArrayView1<f64>,
        options: SolverOptions,
    ) -> FeosResult<Vec<PhaseEquilibrium<E, 2>>> {
        let mut states = Vec::with_capacity(temperatures.len());
        let mut vle = None;
        for ti in temperatures {
            vle =
                PhaseEquilibrium::pure(eos, Temperature::from_reduced(*ti), vle.as_ref(), options)
                    .ok();
            if let Some(vle) = vle.as_ref() {
                states.push(vle.clone());
            }
        }
        Ok(states)
    }

    pub fn par_pure(
        eos: &E,
        min_temperature: Temperature,
        npoints: usize,
        chunksize: usize,
        thread_pool: ThreadPool,
        critical_temperature: Option<Temperature>,
        options: SolverOptions,
    ) -> FeosResult<Self>
    where
        E: Send + Sync,
    {
        let sc = State::critical_point(
            eos,
            (),
            critical_temperature,
            None,
            SolverOptions::default(),
        )?;

        let max_temperature = min_temperature
            + (sc.temperature - min_temperature) * ((npoints - 2) as f64 / (npoints - 1) as f64);
        let temperatures = Array1::linspace(
            min_temperature.to_reduced(),
            max_temperature.to_reduced(),
            npoints - 1,
        );

        let mut states: Vec<PhaseEquilibrium<E, 2>> = thread_pool.install(|| {
            temperatures
                .axis_chunks_iter(Axis(0), chunksize)
                .into_par_iter()
                .filter_map(|t| Self::solve_temperatures(eos, t, options).ok())
                .flatten()
                .collect()
        });

        states.push(PhaseEquilibrium::from_states(sc.clone(), sc));
        Ok(PhaseDiagram::new(states))
    }
}
