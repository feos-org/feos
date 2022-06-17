use super::{PhaseEquilibrium, SolverOptions};
use crate::equation_of_state::EquationOfState;
use crate::errors::EosResult;
use crate::state::{State, StateVec};
use crate::EosUnit;
use quantity::{QuantityArray1, QuantityScalar};
use std::rc::Rc;

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
        eos: &Rc<E>,
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
