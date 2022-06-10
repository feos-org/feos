use super::{PhaseEquilibrium, SolverOptions};
use crate::equation_of_state::EquationOfState;
use crate::errors::EosResult;
use crate::state::{State, StateVec};
use crate::{Contributions, EosUnit};
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

    pub fn mix(
        eos: &Rc<E>,
        moles: &QuantityArray1<U>,
        min_pressure: QuantityScalar<U>,
        npoints: usize,
        critical_temperature: Option<QuantityScalar<U>>,
        options: (SolverOptions, SolverOptions),
    ) -> EosResult<Self>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let mut states = Vec::with_capacity(npoints);

        let sc = State::critical_point(
            eos,
            Some(moles),
            critical_temperature,
            SolverOptions::default(),
        )?;

        let max_pressure = min_pressure
            + (sc.pressure(Contributions::Total) - min_pressure)
                * ((npoints - 2) as f64 / (npoints - 1) as f64);
        let pressures = QuantityArray1::linspace(min_pressure, max_pressure, npoints - 1)?;
        let molefracs = moles.to_reduced(moles.sum())?;

        let mut vle_liquid: Option<PhaseEquilibrium<U, E, 2>> = None;
        let mut vle_vapor: Option<PhaseEquilibrium<U, E, 2>> = None;
        for ti in &pressures {
            // calculate new liquid point
            let t_init = vle_liquid.as_ref().map(|vle| vle.vapor().temperature);
            let vapor_molefracs = vle_liquid.as_ref().map(|vle| &vle.vapor().molefracs);
            vle_liquid = PhaseEquilibrium::bubble_point(
                eos,
                ti,
                &molefracs,
                t_init,
                vapor_molefracs,
                options,
            )
            .ok();

            // calculate new vapor point
            let t_init = vle_vapor.as_ref().map(|vle| vle.liquid().temperature);
            let liquid_molefracs = vle_vapor.as_ref().map(|vle| &vle.liquid().molefracs);
            vle_vapor =
                PhaseEquilibrium::dew_point(eos, ti, &molefracs, t_init, liquid_molefracs, options)
                    .ok();
            if let (Some(vle_liquid), Some(vle_vapor)) = (vle_liquid.as_ref(), vle_vapor.as_ref()) {
                states.push(PhaseEquilibrium::from_states(
                    vle_liquid.liquid().clone(),
                    vle_vapor.vapor().clone(),
                ));
            }
        }
        states.push(PhaseEquilibrium::from_states(sc.clone(), sc));

        Ok(PhaseDiagram { states })
    }
}
