use super::{PhaseDiagram, PhaseEquilibrium, SolverOptions};
use crate::equation_of_state::EquationOfState;
use crate::errors::EosResult;
use crate::state::State;
use crate::{Contributions, EosUnit};
use quantity::{QuantityArray1, QuantityScalar};
use std::rc::Rc;

impl<U: EosUnit, E: EquationOfState> PhaseDiagram<U, E> {
    /// Calculate the bubble point line of a mixture with given composition.
    pub fn bubble_point_line(
        eos: &Rc<E>,
        moles: &QuantityArray1<U>,
        min_temperature: QuantityScalar<U>,
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

        let max_temperature = min_temperature
            + (sc.temperature - min_temperature) * ((npoints - 2) as f64 / (npoints - 1) as f64);
        let temperatures = QuantityArray1::linspace(min_temperature, max_temperature, npoints - 1)?;
        let molefracs = moles.to_reduced(moles.sum())?;

        let mut vle: Option<PhaseEquilibrium<U, E, 2>> = None;
        for ti in &temperatures {
            // calculate new liquid point
            let p_init = vle
                .as_ref()
                .map(|vle| vle.vapor().pressure(Contributions::Total));
            let vapor_molefracs = vle.as_ref().map(|vle| &vle.vapor().molefracs);
            vle = PhaseEquilibrium::bubble_point(
                eos,
                ti,
                &molefracs,
                p_init,
                vapor_molefracs,
                options,
            )
            .ok();

            if let Some(vle) = vle.as_ref() {
                states.push(vle.clone());
            }
        }
        states.push(PhaseEquilibrium::from_states(sc.clone(), sc));

        Ok(PhaseDiagram { states })
    }

    /// Calculate the dew point line of a mixture with given composition.
    pub fn dew_point_line(
        eos: &Rc<E>,
        moles: &QuantityArray1<U>,
        min_temperature: QuantityScalar<U>,
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

        let n_t = npoints / 2;
        let max_temperature = min_temperature
            + (sc.temperature - min_temperature) * ((n_t - 2) as f64 / (n_t - 1) as f64);
        let temperatures = QuantityArray1::linspace(min_temperature, max_temperature, n_t - 1)?;
        let molefracs = moles.to_reduced(moles.sum())?;

        let mut vle: Option<PhaseEquilibrium<U, E, 2>> = None;
        for ti in &temperatures {
            let p_init = vle
                .as_ref()
                .map(|vle| vle.vapor().pressure(Contributions::Total));
            let liquid_molefracs = vle.as_ref().map(|vle| &vle.liquid().molefracs);
            vle =
                PhaseEquilibrium::dew_point(eos, ti, &molefracs, p_init, liquid_molefracs, options)
                    .ok();
            if let Some(vle) = vle.as_ref() {
                states.push(vle.clone());
            }
        }

        let n_p = npoints - n_t;
        if vle.is_none() {
            return Ok(PhaseDiagram { states });
        }

        let min_pressure = vle.as_ref().unwrap().vapor().pressure(Contributions::Total);
        let p_c = sc.pressure(Contributions::Total);
        let max_pressure =
            min_pressure + (p_c - min_pressure) * ((n_p - 2) as f64 / (n_p - 1) as f64);
        let pressures = QuantityArray1::linspace(min_pressure, max_pressure, n_p)?;

        for pi in &pressures {
            let t_init = vle.as_ref().map(|vle| vle.vapor().temperature);
            let liquid_molefracs = vle.as_ref().map(|vle| &vle.liquid().molefracs);
            vle =
                PhaseEquilibrium::dew_point(eos, pi, &molefracs, t_init, liquid_molefracs, options)
                    .ok();
            if let Some(vle) = vle.as_ref() {
                states.push(vle.clone());
            }
        }

        states.push(PhaseEquilibrium::from_states(sc.clone(), sc));

        Ok(PhaseDiagram { states })
    }

    /// Calculate the spinodal lines for a mixture with fixed composition.
    pub fn spinodal(
        eos: &Rc<E>,
        moles: &QuantityArray1<U>,
        min_temperature: QuantityScalar<U>,
        npoints: usize,
        critical_temperature: Option<QuantityScalar<U>>,
        options: SolverOptions,
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

        let max_temperature = min_temperature
            + (sc.temperature - min_temperature) * ((npoints - 2) as f64 / (npoints - 1) as f64);
        let temperatures = QuantityArray1::linspace(min_temperature, max_temperature, npoints - 1)?;

        for ti in &temperatures {
            let spinodal = State::spinodal(eos, ti, Some(moles), options).ok();
            if let Some(spinodal) = spinodal {
                states.push(PhaseEquilibrium(spinodal));
            }
        }
        states.push(PhaseEquilibrium::from_states(sc.clone(), sc));

        Ok(PhaseDiagram { states })
    }
}
