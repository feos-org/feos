use crate::equation_of_state::Residual;
use crate::errors::{EosError, EosResult};
use crate::si::{Dimensionless, Energy, Moles, Pressure, Temperature, RGAS};
use crate::state::{DensityInitialization, State};
use crate::Contributions;
use ndarray::Array1;
use std::fmt;
use std::fmt::Write;
use std::sync::Arc;

mod bubble_dew;
mod phase_diagram_binary;
mod phase_diagram_pure;
mod phase_envelope;
mod stability_analysis;
mod tp_flash;
mod vle_pure;
pub use bubble_dew::TemperatureOrPressure;
pub use phase_diagram_binary::PhaseDiagramHetero;
pub use phase_diagram_pure::PhaseDiagram;

/// A thermodynamic equilibrium state.
///
/// The struct is parametrized over the number of phases with most features
/// being implemented for the two phase vapor/liquid or liquid/liquid case.
///
/// ## Contents
///
/// + [Bubble and dew point calculations](#bubble-and-dew-point-calculations)
/// + [Heteroazeotropes](#heteroazeotropes)
/// + [Flash calculations](#flash-calculations)
/// + [Pure component phase equilibria](#pure-component-phase-equilibria)
/// + [Utility functions](#utility-functions)
#[derive(Debug)]
pub struct PhaseEquilibrium<E, const N: usize>([State<E>; N]);

impl<E, const N: usize> Clone for PhaseEquilibrium<E, N> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<E: Residual, const N: usize> fmt::Display for PhaseEquilibrium<E, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, s) in self.0.iter().enumerate() {
            writeln!(f, "phase {}: {}", i, s)?;
        }
        Ok(())
    }
}

impl<E: Residual, const N: usize> PhaseEquilibrium<E, N> {
    pub fn _repr_markdown_(&self) -> String {
        if self.0[0].eos.components() == 1 {
            let mut res = "||temperature|density|\n|-|-|-|\n".to_string();
            for (i, s) in self.0.iter().enumerate() {
                writeln!(
                    res,
                    "|phase {}|{:.5}|{:.5}|",
                    i + 1,
                    s.temperature,
                    s.density
                )
                .unwrap();
            }
            res
        } else {
            let mut res = "||temperature|density|molefracs|\n|-|-|-|-|\n".to_string();
            for (i, s) in self.0.iter().enumerate() {
                writeln!(
                    res,
                    "|phase {}|{:.5}|{:.5}|{:.5}|",
                    i + 1,
                    s.temperature,
                    s.density,
                    s.molefracs
                )
                .unwrap();
            }
            res
        }
    }
}

impl<E> PhaseEquilibrium<E, 2> {
    pub fn vapor(&self) -> &State<E> {
        &self.0[0]
    }

    pub fn liquid(&self) -> &State<E> {
        &self.0[1]
    }
}

impl<E> PhaseEquilibrium<E, 3> {
    pub fn vapor(&self) -> &State<E> {
        &self.0[0]
    }

    pub fn liquid1(&self) -> &State<E> {
        &self.0[1]
    }

    pub fn liquid2(&self) -> &State<E> {
        &self.0[2]
    }
}

impl<E: Residual> PhaseEquilibrium<E, 2> {
    pub(super) fn from_states(state1: State<E>, state2: State<E>) -> Self {
        let (vapor, liquid) = if state1.density < state2.density {
            (state1, state2)
        } else {
            (state2, state1)
        };
        Self([vapor, liquid])
    }

    /// Creates a new PhaseEquilibrium that contains two states at the
    /// specified temperature, pressure and moles.
    ///
    /// The constructor can be used in custom phase equilibrium solvers or,
    /// e.g., to generate initial guesses for an actual VLE solver.
    /// In general, the two states generated are NOT in an equilibrium.
    pub fn new_npt(
        eos: &Arc<E>,
        temperature: Temperature,
        pressure: Pressure,
        vapor_moles: &Moles<Array1<f64>>,
        liquid_moles: &Moles<Array1<f64>>,
    ) -> EosResult<Self> {
        let liquid = State::new_npt(
            eos,
            temperature,
            pressure,
            liquid_moles,
            DensityInitialization::Liquid,
        )?;
        let vapor = State::new_npt(
            eos,
            temperature,
            pressure,
            vapor_moles,
            DensityInitialization::Vapor,
        )?;
        Ok(Self([vapor, liquid]))
    }

    pub(super) fn vapor_phase_fraction(&self) -> f64 {
        (self.vapor().total_moles / (self.vapor().total_moles + self.liquid().total_moles))
            .into_value()
    }
}

impl<E: Residual, const N: usize> PhaseEquilibrium<E, N> {
    pub(super) fn update_pressure(
        mut self,
        temperature: Temperature,
        pressure: Pressure,
    ) -> EosResult<Self> {
        for s in self.0.iter_mut() {
            *s = State::new_npt(
                &s.eos,
                temperature,
                pressure,
                &s.moles,
                DensityInitialization::InitialDensity(s.density),
            )?;
        }
        Ok(self)
    }

    pub(super) fn update_moles(
        &mut self,
        pressure: Pressure,
        moles: [&Moles<Array1<f64>>; N],
    ) -> EosResult<()> {
        for (i, s) in self.0.iter_mut().enumerate() {
            *s = State::new_npt(
                &s.eos,
                s.temperature,
                pressure,
                moles[i],
                DensityInitialization::InitialDensity(s.density),
            )?;
        }
        Ok(())
    }

    // Total Gibbs energy excluding the constant contribution RT sum_i N_i ln(\Lambda_i^3)
    pub(super) fn total_gibbs_energy(&self) -> Energy {
        self.0.iter().fold(Energy::from_reduced(0.0), |acc, s| {
            let ln_rho = s.partial_density.to_reduced().mapv(f64::ln);
            acc + s.residual_helmholtz_energy()
                + s.pressure(Contributions::Total) * s.volume
                + RGAS * s.temperature * (s.moles.clone() * Dimensionless::from(ln_rho - 1.0)).sum()
        })
    }
}

const TRIVIAL_REL_DEVIATION: f64 = 1e-5;

/// # Utility functions
impl<E: Residual> PhaseEquilibrium<E, 2> {
    pub(super) fn check_trivial_solution(self) -> EosResult<Self> {
        if Self::is_trivial_solution(self.vapor(), self.liquid()) {
            Err(EosError::TrivialSolution)
        } else {
            Ok(self)
        }
    }

    /// Check if the two states form a trivial solution
    pub fn is_trivial_solution(state1: &State<E>, state2: &State<E>) -> bool {
        let rho1 = state1.partial_density.to_reduced();
        let rho2 = state2.partial_density.to_reduced();

        rho1.iter()
            .zip(rho2.iter())
            .fold(0.0, |acc, (&rho1, &rho2)| {
                (rho2 / rho1 - 1.0).abs().max(acc)
            })
            < TRIVIAL_REL_DEVIATION
    }
}
