use crate::equation_of_state::EquationOfState;
use crate::errors::{EosError, EosResult};
use crate::state::{Contributions, DensityInitialization, State};
use crate::EosUnit;
use quantity::si::{SIArray1, SINumber, SIUnit};
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
pub use phase_diagram_binary::PhaseDiagramHetero;
pub use phase_diagram_pure::PhaseDiagram;

/// Level of detail in the iteration output.
#[derive(Copy, Clone, PartialOrd, PartialEq, Eq)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub enum Verbosity {
    /// Do not print output.
    None,
    /// Print information about the success of failure of the iteration.
    Result,
    /// Print a detailed outpur for every iteration.
    Iter,
}

impl Default for Verbosity {
    fn default() -> Self {
        Self::None
    }
}

/// Options for the various phase equilibria solvers.
///
/// If the values are [None], solver specific default
/// values are used.
#[derive(Copy, Clone, Default)]
pub struct SolverOptions {
    /// Maximum number of iterations.
    pub max_iter: Option<usize>,
    /// Tolerance.
    pub tol: Option<f64>,
    /// Iteration outpput indicated by the [Verbosity] enum.
    pub verbosity: Verbosity,
}

impl From<(Option<usize>, Option<f64>, Option<Verbosity>)> for SolverOptions {
    fn from(options: (Option<usize>, Option<f64>, Option<Verbosity>)) -> Self {
        Self {
            max_iter: options.0,
            tol: options.1,
            verbosity: options.2.unwrap_or(Verbosity::None),
        }
    }
}

impl SolverOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = Some(max_iter);
        self
    }

    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = Some(tol);
        self
    }

    pub fn verbosity(mut self, verbosity: Verbosity) -> Self {
        self.verbosity = verbosity;
        self
    }

    pub fn unwrap_or(self, max_iter: usize, tol: f64) -> (usize, f64, Verbosity) {
        (
            self.max_iter.unwrap_or(max_iter),
            self.tol.unwrap_or(tol),
            self.verbosity,
        )
    }
}

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

impl<E, const N: usize> fmt::Display for PhaseEquilibrium<E, N>
where
    SINumber: fmt::Display,
    SIArray1: fmt::Display,
    E: EquationOfState,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, s) in self.0.iter().enumerate() {
            writeln!(f, "phase {}: {}", i, s)?;
        }
        Ok(())
    }
}

impl<E, const N: usize> PhaseEquilibrium<E, N>
where
    SINumber: fmt::Display,
    SIArray1: fmt::Display,
    E: EquationOfState,
{
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

impl<E: EquationOfState> PhaseEquilibrium<E, 2> {
    pub fn vapor(&self) -> &State<E> {
        &self.0[0]
    }

    pub fn liquid(&self) -> &State<E> {
        &self.0[1]
    }
}

impl<E: EquationOfState> PhaseEquilibrium<E, 3> {
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

impl<E: EquationOfState> PhaseEquilibrium<E, 2> {
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
        temperature: SINumber,
        pressure: SINumber,
        vapor_moles: &SIArray1,
        liquid_moles: &SIArray1,
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
            .unwrap()
    }
}

impl<E: EquationOfState, const N: usize> PhaseEquilibrium<E, N> {
    pub(super) fn update_pressure(
        mut self,
        temperature: SINumber,
        pressure: SINumber,
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
        pressure: SINumber,
        moles: [&SIArray1; N],
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

    pub fn update_chemical_potential(&mut self, chemical_potential: &SIArray1) -> EosResult<()> {
        for s in self.0.iter_mut() {
            s.update_chemical_potential(chemical_potential)?;
        }
        Ok(())
    }

    pub(super) fn total_gibbs_energy(&self) -> SINumber {
        self.0
            .iter()
            .fold(0.0 * SIUnit::reference_energy(), |acc, s| {
                acc + s.gibbs_energy(Contributions::Total)
            })
    }
}

const TRIVIAL_REL_DEVIATION: f64 = 1e-5;

/// # Utility functions
impl<E: EquationOfState> PhaseEquilibrium<E, 2> {
    pub(super) fn check_trivial_solution(self) -> EosResult<Self> {
        if Self::is_trivial_solution(self.vapor(), self.liquid()) {
            Err(EosError::TrivialSolution)
        } else {
            Ok(self)
        }
    }

    /// Check if the two states form a trivial solution
    pub fn is_trivial_solution(state1: &State<E>, state2: &State<E>) -> bool {
        let rho1 = state1
            .partial_density
            .to_reduced(SIUnit::reference_density())
            .unwrap();
        let rho2 = state2
            .partial_density
            .to_reduced(SIUnit::reference_density())
            .unwrap();

        rho1.iter()
            .zip(rho2.iter())
            .fold(0.0, |acc, (&rho1, &rho2)| {
                (rho2 / rho1 - 1.0).abs().max(acc)
            })
            < TRIVIAL_REL_DEVIATION
    }
}
