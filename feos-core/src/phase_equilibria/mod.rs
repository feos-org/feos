use crate::equation_of_state::EquationOfState;
use crate::errors::{EosError, EosResult};
use crate::state::{Contributions, DensityInitialization, State};
use crate::EosUnit;
use quantity::{QuantityArray1, QuantityScalar};
use std::fmt;
use std::fmt::Write;
use std::rc::Rc;

mod bubble_dew;
mod phase_diagram_binary;
mod phase_diagram_pure;
mod stability_analysis;
mod tp_flash;
mod vle_pure;
pub use phase_diagram_binary::PhaseDiagramHetero;
pub use phase_diagram_pure::PhaseDiagram;

/// Level of detail in the iteration output.
#[derive(Copy, Clone, PartialOrd, PartialEq)]
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
///  values are used.
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
pub struct PhaseEquilibrium<U, E, const N: usize>([State<U, E>; N]);

impl<U: Clone, E, const N: usize> Clone for PhaseEquilibrium<U, E, N> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<U, E, const N: usize> fmt::Display for PhaseEquilibrium<U, E, N>
where
    QuantityScalar<U>: fmt::Display,
    QuantityArray1<U>: fmt::Display,
    E: EquationOfState,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, s) in self.0.iter().enumerate() {
            writeln!(f, "phase {}: {}", i, s)?;
        }
        Ok(())
    }
}

impl<U, E, const N: usize> PhaseEquilibrium<U, E, N>
where
    QuantityScalar<U>: fmt::Display,
    QuantityArray1<U>: fmt::Display,
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

impl<U: EosUnit, E: EquationOfState> PhaseEquilibrium<U, E, 2> {
    pub fn vapor(&self) -> &State<U, E> {
        &self.0[0]
    }

    pub fn liquid(&self) -> &State<U, E> {
        &self.0[1]
    }
}

impl<U: EosUnit, E: EquationOfState> PhaseEquilibrium<U, E, 3> {
    pub fn vapor(&self) -> &State<U, E> {
        &self.0[0]
    }

    pub fn liquid1(&self) -> &State<U, E> {
        &self.0[1]
    }

    pub fn liquid2(&self) -> &State<U, E> {
        &self.0[2]
    }
}

impl<U: EosUnit, E: EquationOfState> PhaseEquilibrium<U, E, 2> {
    pub(super) fn from_states(state1: State<U, E>, state2: State<U, E>) -> Self {
        let (vapor, liquid) = if state1.density < state2.density {
            (state1, state2)
        } else {
            (state2, state1)
        };
        Self([vapor, liquid])
    }

    pub(super) fn new_npt(
        eos: &Rc<E>,
        temperature: QuantityScalar<U>,
        pressure: QuantityScalar<U>,
        vapor_moles: &QuantityArray1<U>,
        liquid_moles: &QuantityArray1<U>,
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

impl<U: EosUnit, E: EquationOfState, const N: usize> PhaseEquilibrium<U, E, N> {
    pub(super) fn update_pressure(
        mut self,
        temperature: QuantityScalar<U>,
        pressure: QuantityScalar<U>,
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
        pressure: QuantityScalar<U>,
        moles: [&QuantityArray1<U>; N],
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

    pub fn update_chemical_potential(
        &mut self,
        chemical_potential: &QuantityArray1<U>,
    ) -> EosResult<()> {
        for s in self.0.iter_mut() {
            s.update_chemical_potential(chemical_potential)?;
        }
        Ok(())
    }

    pub(super) fn total_gibbs_energy(&self) -> QuantityScalar<U> {
        self.0.iter().fold(0.0 * U::reference_energy(), |acc, s| {
            acc + s.gibbs_energy(Contributions::Total)
        })
    }
}

const TRIVIAL_REL_DEVIATION: f64 = 1e-5;

/// # Utility functions
impl<U: EosUnit, E: EquationOfState> PhaseEquilibrium<U, E, 2> {
    pub(super) fn check_trivial_solution(self) -> EosResult<Self> {
        if Self::is_trivial_solution(self.vapor(), self.liquid()) {
            Err(EosError::TrivialSolution)
        } else {
            Ok(self)
        }
    }

    /// Check if the two states form a trivial solution
    pub fn is_trivial_solution(state1: &State<U, E>, state2: &State<U, E>) -> bool {
        let rho1 = state1
            .partial_density
            .to_reduced(U::reference_density())
            .unwrap();
        let rho2 = state2
            .partial_density
            .to_reduced(U::reference_density())
            .unwrap();

        rho1.iter()
            .zip(rho2.iter())
            .fold(0.0, |acc, (&rho1, &rho2)| {
                (rho2 / rho1 - 1.0).abs().max(acc)
            })
            < TRIVIAL_REL_DEVIATION
    }
}
