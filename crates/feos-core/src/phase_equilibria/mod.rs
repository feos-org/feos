use crate::FeosError;
use crate::equation_of_state::Residual;
use crate::errors::FeosResult;
use crate::state::State;
use crate::{Contributions::Total as Tot, ReferenceSystem, Total};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, Dyn};
use num_dual::{DualNum, Gradients};
use quantity::{Dimensionless, Energy, Entropy, MolarEnergy, MolarEntropy, Moles};
use std::fmt;
use std::fmt::Write;

mod bubble_dew;
#[cfg(feature = "ndarray")]
mod phase_diagram_binary;
#[cfg(feature = "ndarray")]
mod phase_diagram_pure;
#[cfg(feature = "ndarray")]
mod phase_envelope;
mod stability_analysis;
mod tp_flash;
mod vle_pure;
pub use bubble_dew::TemperatureOrPressure;
#[cfg(feature = "ndarray")]
pub use phase_diagram_binary::PhaseDiagramHetero;
#[cfg(feature = "ndarray")]
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
#[derive(Debug, Clone)]
pub struct PhaseEquilibrium<E, const P: usize, N: Dim = Dyn, D: DualNum<f64> + Copy = f64>
where
    DefaultAllocator: Allocator<N>,
{
    states: [State<E, N, D>; P],
    pub phase_fractions: [D; P],
    total_moles: Option<Moles<D>>,
}

impl<E: Residual, const P: usize> fmt::Display for PhaseEquilibrium<E, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, s) in self.states.iter().enumerate() {
            writeln!(f, "phase {i}: {s}")?;
        }
        Ok(())
    }
}

impl<E: Residual, const P: usize> PhaseEquilibrium<E, P> {
    pub fn _repr_markdown_(&self) -> String {
        if self.states[0].eos.components() == 1 {
            let mut res = "||temperature|density|\n|-|-|-|\n".to_string();
            for (i, s) in self.states.iter().enumerate() {
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
            for (i, s) in self.states.iter().enumerate() {
                writeln!(
                    res,
                    "|phase {}|{:.5}|{:.5}|{:.5?}|",
                    i + 1,
                    s.temperature,
                    s.density,
                    s.molefracs.as_slice()
                )
                .unwrap();
            }
            res
        }
    }
}

impl<E: Residual<N, D>, N: Dim, D: DualNum<f64> + Copy> PhaseEquilibrium<E, 2, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    pub fn vapor(&self) -> &State<E, N, D> {
        &self.states[0]
    }

    pub fn liquid(&self) -> &State<E, N, D> {
        &self.states[1]
    }

    pub fn vapor_phase_fraction(&self) -> D {
        self.phase_fractions[0]
    }
}

impl<E> PhaseEquilibrium<E, 3> {
    pub fn vapor(&self) -> &State<E> {
        &self.states[0]
    }

    pub fn liquid1(&self) -> &State<E> {
        &self.states[1]
    }

    pub fn liquid2(&self) -> &State<E> {
        &self.states[2]
    }
}

impl<E: Residual<N, D>, N: Dim, D: DualNum<f64> + Copy> PhaseEquilibrium<E, 2, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    pub(super) fn single_phase(state: State<E, N, D>) -> Self {
        let total_moles = state.total_moles;
        Self::with_vapor_phase_fraction(state.clone(), state, D::from(1.0), total_moles)
    }

    pub(super) fn two_phase(vapor: State<E, N, D>, liquid: State<E, N, D>) -> Self {
        let (beta, total_moles) =
            if let (Some(nv), Some(nl)) = (vapor.total_moles, liquid.total_moles) {
                (nv.convert_into(nl + nv), Some(nl + nv))
            } else {
                (D::from(1.0), None)
            };
        Self::with_vapor_phase_fraction(vapor, liquid, beta, total_moles)
    }

    pub(super) fn with_vapor_phase_fraction(
        vapor: State<E, N, D>,
        liquid: State<E, N, D>,
        vapor_phase_fraction: D,
        total_moles: Option<Moles<D>>,
    ) -> Self {
        Self {
            states: [vapor, liquid],
            phase_fractions: [vapor_phase_fraction, -vapor_phase_fraction + 1.0],
            total_moles,
        }
    }
}

impl<E: Residual<N, D>, N: Dim, D: DualNum<f64> + Copy> PhaseEquilibrium<E, 3, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    pub(super) fn new(
        vapor: State<E, N, D>,
        liquid1: State<E, N, D>,
        liquid2: State<E, N, D>,
    ) -> Self {
        Self {
            states: [vapor, liquid1, liquid2],
            phase_fractions: [D::from(1.0), D::from(0.0), D::from(0.0)],
            total_moles: None,
        }
    }
}

impl<E: Total<N, D>, N: Gradients, const P: usize, D: DualNum<f64> + Copy>
    PhaseEquilibrium<E, P, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    pub fn total_moles(&self) -> FeosResult<Moles<D>> {
        self.total_moles.ok_or(FeosError::IntensiveState)
    }

    pub fn molar_enthalpy(&self) -> MolarEnergy<D> {
        self.states
            .iter()
            .zip(&self.phase_fractions)
            .map(|(s, x)| s.molar_enthalpy(Tot) * Dimensionless::new(x))
            .reduce(|a, b| a + b)
            .unwrap()
    }

    pub fn enthalpy(&self) -> FeosResult<Energy<D>> {
        Ok(self.total_moles()? * self.molar_enthalpy())
    }

    pub fn molar_entropy(&self) -> MolarEntropy<D> {
        self.states
            .iter()
            .zip(&self.phase_fractions)
            .map(|(s, x)| s.molar_entropy(Tot) * Dimensionless::new(x))
            .reduce(|a, b| a + b)
            .unwrap()
    }

    pub fn entropy(&self) -> FeosResult<Entropy<D>> {
        Ok(self.total_moles()? * self.molar_entropy())
    }
}

const TRIVIAL_REL_DEVIATION: f64 = 1e-5;

/// # Utility functions
impl<E: Residual<N>, N: Dim> PhaseEquilibrium<E, 2, N>
where
    DefaultAllocator: Allocator<N>,
{
    /// Check if the two states form a trivial solution
    pub fn is_trivial_solution(state1: &State<E, N>, state2: &State<E, N>) -> bool {
        let rho1 = state1.molefracs.clone() * state1.density.into_reduced();
        let rho2 = state2.molefracs.clone() * state2.density.into_reduced();

        rho1.into_iter()
            .zip(&rho2)
            .fold(0.0, |acc, (rho1, rho2)| (rho2 / rho1 - 1.0).abs().max(acc))
            < TRIVIAL_REL_DEVIATION
    }
}
