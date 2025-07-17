use crate::equation_of_state::Residual;
use crate::errors::{FeosError, FeosResult};
use crate::state::{Cache, DensityInitialization, State};
use crate::{Contributions, HelmholtzEnergyDerivatives, ReferenceSystem, StateGeneric};
use ndarray::Array1;
use num_dual::DualNum;
use quantity::{Dimensionless, Energy, Moles, Pressure, RGAS, Temperature};
use std::fmt;
use std::fmt::Write;
use std::sync::{Arc, Mutex};

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
pub struct PhaseEquilibriumGeneric<E, D: DualNum<f64> + Copy, M, C, const N: usize>(
    [StateGeneric<E, D, M, C>; N],
);
pub type PhaseEquilibrium<E, const N: usize> =
    PhaseEquilibriumGeneric<Arc<E>, f64, Array1<f64>, Mutex<Cache>, N>;

impl<E, const N: usize> Clone for PhaseEquilibrium<E, N> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<E: Residual, const N: usize> fmt::Display for PhaseEquilibrium<E, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, s) in self.0.iter().enumerate() {
            writeln!(f, "phase {i}: {s}")?;
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

impl<E: HelmholtzEnergyDerivatives<D>, D: DualNum<f64> + Copy>
    PhaseEquilibriumGeneric<E, D, E::Molefracs, E::Cache, 2>
{
    pub fn vapor(&self) -> &StateGeneric<E, D, E::Molefracs, E::Cache> {
        &self.0[0]
    }

    pub fn liquid(&self) -> &StateGeneric<E, D, E::Molefracs, E::Cache> {
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

impl<E: HelmholtzEnergyDerivatives<f64>>
    PhaseEquilibriumGeneric<E, f64, E::Molefracs, E::Cache, 2>
{
    pub(super) fn from_states(
        state1: StateGeneric<E, f64, E::Molefracs, E::Cache>,
        state2: StateGeneric<E, f64, E::Molefracs, E::Cache>,
    ) -> Self {
        let (vapor, liquid) = if state1.density.re() < state2.density.re() {
            (state1, state2)
        } else {
            (state2, state1)
        };
        Self([vapor, liquid])
    }

    /// Creates a new PhaseEquilibrium that contains two states at the
    /// specified temperature, pressure and molefracs.
    ///
    /// The constructor can be used in custom phase equilibrium solvers or,
    /// e.g., to generate initial guesses for an actual VLE solver.
    /// In general, the two states generated are NOT in an equilibrium.
    pub fn new_xpt(
        eos: &E,
        temperature: Temperature,
        pressure: Pressure,
        vapor_molefracs: &E::Molefracs,
        liquid_molefracs: &E::Molefracs,
    ) -> FeosResult<Self> {
        let liquid = StateGeneric::new_xpt(
            eos,
            temperature,
            pressure,
            liquid_molefracs,
            DensityInitialization::Liquid,
        )?;
        let vapor = StateGeneric::new_xpt(
            eos,
            temperature,
            pressure,
            vapor_molefracs,
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
    ) -> FeosResult<Self> {
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
    ) -> FeosResult<()> {
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
                + RGAS * s.temperature * (s.moles.clone() * Dimensionless::new(ln_rho - 1.0)).sum()
        })
    }
}

const TRIVIAL_REL_DEVIATION: f64 = 1e-5;

/// # Utility functions
impl<E: HelmholtzEnergyDerivatives<f64>>
    PhaseEquilibriumGeneric<E, f64, E::Molefracs, E::Cache, 2>
{
    pub(super) fn check_trivial_solution(self) -> FeosResult<Self> {
        if Self::is_trivial_solution(self.vapor(), self.liquid()) {
            Err(FeosError::TrivialSolution)
        } else {
            Ok(self)
        }
    }

    /// Check if the two states form a trivial solution
    pub fn is_trivial_solution(
        state1: &StateGeneric<E, f64, E::Molefracs, E::Cache>,
        state2: &StateGeneric<E, f64, E::Molefracs, E::Cache>,
    ) -> bool {
        let rho1 = state1.molefracs.clone() * state1.density.into_reduced();
        let rho2 = state2.molefracs.clone() * state2.density.into_reduced();

        E::iter_molefracs(&rho1)
            .zip(E::iter_molefracs(&rho2))
            .fold(0.0, |acc, (rho1, rho2)| (rho2 / rho1 - 1.0).abs().max(acc))
            < TRIVIAL_REL_DEVIATION
    }
}
