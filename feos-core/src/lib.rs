#![warn(clippy::all)]
#![allow(clippy::reversed_empty_ranges)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]
#![allow(deprecated)]

use quantity::si::*;
use quantity::*;

/// Print messages with level `Verbosity::Iter` or higher.
#[macro_export]
macro_rules! log_iter {
    ($verbosity:expr, $($arg:tt)*) => {
        if $verbosity >= Verbosity::Iter {
            println!($($arg)*);
        }
    }
}

/// Print messages with level `Verbosity::Result` or higher.
#[macro_export]
macro_rules! log_result {
    ($verbosity:expr, $($arg:tt)*) => {
        if $verbosity >= Verbosity::Result {
            println!($($arg)*);
        }
    }
}

pub mod cubic;
mod density_iteration;
mod equation_of_state;
mod errors;
pub mod joback;
pub mod parameter;
mod phase_equilibria;
mod state;
pub use equation_of_state::{
    Components, DeBroglieWavelength, DeBroglieWavelengthDual, EntropyScaling, EquationOfState,
    HelmholtzEnergy, HelmholtzEnergyDual, IdealGas, MolarWeight, Residual,
};
pub use errors::{EosError, EosResult};
pub use phase_equilibria::{PhaseDiagram, PhaseDiagramHetero, PhaseEquilibrium};
pub use state::{
    Contributions, DensityInitialization, Derivative, State, StateBuilder, StateHD, StateVec,
};

#[cfg(feature = "python")]
pub mod python;

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

/// Consistent conversions between quantities and reduced properties.
pub trait EosUnit: Unit + Send + Sync {
    fn reference_temperature() -> QuantityScalar<Self>;
    fn reference_length() -> QuantityScalar<Self>;
    fn reference_density() -> QuantityScalar<Self>;
    fn reference_time() -> QuantityScalar<Self>;
    fn gas_constant() -> QuantityScalar<Self>;
    fn reference_volume() -> QuantityScalar<Self> {
        Self::reference_length().powi(3)
    }
    fn reference_velocity() -> QuantityScalar<Self> {
        Self::reference_length() / Self::reference_time()
    }
    fn reference_moles() -> QuantityScalar<Self> {
        Self::reference_density() * Self::reference_volume()
    }
    fn reference_mass() -> QuantityScalar<Self> {
        Self::reference_energy() * Self::reference_velocity().powi(-2)
    }
    fn reference_energy() -> QuantityScalar<Self> {
        Self::gas_constant() * Self::reference_temperature() * Self::reference_moles()
    }
    fn reference_pressure() -> QuantityScalar<Self> {
        Self::reference_energy() / Self::reference_volume()
    }
    fn reference_entropy() -> QuantityScalar<Self> {
        Self::reference_energy() / Self::reference_temperature()
    }
    fn reference_molar_energy() -> QuantityScalar<Self> {
        Self::reference_energy() / Self::reference_moles()
    }
    fn reference_molar_entropy() -> QuantityScalar<Self> {
        Self::reference_entropy() / Self::reference_moles()
    }
    fn reference_surface_tension() -> QuantityScalar<Self> {
        Self::reference_pressure() * Self::reference_length()
    }
    fn reference_influence_parameter() -> QuantityScalar<Self> {
        Self::reference_temperature() * Self::gas_constant() * Self::reference_length().powi(2)
            / Self::reference_density()
    }
    fn reference_molar_mass() -> QuantityScalar<Self> {
        Self::reference_mass() / Self::reference_moles()
    }
    fn reference_viscosity() -> QuantityScalar<Self> {
        Self::reference_pressure() * Self::reference_time()
    }
    fn reference_diffusion() -> QuantityScalar<Self> {
        Self::reference_length().powi(2) / Self::reference_time()
    }
    fn reference_momentum() -> QuantityScalar<Self> {
        Self::reference_molar_mass() * Self::reference_density() * Self::reference_velocity()
    }
}

impl EosUnit for SIUnit {
    fn reference_temperature() -> SINumber {
        KELVIN
    }
    fn reference_length() -> SINumber {
        ANGSTROM
    }
    fn reference_density() -> SINumber {
        ANGSTROM.powi(-3) / NAV
    }
    fn reference_time() -> SINumber {
        PICO * SECOND
    }
    fn gas_constant() -> SINumber {
        RGAS
    }
}

#[cfg(test)]
mod tests {
    use crate::cubic::*;
    use crate::equation_of_state::EquationOfState;
    use crate::joback::{Joback, JobackParameters, JobackRecord};
    use crate::parameter::*;
    use crate::Contributions;
    use crate::EosResult;
    use crate::StateBuilder;
    use approx::*;
    use ndarray::Array2;
    use quantity::si::*;
    use std::sync::Arc;

    fn pure_record_vec() -> Vec<PureRecord<PengRobinsonRecord>> {
        let records = r#"[
            {
                "identifier": {
                    "cas": "74-98-6",
                    "name": "propane",
                    "iupac_name": "propane",
                    "smiles": "CCC",
                    "inchi": "InChI=1/C3H8/c1-3-2/h3H2,1-2H3",
                    "formula": "C3H8"
                },
                "model_record": {
                    "tc": 369.96,
                    "pc": 4250000.0,
                    "acentric_factor": 0.153
                },
                "molarweight": 44.0962
            },
            {
                "identifier": {
                    "cas": "106-97-8",
                    "name": "butane",
                    "iupac_name": "butane",
                    "smiles": "CCCC",
                    "inchi": "InChI=1/C4H10/c1-3-4-2/h3-4H2,1-2H3",
                    "formula": "C4H10"
                },
                "model_record": {
                    "tc": 425.2,
                    "pc": 3800000.0,
                    "acentric_factor": 0.199
                },
                "molarweight": 58.123
            }
        ]"#;
        serde_json::from_str(records).expect("Unable to parse json.")
    }

    #[test]
    fn validate_residual_properties() -> EosResult<()> {
        let mixture = pure_record_vec();
        let propane = mixture[0].clone();
        let parameters = PengRobinsonParameters::from_records(vec![propane], Array2::zeros((1, 1)));
        let residual = Arc::new(PengRobinson::new(Arc::new(parameters)));
        let joback_parameters = Arc::new(JobackParameters::new_pure(PureRecord::new(
            Identifier::default(),
            1.0,
            JobackRecord::new(0.0, 0.0, 0.0, 0.0, 0.0),
        )));
        let ideal_gas = Arc::new(Joback::new(joback_parameters));
        let eos = Arc::new(EquationOfState::new(ideal_gas, residual.clone()));

        let sr = StateBuilder::new(&residual)
            .temperature(300.0 * KELVIN)
            .pressure(1.0 * BAR)
            .build()?;

        let s = StateBuilder::new(&eos)
            .temperature(300.0 * KELVIN)
            .pressure(1.0 * BAR)
            .build()?;

        // pressure
        assert_relative_eq!(
            s.pressure(Contributions::Total),
            sr.pressure(Contributions::Total),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.pressure(Contributions::Residual),
            sr.pressure(Contributions::Residual),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.compressibility(Contributions::Total),
            sr.compressibility(Contributions::Total),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.compressibility(Contributions::Residual),
            sr.compressibility(Contributions::Residual),
            max_relative = 1e-15
        );

        // residual properties
        assert_relative_eq!(
            s.helmholtz_energy(Contributions::Residual),
            sr.residual_helmholtz_energy(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.entropy(Contributions::Residual),
            sr.residual_entropy(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.enthalpy(Contributions::Residual),
            sr.residual_enthalpy(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.internal_energy(Contributions::Residual),
            sr.residual_internal_energy(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.gibbs_energy(Contributions::Residual),
            sr.residual_gibbs_energy(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.chemical_potential(Contributions::Residual),
            sr.residual_chemical_potential(),
            max_relative = 1e-15
        );

        // pressure derivatives
        assert_relative_eq!(
            s.structure_factor(),
            sr.structure_factor(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.dp_dt(Contributions::Total),
            sr.dp_dt(Contributions::Total),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.dp_dt(Contributions::Residual),
            sr.dp_dt(Contributions::Residual),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.dp_dv(Contributions::Total),
            sr.dp_dv(Contributions::Total),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.dp_dv(Contributions::Residual),
            sr.dp_dv(Contributions::Residual),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.dp_drho(Contributions::Total),
            sr.dp_drho(Contributions::Total),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.dp_drho(Contributions::Residual),
            sr.dp_drho(Contributions::Residual),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.d2p_dv2(Contributions::Total),
            sr.d2p_dv2(Contributions::Total),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.d2p_dv2(Contributions::Residual),
            sr.d2p_dv2(Contributions::Residual),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.d2p_drho2(Contributions::Total),
            sr.d2p_drho2(Contributions::Total),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.d2p_drho2(Contributions::Residual),
            sr.d2p_drho2(Contributions::Residual),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.dp_dni(Contributions::Total),
            sr.dp_dni(Contributions::Total),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.dp_dni(Contributions::Residual),
            sr.dp_dni(Contributions::Residual),
            max_relative = 1e-15
        );

        // entropy
        assert_relative_eq!(
            s.ds_dt(Contributions::Residual),
            sr.ds_res_dt(),
            max_relative = 1e-15
        );

        // chemical potential
        assert_relative_eq!(
            s.dmu_dt(Contributions::Residual),
            sr.dmu_res_dt(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.dmu_dni(Contributions::Residual),
            sr.dmu_dni(Contributions::Residual),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.dmu_dt(Contributions::Residual),
            sr.dmu_res_dt(),
            max_relative = 1e-15
        );

        // fugacity
        assert_relative_eq!(s.ln_phi(), sr.ln_phi(), max_relative = 1e-15);
        assert_relative_eq!(s.dln_phi_dt(), sr.dln_phi_dt(), max_relative = 1e-15);
        assert_relative_eq!(s.dln_phi_dp(), sr.dln_phi_dp(), max_relative = 1e-15);
        assert_relative_eq!(s.dln_phi_dnj(), sr.dln_phi_dnj(), max_relative = 1e-15);
        assert_relative_eq!(
            s.thermodynamic_factor(),
            sr.thermodynamic_factor(),
            max_relative = 1e-15
        );

        // residual properties using multiple derivatives
        assert_relative_eq!(
            s.c_v(Contributions::Residual),
            sr.c_v_res(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.dc_v_dt(Contributions::Residual),
            sr.dc_v_res_dt(),
            max_relative = 1e-15
        );
        println!(
            "{}\n{}\n{}",
            s.c_p(Contributions::Residual),
            s.c_p(Contributions::IdealGas),
            s.c_p(Contributions::Total)
        );
        assert_relative_eq!(
            s.c_p(Contributions::Residual),
            sr.c_p_res(),
            max_relative = 1e-14
        );
        Ok(())
    }
}
