#![warn(clippy::all)]
#![allow(clippy::reversed_empty_ranges)]
#![warn(clippy::allow_attributes)]
use quantity::{Quantity, SIUnit};
use std::ops::{Div, Mul};
use typenum::Integer;

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
pub mod parameter;
mod parameter_fit;
mod phase_equilibria;
mod state;
pub use equation_of_state::{
    EquationOfState, IdealGas, IdealGasDyn, Molarweight, NoResidual, Residual, ResidualConst,
    ResidualDyn, Subset, Total,
};
pub use errors::{FeosError, FeosResult};
pub use parameter_fit::{BinaryModel, ParametersAD, PureModel};
#[cfg(feature = "ndarray")]
pub use phase_equilibria::{PhaseDiagram, PhaseDiagramHetero};
pub use phase_equilibria::{PhaseEquilibrium, TemperatureOrPressure};
pub use state::{Contributions, DensityInitialization, State, StateBuilder, StateHD, StateVec};

/// Level of detail in the iteration output.
#[derive(Copy, Clone, PartialOrd, PartialEq, Eq)]
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

/// Reference values used for reduced properties in feos
const REFERENCE_VALUES: [f64; 7] = [
    1e-12,               // 1 ps
    1e-10,               // 1 Å
    1.380649e-27,        // Fixed through k_B
    1.0,                 // 1 A
    1.0,                 // 1 K
    1.0 / 6.02214076e23, // 1/N_AV
    1.0,                 // 1 Cd
];

const fn powi(x: f64, n: i32) -> f64 {
    match n {
        ..=-1 => powi(1.0 / x, -n),
        0 => 1.0,
        n if n % 2 == 0 => powi(x * x, n / 2),
        n => x * powi(x * x, (n - 1) / 2),
    }
}

pub trait ReferenceSystem {
    type Inner;
    type T: Integer;
    type L: Integer;
    type M: Integer;
    type I: Integer;
    type THETA: Integer;
    type N: Integer;
    type J: Integer;
    const FACTOR: f64 = powi(REFERENCE_VALUES[0], Self::T::I32)
        * powi(REFERENCE_VALUES[1], Self::L::I32)
        * powi(REFERENCE_VALUES[2], Self::M::I32)
        * powi(REFERENCE_VALUES[3], Self::I::I32)
        * powi(REFERENCE_VALUES[4], Self::THETA::I32)
        * powi(REFERENCE_VALUES[5], Self::N::I32)
        * powi(REFERENCE_VALUES[6], Self::J::I32);

    fn from_reduced(value: Self::Inner) -> Self
    where
        Self::Inner: Mul<f64, Output = Self::Inner>;

    fn to_reduced(&self) -> Self::Inner
    where
        for<'a> &'a Self::Inner: Div<f64, Output = Self::Inner>;

    fn into_reduced(self) -> Self::Inner
    where
        Self::Inner: Div<f64, Output = Self::Inner>;
}

/// Conversion to and from reduced units
impl<Inner, T: Integer, L: Integer, M: Integer, I: Integer, THETA: Integer, N: Integer, J: Integer>
    ReferenceSystem for Quantity<Inner, SIUnit<T, L, M, I, THETA, N, J>>
{
    type Inner = Inner;
    type T = T;
    type L = L;
    type M = M;
    type I = I;
    type THETA = THETA;
    type N = N;
    type J = J;
    fn from_reduced(value: Inner) -> Self
    where
        Inner: Mul<f64, Output = Inner>,
    {
        Self::new(value * Self::FACTOR)
    }

    fn to_reduced(&self) -> Inner
    where
        for<'a> &'a Inner: Div<f64, Output = Inner>,
    {
        self.convert_to(Quantity::new(Self::FACTOR))
    }

    fn into_reduced(self) -> Inner
    where
        Inner: Div<f64, Output = Inner>,
    {
        self.convert_into(Quantity::new(Self::FACTOR))
    }
}

#[cfg(test)]
mod tests {
    use crate::Contributions;
    use crate::FeosResult;
    use crate::StateBuilder;
    use crate::cubic::*;
    use crate::equation_of_state::{EquationOfState, IdealGasDyn};
    use crate::parameter::*;
    use approx::*;
    use num_dual::DualNum;
    use quantity::{BAR, KELVIN, MOL, RGAS};

    // Only to be able to instantiate an `EquationOfState`
    #[derive(Clone, Copy)]
    struct NoIdealGas;

    impl IdealGasDyn for NoIdealGas {
        fn ideal_gas_model(&self) -> &'static str {
            "NoIdealGas"
        }

        fn ln_lambda3<D: DualNum<f64> + Copy>(&self, _: D) -> D {
            unreachable!()
        }
    }

    fn pure_record_vec() -> Vec<PureRecord<PengRobinsonRecord, ()>> {
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
                "tc": 369.96,
                "pc": 4250000.0,
                "acentric_factor": 0.153,
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
                "tc": 425.2,
                "pc": 3800000.0,
                "acentric_factor": 0.199,
                "molarweight": 58.123
            }
        ]"#;
        serde_json::from_str(records).expect("Unable to parse json.")
    }

    #[test]
    fn validate_residual_properties() -> FeosResult<()> {
        let mixture = pure_record_vec();
        let propane = &mixture[0];
        let parameters = PengRobinsonParameters::new_pure(propane.clone())?;
        let residual = PengRobinson::new(parameters);

        let sr = StateBuilder::new(&&residual)
            .temperature(300.0 * KELVIN)
            .pressure(1.0 * BAR)
            .total_moles(2.0 * MOL)
            .build()?;

        let parameters = PengRobinsonParameters::new_pure(propane.clone())?;
        let residual = PengRobinson::new(parameters);
        let eos = EquationOfState::new(vec![NoIdealGas], residual);
        let s = StateBuilder::new(&&eos)
            .temperature(300.0 * KELVIN)
            .pressure(1.0 * BAR)
            .total_moles(2.0 * MOL)
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
            s.molar_helmholtz_energy(Contributions::Residual),
            sr.residual_molar_helmholtz_energy(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.entropy(Contributions::Residual),
            sr.residual_entropy(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.molar_entropy(Contributions::Residual),
            sr.residual_molar_entropy(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.enthalpy(Contributions::Residual),
            sr.residual_enthalpy(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.molar_enthalpy(Contributions::Residual),
            sr.residual_molar_enthalpy(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.internal_energy(Contributions::Residual),
            sr.residual_internal_energy(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.molar_internal_energy(Contributions::Residual),
            sr.residual_molar_internal_energy(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.gibbs_energy(Contributions::Residual)
                - s.total_moles
                    * RGAS
                    * s.temperature
                    * s.compressibility(Contributions::Total).ln(),
            sr.residual_gibbs_energy(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.molar_gibbs_energy(Contributions::Residual)
                - RGAS * s.temperature * s.compressibility(Contributions::Total).ln(),
            sr.residual_molar_gibbs_energy(),
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
            s.molar_isochoric_heat_capacity(Contributions::Residual),
            sr.residual_molar_isochoric_heat_capacity(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.dc_v_dt(Contributions::Residual),
            sr.dc_v_res_dt(),
            max_relative = 1e-15
        );
        assert_relative_eq!(
            s.molar_isobaric_heat_capacity(Contributions::Residual),
            sr.residual_molar_isobaric_heat_capacity(),
            max_relative = 1e-15
        );
        Ok(())
    }
}
