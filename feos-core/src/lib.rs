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
// mod phase_equilibria;
mod state;
pub use equation_of_state::{
    DeBroglieWavelength, DeBroglieWavelengthDual, EntropyScaling, EquationOfState, HelmholtzEnergy,
    HelmholtzEnergyDual, IdealGas, MolarWeight, Residual,
};
pub use errors::{EosError, EosResult};
// pub use phase_equilibria::{
//     PhaseDiagram, PhaseDiagramHetero, PhaseEquilibrium, SolverOptions, Verbosity,
// };
pub use state::{
    Contributions, DensityInitialization, Derivative, SolverOptions, State, StateBuilder, StateHD,
    StateVec, Verbosity,
};

#[cfg(feature = "python")]
pub mod python;

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
    use crate::equation_of_state::ideal_gas;
    use crate::equation_of_state::EquationOfState;
    use crate::joback::Joback;
    use crate::joback::JobackRecord;
    use crate::parameter::*;
    use crate::state::State;
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
        let ideal_gas = Arc::new(Joback::new(Arc::new(vec![JobackRecord::new(
            1.0, 1.0, 1.0, 1.0, 1.0,
        )])));
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
            sr.dmu_res_dni(),
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
        assert_relative_eq!(
            s.c_p(Contributions::Residual),
            sr.c_p_res(),
            max_relative = 1e-15
        );
        Ok(())
    }
}
