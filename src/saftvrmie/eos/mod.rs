use crate::hard_sphere::HardSphere;

use self::dispersion::{a_disp, a_disp_chain, Properties};

use super::SaftVRMieParameters;
use association::Association;
use feos_core::{
    parameter::Parameter,
    si::{MolarWeight, GRAM, MOL},
    Components, Residual, StateHD,
};
use ndarray::{Array1, ScalarOperand};
use num_dual::DualNum;
use std::{f64::consts::FRAC_PI_6, sync::Arc};

pub(super) mod association;
mod dispersion;

/// Customization options for the PC-SAFT equation of state and functional.
#[derive(Copy, Clone)]
pub struct SaftVRMieOptions {
    pub max_eta: f64,
    pub max_iter_cross_assoc: usize,
    pub tol_cross_assoc: f64,
}

impl Default for SaftVRMieOptions {
    fn default() -> Self {
        Self {
            max_eta: 0.5,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
        }
    }
}

pub struct SaftVRMie {
    parameters: Arc<SaftVRMieParameters>,
    options: SaftVRMieOptions,
    hard_sphere: HardSphere<SaftVRMieParameters>,
    chain: bool,
    association: Option<Association<SaftVRMieParameters>>,
}

impl SaftVRMie {
    pub fn new(parameters: Arc<SaftVRMieParameters>) -> Self {
        Self::with_options(parameters, SaftVRMieOptions::default())
    }

    pub fn with_options(parameters: Arc<SaftVRMieParameters>, options: SaftVRMieOptions) -> Self {
        let association = if !parameters.association.is_empty() {
            Some(Association::new(
                &parameters,
                &parameters.association,
                options.max_iter_cross_assoc,
                options.tol_cross_assoc,
            ))
        } else {
            None
        };
        Self {
            parameters: parameters.clone(),
            options,
            hard_sphere: HardSphere::new(&parameters),
            chain: parameters.m.iter().any(|&m| m > 1.0),
            association,
        }
    }
}

impl Components for SaftVRMie {
    fn components(&self) -> usize {
        self.parameters.m.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::new(Arc::new(self.parameters.subset(component_list)))
    }
}

impl Residual for SaftVRMie {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * &self.parameters.m * self.parameters.sigma.mapv(|v| v.powi(3)) * moles)
                .sum()
    }

    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molarweight.clone() * GRAM / MOL
    }

    fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)> {
        let mut a = Vec::with_capacity(5);

        let (a_hs, _, diameter) = self.hard_sphere.helmholtz_energy_and_properties(state);
        a.push(("hard sphere".to_string(), a_hs));

        let properties = Properties::new(&self.parameters, state, &diameter);
        if self.chain {
            let a_disp_chain = a_disp_chain(&self.parameters, &properties, state);
            a.push(("dispersion + chain".to_string(), a_disp_chain));
        } else {
            let a_disp = a_disp(&self.parameters, &properties, state);
            a.push(("dispersion".to_string(), a_disp));
        }

        if let Some(assoc) = self.association.as_ref() {
            let reduced_temperature = self
                .parameters
                .epsilon_k_ij
                .mapv(|eps| state.temperature / eps);
            a.push((
                "association".to_string(),
                assoc.helmholtz_energy(
                    state,
                    properties.segment_density * properties.zeta_x_bar,
                    &reduced_temperature,
                ),
            ));
        }
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::saftvrmie::{ethane, methane, methane_ethane, methanol, methanol_propanol};
    use approx::assert_relative_eq;
    use feos_core::{si::*, SolverOptions, State, StateBuilder};
    use ndarray::{arr1, Array1};
    use num_traits::pow;
    use typenum::P3;

    #[test]
    fn test_free_energy() {
        let parameters = Arc::new(ethane());
        let mw = parameters.molarweight[0] * GRAM / MOL;
        let eos = Arc::new(SaftVRMie::new(parameters));
        let (rhoc, tc, pc) = (
            205.84 * KILO * GRAM / METER.powi::<P3>(),
            311.38 * KELVIN,
            5.49 * MEGA * PASCAL,
        );
        let cp = &State::critical_point_pure(&eos, None, SolverOptions::default()).unwrap()[0];
        dbg!(cp.temperature);
        dbg!(cp.moles.sum());
        dbg!(cp.mass_density());
        dbg!(cp
            .pressure(feos_core::Contributions::Total)
            .convert_into(MEGA * PASCAL));
        println!(
            "delta Tc: {}",
            ((cp.temperature - tc) / tc).to_reduced() * 100.0
        );
        println!(
            "delta pc: {}",
            ((cp.pressure(feos_core::Contributions::Total) - pc) / pc).to_reduced() * 100.0
        );
        println!(
            "delta rhoc: {}",
            ((cp.mass_density() - rhoc) / rhoc).to_reduced() * 100.0
        );

        let state = StateBuilder::new(&eos)
            .temperature(200.0 * KELVIN)
            .density(200.0 * KILOGRAM / METER.powi::<P3>() / mw)
            .total_moles(MOL)
            .build()
            .unwrap();
        let a = state.residual_molar_helmholtz_energy();
        dbg!(a);
        dbg!(state.temperature);
        dbg!(state.moles.sum());
        dbg!(state.mass_density());
        dbg!(state
            .pressure(feos_core::Contributions::Total)
            .convert_into(MEGA * PASCAL));
        let contributions = eos.residual_helmholtz_energy_contributions(&state.derive0());
        dbg!(contributions);
        let state = StateBuilder::new(&eos)
            .temperature(200.0 * KELVIN)
            .volume(1.0 * METER.powi::<P3>())
            .total_moles(MOL)
            .build()
            .unwrap();
        let a = state.residual_molar_helmholtz_energy() / (RGAS * 200.0 * KELVIN);
        dbg!(a);
        assert!(1 == 2)
    }

    #[test]
    fn test_methane_ethane() {
        let parameters = Arc::new(methane_ethane());
        let eos = Arc::new(SaftVRMie::new(parameters));
        dbg!(&eos.association.is_some());
        let x = Array1::from_vec(vec![0.5, 0.5]);
        let n = &x * 1.0 * MOL;
        let state = StateBuilder::new(&eos)
            .temperature(200.0 * KELVIN)
            .volume(1.0 * METER.powi::<P3>())
            .moles(&n)
            .build()
            .unwrap();
        dbg!(state.total_moles);
        let a = state.residual_molar_helmholtz_energy() / (RGAS * 200.0 * KELVIN);
        let contributions = eos.residual_helmholtz_energy_contributions(&state.derive0());

        dbg!(&contributions);
        dbg!(a);

        let state = State::critical_point(&eos, Some(&n), None, SolverOptions::default()).unwrap();
        dbg!(state.temperature);
        dbg!(state
            .pressure(feos_core::Contributions::Total)
            .convert_into(PASCAL));
        dbg!(state.volume.convert_into(METER.powi::<P3>()));
        assert!(1 == 2)
    }

    #[test]
    fn test_methanol() {
        let parameters = Arc::new(methanol());
        let eos = Arc::new(SaftVRMie::with_options(
            parameters,
            SaftVRMieOptions::default(),
        ));
        dbg!(&eos.association.is_some());
        let state = StateBuilder::new(&eos)
            .temperature(200.0 * KELVIN)
            .volume(1.0 * METER.powi::<P3>())
            .total_moles(MOL)
            .build()
            .unwrap();
        let a = state.residual_molar_helmholtz_energy() / (RGAS * 200.0 * KELVIN);
        let contributions = eos.residual_helmholtz_energy_contributions(&state.derive0());

        dbg!(&contributions);
        dbg!(a);

        // let state = State::critical_point(&eos, None, None, SolverOptions::default()).unwrap();
        // dbg!(state.temperature);
        // dbg!(state
        //     .pressure(feos_core::Contributions::Total)
        //     .convert_into(PASCAL));
        // dbg!(state.volume.convert_into(ANGSTROM.powi::<P3>()));
        assert!(1 == 2)
    }

    #[test]
    fn test_methanol_propanol() {
        let parameters = Arc::new(methanol_propanol());
        let eos = Arc::new(SaftVRMie::with_options(
            parameters,
            SaftVRMieOptions::default(),
        ));
        dbg!(&eos.association.is_some());
        let moles = &arr1(&[0.5, 0.5]) * MOL;
        let state = StateBuilder::new(&eos)
            .temperature(200.0 * KELVIN)
            .volume(1.0 * METER.powi::<P3>())
            .moles(&moles)
            .build()
            .unwrap();
        let a = state.residual_molar_helmholtz_energy() / (RGAS * 200.0 * KELVIN);
        let contributions = eos.residual_helmholtz_energy_contributions(&state.derive0());

        dbg!(&contributions);
        dbg!(a);

        let state =
            State::critical_point(&eos, Some(&moles), None, SolverOptions::default()).unwrap();
        dbg!(state.temperature);
        dbg!(state
            .pressure(feos_core::Contributions::Total)
            .convert_into(PASCAL));
        dbg!(state.volume.convert_into(ANGSTROM.powi::<P3>()));
        assert!(1 == 2)
    }

    #[test]
    fn test_ethane() {
        let parameters = Arc::new(ethane());
        let eos = Arc::new(SaftVRMie::new(parameters));
        let n = 1.0 * MOL;
        let state = StateBuilder::new(&eos)
            .temperature(200.0 * KELVIN)
            .volume(1.0 * METER.powi::<P3>())
            .total_moles(n)
            .build()
            .unwrap();
        let a = state.residual_molar_helmholtz_energy() / (RGAS * 200.0 * KELVIN);
        let contributions = eos.residual_helmholtz_energy_contributions(&state.derive0());

        dbg!(&contributions);
        dbg!(a);
    }
}
