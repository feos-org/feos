use crate::hard_sphere::HardSphere;

use super::SaftVRMieParameters;
use association::Association;
use feos_core::parameter::Parameter;
use feos_core::{Components, Molarweight, Residual, StateHD};
use ndarray::{Array1, ScalarOperand};
use num_dual::DualNum;
use quantity::{MolarWeight, GRAM, MOL};
use std::{f64::consts::FRAC_PI_6, sync::Arc};

pub(super) mod association;
pub(crate) mod dispersion;
use dispersion::{a_disp, a_disp_chain, Properties};

/// Customization options for the SAFT-VR Mie equation of state.
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

/// SAFT-VR Mie equation of state.
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

    fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)> {
        let mut a = Vec::with_capacity(7);

        let (a_hs, _, d) = self.hard_sphere.helmholtz_energy_and_properties(state);
        a.push(("Hard Sphere".to_string(), a_hs));

        let properties = Properties::new(&self.parameters, state, &d);
        if self.chain {
            let a_disp_chain = a_disp_chain(&self.parameters, &properties, state);
            a.push(("Dispersion + Chain".to_string(), a_disp_chain));
        } else {
            let a_disp = a_disp(&self.parameters, &properties, state);
            a.push(("Dispersion".to_string(), a_disp));
        }
        if let Some(assoc) = self.association.as_ref() {
            a.push(("Association".to_string(), assoc.helmholtz_energy(state, &d)));
        }
        a
    }
}

impl Molarweight for SaftVRMie {
    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}
