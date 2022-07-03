use feos_core::joback::Joback;
use feos_core::parameter::ParameterHetero;
use feos_core::{EquationOfState, HelmholtzEnergy, IdealGasContribution, MolarWeight};
use feos_saft::HardSphere;
use ndarray::Array1;
use quantity::si::*;
use std::f64::consts::FRAC_PI_6;
use std::rc::Rc;

pub(crate) mod association;
pub(crate) mod dispersion;
mod hard_chain;
mod parameter;
mod polar;
use association::{Association, CrossAssociation};
use dispersion::Dispersion;
use hard_chain::HardChain;
pub use parameter::{GcPcSaftChemicalRecord, GcPcSaftEosParameters};
use polar::Dipole;

/// Customization options for the gc-PC-SAFT equation of state and functional.
#[derive(Copy, Clone)]
pub struct GcPcSaftOptions {
    /// maximum packing fraction
    pub max_eta: f64,
    /// maximum number of iterations for cross association calculation
    pub max_iter_cross_assoc: usize,
    /// tolerance for cross association calculation
    pub tol_cross_assoc: f64,
}

impl Default for GcPcSaftOptions {
    fn default() -> Self {
        Self {
            max_eta: 0.5,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
        }
    }
}

/// gc-PC-SAFT equation of state
pub struct GcPcSaft {
    pub parameters: Rc<GcPcSaftEosParameters>,
    options: GcPcSaftOptions,
    contributions: Vec<Box<dyn HelmholtzEnergy>>,
    joback: Joback,
}

impl GcPcSaft {
    pub fn new(parameters: Rc<GcPcSaftEosParameters>) -> Self {
        Self::with_options(parameters, GcPcSaftOptions::default())
    }

    pub fn with_options(parameters: Rc<GcPcSaftEosParameters>, options: GcPcSaftOptions) -> Self {
        let mut contributions: Vec<Box<dyn HelmholtzEnergy>> = Vec::with_capacity(7);
        contributions.push(Box::new(HardSphere {
            parameters: parameters.clone(),
        }));
        contributions.push(Box::new(HardChain {
            parameters: parameters.clone(),
        }));
        contributions.push(Box::new(Dispersion {
            parameters: parameters.clone(),
        }));
        match parameters.assoc_segment.len() {
            0 => (),
            1 => contributions.push(Box::new(Association {
                parameters: parameters.clone(),
            })),
            _ => contributions.push(Box::new(CrossAssociation {
                parameters: parameters.clone(),
                max_iter: options.max_iter_cross_assoc,
                tol: options.tol_cross_assoc,
            })),
        };
        if !parameters.dipole_comp.is_empty() {
            contributions.push(Box::new(Dipole::new(&parameters)))
        }
        Self {
            parameters: parameters.clone(),
            options,
            contributions,
            joback: parameters.joback_records.clone().map_or_else(
                || Joback::default(parameters.chemical_records.len()),
                Joback::new,
            ),
        }
    }
}

impl EquationOfState for GcPcSaft {
    fn components(&self) -> usize {
        self.parameters.molarweight.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            Rc::new(self.parameters.subset(component_list)),
            self.options,
        )
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        let p = &self.parameters;
        let moles_segments: Array1<f64> = p.component_index.iter().map(|&i| moles[i]).collect();
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * &p.m * p.sigma.mapv(|v| v.powi(3)) * moles_segments).sum()
    }

    fn residual(&self) -> &[Box<dyn HelmholtzEnergy>] {
        &self.contributions
    }

    fn ideal_gas(&self) -> &dyn IdealGasContribution {
        &self.joback
    }
}

impl MolarWeight<SIUnit> for GcPcSaft {
    fn molar_weight(&self) -> SIArray1 {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}
