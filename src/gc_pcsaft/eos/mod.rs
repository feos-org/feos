use crate::association::Association;
use crate::hard_sphere::HardSphere;
use feos_core::joback::Joback;
use feos_core::parameter::ParameterHetero;
use feos_core::{EquationOfState, HelmholtzEnergy, IdealGasContribution, MolarWeight};
use ndarray::Array1;
use quantity::si::*;
use std::f64::consts::FRAC_PI_6;
use std::sync::Arc;

pub(crate) mod dispersion;
mod hard_chain;
pub(crate) mod parameter;
mod polar;
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
    pub parameters: Arc<GcPcSaftEosParameters>,
    options: GcPcSaftOptions,
    contributions: Vec<Box<dyn HelmholtzEnergy>>,
    joback: Joback,
}

impl GcPcSaft {
    pub fn new(parameters: Arc<GcPcSaftEosParameters>) -> Self {
        Self::with_options(parameters, GcPcSaftOptions::default())
    }

    pub fn with_options(parameters: Arc<GcPcSaftEosParameters>, options: GcPcSaftOptions) -> Self {
        let mut contributions: Vec<Box<dyn HelmholtzEnergy>> = Vec::with_capacity(7);
        contributions.push(Box::new(HardSphere::new(&parameters)));
        contributions.push(Box::new(HardChain {
            parameters: parameters.clone(),
        }));
        contributions.push(Box::new(Dispersion {
            parameters: parameters.clone(),
        }));
        if !parameters.association.assoc_comp.is_empty() {
            contributions.push(Box::new(Association::new(
                &parameters,
                &parameters.association,
                options.max_iter_cross_assoc,
                options.tol_cross_assoc,
            )));
        }
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
            Arc::new(self.parameters.subset(component_list)),
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

impl MolarWeight for GcPcSaft {
    fn molar_weight(&self) -> SIArray1 {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gc_pcsaft::eos::parameter::test::*;
    use approx::assert_relative_eq;
    use feos_core::{EosUnit, HelmholtzEnergyDual, StateHD};
    use ndarray::arr1;
    use num_dual::Dual64;
    use quantity::si::{METER, MOL, PASCAL};

    #[test]
    fn hs_propane() {
        let parameters = propane();
        let contrib = HardSphere::new(&Arc::new(parameters));
        let temperature = 300.0;
        let volume = METER
            .powi(3)
            .to_reduced(EosUnit::reference_volume())
            .unwrap();
        let moles = (1.5 * MOL).to_reduced(EosUnit::reference_moles()).unwrap();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derive(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let pressure =
            -contrib.helmholtz_energy(&state).eps[0] * temperature * EosUnit::reference_pressure();
        assert_relative_eq!(pressure, 1.5285037907989527 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn hs_propanol() {
        let parameters = propanol();
        let contrib = HardSphere::new(&Arc::new(parameters));
        let temperature = 300.0;
        let volume = METER
            .powi(3)
            .to_reduced(EosUnit::reference_volume())
            .unwrap();
        let moles = (1.5 * MOL).to_reduced(EosUnit::reference_moles()).unwrap();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derive(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let pressure =
            -contrib.helmholtz_energy(&state).eps[0] * temperature * EosUnit::reference_pressure();
        assert_relative_eq!(pressure, 2.3168212018200243 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn assoc_propanol() {
        let parameters = Arc::new(propanol());
        let contrib = Association::new(&parameters, &parameters.association, 50, 1e-10);
        let temperature = 300.0;
        let volume = METER
            .powi(3)
            .to_reduced(EosUnit::reference_volume())
            .unwrap();
        let moles = (1.5 * MOL).to_reduced(EosUnit::reference_moles()).unwrap();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derive(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let pressure =
            -contrib.helmholtz_energy(&state).eps[0] * temperature * EosUnit::reference_pressure();
        assert_relative_eq!(pressure, -3.6819598891967344 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn cross_assoc_propanol() {
        let parameters = Arc::new(propanol());
        let contrib =
            Association::new_cross_association(&parameters, &parameters.association, 50, 1e-10);
        let temperature = 300.0;
        let volume = METER
            .powi(3)
            .to_reduced(EosUnit::reference_volume())
            .unwrap();
        let moles = (1.5 * MOL).to_reduced(EosUnit::reference_moles()).unwrap();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derive(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let pressure =
            -contrib.helmholtz_energy(&state).eps[0] * temperature * EosUnit::reference_pressure();
        assert_relative_eq!(pressure, -3.6819598891967344 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn cross_assoc_ethanol_propanol() {
        let parameters = Arc::new(ethanol_propanol(false));
        let contrib = Association::new(&parameters, &parameters.association, 50, 1e-10);
        let temperature = 300.0;
        let volume = METER
            .powi(3)
            .to_reduced(EosUnit::reference_volume())
            .unwrap();
        let moles = (arr1(&[1.5, 2.5]) * MOL)
            .to_reduced(EosUnit::reference_moles())
            .unwrap();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derive(),
            moles.mapv(Dual64::from_re),
        );
        let pressure =
            -contrib.helmholtz_energy(&state).eps[0] * temperature * EosUnit::reference_pressure();
        assert_relative_eq!(pressure, -26.105606376765632 * PASCAL, max_relative = 1e-10);
    }
}
