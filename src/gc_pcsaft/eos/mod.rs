use crate::association::Association;
use crate::hard_sphere::{HardSphere, HardSphereProperties};
use feos_core::parameter::ParameterHetero;
use feos_core::si::{MolarWeight, GRAM, MOL};
use feos_core::{Components, Residual};
use ndarray::Array1;
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
    hard_sphere: HardSphere<GcPcSaftEosParameters>,
    hard_chain: HardChain,
    dispersion: Dispersion,
    association: Option<Association<GcPcSaftEosParameters>>,
    dipole: Option<Dipole>,
}

impl GcPcSaft {
    pub fn new(parameters: Arc<GcPcSaftEosParameters>) -> Self {
        Self::with_options(parameters, GcPcSaftOptions::default())
    }

    pub fn with_options(parameters: Arc<GcPcSaftEosParameters>, options: GcPcSaftOptions) -> Self {
        let hard_sphere = HardSphere::new(&parameters);
        let hard_chain = HardChain {
            parameters: parameters.clone(),
        };
        let dispersion = Dispersion {
            parameters: parameters.clone(),
        };
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
        let dipole = if !parameters.dipole_comp.is_empty() {
            Some(Dipole::new(&parameters))
        } else {
            None
        };
        Self {
            parameters,
            options,
            hard_sphere,
            hard_chain,
            dispersion,
            association,
            dipole,
        }
    }
}

impl Components for GcPcSaft {
    fn components(&self) -> usize {
        self.parameters.molarweight.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            Arc::new(self.parameters.subset(component_list)),
            self.options,
        )
    }
}

impl Residual for GcPcSaft {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        let p = &self.parameters;
        let moles_segments: Array1<f64> = p.component_index.iter().map(|&i| moles[i]).collect();
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * &p.m * p.sigma.mapv(|v| v.powi(3)) * moles_segments).sum()
    }

    fn residual_helmholtz_energy_contributions<D: num_dual::DualNum<f64> + Copy>(
        &self,
        state: &feos_core::StateHD<D>,
    ) -> Vec<(String, D)> {
        let mut v = Vec::with_capacity(7);
        let d = self.parameters.hs_diameter(state.temperature);

        v.push((
            "Hard Sphere".to_string(),
            self.hard_sphere.helmholtz_energy(&state),
        ));
        v.push((
            "Hard Sphere".to_string(),
            self.hard_chain.helmholtz_energy(&state),
        ));
        v.push((
            "Dispersion".to_string(),
            self.dispersion.helmholtz_energy(&state),
        ));
        if let Some(dipole) = self.dipole.as_ref() {
            v.push(("Dipole".to_string(), dipole.helmholtz_energy(&state)))
        }
        if let Some(association) = self.association.as_ref() {
            v.push((
                "Association".to_string(),
                association.helmholtz_energy(&state, &d),
            ))
        }
        v
    }

    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gc_pcsaft::eos::parameter::test::*;
    use crate::hard_sphere::HardSphereProperties;
    use approx::assert_relative_eq;
    use feos_core::si::{Pressure, METER, MOL, PASCAL};
    use feos_core::StateHD;
    use ndarray::arr1;
    use num_dual::Dual64;
    use typenum::P3;

    #[test]
    fn hs_propane() {
        let parameters = propane();
        let contrib = HardSphere::new(&Arc::new(parameters));
        let temperature = 300.0;
        let volume = METER.powi::<P3>().to_reduced();
        let moles = (1.5 * MOL).to_reduced();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derivative(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let pressure = Pressure::from_reduced(-contrib.helmholtz_energy(&state).eps * temperature);
        assert_relative_eq!(pressure, 1.5285037907989527 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn hs_propanol() {
        let parameters = propanol();
        let contrib = HardSphere::new(&Arc::new(parameters));
        let temperature = 300.0;
        let volume = METER.powi::<P3>().to_reduced();
        let moles = (1.5 * MOL).to_reduced();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derivative(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let pressure = Pressure::from_reduced(-contrib.helmholtz_energy(&state).eps * temperature);
        assert_relative_eq!(pressure, 2.3168212018200243 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn assoc_propanol() {
        let parameters = Arc::new(propanol());
        let contrib = Association::new(&parameters, &parameters.association, 50, 1e-10);
        let temperature = 300.0;
        let volume = METER.powi::<P3>().to_reduced();
        let moles = (1.5 * MOL).to_reduced();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derivative(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let diameter = parameters.hs_diameter(state.temperature);
        let pressure =
            Pressure::from_reduced(-contrib.helmholtz_energy(&state, &diameter).eps * temperature);
        assert_relative_eq!(pressure, -3.6819598891967344 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn cross_assoc_propanol() {
        let parameters = Arc::new(propanol());
        let contrib =
            Association::new_cross_association(&parameters, &parameters.association, 50, 1e-10);
        let temperature = 300.0;
        let volume = METER.powi::<P3>().to_reduced();
        let moles = (1.5 * MOL).to_reduced();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derivative(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let diameter = parameters.hs_diameter(state.temperature);
        let pressure =
            Pressure::from_reduced(-contrib.helmholtz_energy(&state, &diameter).eps * temperature);
        assert_relative_eq!(pressure, -3.6819598891967344 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn cross_assoc_ethanol_propanol() {
        let parameters = Arc::new(ethanol_propanol(false));
        let contrib = Association::new(&parameters, &parameters.association, 50, 1e-10);
        let temperature = 300.0;
        let volume = METER.powi::<P3>().to_reduced();
        let moles = (arr1(&[1.5, 2.5]) * MOL).to_reduced();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derivative(),
            moles.mapv(Dual64::from_re),
        );
        let diameter = parameters.hs_diameter(state.temperature);
        let pressure =
            Pressure::from_reduced(-contrib.helmholtz_energy(&state, &diameter).eps * temperature);
        assert_relative_eq!(pressure, -26.105606376765632 * PASCAL, max_relative = 1e-10);
    }
}
