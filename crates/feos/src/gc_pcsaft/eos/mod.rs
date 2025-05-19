use crate::association::Association;
use crate::hard_sphere::{HardSphere, HardSphereProperties};
use feos_core::parameter::ParameterHetero;
use feos_core::{Components, Molarweight, Residual, StateHD};
use ndarray::Array1;
use num_dual::DualNum;
use quantity::{GRAM, MOL, MolarWeight};
use std::f64::consts::FRAC_PI_6;

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
    pub parameters: GcPcSaftEosParameters,
    options: GcPcSaftOptions,
    association: Option<Association>,
    dipole: Option<Dipole>,
}

impl GcPcSaft {
    pub fn new(parameters: GcPcSaftEosParameters) -> Self {
        Self::with_options(parameters, GcPcSaftOptions::default())
    }

    pub fn with_options(parameters: GcPcSaftEosParameters, options: GcPcSaftOptions) -> Self {
        let association = (!parameters.association.is_empty()).then_some(Association::new(
            options.max_iter_cross_assoc,
            options.tol_cross_assoc,
        ));
        let dipole = (!parameters.dipole_comp.is_empty()).then(|| Dipole::new(&parameters));
        Self {
            parameters,
            options,
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
        Self::with_options(self.parameters.subset(component_list), self.options)
    }
}

impl Residual for GcPcSaft {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        let p = &self.parameters;
        let moles_segments: Array1<f64> = p.component_index.iter().map(|&i| moles[i]).collect();
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * &p.m * p.sigma.mapv(|v| v.powi(3)) * moles_segments).sum()
    }

    fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)> {
        let mut v = Vec::with_capacity(7);
        let d = self.parameters.hs_diameter(state.temperature);

        v.push((
            HardSphere.to_string(),
            HardSphere.helmholtz_energy(&self.parameters, state),
        ));
        v.push((
            HardChain.to_string(),
            HardChain.helmholtz_energy(&self.parameters, state),
        ));
        v.push((
            Dispersion.to_string(),
            Dispersion.helmholtz_energy(&self.parameters, state),
        ));
        if let Some(dipole) = self.dipole.as_ref() {
            v.push((
                dipole.to_string(),
                dipole.helmholtz_energy(&self.parameters, state),
            ))
        }
        if let Some(association) = self.association.as_ref() {
            v.push((
                association.to_string(),
                association.helmholtz_energy(&self.parameters, state, &d),
            ))
        }
        v
    }
}

impl Molarweight for GcPcSaft {
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
    use feos_core::ReferenceSystem;
    use feos_core::StateHD;
    use ndarray::arr1;
    use num_dual::Dual64;
    use quantity::{METER, MOL, PASCAL, Pressure};
    use typenum::P3;

    #[test]
    fn hs_propane() {
        let parameters = propane();
        let temperature = 300.0;
        let volume = METER.powi::<P3>().to_reduced();
        let moles = (1.5 * MOL).to_reduced();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derivative(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let pressure = Pressure::from_reduced(
            -HardSphere.helmholtz_energy(&parameters, &state).eps * temperature,
        );
        assert_relative_eq!(pressure, 1.5285037907989527 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn hs_propanol() {
        let parameters = propanol();
        let temperature = 300.0;
        let volume = METER.powi::<P3>().to_reduced();
        let moles = (1.5 * MOL).to_reduced();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derivative(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let pressure = Pressure::from_reduced(
            -HardSphere.helmholtz_energy(&parameters, &state).eps * temperature,
        );
        assert_relative_eq!(pressure, 2.3168212018200243 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn assoc_propanol() {
        let parameters = propanol();
        let contrib = Association::new(50, 1e-10);
        let temperature = 300.0;
        let volume = METER.powi::<P3>().to_reduced();
        let moles = (1.5 * MOL).to_reduced();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derivative(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let diameter = parameters.hs_diameter(state.temperature);
        let pressure = Pressure::from_reduced(
            -contrib.helmholtz_energy(&parameters, &state, &diameter).eps * temperature,
        );
        assert_relative_eq!(pressure, -3.6819598891967344 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn cross_assoc_propanol() {
        let parameters = propanol();
        let contrib = Association::new_cross_association(50, 1e-10);
        let temperature = 300.0;
        let volume = METER.powi::<P3>().to_reduced();
        let moles = (1.5 * MOL).to_reduced();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derivative(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let diameter = parameters.hs_diameter(state.temperature);
        let pressure = Pressure::from_reduced(
            -contrib.helmholtz_energy(&parameters, &state, &diameter).eps * temperature,
        );
        assert_relative_eq!(pressure, -3.6819598891967344 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn cross_assoc_ethanol_propanol() {
        let parameters = ethanol_propanol(false);
        let contrib = Association::new(50, 1e-10);
        let temperature = 300.0;
        let volume = METER.powi::<P3>().to_reduced();
        let moles = (arr1(&[1.5, 2.5]) * MOL).to_reduced();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derivative(),
            moles.mapv(Dual64::from_re),
        );
        let diameter = parameters.hs_diameter(state.temperature);
        let pressure = Pressure::from_reduced(
            -contrib.helmholtz_energy(&parameters, &state, &diameter).eps * temperature,
        );
        assert_relative_eq!(pressure, -26.105606376765632 * PASCAL, max_relative = 1e-10);
    }
}
