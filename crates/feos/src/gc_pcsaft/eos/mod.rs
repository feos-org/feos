use super::record::GcPcSaftParameters;
use crate::association::Association;
use crate::hard_sphere::{HardSphere, HardSphereProperties};
use feos_core::{Molarweight, ResidualDyn, Subset};
use nalgebra::DVector;
use num_dual::DualNum;
use quantity::MolarWeight;
use std::f64::consts::FRAC_PI_6;

mod ad;
pub(crate) mod dispersion;
mod hard_chain;
pub(crate) mod parameter;
mod polar;
pub use ad::{GcPcSaftAD, GcPcSaftADParameters};
use dispersion::Dispersion;
use hard_chain::HardChain;
pub use parameter::GcPcSaftEosParameters;
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
    parameters: GcPcSaftParameters<f64>,
    params: GcPcSaftEosParameters,
    options: GcPcSaftOptions,
    association: Option<Association<GcPcSaftEosParameters>>,
    dipole: Option<Dipole>,
}

impl GcPcSaft {
    pub fn new(parameters: GcPcSaftParameters<f64>) -> Self {
        Self::with_options(parameters, GcPcSaftOptions::default())
    }

    pub fn with_options(parameters: GcPcSaftParameters<f64>, options: GcPcSaftOptions) -> Self {
        let params = GcPcSaftEosParameters::new(&parameters);
        let association = Association::new(
            &parameters,
            options.max_iter_cross_assoc,
            options.tol_cross_assoc,
        )
        .unwrap();
        let dipole = if !params.dipole_comp.is_empty() {
            Some(Dipole::new(&params))
        } else {
            None
        };
        Self {
            parameters,
            params,
            options,
            association,
            dipole,
        }
    }
}

impl ResidualDyn for GcPcSaft {
    fn components(&self) -> usize {
        self.parameters.molar_weight.len()
    }

    fn compute_max_density<D: DualNum<f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        let p = &self.params;
        let molefracs_segments = DVector::from(p.component_index.clone()).map(|i| molefracs[i]);
        (p.m.component_mul(&p.sigma.map(|v| v.powi(3)))
            .map(D::from)
            .dot(&molefracs_segments)
            * FRAC_PI_6)
            .recip()
            * self.options.max_eta
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &feos_core::StateHD<D>,
    ) -> Vec<(String, D)> {
        let mut v = Vec::with_capacity(7);
        let d = self.params.hs_diameter(state.temperature);

        v.push((
            "Hard Sphere".to_string(),
            HardSphere.helmholtz_energy_density(&self.params, state),
        ));
        v.push((
            "Hard Chain".to_string(),
            HardChain.helmholtz_energy_density(&self.params, state),
        ));
        v.push((
            "Dispersion".to_string(),
            Dispersion.helmholtz_energy_density(&self.params, state),
        ));
        if let Some(dipole) = self.dipole.as_ref() {
            v.push((
                "Dipole".to_string(),
                dipole.helmholtz_energy_density(&self.params, state),
            ))
        }
        if let Some(association) = self.association.as_ref() {
            v.push((
                "Association".to_string(),
                association.helmholtz_energy_density(
                    &self.params,
                    &self.parameters.association,
                    state,
                    &d,
                ),
            ))
        }
        v
    }
}

impl Subset for GcPcSaft {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(self.parameters.subset(component_list), self.options)
    }
}

impl Molarweight for GcPcSaft {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gc_pcsaft::eos::parameter::test::*;
    use approx::assert_relative_eq;
    use feos_core::ReferenceSystem;
    use feos_core::StateHD;
    use nalgebra::dvector;
    use num_dual::Dual64;
    use quantity::{METER, MOL, PASCAL, Pressure};
    use typenum::P3;

    #[test]
    fn hs_propane() {
        let parameters = propane();
        let temperature = 300.0;
        let volume = Dual64::from_re(METER.powi::<P3>().to_reduced()).derivative();
        let moles = Dual64::from_re((1.5 * MOL).to_reduced());
        let molar_volume = volume / moles;
        let state = StateHD::new(
            Dual64::from_re(temperature),
            molar_volume,
            &dvector![Dual64::from_re(1.0)],
        );
        let pressure = Pressure::from_reduced(
            -(HardSphere.helmholtz_energy_density(&parameters, &state) * volume).eps * temperature,
        );
        assert_relative_eq!(pressure, 1.5285037907989527 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn hs_propanol() {
        let parameters = GcPcSaftEosParameters::new(&propanol());
        let temperature = 300.0;
        let volume = Dual64::from_re(METER.powi::<P3>().to_reduced()).derivative();
        let moles = Dual64::from_re((1.5 * MOL).to_reduced());
        let molar_volume = volume / moles;
        let state = StateHD::new(
            Dual64::from_re(temperature),
            molar_volume,
            &dvector![Dual64::from_re(1.0)],
        );
        let pressure = Pressure::from_reduced(
            -(HardSphere.helmholtz_energy_density(&parameters, &state) * volume).eps * temperature,
        );
        assert_relative_eq!(pressure, 2.3168212018200243 * PASCAL, max_relative = 1e-10);
    }
}
