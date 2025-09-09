use super::SaftVRMieParameters;
use crate::association::{Association, AssociationStrength};
use crate::hard_sphere::HardSphereProperties;
use crate::saftvrmie::SaftVRMieRecord;
use crate::saftvrmie::parameters::SaftVRMieAssociationRecord;
use crate::{hard_sphere::HardSphere, saftvrmie::parameters::SaftVRMiePars};
use feos_core::{Molarweight, ResidualDyn, StateHD, Subset};
use nalgebra::DVector;
use num_dual::DualNum;
use quantity::MolarWeight;
use std::f64::consts::{FRAC_PI_6, PI};

// pub(super) mod association;
pub(crate) mod dispersion;
use dispersion::{Properties, helmholtz_energy_density_disp, helmholtz_energy_density_disp_chain};

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
    pub parameters: SaftVRMieParameters,
    pub params: SaftVRMiePars,
    pub options: SaftVRMieOptions,
    pub chain: bool,
    pub association: Option<Association<SaftVRMiePars>>,
}

impl SaftVRMie {
    pub fn new(parameters: SaftVRMieParameters) -> Self {
        Self::with_options(parameters, SaftVRMieOptions::default())
    }

    pub fn with_options(parameters: SaftVRMieParameters, options: SaftVRMieOptions) -> Self {
        let params = SaftVRMiePars::new(&parameters);
        let chain = params.m.iter().any(|&m| m > 1.0);

        let association = Association::new(
            &parameters,
            options.max_iter_cross_assoc,
            options.tol_cross_assoc,
        )
        .unwrap();
        Self {
            parameters,
            params,
            options,
            chain,
            association,
        }
    }
}

// impl Components for SaftVRMie {
//     fn components(&self) -> usize {
//         self.params.m.len()
//     }

//     fn subset(&self, component_list: &[usize]) -> Self {
//         Self::new(self.parameters.subset(component_list))
//     }
// }

impl ResidualDyn for SaftVRMie {
    fn components(&self) -> usize {
        self.params.m.len()
    }

    fn compute_max_density<D: DualNum<f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        let msigma3 = self
            .params
            .m
            .component_mul(&self.params.sigma.map(|v| v.powi(3)));
        (msigma3.map(D::from).dot(molefracs) * FRAC_PI_6).recip() * self.options.max_eta

        // self.options.max_eta * moles.sum()
        //     / (FRAC_PI_6 * &self.params.m * self.params.sigma.mapv(|v| v.powi(3)) * moles).sum()
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        let mut a = Vec::with_capacity(4);

        let (a_hs, _, d) = HardSphere.helmholtz_energy_density_and_properties(&self.params, state);
        a.push(("Hard Sphere", a_hs));

        let properties = Properties::new(&self.params, state, &d);
        if self.chain {
            let a_disp_chain =
                helmholtz_energy_density_disp_chain(&self.params, &properties, state);
            a.push(("Dispersion + Chain", a_disp_chain));
        } else {
            let a_disp = helmholtz_energy_density_disp(&self.params, &properties, state);
            a.push(("Dispersion", a_disp));
        }
        if let Some(assoc) = self.association.as_ref() {
            a.push((
                "Association",
                assoc.helmholtz_energy_density(
                    &self.params,
                    &self.parameters.association,
                    state,
                    &d,
                ),
            ));
        }
        a
    }
}

impl Subset for SaftVRMie {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(self.parameters.subset(component_list), self.options)
    }
}

impl Molarweight for SaftVRMie {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

impl AssociationStrength for SaftVRMiePars {
    type Pure = SaftVRMieRecord;
    type Record = SaftVRMieAssociationRecord;

    fn association_strength<D: DualNum<f64> + Copy>(
        &self,
        temperature: D,
        comp_i: usize,
        comp_j: usize,
        assoc_ij: &Self::Record,
    ) -> D {
        let diameter = self.hs_diameter(temperature);
        let di = diameter[comp_i];
        let dj = diameter[comp_j];
        let d = (di + dj) * 0.5;
        // temperature dependent association volume
        // rc and rd are dimensioned in units of Angstrom
        let rc = assoc_ij.rc_ab;
        // rd is the distance between an association site and the segment centre.
        // It is fixed at 0.4 sigma, leading to 0.4 * 0.5 = 0.2 in the combining rule.
        let rd = (self.sigma[comp_i] + self.sigma[comp_j]) * 0.2;
        let v = d * d * PI * 4.0 / (72.0 * rd.powi(2))
            * ((d.recip() * (rc + 2.0 * rd)).ln()
                * (6.0 * rc.powi(3) + 18.0 * rc.powi(2) * rd - 24.0 * rd.powi(3))
                + (-d + rc + 2.0 * rd)
                    * (d.powi(2) + d * rc + 22.0 * rd.powi(2)
                        - 5.0 * rc * rd
                        - d * 7.0 * rd
                        - 8.0 * rc.powi(2)));
        v * (temperature.recip() * assoc_ij.epsilon_k_ab).exp_m1()
    }

    fn combining_rule(
        pure_i: &Self::Pure,
        pure_j: &Self::Pure,
        parameters_i: &Self::Record,
        parameters_j: &Self::Record,
    ) -> Self::Record {
        let rc_ab = (parameters_i.rc_ab * pure_i.sigma + parameters_j.rc_ab * pure_j.sigma) * 0.5;
        let epsilon_k_ab = (parameters_i.epsilon_k_ab * parameters_j.epsilon_k_ab).sqrt();
        Self::Record {
            rc_ab,
            epsilon_k_ab,
        }
    }
}
