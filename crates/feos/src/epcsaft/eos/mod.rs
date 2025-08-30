use crate::association::Association;
use crate::epcsaft::parameters::{ElectrolytePcSaftParameters, ElectrolytePcSaftPars};
use crate::hard_sphere::{HardSphere, HardSphereProperties};
use feos_core::{FeosResult, ResidualDyn, Subset};
use feos_core::{Molarweight, StateHD};
use nalgebra::DVector;
use num_dual::DualNum;
use quantity::*;
use std::f64::consts::FRAC_PI_6;

pub(crate) mod born;
pub(crate) mod dispersion;
pub(crate) mod hard_chain;
pub(crate) mod ionic;
pub(crate) mod permittivity;
use born::Born;
use dispersion::Dispersion;
use hard_chain::HardChain;
use ionic::Ionic;

/// Implemented variants of the ePC-SAFT equation of state.
#[derive(Copy, Clone, PartialEq)]
pub enum ElectrolytePcSaftVariants {
    Advanced,
    Revised,
}

/// Customization options for the ePC-SAFT equation of state.
#[derive(Copy, Clone)]
pub struct ElectrolytePcSaftOptions {
    pub max_eta: f64,
    pub max_iter_cross_assoc: usize,
    pub tol_cross_assoc: f64,
    pub epcsaft_variant: ElectrolytePcSaftVariants,
}

impl Default for ElectrolytePcSaftOptions {
    fn default() -> Self {
        Self {
            max_eta: 0.5,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
            epcsaft_variant: ElectrolytePcSaftVariants::Advanced,
        }
    }
}

/// electrolyte PC-SAFT (ePC-SAFT) equation of state.
pub struct ElectrolytePcSaft {
    pub parameters: ElectrolytePcSaftParameters,
    pub params: ElectrolytePcSaftPars,
    pub options: ElectrolytePcSaftOptions,
    hard_chain: bool,
    association: Option<Association<ElectrolytePcSaftPars>>,
    ionic: bool,
    born: bool,
}

impl ElectrolytePcSaft {
    pub fn new(parameters: ElectrolytePcSaftParameters) -> FeosResult<Self> {
        Self::with_options(parameters, ElectrolytePcSaftOptions::default())
    }

    pub fn with_options(
        parameters: ElectrolytePcSaftParameters,
        options: ElectrolytePcSaftOptions,
    ) -> FeosResult<Self> {
        let params = ElectrolytePcSaftPars::new(&parameters)?;
        let hard_chain = params.m.iter().any(|m| (m - 1.0).abs() > 1e-15);

        let association = Association::new(
            &parameters,
            options.max_iter_cross_assoc,
            options.tol_cross_assoc,
        )?;

        let ionic = params.nionic > 0;
        let born = ionic && matches!(options.epcsaft_variant, ElectrolytePcSaftVariants::Advanced);

        Ok(Self {
            parameters,
            params,
            options,
            hard_chain,
            association,
            ionic,
            born,
        })
    }
}

impl Subset for ElectrolytePcSaft {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(self.parameters.subset(component_list), self.options).unwrap()
    }
}

impl ResidualDyn for ElectrolytePcSaft {
    fn components(&self) -> usize {
        self.parameters.pure.len()
    }

    fn compute_max_density<D: DualNum<f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        let msigma3 = self
            .params
            .m
            .component_mul(&self.params.sigma.map(|v| v.powi(3)));
        (msigma3.map(D::from).dot(molefracs) * FRAC_PI_6).recip() * self.options.max_eta
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)> {
        let mut v = Vec::with_capacity(7);
        let d = self.params.hs_diameter(state.temperature);

        v.push((
            "Hard Sphere".to_string(),
            HardSphere.helmholtz_energy_density(&self.params, state),
        ));
        if self.hard_chain {
            v.push((
                "Hard Chain".to_string(),
                HardChain.helmholtz_energy_density(&self.params, state),
            ))
        }
        v.push((
            "Dispersion".to_string(),
            Dispersion.helmholtz_energy_density(&self.params, state, &d),
        ));
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
        if self.ionic {
            v.push((
                "Ionic".to_string(),
                Ionic.helmholtz_energy_density(
                    &self.params,
                    state,
                    &d,
                    self.options.epcsaft_variant,
                ),
            ))
        };
        if self.born {
            v.push((
                "Born".to_string(),
                Born.helmholtz_energy_density(&self.params, state, &d),
            ))
        };
        v
    }
}

impl Molarweight for ElectrolytePcSaft {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epcsaft::parameters::utils::{
        butane_parameters, propane_butane_parameters, propane_parameters,
    };
    use approx::assert_relative_eq;
    use feos_core::*;
    use nalgebra::dvector;
    use std::sync::Arc;
    use typenum::P3;

    #[test]
    fn ideal_gas_pressure() {
        let e = Arc::new(ElectrolytePcSaft::new(propane_parameters()).unwrap());
        let t = 200.0 * KELVIN;
        let v = 1e-3 * METER.powi::<P3>();
        let n = dvector![1.0] * MOL;
        let s = State::new_nvt(&e, t, v, &n).unwrap();
        let p_ig = s.total_moles * RGAS * t / v;
        assert_relative_eq!(s.pressure(Contributions::IdealGas), p_ig, epsilon = 1e-10);
        assert_relative_eq!(
            s.pressure(Contributions::IdealGas) + s.pressure(Contributions::Residual),
            s.pressure(Contributions::Total),
            epsilon = 1e-10
        );
    }

    #[test]
    fn ideal_gas_heat_capacity_joback() {
        let e = Arc::new(ElectrolytePcSaft::new(propane_parameters()).unwrap());
        let t = 200.0 * KELVIN;
        let v = 1e-3 * METER.powi::<P3>();
        let n = dvector![1.0] * MOL;
        let s = State::new_nvt(&e, t, v, &n).unwrap();
        let p_ig = s.total_moles * RGAS * t / v;
        assert_relative_eq!(s.pressure(Contributions::IdealGas), p_ig, epsilon = 1e-10);
        assert_relative_eq!(
            s.pressure(Contributions::IdealGas) + s.pressure(Contributions::Residual),
            s.pressure(Contributions::Total),
            epsilon = 1e-10
        );
    }

    #[test]
    fn hard_sphere() {
        let p = ElectrolytePcSaftPars::new(&propane_parameters()).unwrap();
        let t = 250.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, &dvector![n]);
        let a_rust = HardSphere.helmholtz_energy_density(&p, &s) * v;
        assert_relative_eq!(a_rust, 0.410610492598808, epsilon = 1e-10);
    }

    #[test]
    fn hard_sphere_mix() {
        let p1 = ElectrolytePcSaftPars::new(&propane_parameters()).unwrap();
        let p2 = ElectrolytePcSaftPars::new(&butane_parameters()).unwrap();
        let p12 = ElectrolytePcSaftPars::new(&propane_butane_parameters()).unwrap();
        let t = 250.0;
        let v = 2.5e28;
        let n = 1.0;
        let s = StateHD::new(t, v, &dvector![n]);
        let a1 = HardSphere.helmholtz_energy_density(&p1, &s);
        let a2 = HardSphere.helmholtz_energy_density(&p2, &s);
        let s1m = StateHD::new(t, v, &dvector![n, 0.0]);
        let a1m = HardSphere.helmholtz_energy_density(&p12, &s1m);
        let s2m = StateHD::new(t, v, &dvector![0.0, n]);
        let a2m = HardSphere.helmholtz_energy_density(&p12, &s2m);
        assert_relative_eq!(a1, a1m, epsilon = 1e-14);
        assert_relative_eq!(a2, a2m, epsilon = 1e-14);
    }

    #[test]
    fn new_tpn() {
        let e = Arc::new(ElectrolytePcSaft::new(propane_parameters()).unwrap());
        let t = 300.0 * KELVIN;
        let p = BAR;
        let m = dvector![1.0] * MOL;
        let s = State::new_npt(&e, t, p, &m, None);
        let p_calc = if let Ok(state) = s {
            state.pressure(Contributions::Total)
        } else {
            0.0 * PASCAL
        };
        assert_relative_eq!(p, p_calc, epsilon = 1e-6);
    }

    #[test]
    fn vle_pure() {
        let e = Arc::new(ElectrolytePcSaft::new(propane_parameters()).unwrap());
        let t = 300.0 * KELVIN;
        let vle = PhaseEquilibrium::pure(&e, t, None, Default::default());
        if let Ok(v) = vle {
            assert_relative_eq!(
                v.vapor().pressure(Contributions::Total),
                v.liquid().pressure(Contributions::Total),
                epsilon = 1e-6
            )
        }
    }

    #[test]
    fn critical_point() {
        let e = Arc::new(ElectrolytePcSaft::new(propane_parameters()).unwrap());
        let t = 300.0 * KELVIN;
        let cp = State::critical_point(&e, None, Some(t), Default::default());
        if let Ok(v) = cp {
            assert_relative_eq!(v.temperature, 375.1244078318015 * KELVIN, epsilon = 1e-8)
        }
    }

    #[test]
    fn mix_single() {
        let e1 = Arc::new(ElectrolytePcSaft::new(propane_parameters()).unwrap());
        let e2 = Arc::new(ElectrolytePcSaft::new(butane_parameters()).unwrap());
        let e12 = Arc::new(ElectrolytePcSaft::new(propane_butane_parameters()).unwrap());
        let t = 300.0 * KELVIN;
        let v = 0.02456883872966545 * METER.powi::<P3>();
        let m1 = dvector![2.0] * MOL;
        let m1m = dvector![2.0, 0.0] * MOL;
        let m2m = dvector![0.0, 2.0] * MOL;
        let s1 = State::new_nvt(&e1, t, v, &m1).unwrap();
        let s2 = State::new_nvt(&e2, t, v, &m1).unwrap();
        let s1m = State::new_nvt(&e12, t, v, &m1m).unwrap();
        let s2m = State::new_nvt(&e12, t, v, &m2m).unwrap();
        assert_relative_eq!(
            s1.pressure(Contributions::Total),
            s1m.pressure(Contributions::Total),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            s2.pressure(Contributions::Total),
            s2m.pressure(Contributions::Total),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            s2.pressure(Contributions::Total),
            s2m.pressure(Contributions::Total),
            epsilon = 1e-12
        )
    }
}
