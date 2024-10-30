use crate::association::Association;
use crate::epcsaft::parameters::ElectrolytePcSaftParameters;
use crate::hard_sphere::{HardSphere, HardSphereProperties};
use feos_core::parameter::Parameter;
use feos_core::{Components, Residual};
use feos_core::{Molarweight, StateHD};
use ndarray::Array1;
use num_dual::DualNum;
use quantity::*;
use std::f64::consts::FRAC_PI_6;
use std::fmt;
use std::sync::Arc;

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
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
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
    pub parameters: Arc<ElectrolytePcSaftParameters>,
    pub options: ElectrolytePcSaftOptions,
    hard_sphere: HardSphere<ElectrolytePcSaftParameters>,
    hard_chain: Option<HardChain>,
    dispersion: Dispersion,
    association: Option<Association<ElectrolytePcSaftParameters>>,
    ionic: Option<Ionic>,
    born: Option<Born>,
}

impl ElectrolytePcSaft {
    pub fn new(parameters: Arc<ElectrolytePcSaftParameters>) -> Self {
        Self::with_options(parameters, ElectrolytePcSaftOptions::default())
    }

    pub fn with_options(
        parameters: Arc<ElectrolytePcSaftParameters>,
        options: ElectrolytePcSaftOptions,
    ) -> Self {
        let hard_sphere = HardSphere::new(&parameters);
        let dispersion = Dispersion {
            parameters: parameters.clone(),
        };
        let hard_chain = if parameters.m.iter().any(|m| (m - 1.0).abs() > 1e-15) {
            Some(HardChain {
                parameters: parameters.clone(),
            })
        } else {
            None
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

        let ionic = if parameters.nionic > 0 {
            Some(Ionic {
                parameters: parameters.clone(),
                variant: options.epcsaft_variant,
            })
        } else {
            None
        };

        let born = if parameters.nionic > 0 {
            match options.epcsaft_variant {
                ElectrolytePcSaftVariants::Revised => None,
                ElectrolytePcSaftVariants::Advanced => Some(Born {
                    parameters: parameters.clone(),
                }),
            }
        } else {
            None
        };

        match options.epcsaft_variant {
            ElectrolytePcSaftVariants::Revised => {
                if ionic.is_some() {
                    panic!("Ionic contribution is not available in the revised ePC-SAFT variant.")
                }
            }
            ElectrolytePcSaftVariants::Advanced => (),
        }

        Self {
            parameters,
            options,
            hard_sphere,
            hard_chain,
            dispersion,
            association,
            ionic,
            born,
        }
    }
}

impl Components for ElectrolytePcSaft {
    fn components(&self) -> usize {
        self.parameters.pure_records.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            Arc::new(self.parameters.subset(component_list)),
            self.options,
        )
    }
}

impl Residual for ElectrolytePcSaft {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * &self.parameters.m * self.parameters.sigma.mapv(|v| v.powi(3)) * moles)
                .sum()
    }

    fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)> {
        let mut v = Vec::with_capacity(7);
        let d = self.parameters.hs_diameter(state.temperature);

        v.push((
            self.hard_sphere.to_string(),
            self.hard_sphere.helmholtz_energy(state),
        ));
        if let Some(hc) = self.hard_chain.as_ref() {
            v.push((hc.to_string(), hc.helmholtz_energy(state)))
        }
        v.push((
            self.dispersion.to_string(),
            self.dispersion.helmholtz_energy(state, &d),
        ));
        if let Some(association) = self.association.as_ref() {
            v.push((
                association.to_string(),
                association.helmholtz_energy(state, &d),
            ))
        }
        if let Some(ionic) = self.ionic.as_ref() {
            v.push((ionic.to_string(), ionic.helmholtz_energy(state, &d)))
        };
        if let Some(born) = self.born.as_ref() {
            v.push((born.to_string(), born.helmholtz_energy(state, &d)))
        };
        v
    }
}

impl Molarweight for ElectrolytePcSaft {
    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}

impl fmt::Display for ElectrolytePcSaft {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ePC-SAFT")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epcsaft::parameters::utils::{
        butane_parameters, propane_butane_parameters, propane_parameters, water_parameters,
    };
    use approx::assert_relative_eq;
    use feos_core::*;
    use ndarray::arr1;
    use typenum::P3;

    #[test]
    fn ideal_gas_pressure() {
        let e = Arc::new(ElectrolytePcSaft::new(propane_parameters()));
        let t = 200.0 * KELVIN;
        let v = 1e-3 * METER.powi::<P3>();
        let n = arr1(&[1.0]) * MOL;
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
        let e = Arc::new(ElectrolytePcSaft::new(propane_parameters()));
        let t = 200.0 * KELVIN;
        let v = 1e-3 * METER.powi::<P3>();
        let n = arr1(&[1.0]) * MOL;
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
        let hs = HardSphere::new(&propane_parameters());
        let t = 250.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a_rust = hs.helmholtz_energy(&s);
        assert_relative_eq!(a_rust, 0.410610492598808, epsilon = 1e-10);
    }

    #[test]
    fn hard_sphere_mix() {
        let c1 = HardSphere::new(&propane_parameters());
        let c2 = HardSphere::new(&butane_parameters());
        let c12 = HardSphere::new(&propane_butane_parameters());
        let t = 250.0;
        let v = 2.5e28;
        let n = 1.0;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a1 = c1.helmholtz_energy(&s);
        let a2 = c2.helmholtz_energy(&s);
        let s1m = StateHD::new(t, v, arr1(&[n, 0.0]));
        let a1m = c12.helmholtz_energy(&s1m);
        let s2m = StateHD::new(t, v, arr1(&[0.0, n]));
        let a2m = c12.helmholtz_energy(&s2m);
        assert_relative_eq!(a1, a1m, epsilon = 1e-14);
        assert_relative_eq!(a2, a2m, epsilon = 1e-14);
    }

    #[test]
    fn association() {
        let parameters = water_parameters();
        let assoc = Association::new(&parameters, &parameters.association, 50, 1e-10);
        let t = 350.0;
        let v = 41.248289328513216;
        let n = 1.23;
        let s = StateHD::new(t, v, arr1(&[n]));
        let d = parameters.hs_diameter(t);
        let a_rust = assoc.helmholtz_energy(&s, &d) / n;
        assert_relative_eq!(a_rust, -4.229878997054543, epsilon = 1e-10);
    }

    #[test]
    fn cross_association() {
        let parameters = water_parameters();
        let assoc =
            Association::new_cross_association(&parameters, &parameters.association, 50, 1e-10);
        let t = 350.0;
        let v = 41.248289328513216;
        let n = 1.23;
        let s = StateHD::new(t, v, arr1(&[n]));
        let d = parameters.hs_diameter(t);
        let a_rust = assoc.helmholtz_energy(&s, &d) / n;
        assert_relative_eq!(a_rust, -4.229878997054543, epsilon = 1e-10);
    }

    #[test]
    fn new_tpn() {
        let e = Arc::new(ElectrolytePcSaft::new(propane_parameters()));
        let t = 300.0 * KELVIN;
        let p = BAR;
        let m = arr1(&[1.0]) * MOL;
        let s = State::new_npt(&e, t, p, &m, DensityInitialization::None);
        let p_calc = if let Ok(state) = s {
            state.pressure(Contributions::Total)
        } else {
            0.0 * PASCAL
        };
        assert_relative_eq!(p, p_calc, epsilon = 1e-6);
    }

    #[test]
    fn vle_pure() {
        let e = Arc::new(ElectrolytePcSaft::new(propane_parameters()));
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
        let e = Arc::new(ElectrolytePcSaft::new(propane_parameters()));
        let t = 300.0 * KELVIN;
        let cp = State::critical_point(&e, None, Some(t), Default::default());
        if let Ok(v) = cp {
            assert_relative_eq!(v.temperature, 375.1244078318015 * KELVIN, epsilon = 1e-8)
        }
    }

    #[test]
    fn mix_single() {
        let e1 = Arc::new(ElectrolytePcSaft::new(propane_parameters()));
        let e2 = Arc::new(ElectrolytePcSaft::new(butane_parameters()));
        let e12 = Arc::new(ElectrolytePcSaft::new(propane_butane_parameters()));
        let t = 300.0 * KELVIN;
        let v = 0.02456883872966545 * METER.powi::<P3>();
        let m1 = arr1(&[2.0]) * MOL;
        let m1m = arr1(&[2.0, 0.0]) * MOL;
        let m2m = arr1(&[0.0, 2.0]) * MOL;
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
