use super::parameters::PetsParameters;
use crate::hard_sphere::HardSphere;
use feos_core::parameter::Parameter;
use feos_core::si::{MolarWeight, GRAM, MOL};
use feos_core::{Components, HelmholtzEnergy, Residual};
use ndarray::Array1;
use std::f64::consts::FRAC_PI_6;
use std::sync::Arc;

pub(crate) mod dispersion;
use dispersion::Dispersion;

/// Configuration options for the PeTS equation of state and Helmholtz energy functional.
///
/// The maximum packing fraction is used to infer initial values
/// for routines that depend on starting values for the system density.
#[derive(Copy, Clone)]
pub struct PetsOptions {
    /// maximum packing fraction
    pub max_eta: f64,
}

impl Default for PetsOptions {
    fn default() -> Self {
        Self { max_eta: 0.5 }
    }
}

/// PeTS equation of state.
pub struct Pets {
    parameters: Arc<PetsParameters>,
    options: PetsOptions,
    contributions: Vec<Box<dyn HelmholtzEnergy>>,
}

impl Pets {
    /// PeTS equation of state with default options.
    pub fn new(parameters: Arc<PetsParameters>) -> Self {
        Self::with_options(parameters, PetsOptions::default())
    }

    /// PeTS equation of state with provided options.
    pub fn with_options(parameters: Arc<PetsParameters>, options: PetsOptions) -> Self {
        let contributions: Vec<Box<dyn HelmholtzEnergy>> = vec![
            Box::new(HardSphere::new(&parameters)),
            Box::new(Dispersion {
                parameters: parameters.clone(),
            }),
        ];
        Self {
            parameters,
            options,
            contributions,
        }
    }
}

impl Components for Pets {
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

impl Residual for Pets {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * self.parameters.sigma.mapv(|v| v.powi(3)) * moles).sum()
    }

    fn contributions(&self) -> &[Box<dyn HelmholtzEnergy>] {
        &self.contributions
    }

    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}

// fn omega11(t: f64) -> f64 {
//     1.06036 * t.powf(-0.15610)
//         + 0.19300 * (-0.47635 * t).exp()
//         + 1.03587 * (-1.52996 * t).exp()
//         + 1.76474 * (-3.89411 * t).exp()
// }

// fn omega22(t: f64) -> f64 {
//     1.16145 * t.powf(-0.14874) + 0.52487 * (-0.77320 * t).exp() + 2.16178 * (-2.43787 * t).exp()
//         - 6.435e-4 * t.powf(0.14874) * (18.0323 * t.powf(-0.76830) - 7.27371).sin()
// }

// impl EntropyScaling for Pets {
//     fn viscosity_reference(
//         &self,
//         temperature: SINumber,
//         _: SINumber,
//         moles: &SIArray1,
//     ) -> EosResult<SINumber> {
//         let x = moles.to_reduced(moles.sum())?;
//         let p = &self.parameters;
//         let mw = &p.molarweight;
//         let ce: Array1<SINumber> = (0..self.components())
//             .map(|i| {
//                 let tr = (temperature / p.epsilon_k[i] / KELVIN)
//                     .into_value()
//                     .unwrap();
//                 5.0 / 16.0
//                     * (mw[i] * GRAM / MOL * KB / NAV * temperature / PI)
//                         .sqrt()
//                         .unwrap()
//                     / omega22(tr)
//                     / (p.sigma[i] * ANGSTROM).powi(2)
//             })
//             .collect();
//         let mut ce_mix = 0.0 * MILLI * PASCAL * SECOND;
//         for i in 0..self.components() {
//             let denom: f64 = (0..self.components())
//                 .map(|j| {
//                     x[j] * (1.0
//                         + (ce[i] / ce[j]).into_value().unwrap().sqrt()
//                             * (mw[j] / mw[i]).powf(1.0 / 4.0))
//                     .powi(2)
//                         / (8.0 * (1.0 + mw[i] / mw[j])).sqrt()
//                 })
//                 .sum();
//             ce_mix += ce[i] * x[i] / denom
//         }
//         Ok(ce_mix)
//     }

//     fn viscosity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
//         let coefficients = self
//             .parameters
//             .viscosity
//             .as_ref()
//             .expect("Missing viscosity coefficients.");
//         let a: f64 = (&coefficients.row(0) * x).sum();
//         let b: f64 = (&coefficients.row(1) * x).sum();
//         let c: f64 = (&coefficients.row(2) * x).sum();
//         let d: f64 = (&coefficients.row(3) * x).sum();
//         Ok(a + b * s_res + c * s_res.powi(2) + d * s_res.powi(3))
//     }

//     fn diffusion_reference(
//         &self,
//         temperature: SINumber,
//         volume: SINumber,
//         moles: &SIArray1,
//     ) -> EosResult<SINumber> {
//         if self.components() != 1 {
//             return Err(EosError::IncompatibleComponents(self.components(), 1));
//         }
//         let p = &self.parameters;
//         let density = moles.sum() / volume;
//         let res: Array1<SINumber> = (0..self.components())
//             .map(|i| {
//                 let tr = (temperature / p.epsilon_k[i] / KELVIN)
//                     .into_value()
//                     .unwrap();
//                 3.0 / 8.0 / (p.sigma[i] * ANGSTROM).powi(2) / omega11(tr) / (density * NAV)
//                     * (temperature * RGAS / PI / (p.molarweight[i] * GRAM / MOL))
//                         .sqrt()
//                         .unwrap()
//             })
//             .collect();
//         Ok(res[0])
//     }

//     fn diffusion_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
//         if self.components() != 1 {
//             return Err(EosError::IncompatibleComponents(self.components(), 1));
//         }
//         let coefficients = self
//             .parameters
//             .diffusion
//             .as_ref()
//             .expect("Missing diffusion coefficients.");
//         let a: f64 = (&coefficients.row(0) * x).sum();
//         let b: f64 = (&coefficients.row(1) * x).sum();
//         let c: f64 = (&coefficients.row(2) * x).sum();
//         let d: f64 = (&coefficients.row(3) * x).sum();
//         let e: f64 = (&coefficients.row(4) * x).sum();
//         Ok(a + b * s_res
//             - c * (1.0 - s_res.exp()) * s_res.powi(2)
//             - d * s_res.powi(4)
//             - e * s_res.powi(8))
//     }

//     // fn thermal_conductivity_reference(
//     //     &self,
//     //     state: &State<E>,
//     // ) -> EosResult<SINumber> {
//     //     if self.components() != 1 {
//     //         return Err(EosError::IncompatibleComponents(self.components(), 1));
//     //     }
//     //     let p = &self.parameters;
//     //     let res: Array1<SINumber> = (0..self.components())
//     //         .map(|i| {
//     //             let tr = (state.temperature / p.epsilon_k[i] / KELVIN)
//     //                 .into_value()
//     //                 .unwrap();
//     //             let cp = State::critical_point_pure(&state.eos, Some(state.temperature)).unwrap();
//     //             let s_res_cp_reduced = cp
//     //                 .entropy(Contributions::Residual)
//     //                 .to_reduced(SIUnit::reference_entropy())
//     //                 .unwrap();
//     //             let s_res_reduced = cp
//     //                 .entropy(Contributions::Residual)
//     //                 .to_reduced(SIUnit::reference_entropy())
//     //                 .unwrap();
//     //             let ref_ce = 0.083235
//     //                 * ((state.temperature / KELVIN).into_value().unwrap()
//     //                     / (p.molarweight[0]))
//     //                     .sqrt()
//     //                 / p.sigma[0]
//     //                 / p.sigma[0]
//     //                 / omega22(tr);
//     //             let alpha_visc = (-s_res_reduced / s_res_cp_reduced).exp();
//     //             let ref_ts = (-0.0167141 * tr + 0.0470581 * (tr).powi(2))
//     //                 * (p.sigma[i].powi(3) * p.epsilon_k[0])
//     //                 / 100000.0;
//     //             (ref_ce + ref_ts * alpha_visc) * WATT / METER / KELVIN
//     //         })
//     //         .collect();
//     //     Ok(res[0])
//     // }

//     // Equation 11 of DOI: 10.1021/acs.iecr.9b03998
//     fn thermal_conductivity_reference(
//         &self,
//         temperature: SINumber,
//         volume: SINumber,
//         moles: &SIArray1,
//     ) -> EosResult<SINumber> {
//         if self.components() != 1 {
//             return Err(EosError::IncompatibleComponents(self.components(), 1));
//         }
//         let p = &self.parameters;
//         let state = State::new_nvt(
//             &Arc::new(Self::new(self.parameters.clone())),
//             temperature,
//             volume,
//             moles,
//         )?;
//         let res: Array1<SINumber> = (0..self.components())
//             .map(|i| {
//                 let tr = (temperature / p.epsilon_k[i] / KELVIN)
//                     .into_value()
//                     .unwrap();
//                 let ce = 83.235
//                     * f64::powf(10.0, -1.5)
//                     * ((temperature / KELVIN).into_value().unwrap() / p.molarweight[0]).sqrt()
//                     / (p.sigma[0] * p.sigma[0])
//                     / omega22(tr);
//                 ce * WATT / METER / KELVIN
//                     + state.density
//                         * self
//                             .diffusion_reference(temperature, volume, moles)
//                             .unwrap()
//                         * self
//                             .diffusion_correlation(
//                                 state
//                                     .residual_molar_entropy()
//                                     .to_reduced(SIUnit::reference_molar_entropy())
//                                     .unwrap(),
//                                 &state.molefracs,
//                             )
//                             .unwrap()
//                         * (state.c_v(Contributions::Total) - 1.5 * RGAS)
//             })
//             .collect();
//         Ok(res[0])
//     }

//     fn thermal_conductivity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
//         if self.components() != 1 {
//             return Err(EosError::IncompatibleComponents(self.components(), 1));
//         }
//         let coefficients = self
//             .parameters
//             .thermal_conductivity
//             .as_ref()
//             .expect("Missing thermal conductivity coefficients");
//         let a: f64 = (&coefficients.row(0) * x).sum();
//         let b: f64 = (&coefficients.row(1) * x).sum();
//         let c: f64 = (&coefficients.row(2) * x).sum();
//         let d: f64 = (&coefficients.row(3) * x).sum();
//         Ok(a + b * s_res + c * (1.0 - s_res.exp()) + d * s_res.powi(2))
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pets::parameters::utils::{
        argon_krypton_parameters, argon_parameters, krypton_parameters,
    };
    use approx::assert_relative_eq;
    use feos_core::si::{BAR, KELVIN, METER, PASCAL, RGAS};
    use feos_core::{
        Contributions, DensityInitialization, HelmholtzEnergyDual, PhaseEquilibrium, State, StateHD,
    };
    use ndarray::arr1;
    use typenum::P3;

    #[test]
    fn ideal_gas_pressure() {
        let e = Arc::new(Pets::new(argon_parameters()));
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
    fn hard_sphere_mix() {
        let c1 = HardSphere::new(&argon_parameters());
        let c2 = HardSphere::new(&krypton_parameters());
        let c12 = HardSphere::new(&argon_krypton_parameters());
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
    fn new_tpn() {
        let e = Arc::new(Pets::new(argon_parameters()));
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
    fn vle_pure_t() {
        let e = Arc::new(Pets::new(argon_parameters()));
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

    // #[test]
    // fn critical_point() {
    //     let e = Arc::new(Pets::new(argon_parameters()));
    //     let t = 300.0 * KELVIN;
    //     let cp = State::critical_point(&e, None, Some(t), Default::default());
    //     if let Ok(v) = cp {
    //         assert_relative_eq!(v.temperature, 375.1244078318015 * KELVIN, epsilon = 1e-8)
    //     }
    // }

    // #[test]
    // fn speed_of_sound() {
    //     let e = Arc::new(Pets::new(argon_parameters()));
    //     let t = 300.0 * KELVIN;
    //     let p = BAR;
    //     let m = arr1(&[1.0]) * MOL;
    //     let s = State::new_npt(&e, t, p, &m, DensityInitialization::None).unwrap();
    //     assert_relative_eq!(
    //         s.speed_of_sound(),
    //         245.00185709137546 * METER / SECOND,
    //         epsilon = 1e-4
    //     )
    // }

    #[test]
    fn mix_single() {
        let e1 = Arc::new(Pets::new(argon_parameters()));
        let e2 = Arc::new(Pets::new(krypton_parameters()));
        let e12 = Arc::new(Pets::new(argon_krypton_parameters()));
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
        )
    }

    // #[test]
    // fn viscosity() -> EosResult<()> {
    //     let e = Arc::new(Pets::new(argon_parameters()));
    //     let t = 300.0 * KELVIN;
    //     let p = BAR;
    //     let n = arr1(&[1.0]) * MOL;
    //     let s = State::new_npt(&e, t, p, &n, DensityInitialization::None).unwrap();
    //     assert_relative_eq!(
    //         s.viscosity()?,
    //         0.00797 * MILLI * PASCAL * SECOND,
    //         epsilon = 1e-5
    //     );
    //     assert_relative_eq!(
    //         s.ln_viscosity_reduced()?,
    //         (s.viscosity()? / e.viscosity_reference(s.temperature, s.volume, &s.moles)?)
    //             .into_value()
    //             .unwrap()
    //             .ln(),
    //         epsilon = 1e-15
    //     );
    //     Ok(())
    // }

    // #[test]
    // fn diffusion() -> EosResult<()> {
    //     let e = Arc::new(Pets::new(argon_parameters()));
    //     let t = 300.0 * KELVIN;
    //     let p = BAR;
    //     let n = arr1(&[1.0]) * MOL;
    //     let s = State::new_npt(&e, t, p, &n, DensityInitialization::None).unwrap();
    //     assert_relative_eq!(
    //         s.diffusion()?,
    //         0.01505 * (CENTI * METER).powi(2) / SECOND,
    //         epsilon = 1e-5
    //     );
    //     assert_relative_eq!(
    //         s.ln_diffusion_reduced()?,
    //         (s.diffusion()? / e.diffusion_reference(s.temperature, s.volume, &s.moles)?)
    //             .into_value()
    //             .unwrap()
    //             .ln(),
    //         epsilon = 1e-15
    //     );
    //     Ok(())
    // }
}
