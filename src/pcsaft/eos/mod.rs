use super::parameters::PcSaftParameters;
use crate::association::Association;
use crate::hard_sphere::HardSphere;
use feos_core::parameter::Parameter;
use feos_core::si::*;
use feos_core::{
    Components, EntropyScaling, EosError, EosResult, HelmholtzEnergy, Residual, State,
};
use ndarray::Array1;
use std::f64::consts::{FRAC_PI_6, PI};
use std::fmt;
use std::sync::Arc;
use typenum::P2;

pub(crate) mod dispersion;
pub(crate) mod hard_chain;
pub(crate) mod polar;
use dispersion::Dispersion;
use hard_chain::HardChain;
pub use polar::DQVariants;
use polar::{Dipole, DipoleQuadrupole, Quadrupole};

/// Customization options for the PC-SAFT equation of state and functional.
#[derive(Copy, Clone)]
pub struct PcSaftOptions {
    pub max_eta: f64,
    pub max_iter_cross_assoc: usize,
    pub tol_cross_assoc: f64,
    pub dq_variant: DQVariants,
}

impl Default for PcSaftOptions {
    fn default() -> Self {
        Self {
            max_eta: 0.5,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
            dq_variant: DQVariants::DQ35,
        }
    }
}

/// PC-SAFT equation of state.
pub struct PcSaft {
    parameters: Arc<PcSaftParameters>,
    options: PcSaftOptions,
    contributions: Vec<Box<dyn HelmholtzEnergy>>,
}

impl PcSaft {
    pub fn new(parameters: Arc<PcSaftParameters>) -> Self {
        Self::with_options(parameters, PcSaftOptions::default())
    }

    pub fn with_options(parameters: Arc<PcSaftParameters>, options: PcSaftOptions) -> Self {
        let mut contributions: Vec<Box<dyn HelmholtzEnergy>> = Vec::with_capacity(7);
        contributions.push(Box::new(HardSphere::new(&parameters)));
        contributions.push(Box::new(HardChain {
            parameters: parameters.clone(),
        }));
        contributions.push(Box::new(Dispersion {
            parameters: parameters.clone(),
        }));
        if parameters.ndipole > 0 {
            contributions.push(Box::new(Dipole {
                parameters: parameters.clone(),
            }));
        };
        if parameters.nquadpole > 0 {
            contributions.push(Box::new(Quadrupole {
                parameters: parameters.clone(),
            }));
        };
        if parameters.ndipole > 0 && parameters.nquadpole > 0 {
            contributions.push(Box::new(DipoleQuadrupole {
                parameters: parameters.clone(),
                variant: options.dq_variant,
            }));
        };
        if !parameters.association.is_empty() {
            contributions.push(Box::new(Association::new(
                &parameters,
                &parameters.association,
                options.max_iter_cross_assoc,
                options.tol_cross_assoc,
            )));
        };

        Self {
            parameters,
            options,
            contributions,
        }
    }
}

impl Components for PcSaft {
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

impl Residual for PcSaft {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * &self.parameters.m * self.parameters.sigma.mapv(|v| v.powi(3)) * moles)
                .sum()
    }

    fn contributions(&self) -> &[Box<dyn HelmholtzEnergy>] {
        &self.contributions
    }

    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}

impl fmt::Display for PcSaft {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PC-SAFT")
    }
}

fn omega11(t: f64) -> f64 {
    1.06036 * t.powf(-0.15610)
        + 0.19300 * (-0.47635 * t).exp()
        + 1.03587 * (-1.52996 * t).exp()
        + 1.76474 * (-3.89411 * t).exp()
}

fn omega22(t: f64) -> f64 {
    1.16145 * t.powf(-0.14874) + 0.52487 * (-0.77320 * t).exp() + 2.16178 * (-2.43787 * t).exp()
        - 6.435e-4 * t.powf(0.14874) * (18.0323 * t.powf(-0.76830) - 7.27371).sin()
}

#[inline]
fn chapman_enskog_thermal_conductivity(
    temperature: Temperature,
    molarweight: MolarWeight,
    m: f64,
    sigma: f64,
    epsilon_k: f64,
) -> ThermalConductivity {
    let t = temperature.to_reduced();
    0.083235 * (t * m / (molarweight / (GRAM / MOL)).into_value()).sqrt()
        / sigma.powi(2)
        / omega22(t / epsilon_k)
        * WATT
        / METER
        / KELVIN
}

impl EntropyScaling for PcSaft {
    fn viscosity_reference(
        &self,
        temperature: Temperature,
        _: Volume,
        moles: &Moles<Array1<f64>>,
    ) -> EosResult<Viscosity> {
        let p = &self.parameters;
        let mw = &p.molarweight;
        let x = (moles / moles.sum()).into_value();
        let ce: Array1<_> = (0..self.components())
            .map(|i| {
                let tr = (temperature / p.epsilon_k[i] / KELVIN).into_value();
                5.0 / 16.0 * (mw[i] * GRAM / MOL * KB / NAV * temperature / PI).sqrt()
                    / omega22(tr)
                    / (p.sigma[i] * ANGSTROM).powi::<P2>()
            })
            .collect();
        let mut ce_mix = 0.0 * MILLI * PASCAL * SECOND;
        for i in 0..self.components() {
            let denom: f64 = (0..self.components())
                .map(|j| {
                    x[j] * (1.0
                        + (ce[i] / ce[j]).into_value().sqrt() * (mw[j] / mw[i]).powf(1.0 / 4.0))
                    .powi(2)
                        / (8.0 * (1.0 + mw[i] / mw[j])).sqrt()
                })
                .sum();
            ce_mix += ce[i] * x[i] / denom
        }
        Ok(ce_mix)
    }

    fn viscosity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        let coefficients = self
            .parameters
            .viscosity
            .as_ref()
            .expect("Missing viscosity coefficients.");
        let m = (x * &self.parameters.m).sum();
        let s = s_res / m;
        let pref = (x * &self.parameters.m) / m;
        let a: f64 = (&coefficients.row(0) * x).sum();
        let b: f64 = (&coefficients.row(1) * &pref).sum();
        let c: f64 = (&coefficients.row(2) * &pref).sum();
        let d: f64 = (&coefficients.row(3) * &pref).sum();
        Ok(a + b * s + c * s.powi(2) + d * s.powi(3))
    }

    fn diffusion_reference(
        &self,
        temperature: Temperature,
        volume: Volume,
        moles: &Moles<Array1<f64>>,
    ) -> EosResult<Diffusivity> {
        if self.components() != 1 {
            return Err(EosError::IncompatibleComponents(self.components(), 1));
        }
        let p = &self.parameters;
        let density = moles.sum() / volume;
        let res: Array1<_> = (0..self.components())
            .map(|i| {
                let tr = (temperature / p.epsilon_k[i] / KELVIN).into_value();
                3.0 / 8.0 / (p.sigma[i] * ANGSTROM).powi::<P2>() / omega11(tr) / (density * NAV)
                    * (temperature * RGAS / PI / (p.molarweight[i] * GRAM / MOL)).sqrt()
            })
            .collect();
        Ok(res[0])
    }

    fn diffusion_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        if self.components() != 1 {
            return Err(EosError::IncompatibleComponents(self.components(), 1));
        }
        let coefficients = self
            .parameters
            .diffusion
            .as_ref()
            .expect("Missing diffusion coefficients.");
        let m = (x * &self.parameters.m).sum();
        let s = s_res / m;
        let pref = (x * &self.parameters.m).mapv(|v| v / m);
        let a: f64 = (&coefficients.row(0) * x).sum();
        let b: f64 = (&coefficients.row(1) * &pref).sum();
        let c: f64 = (&coefficients.row(2) * &pref).sum();
        let d: f64 = (&coefficients.row(3) * &pref).sum();
        let e: f64 = (&coefficients.row(4) * &pref).sum();
        Ok(a + b * s - c * (1.0 - s.exp()) * s.powi(2) - d * s.powi(4) - e * s.powi(8))
    }

    // Equation 4 of DOI: 10.1021/acs.iecr.9b04289
    fn thermal_conductivity_reference(
        &self,
        temperature: Temperature,
        volume: Volume,
        moles: &Moles<Array1<f64>>,
    ) -> EosResult<ThermalConductivity> {
        if self.components() != 1 {
            return Err(EosError::IncompatibleComponents(self.components(), 1));
        }
        let p = &self.parameters;
        let mws = self.molar_weight();
        let state = State::new_nvt(&Arc::new(Self::new(p.clone())), temperature, volume, moles)?;
        let res: Array1<_> = (0..self.components())
            .map(|i| {
                let tr = (temperature / p.epsilon_k[i] / KELVIN).into_value();
                let s_res_reduced = state.residual_molar_entropy().to_reduced() / p.m[i];
                let ref_ce = chapman_enskog_thermal_conductivity(
                    temperature,
                    mws.get(i),
                    p.m[i],
                    p.sigma[i],
                    p.epsilon_k[i],
                );
                let alpha_visc = (-s_res_reduced / -0.5).exp();
                let ref_ts = (-0.0167141 * tr / p.m[i] + 0.0470581 * (tr / p.m[i]).powi(2))
                    * (p.m[i] * p.m[i] * p.sigma[i].powi(3) * p.epsilon_k[i])
                    * 1e-5
                    * WATT
                    / METER
                    / KELVIN;
                ref_ce + ref_ts * alpha_visc
            })
            .collect();
        Ok(res[0])
    }

    fn thermal_conductivity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        if self.components() != 1 {
            return Err(EosError::IncompatibleComponents(self.components(), 1));
        }
        let coefficients = self
            .parameters
            .thermal_conductivity
            .as_ref()
            .expect("Missing thermal conductivity coefficients");
        let m = (x * &self.parameters.m).sum();
        let s = s_res / m;
        let pref = (x * &self.parameters.m).mapv(|v| v / m);
        let a: f64 = (&coefficients.row(0) * x).sum();
        let b: f64 = (&coefficients.row(1) * &pref).sum();
        let c: f64 = (&coefficients.row(2) * &pref).sum();
        let d: f64 = (&coefficients.row(3) * &pref).sum();
        Ok(a + b * s + c * (1.0 - s.exp()) + d * s.powi(2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcsaft::parameters::utils::{
        butane_parameters, propane_butane_parameters, propane_parameters, water_parameters,
    };
    use approx::assert_relative_eq;
    use feos_core::si::{BAR, KELVIN, METER, MILLI, PASCAL, RGAS, SECOND};
    use feos_core::*;
    use ndarray::arr1;
    use typenum::P3;

    #[test]
    fn ideal_gas_pressure() {
        let e = Arc::new(PcSaft::new(propane_parameters()));
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
        let e = Arc::new(PcSaft::new(propane_parameters()));
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
        let parameters = Arc::new(water_parameters());
        let assoc = Association::new(&parameters, &parameters.association, 50, 1e-10);
        let t = 350.0;
        let v = 41.248289328513216;
        let n = 1.23;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a_rust = assoc.helmholtz_energy(&s) / n;
        assert_relative_eq!(a_rust, -4.229878997054543, epsilon = 1e-10);
    }

    #[test]
    fn cross_association() {
        let parameters = Arc::new(water_parameters());
        let assoc =
            Association::new_cross_association(&parameters, &parameters.association, 50, 1e-10);
        let t = 350.0;
        let v = 41.248289328513216;
        let n = 1.23;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a_rust = assoc.helmholtz_energy(&s) / n;
        assert_relative_eq!(a_rust, -4.229878997054543, epsilon = 1e-10);
    }

    #[test]
    fn new_tpn() {
        let e = Arc::new(PcSaft::new(propane_parameters()));
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
        let e = Arc::new(PcSaft::new(propane_parameters()));
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
        let e = Arc::new(PcSaft::new(propane_parameters()));
        let t = 300.0 * KELVIN;
        let cp = State::critical_point(&e, None, Some(t), Default::default());
        if let Ok(v) = cp {
            assert_relative_eq!(v.temperature, 375.1244078318015 * KELVIN, epsilon = 1e-8)
        }
    }

    #[test]
    fn mix_single() {
        let e1 = Arc::new(PcSaft::new(propane_parameters()));
        let e2 = Arc::new(PcSaft::new(butane_parameters()));
        let e12 = Arc::new(PcSaft::new(propane_butane_parameters()));
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

    #[test]
    fn viscosity() -> EosResult<()> {
        let e = Arc::new(PcSaft::new(propane_parameters()));
        let t = 300.0 * KELVIN;
        let p = BAR;
        let n = arr1(&[1.0]) * MOL;
        let s = State::new_npt(&e, t, p, &n, DensityInitialization::None).unwrap();
        assert_relative_eq!(
            s.viscosity()?,
            0.00797 * MILLI * PASCAL * SECOND,
            epsilon = 1e-5
        );
        assert_relative_eq!(
            s.ln_viscosity_reduced()?,
            (s.viscosity()? / e.viscosity_reference(s.temperature, s.volume, &s.moles)?)
                .into_value()
                .ln(),
            epsilon = 1e-15
        );
        Ok(())
    }

    #[test]
    fn diffusion() -> EosResult<()> {
        let e = Arc::new(PcSaft::new(propane_parameters()));
        let t = 300.0 * KELVIN;
        let p = BAR;
        let n = arr1(&[1.0]) * MOL;
        let s = State::new_npt(&e, t, p, &n, DensityInitialization::None).unwrap();
        assert_relative_eq!(
            s.diffusion()?,
            0.01505 * (CENTI * METER).powi::<P2>() / SECOND,
            epsilon = 1e-5
        );
        assert_relative_eq!(
            s.ln_diffusion_reduced()?,
            (s.diffusion()? / e.diffusion_reference(s.temperature, s.volume, &s.moles)?)
                .into_value()
                .ln(),
            epsilon = 1e-15
        );
        Ok(())
    }
}
