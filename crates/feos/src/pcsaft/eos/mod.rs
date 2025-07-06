use super::parameters::{PcSaftAssociationRecord, PcSaftParameters, PcSaftPars};
use crate::association::{Association, AssociationStrength};
use crate::hard_sphere::{HardSphere, HardSphereProperties, MonomerShape};
use crate::pcsaft::PcSaftRecord;
use feos_core::{
    Components, EntropyScaling, FeosError, FeosResult, Molarweight, ReferenceSystem, Residual,
    StateHD,
};
use ndarray::Array1;
use num_dual::{Dual64, DualNum};
use quantity::*;
use std::f64::consts::{FRAC_PI_6, PI};
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
    parameters: PcSaftParameters,
    params: PcSaftPars,
    options: PcSaftOptions,
    hard_chain: bool,
    dipole: bool,
    quadrupole: bool,
    dipole_quadrupole: bool,
    association: Option<Association<PcSaftPars>>,
}

impl PcSaft {
    pub fn new(parameters: PcSaftParameters) -> Self {
        Self::with_options(parameters, PcSaftOptions::default())
    }

    pub fn with_options(parameters: PcSaftParameters, options: PcSaftOptions) -> Self {
        let params = PcSaftPars::new(&parameters);
        let hard_chain = params.m.iter().any(|m| (m - 1.0).abs() > 1e-15);

        let dipole = params.ndipole > 0;
        let quadrupole = params.nquadpole > 0;
        let dipole_quadrupole = params.ndipole > 0 && params.nquadpole > 0;

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
            hard_chain,
            dipole,
            quadrupole,
            dipole_quadrupole,
            association,
        }
    }
}

impl Components for PcSaft {
    fn components(&self) -> usize {
        self.parameters.pure.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(self.parameters.subset(component_list), self.options)
    }
}

impl Residual for PcSaft {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * &self.params.m * self.params.sigma.mapv(|v| v.powi(3)) * moles).sum()
    }

    fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)> {
        let mut v = Vec::with_capacity(7);
        let d = self.params.hs_diameter(state.temperature);

        v.push((
            "Hard Sphere".to_string(),
            HardSphere.helmholtz_energy(&self.params, state),
        ));
        if self.hard_chain {
            v.push((
                "Hard Chain".to_string(),
                HardChain.helmholtz_energy(&self.params, state),
            ))
        }
        v.push((
            "Dispersion".to_string(),
            Dispersion.helmholtz_energy(&self.params, state, &d),
        ));
        if self.dipole {
            v.push((
                "Dipole".to_string(),
                Dipole.helmholtz_energy(&self.params, state, &d),
            ))
        }
        if self.quadrupole {
            v.push((
                "Quadrupole".to_string(),
                Quadrupole.helmholtz_energy(&self.params, state, &d),
            ))
        }
        if self.dipole_quadrupole {
            v.push((
                "DipoleQuadrupole".to_string(),
                DipoleQuadrupole.helmholtz_energy(&self.params, state, &d, self.options.dq_variant),
            ))
        }
        if let Some(association) = self.association.as_ref() {
            v.push((
                "Association".to_string(),
                association.helmholtz_energy(&self.params, &self.parameters.association, state, &d),
            ))
        }
        v
    }
}

impl Molarweight for PcSaft {
    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molar_weight.clone()
    }
}

impl HardSphereProperties for PcSaftPars {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<N> {
        MonomerShape::NonSpherical(self.m.mapv(N::from))
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D> {
        let ti = temperature.recip() * -3.0;
        Array1::from_shape_fn(self.sigma.len(), |i| {
            -((ti * self.epsilon_k[i]).exp() * 0.12 - 1.0) * self.sigma[i]
        })
    }
}

impl AssociationStrength for PcSaftPars {
    type Pure = PcSaftRecord;
    type Record = PcSaftAssociationRecord;

    fn association_strength<D: DualNum<f64> + Copy>(
        &self,
        temperature: D,
        comp_i: usize,
        comp_j: usize,
        assoc_ij: &Self::Record,
    ) -> D {
        let si = self.sigma[comp_i];
        let sj = self.sigma[comp_j];
        (temperature.recip() * assoc_ij.epsilon_k_ab).exp_m1()
            * assoc_ij.kappa_ab
            * (si * sj).powf(1.5)
    }

    fn combining_rule(
        _: &Self::Pure,
        _: &Self::Pure,
        parameters_i: &Self::Record,
        parameters_j: &Self::Record,
    ) -> Self::Record {
        let kappa_ab = (parameters_i.kappa_ab * parameters_j.kappa_ab).sqrt();
        let epsilon_k_ab = 0.5 * (parameters_i.epsilon_k_ab + parameters_j.epsilon_k_ab);
        Self::Record {
            kappa_ab,
            epsilon_k_ab,
        }
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
    0.083235 * (t * m / molarweight.convert_to(GRAM / MOL)).sqrt()
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
    ) -> FeosResult<Viscosity> {
        let p = &self.params;
        let mw = &self.parameters.molar_weight;
        let x = (moles / moles.sum()).into_value();
        let ce: Array1<_> = (0..self.components())
            .map(|i| {
                let tr = (temperature / p.epsilon_k[i] / KELVIN).into_value();
                5.0 / 16.0 * (mw.get(i) * KB / NAV * temperature / PI).sqrt()
                    / omega22(tr)
                    / (p.sigma[i] * ANGSTROM).powi::<P2>()
            })
            .collect();
        let mut ce_mix = 0.0 * MILLI * PASCAL * SECOND;
        for i in 0..self.components() {
            let denom: f64 = (0..self.components())
                .map(|j| {
                    x[j] * (1.0
                        + (ce[i] / ce[j]).into_value().sqrt()
                            * (mw.get(j) / mw.get(i)).powf(1.0 / 4.0))
                    .powi(2)
                        / (8.0 * (1.0 + (mw.get(i) / mw.get(j)).into_value())).sqrt()
                })
                .sum();
            ce_mix += ce[i] * x[i] / denom
        }
        Ok(ce_mix)
    }

    fn viscosity_correlation(&self, s_res: f64, x: &Array1<f64>) -> FeosResult<f64> {
        let coefficients = self
            .params
            .viscosity
            .as_ref()
            .expect("Missing viscosity coefficients.");
        let m = (x * &self.params.m).sum();
        let s = s_res / m;
        let pref = (x * &self.params.m) / m;
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
    ) -> FeosResult<Diffusivity> {
        if self.components() != 1 {
            return Err(FeosError::IncompatibleComponents(self.components(), 1));
        }
        let p = &self.params;
        let mw = &self.parameters.molar_weight;
        let density = moles.sum() / volume;
        let res: Array1<_> = (0..self.components())
            .map(|i| {
                let tr = (temperature / p.epsilon_k[i] / KELVIN).into_value();
                3.0 / 8.0 / (p.sigma[i] * ANGSTROM).powi::<P2>() / omega11(tr) / (density * NAV)
                    * (temperature * RGAS / PI / mw.get(i) / p.m[i]).sqrt()
            })
            .collect();
        Ok(res[0])
    }

    fn diffusion_correlation(&self, s_res: f64, x: &Array1<f64>) -> FeosResult<f64> {
        if self.components() != 1 {
            return Err(FeosError::IncompatibleComponents(self.components(), 1));
        }
        let coefficients = self
            .params
            .diffusion
            .as_ref()
            .expect("Missing diffusion coefficients.");
        let m = (x * &self.params.m).sum();
        let s = s_res / m;
        let pref = (x * &self.params.m).mapv(|v| v / m);
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
    ) -> FeosResult<ThermalConductivity> {
        if self.components() != 1 {
            return Err(FeosError::IncompatibleComponents(self.components(), 1));
        }
        let p = &self.params;
        let mws = self.molar_weight();
        let t = Dual64::from(temperature.into_reduced()).derivative();
        let v = Dual64::from(volume.into_reduced());
        let n = moles.to_reduced().mapv(Dual64::from);
        let n_tot = n.sum();
        let state = StateHD::new(t, v, n);
        let s_res = -(self.residual_helmholtz_energy(&state) * t / n_tot).eps;
        let res: Array1<_> = (0..self.components())
            .map(|i| {
                let tr = (temperature / p.epsilon_k[i] / KELVIN).into_value();
                let s_res_reduced = s_res / p.m[i];
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

    fn thermal_conductivity_correlation(&self, s_res: f64, x: &Array1<f64>) -> FeosResult<f64> {
        if self.components() != 1 {
            return Err(FeosError::IncompatibleComponents(self.components(), 1));
        }
        let coefficients = self
            .params
            .thermal_conductivity
            .as_ref()
            .expect("Missing thermal conductivity coefficients");
        let m = (x * &self.params.m).sum();
        let s = s_res / m;
        let pref = (x * &self.params.m).mapv(|v| v / m);
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
        butane_parameters, propane_butane_parameters, propane_parameters,
    };
    use approx::assert_relative_eq;
    use feos_core::*;
    use ndarray::arr1;
    use quantity::{BAR, KELVIN, METER, MILLI, PASCAL, RGAS, SECOND};
    use typenum::P3;

    #[test]
    fn ideal_gas_pressure() {
        let e = propane_parameters();
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
        let e = propane_parameters();
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
        let t = 250.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a_rust = HardSphere.helmholtz_energy(&propane_parameters().params as &PcSaftPars, &s);
        assert_relative_eq!(a_rust, 0.410610492598808, epsilon = 1e-10);
    }

    #[test]
    fn hard_sphere_mix() {
        let t = 250.0;
        let v = 2.5e28;
        let n = 1.0;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a1 = HardSphere.helmholtz_energy(&propane_parameters().params as &PcSaftPars, &s);
        let a2 = HardSphere.helmholtz_energy(&butane_parameters().params as &PcSaftPars, &s);
        let s1m = StateHD::new(t, v, arr1(&[n, 0.0]));
        let a1m =
            HardSphere.helmholtz_energy(&propane_butane_parameters().params as &PcSaftPars, &s1m);
        let s2m = StateHD::new(t, v, arr1(&[0.0, n]));
        let a2m =
            HardSphere.helmholtz_energy(&propane_butane_parameters().params as &PcSaftPars, &s2m);
        assert_relative_eq!(a1, a1m, epsilon = 1e-14);
        assert_relative_eq!(a2, a2m, epsilon = 1e-14);
    }

    #[test]
    fn new_tpn() {
        let e = propane_parameters();
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
        let e = propane_parameters();
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
        let e = propane_parameters();
        let t = 300.0 * KELVIN;
        let cp = State::critical_point(&e, None, Some(t), Default::default());
        if let Ok(v) = cp {
            assert_relative_eq!(v.temperature, 375.1244078318015 * KELVIN, epsilon = 1e-8)
        }
    }

    #[test]
    fn mix_single() {
        let e1 = propane_parameters();
        let e2 = butane_parameters();
        let e12 = propane_butane_parameters();
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
    fn viscosity() -> FeosResult<()> {
        let e = propane_parameters();
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
    fn diffusion() -> FeosResult<()> {
        let e = propane_parameters();
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
