use super::parameters::{PcSaftAssociationRecord, PcSaftParameters, PcSaftPars};
use crate::association::{Association, AssociationStrength};
use crate::hard_sphere::{HardSphere, HardSphereProperties, MonomerShape};
use feos_core::{
    EntropyScaling, Molarweight, ReferenceSystem, Residual, ResidualDyn, StateHD, Subset,
};
use nalgebra::DVector;
use num_dual::{DualNum, partial2};
use quantity::ad::first_derivative;
use quantity::*;
use std::f64::consts::{FRAC_PI_6, PI};
use typenum::P2;

pub(crate) mod dispersion;
pub(crate) mod hard_chain;
mod pcsaft_binary;
mod pcsaft_pure;
pub(crate) mod polar;
use dispersion::Dispersion;
use hard_chain::HardChain;
pub use pcsaft_binary::PcSaftBinary;
pub use pcsaft_pure::PcSaftPure;
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
    pub parameters: PcSaftParameters,
    params: PcSaftPars,
    options: PcSaftOptions,
    hard_chain: bool,
    dipole: bool,
    quadrupole: bool,
    dipole_quadrupole: bool,
    association: Option<Association>,
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

        let association = (!parameters.association.is_empty())
            .then(|| Association::new(options.max_iter_cross_assoc, options.tol_cross_assoc));

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

impl ResidualDyn for PcSaft {
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
    ) -> Vec<(&'static str, D)> {
        let mut v = Vec::with_capacity(7);
        let d = self.params.hs_diameter(state.temperature);

        v.push((
            "Hard Sphere",
            HardSphere.helmholtz_energy_density(&self.params, state),
        ));
        if self.hard_chain {
            v.push((
                "Hard Chain",
                HardChain.helmholtz_energy_density(&self.params, state),
            ))
        }
        v.push((
            "Dispersion",
            Dispersion.helmholtz_energy_density(&self.params, state),
        ));
        if self.dipole {
            v.push((
                "Dipole",
                Dipole.helmholtz_energy_density(&self.params, state),
            ))
        }
        if self.quadrupole {
            v.push((
                "Quadrupole",
                Quadrupole.helmholtz_energy_density(&self.params, state),
            ))
        }
        if self.dipole_quadrupole {
            v.push((
                "DipoleQuadrupole",
                DipoleQuadrupole.helmholtz_energy_density(
                    &self.params,
                    state,
                    self.options.dq_variant,
                ),
            ))
        }
        if let Some(association) = self.association.as_ref() {
            v.push((
                "Association",
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

impl Subset for PcSaft {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(self.parameters.subset(component_list), self.options)
    }
}

impl Molarweight for PcSaft {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

impl HardSphereProperties for PcSaftPars {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<'_, N> {
        MonomerShape::NonSpherical(self.m.map(N::from))
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> DVector<D> {
        let ti = temperature.recip() * -3.0;
        DVector::from_fn(self.sigma.len(), |i, _| {
            -((ti * self.epsilon_k[i]).exp() * 0.12 - 1.0) * self.sigma[i]
        })
    }
}

impl AssociationStrength for PcSaftPars {
    type Record = PcSaftAssociationRecord;

    fn association_strength_ij<D: DualNum<f64> + Copy>(
        &self,
        temperature: D,
        comp_i: usize,
        comp_j: usize,
        association_parameters_ij: &Self::Record,
    ) -> D {
        let f_ab = (temperature.recip() * association_parameters_ij.epsilon_k_ab).exp_m1();
        let k_ab = association_parameters_ij.kappa_ab
            * (self.sigma[comp_i] * self.sigma[comp_j]).powf(1.5);
        f_ab * k_ab
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
    let t = temperature.into_reduced();
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
        moles: &Moles<DVector<f64>>,
    ) -> Viscosity {
        let p = &self.params;
        let mw = &self.parameters.molar_weight;
        let x = (moles / moles.sum()).into_value();
        let ce: Vec<_> = (0..self.components())
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
        ce_mix
    }

    fn viscosity_correlation(&self, s_res: f64, x: &DVector<f64>) -> f64 {
        let coefficients = self
            .params
            .viscosity
            .as_ref()
            .expect("Missing viscosity coefficients.");
        let m = x.dot(&self.params.m);
        let s = s_res / m;
        let pref = x.component_mul(&self.params.m) / m;
        let a = coefficients.row(0).transpose().dot(x);
        let b = coefficients.row(1).transpose().dot(&pref);
        let c = coefficients.row(2).transpose().dot(&pref);
        let d = coefficients.row(3).transpose().dot(&pref);
        a + b * s + c * s.powi(2) + d * s.powi(3)
    }

    fn diffusion_reference(
        &self,
        temperature: Temperature,
        volume: Volume,
        moles: &Moles<DVector<f64>>,
    ) -> Diffusivity {
        if self.components() != 1 {
            panic!("Diffusion coefficients in PC-SAFT are only implemented for pure components!");
        }
        let p = &self.params;
        let mw = &self.parameters.molar_weight;
        let density = moles.sum() / volume;
        let res: Vec<_> = (0..self.components())
            .map(|i| {
                let tr = (temperature / p.epsilon_k[i] / KELVIN).into_value();
                3.0 / 8.0 / (p.sigma[i] * ANGSTROM).powi::<P2>() / omega11(tr) / (density * NAV)
                    * (temperature * RGAS / PI / mw.get(i) / p.m[i]).sqrt()
            })
            .collect();
        res[0]
    }

    fn diffusion_correlation(&self, s_res: f64, x: &DVector<f64>) -> f64 {
        if self.components() != 1 {
            panic!("Diffusion coefficients in PC-SAFT are only implemented for pure components!");
        }
        let coefficients = self
            .params
            .diffusion
            .as_ref()
            .expect("Missing diffusion coefficients.");
        let m = x.dot(&self.params.m);
        let s = s_res / m;
        let pref = x.component_mul(&self.params.m) / m;
        let a = coefficients.row(0).transpose().dot(x);
        let b = coefficients.row(1).transpose().dot(&pref);
        let c = coefficients.row(2).transpose().dot(&pref);
        let d = coefficients.row(3).transpose().dot(&pref);
        let e = coefficients.row(4).transpose().dot(&pref);
        a + b * s - c * (1.0 - s.exp()) * s.powi(2) - d * s.powi(4) - e * s.powi(8)
    }

    // Equation 4 of DOI: 10.1021/acs.iecr.9b04289
    fn thermal_conductivity_reference(
        &self,
        temperature: Temperature,
        volume: Volume,
        moles: &Moles<DVector<f64>>,
    ) -> ThermalConductivity {
        if self.components() != 1 {
            panic!("Thermal conductivity in PC-SAFT is only implemented for pure components!");
        }
        let p = &self.params;
        let mws = self.molar_weight();
        let (_, s_res) = first_derivative(
            partial2(
                |t, &v, n| -self.residual_helmholtz_energy_unit(t, v, n) / n.sum(),
                &volume,
                moles,
            ),
            temperature,
        );
        let res: Vec<_> = (0..self.components())
            .map(|i| {
                let tr = (temperature / p.epsilon_k[i] / KELVIN).into_value();
                let s_res_reduced = s_res.into_reduced() / p.m[i];
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
        res[0]
    }

    fn thermal_conductivity_correlation(&self, s_res: f64, x: &DVector<f64>) -> f64 {
        if self.components() != 1 {
            panic!("Thermal conductivity in PC-SAFT is only implemented for pure components!");
        }
        let coefficients = self
            .params
            .thermal_conductivity
            .as_ref()
            .expect("Missing thermal conductivity coefficients");
        let m = x.dot(&self.params.m);
        let s = s_res / m;
        let pref = x.component_mul(&self.params.m) / m;
        let a = coefficients.row(0).transpose().dot(x);
        let b = coefficients.row(1).transpose().dot(&pref);
        let c = coefficients.row(2).transpose().dot(&pref);
        let d = coefficients.row(3).transpose().dot(&pref);
        a + b * s + c * (1.0 - s.exp()) + d * s.powi(2)
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
    use nalgebra::dvector;
    use quantity::{BAR, KELVIN, METER, PASCAL, RGAS};
    use typenum::{P2, P3};

    #[test]
    fn ideal_gas_pressure() {
        let e = &propane_parameters();
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
        let e = &propane_parameters();
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
        let t = 250.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, &dvector![n]);
        let a_rust = HardSphere
            .helmholtz_energy_density(&propane_parameters().params as &PcSaftPars, &s)
            * v;
        assert_relative_eq!(a_rust, 0.410610492598808, epsilon = 1e-10);
    }

    #[test]
    fn hard_sphere_mix() {
        let t = 250.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, &dvector![n]);
        let a1 =
            HardSphere.helmholtz_energy_density(&propane_parameters().params as &PcSaftPars, &s);
        let a2 =
            HardSphere.helmholtz_energy_density(&butane_parameters().params as &PcSaftPars, &s);
        let s1m = StateHD::new(t, v, &dvector![n, 0.0]);
        let a1m = HardSphere
            .helmholtz_energy_density(&propane_butane_parameters().params as &PcSaftPars, &s1m);
        let s2m = StateHD::new(t, v, &dvector![0.0, n]);
        let a2m = HardSphere
            .helmholtz_energy_density(&propane_butane_parameters().params as &PcSaftPars, &s2m);
        assert_relative_eq!(a1, a1m, epsilon = 1e-14);
        assert_relative_eq!(a2, a2m, epsilon = 1e-14);
    }

    #[test]
    fn new_tpn() {
        let e = &propane_parameters();
        let t = 300.0 * KELVIN;
        let p = BAR;
        let m = dvector![1.5] * MOL;
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
        let e = &propane_parameters();
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
        let e = &propane_parameters();
        let t = 300.0 * KELVIN;
        let cp = State::critical_point(&e, None, Some(t), None, Default::default());
        if let Ok(v) = cp {
            assert_relative_eq!(v.temperature, 375.1244078318015 * KELVIN, epsilon = 1e-8)
        }
    }

    #[test]
    fn mix_single() {
        let e1 = &propane_parameters();
        let e2 = &butane_parameters();
        let e12 = &propane_butane_parameters();
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
        )
    }

    #[test]
    fn viscosity() -> FeosResult<()> {
        let e = &propane_parameters();
        let t = 300.0 * KELVIN;
        let p = BAR;
        let n = dvector![1.0] * MOL;
        let s = State::new_npt(&e, t, p, &n, None)?;
        assert_relative_eq!(
            s.viscosity(),
            0.00797 * MILLI * PASCAL * SECOND,
            epsilon = 1e-5
        );
        assert_relative_eq!(
            s.ln_viscosity_reduced(),
            (s.viscosity() / e.viscosity_reference(s.temperature, s.volume, &s.moles))
                .into_value()
                .ln(),
            epsilon = 1e-15
        );
        Ok(())
    }

    #[test]
    fn viscosity_mix() -> FeosResult<()> {
        let e = &propane_butane_parameters();
        let t = 300.0 * KELVIN;
        let p = 2.0 * BAR;
        let n = dvector![1.0, 0.0] * MOL;
        let s = State::new_npt(&e, t, p, &n, None)?;
        assert_relative_eq!(
            s.viscosity(),
            0.00797 * MILLI * PASCAL * SECOND,
            epsilon = 1e-5
        );
        assert_relative_eq!(
            s.ln_viscosity_reduced(),
            (s.viscosity() / e.viscosity_reference(s.temperature, s.volume, &s.moles))
                .into_value()
                .ln(),
            epsilon = 1e-15
        );
        Ok(())
    }

    #[test]
    fn diffusion() -> FeosResult<()> {
        let e = &propane_parameters();
        let t = 300.0 * KELVIN;
        let p = BAR;
        let n = dvector![1.0] * MOL;
        let s = State::new_npt(&e, t, p, &n, None)?;
        assert_relative_eq!(
            s.diffusion(),
            0.01505 * (CENTI * METER).powi::<P2>() / SECOND,
            epsilon = 1e-5
        );
        assert_relative_eq!(
            s.ln_diffusion_reduced(),
            (s.diffusion() / e.diffusion_reference(s.temperature, s.volume, &s.moles))
                .into_value()
                .ln(),
            epsilon = 1e-15
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests_parameter_fit {
    use super::pcsaft_binary::test::pcsaft_binary;
    use super::pcsaft_pure::test::pcsaft;
    use super::*;
    use approx::assert_relative_eq;
    use feos_core::DensityInitialization::Liquid;
    use feos_core::{Contributions, PropertiesAD, ReferenceSystem};
    use feos_core::{FeosResult, ParametersAD, PhaseEquilibrium, State};
    use nalgebra::{U1, U3, U8, vector};
    use num_dual::{DualStruct, DualVec};
    use quantity::{BAR, KELVIN, LITER, MOL, PASCAL};

    fn pcsaft_non_assoc() -> PcSaftPure<f64, 4> {
        let m = 1.5;
        let sigma = 3.4;
        let epsilon_k = 180.0;
        let mu = 2.2;
        let params = [m, sigma, epsilon_k, mu];
        PcSaftPure(params)
    }

    #[test]
    fn test_vapor_pressure_derivatives() -> FeosResult<()> {
        let pcsaft_params = [
            "m",
            "sigma",
            "epsilon_k",
            "mu",
            "kappa_ab",
            "epsilon_k_ab",
            "na",
            "nb",
        ];
        let (pcsaft, _) = pcsaft()?;
        let pcsaft_ad = pcsaft.named_derivatives(pcsaft_params);
        let temperature = 250.0 * KELVIN;
        let p = pcsaft_ad.vapor_pressure(temperature)?;
        let p = p.convert_into(PASCAL);
        let (p, grad) = (p.re, p.eps.unwrap_generic(U8, U1));

        println!("{p:.5}");
        println!("{grad:.5?}");

        for (i, par) in pcsaft_params.into_iter().enumerate() {
            let mut params = pcsaft.0;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = PcSaftPure(params);
            let (p_h, _) =
                PhaseEquilibrium::pure_t(&pcsaft_h, temperature, None, Default::default())?;
            let dp_h = (p_h.convert_into(PASCAL) - p) / h;
            let dp = grad[i];
            println!(
                "{par:12}: {:11.5} {:11.5} {:.3e}",
                dp_h,
                dp,
                ((dp_h - dp) / dp).abs()
            );
            assert_relative_eq!(dp, dp_h, max_relative = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_vapor_pressure_derivatives_fit() -> FeosResult<()> {
        let pcsaft = pcsaft_non_assoc();
        let pcsaft_ad = pcsaft.named_derivatives(["m", "sigma", "epsilon_k"]);
        let temperature = 150.0 * KELVIN;
        let p = pcsaft_ad.vapor_pressure(temperature)?;
        let p = p.convert_into(PASCAL);
        let (p, grad) = (p.re, p.eps.unwrap_generic(U3, U1));

        println!("{p:.5}");
        println!("{grad:.5?}");

        for (i, par) in ["m", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = pcsaft.0;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = PcSaftPure(params);
            let (p_h, _) =
                PhaseEquilibrium::pure_t(&pcsaft_h, temperature, None, Default::default())?;
            let dp_h = (p_h.convert_into(PASCAL) - p) / h;
            let dp = grad[i];
            println!(
                "{par:12}: {:11.5} {:11.5} {:.3e}",
                dp_h,
                dp,
                ((dp_h - dp) / dp).abs()
            );
            assert_relative_eq!(dp, dp_h, max_relative = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_equilibrium_liquid_density_derivatives_fit() -> FeosResult<()> {
        let pcsaft = pcsaft_non_assoc();
        let pcsaft_ad = pcsaft.named_derivatives(["m", "sigma", "epsilon_k"]);
        let temperature = 150.0 * KELVIN;
        let (p, rho) = pcsaft_ad.equilibrium_liquid_density(temperature)?;
        let p = p.convert_into(PASCAL);
        let rho = rho.convert_into(MOL / LITER);
        let (p, p_grad) = (p.re, p.eps.unwrap_generic(U3, U1));
        let (rho, rho_grad) = (rho.re, rho.eps.unwrap_generic(U3, U1));

        println!("{p:.5} {rho:.5}");
        println!("{p_grad:.5?}");
        println!("{rho_grad:.5?}");

        for (i, par) in ["m", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = pcsaft.0;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = PcSaftPure(params);
            let (p_h, [_, rho_h]) =
                PhaseEquilibrium::pure_t(&pcsaft_h, temperature, None, Default::default())?;
            let dp_h = (p_h.convert_into(PASCAL) - p) / h;
            let drho_h = (rho_h.convert_into(MOL / LITER) - rho) / h;
            let dp = p_grad[i];
            let drho = rho_grad[i];
            println!(
                "{par:12}: {:11.5} {:11.5} {:.3e} {:11.5} {:11.5} {:.3e}",
                dp_h,
                dp,
                ((dp_h - dp) / dp).abs(),
                drho_h,
                drho,
                ((drho_h - drho) / drho).abs()
            );
            assert_relative_eq!(dp, dp_h, max_relative = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_liquid_density_derivatives_fit() -> FeosResult<()> {
        let pcsaft = pcsaft_non_assoc();
        let pcsaft_ad = pcsaft.named_derivatives(["m", "sigma", "epsilon_k"]);
        let temperature = 150.0 * KELVIN;
        let pressure = BAR;
        let rho = pcsaft_ad.liquid_density(temperature, pressure)?;
        let rho = rho.convert_into(MOL / LITER);
        let (rho, grad) = (rho.re, rho.eps.unwrap_generic(U3, U1));

        println!("{rho:.5}");
        println!("{grad:.5?}");

        for (i, par) in ["m", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = pcsaft.0;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = PcSaftPure(params);
            let rho_h = State::new_xpt(
                &pcsaft_h,
                temperature,
                pressure,
                &vector![1.0],
                Some(Liquid),
            )?
            .density;
            let drho_h = (rho_h.convert_into(MOL / LITER) - rho) / h;
            let drho = grad[i];
            println!(
                "{par:12}: {:11.5} {:11.5} {:.3e}",
                drho_h,
                drho,
                ((drho_h - drho) / drho).abs()
            );
            assert_relative_eq!(drho, drho_h, max_relative = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_bubble_point_pressure() -> FeosResult<()> {
        let (pcsaft, _) = pcsaft_binary()?;
        let pcsaft_ad = pcsaft.named_derivatives(["k_ij"]);
        let temperature = 500.0 * KELVIN;
        let x = vector![0.5, 0.5];
        let p = pcsaft_ad.bubble_point_pressure(temperature, None, x)?;
        let p = p.convert_into(BAR);
        let (p, [[grad]]) = (p.re, p.eps.unwrap_generic(U1, U1).data.0);

        println!("{p:.5}");
        println!("{grad:.5?}");

        let (params, mut kij) = pcsaft.0;
        let h = 1e-7;
        kij += h;
        let pcsaft_h = PcSaftBinary::new(params, kij);
        let p_h = PhaseEquilibrium::bubble_point(
            &pcsaft_h,
            temperature,
            &x,
            None,
            None,
            Default::default(),
        )?
        .vapor()
        .pressure(Contributions::Total);
        let dp_h = (p_h.convert_into(BAR) - p) / h;
        println!(
            "k_ij: {:11.5} {:11.5} {:.3e}",
            dp_h,
            grad,
            ((dp_h - grad) / grad).abs()
        );
        assert_relative_eq!(grad, dp_h, max_relative = 1e-6);
        Ok(())
    }

    #[test]
    fn test_dew_point_pressure() -> FeosResult<()> {
        let (pcsaft, _) = pcsaft_binary()?;
        let pcsaft_ad = pcsaft.named_derivatives(["k_ij"]);
        let temperature = 500.0 * KELVIN;
        let y = vector![0.5, 0.5];
        let p = pcsaft_ad.dew_point_pressure(temperature, None, y)?;
        let p = p.convert_into(BAR);
        let (p, [[grad]]) = (p.re, p.eps.unwrap_generic(U1, U1).data.0);

        println!("{p:.5}");
        println!("{grad:.5?}");

        let (params, mut kij) = pcsaft.0;
        let h = 1e-7;
        kij += h;
        let pcsaft_h = PcSaftBinary::new(params, kij);
        let p_h = PhaseEquilibrium::dew_point(
            &pcsaft_h,
            temperature,
            &y,
            None,
            None,
            Default::default(),
        )?
        .vapor()
        .pressure(Contributions::Total);
        let dp_h = (p_h.convert_into(BAR) - p) / h;
        println!(
            "k_ij: {:11.5} {:11.5} {:.3e}",
            dp_h,
            grad,
            ((dp_h - grad) / grad).abs()
        );
        assert_relative_eq!(grad, dp_h, max_relative = 1e-6);
        Ok(())
    }

    #[test]
    fn test_bubble_point_temperature() -> FeosResult<()> {
        let (pcsaft, _) = pcsaft_binary()?;
        let pcsaft_ad = pcsaft.named_derivatives(["k_ij"]);
        let pressure = Pressure::from_reduced(DualVec::from(45. * BAR.into_reduced()));
        let t_init = Temperature::from_reduced(DualVec::from(500.0));
        let x = vector![0.5, 0.5].map(DualVec::from);
        let t = PhaseEquilibrium::bubble_point(
            &pcsaft_ad,
            pressure,
            &x,
            Some(t_init),
            None,
            Default::default(),
        )?
        .vapor()
        .temperature;
        let t = t.convert_into(KELVIN);
        let (t, [[grad]]) = (t.re, t.eps.unwrap_generic(U1, U1).data.0);

        println!("{t:.5}");
        println!("{grad:.5?}");

        let (params, mut kij) = pcsaft.0;
        let h = 1e-7;
        kij += h;
        let pcsaft_h = PcSaftBinary::new(params, kij);
        let t_h = PhaseEquilibrium::bubble_point(
            &pcsaft_h,
            pressure.re(),
            &x.map(|x| x.re()),
            Some(t_init.re()),
            None,
            Default::default(),
        )?
        .vapor()
        .temperature;
        let dt_h = (t_h.convert_into(KELVIN) - t) / h;
        println!(
            "k_ij: {:11.5} {:11.5} {:.3e}",
            dt_h,
            grad,
            ((dt_h - grad) / grad).abs()
        );
        assert_relative_eq!(grad, dt_h, max_relative = 1e-6);
        Ok(())
    }

    #[test]
    fn test_dew_point_temperature() -> FeosResult<()> {
        let (pcsaft, _) = pcsaft_binary()?;
        let pcsaft_ad = pcsaft.named_derivatives(["k_ij"]);
        let pressure = Pressure::from_reduced(DualVec::from(45. * BAR.into_reduced()));
        let t_init = Temperature::from_reduced(DualVec::from(500.0));
        let x = vector![0.5, 0.5].map(DualVec::from);
        let t = PhaseEquilibrium::dew_point(
            &pcsaft_ad,
            pressure,
            &x,
            Some(t_init),
            None,
            Default::default(),
        )?
        .vapor()
        .temperature;
        let t = t.convert_into(KELVIN);
        let (t, [[grad]]) = (t.re, t.eps.unwrap_generic(U1, U1).data.0);

        println!("{t:.5}");
        println!("{grad:.5?}");

        let (params, mut kij) = pcsaft.0;
        let h = 1e-7;
        kij += h;
        let pcsaft_h = PcSaftBinary::new(params, kij);
        let t_h = PhaseEquilibrium::dew_point(
            &pcsaft_h,
            pressure.re(),
            &x.map(|x| x.re()),
            Some(t_init.re()),
            None,
            Default::default(),
        )?
        .vapor()
        .temperature;
        let dt_h = (t_h.convert_into(KELVIN) - t) / h;
        println!(
            "k_ij: {:11.5} {:11.5} {:.3e}",
            dt_h,
            grad,
            ((dt_h - grad) / grad).abs()
        );
        assert_relative_eq!(grad, dt_h, max_relative = 1e-6);
        Ok(())
    }
}
