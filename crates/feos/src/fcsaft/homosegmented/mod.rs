//! Homosegmented formulation of FC-SAFT for chains of identical fused spheres.
use super::FcSaftOptions;
use super::eos::dispersion::Dispersion;
use super::reference::ReferenceFluid;
use crate::association::Association;
use crate::hard_sphere::HardSphereProperties;
use feos_core::{Molarweight, ResidualDyn, StateHD, Subset};
use nalgebra::DVector;
use num_dual::DualNum;
use quantity::MolarWeight;
use std::f64::consts::FRAC_PI_6;

mod fcsaft_binary;
mod fcsaft_pure;
mod parameters;
mod polar;
pub use fcsaft_binary::FcSaftBinary;
pub use fcsaft_pure::FcSaftPure;
pub use parameters::{FcSaftHomoParameters, FcSaftHomoPars, FcSaftHomoRecord};
use polar::Dipole;

/// Homosegmented FC-SAFT equation of state.
pub struct FcSaftHomo {
    pub parameters: FcSaftHomoParameters,
    pub params: FcSaftHomoPars,
    options: FcSaftOptions,
    dispersion: Dispersion,
    dipole: Option<Dipole>,
    association: Option<Association>,
}

impl FcSaftHomo {
    pub fn new(parameters: FcSaftHomoParameters) -> Self {
        Self::with_options(parameters, FcSaftOptions::default())
    }

    pub fn with_options(parameters: FcSaftHomoParameters, options: FcSaftOptions) -> Self {
        let params = FcSaftHomoPars::new(&parameters);
        let dispersion = Dispersion::new(
            &params.component_index,
            &params.sigma_ij,
            &params.epsilon_k_ij,
            None,
        );
        let dipole = (!params.dipole_comp.is_empty()).then_some(Dipole);
        let association = (!parameters.association.is_empty())
            .then(|| Association::new(options.max_iter_cross_assoc, options.tol_cross_assoc));
        Self {
            parameters,
            params,
            options,
            dispersion,
            dipole,
            association,
        }
    }
}

impl ResidualDyn for FcSaftHomo {
    fn components(&self) -> usize {
        self.parameters.pure.len()
    }

    fn compute_max_density<D: DualNum<Primitive = f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        let p = &self.params;
        let [_, m_star] = p.m_values(&p.sigma);
        let msigma3 = m_star.component_mul(&p.sigma.map(|s| s.powi(3)));
        (msigma3.map(D::from).dot(molefracs) * FRAC_PI_6).recip() * self.options.max_eta
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<Primitive = f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        let mut v = Vec::with_capacity(5);
        let p = &self.params;
        let d = p.hs_diameter(state.temperature);

        let density = state.partial_density.sum();
        let m = p.m_values(&d);
        let ([hs, chain], c1) = ReferenceFluid::new_homosegmented(&p.s, &p.l, &d, &m)
            .helmholtz_energy_density(density, &state.molefracs);
        v.push(("Hard Sphere", hs));
        v.push(("Fused-sphere chain", chain));
        v.push((
            "Dispersion",
            self.dispersion.helmholtz_energy_density(state, &d, &m, c1),
        ));
        if let Some(dipole) = self.dipole.as_ref() {
            v.push((
                "Dipole",
                dipole.helmholtz_energy_density(&self.params, state),
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

impl Subset for FcSaftHomo {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(self.parameters.subset(component_list), self.options)
    }
}

impl Molarweight for FcSaftHomo {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::fcsaft::{
        FcSaft, FcSaftAssociationRecord, FcSaftBondRecord, FcSaftParameters, FcSaftRecord,
    };
    use approx::assert_relative_eq;
    use feos_core::FeosResult;
    use feos_core::State;
    use feos_core::parameter::{
        AssociationRecord, BinarySegmentRecord, ChemicalRecord, PureRecord, SegmentRecord,
    };
    use quantity::{KELVIN, METER, MOL};

    #[test]
    fn test_homo() -> FeosResult<()> {
        let s = 3;
        let l = 2.5;
        let sigma = 3.5;
        let epsilon_k = 250.0;
        let association = vec![AssociationRecord::new(
            Some(FcSaftAssociationRecord::new(0.03, 2500.0)),
            1.0,
            1.0,
            0.0,
        )];

        let pr = PureRecord::with_association(
            Default::default(),
            0.0,
            FcSaftHomoRecord::new(s, l, sigma, epsilon_k, 0.0),
            association.clone(),
        );
        let params_homo = FcSaftHomoParameters::new_pure(pr)?;
        let fcsaft_homo = FcSaftHomo::new(params_homo);

        let mut segments = vec!["S".into(); s - 1];
        segments.push("SA".into());
        let cr = ChemicalRecord::new(Default::default(), segments, None);
        let sr = SegmentRecord::new("S".into(), 0.0, FcSaftRecord::new(sigma, epsilon_k, None));
        let sr_a = SegmentRecord::with_association(
            "SA".into(),
            0.0,
            FcSaftRecord::new(sigma, epsilon_k, None),
            association,
        );
        let br = [
            BinarySegmentRecord::new("S".into(), "S".into(), Some(FcSaftBondRecord::new(l))),
            BinarySegmentRecord::new("S".into(), "SA".into(), Some(FcSaftBondRecord::new(l))),
        ];
        let params = FcSaftParameters::from_segments_with_bonds(vec![cr], &[sr, sr_a], None, &br)?;
        let fcsaft = FcSaft::new(params);

        let t = 300.0 * KELVIN;
        let v = METER.powi::<3>();
        let n = 500.0 * MOL;
        let c_homo = State::new_nvt(&&fcsaft_homo, t, v, n)?.pressure_contributions();
        let c = State::new_nvt(&&fcsaft, t, v, n)?.pressure_contributions();
        assert_eq!(c.len(), 5);
        assert_eq!(c_homo.len(), 5);
        for i in 0..5 {
            println!("{:20} {:13.8} {:13.8}", c[i].0, c[i].1, c_homo[i].1);
        }
        for i in 0..5 {
            assert_relative_eq!(c[i].1, c_homo[i].1, max_relative = 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_homo_mixture() -> FeosResult<()> {
        // (s, l, sigma, epsilon_k)
        let components = [(3, 2.5, 3.5, 250.0), (6, 2.0, 3.8, 220.0)];
        let names = ["A", "B"];
        let segment_records: Vec<_> = (0..2)
            .map(|i| {
                let (_, _, sigma, epsilon_k) = components[i];
                SegmentRecord::new(
                    names[i].into(),
                    1.0,
                    FcSaftRecord::new(sigma, epsilon_k, None),
                )
            })
            .collect();
        let bond_records: Vec<_> = (0..2)
            .map(|i| {
                let l = components[i].1;
                BinarySegmentRecord::new(
                    names[i].into(),
                    names[i].into(),
                    Some(FcSaftBondRecord::new(l)),
                )
            })
            .collect();

        for order in [[0, 1], [1, 0]] {
            let pure_records = order
                .iter()
                .map(|&i| {
                    let (s, l, sigma, epsilon_k) = components[i];
                    let record = FcSaftHomoRecord::new(s, l, sigma, epsilon_k, 0.0);
                    PureRecord::new(Default::default(), 1.0, record)
                })
                .collect();
            let fcsaft_homo = FcSaftHomo::new(FcSaftHomoParameters::new(pure_records, vec![])?);

            let chemical_records = order
                .iter()
                .map(|&i| {
                    let segments = vec![names[i].into(); components[i].0];
                    ChemicalRecord::new(Default::default(), segments, None)
                })
                .collect();
            let params = FcSaftParameters::from_segments_with_bonds(
                chemical_records,
                &segment_records,
                None,
                &bond_records,
            )?;
            let fcsaft = FcSaft::new(params);

            let t = 300.0 * KELVIN;
            let v = METER.powi::<3>();
            for x1 in [0.0, 0.1, 0.5, 0.9, 1.0] {
                let n = nalgebra::dvector![x1, 1.0 - x1] * 2000.0 * MOL;
                let c_homo = State::new_nvt(&&fcsaft_homo, t, v, &n)?.pressure_contributions();
                let c = State::new_nvt(&&fcsaft, t, v, &n)?.pressure_contributions();
                assert_eq!(c.len(), c_homo.len());
                for ((name, p), (_, p_homo)) in c.iter().zip(&c_homo) {
                    println!("{order:?} {x1:4.2} {name:20} {p:13.8} {p_homo:13.8}");
                    assert_relative_eq!(p, p_homo, max_relative = 1e-10);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests_parameter_fit {
    use super::fcsaft_pure::test::fcsaft;
    use super::*;
    use approx::assert_relative_eq;
    use feos_core::DensityInitialization::Liquid;
    use feos_core::ad::{
        BoilingTemperature, EquilibriumLiquidDensity, LiquidDensity, PropertyAD, VaporPressure,
    };
    use feos_core::{FeosResult, PhaseEquilibrium, State, ad::ParametersAD};
    use nalgebra::{U1, U3, U9};
    use num_dual::DualStruct;
    use quantity::{BAR, KELVIN, LITER, MOL, PASCAL, Pressure};

    fn fcsaft_non_assoc() -> FcSaftPure<f64, 5> {
        let s = 3.0;
        let l = 2.5;
        let sigma = 3.4;
        let epsilon_k = 180.0;
        let mu = 2.2;
        let params = [s, l, sigma, epsilon_k, mu];
        FcSaftPure(params)
    }

    #[test]
    fn test_vapor_pressure_derivatives() -> FeosResult<()> {
        let fcsaft_params = [
            "s",
            "l",
            "sigma",
            "epsilon_k",
            "mu",
            "kappa_ab",
            "epsilon_k_ab",
            "na",
            "nb",
        ];
        let (fcsaft, _) = fcsaft()?;
        let fcsaft_ad = FcSaftPure::<f64, 9>::seed_derivatives(&fcsaft.0, fcsaft_params);
        let temperature = 300.0 * KELVIN;
        let p = VaporPressure(temperature).evaluate(&fcsaft_ad)?;
        let p = p.convert_into(PASCAL);
        let (p, grad) = (p.re, p.eps.unwrap_generic(U9, U1));

        println!("{p:.5}");
        println!("{grad:.5?}");

        // central differences, because the vapor pressure is strongly curved in some
        // of the parameters (e.g., sigma)
        for (i, par) in fcsaft_params.into_iter().enumerate() {
            let h = fcsaft.0[i] * 1e-6;
            let p_h = |h: f64| {
                let mut params = fcsaft.0;
                params[i] += h;
                PhaseEquilibrium::pure_t(&FcSaftPure(params), temperature, None, Default::default())
                    .map(|(p, _)| p.convert_into(PASCAL))
            };
            let dp_h = (p_h(h)? - p_h(-h)?) / (2.0 * h);
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
        let fcsaft = fcsaft_non_assoc();
        let fcsaft_ad =
            FcSaftPure::<f64, 5>::seed_derivatives(&fcsaft.0, ["l", "sigma", "epsilon_k"]);
        let temperature = 200.0 * KELVIN;
        let p = VaporPressure(temperature).evaluate(&fcsaft_ad)?;
        let p = p.convert_into(PASCAL);
        let (p, grad) = (p.re, p.eps.unwrap_generic(U3, U1));

        println!("{p:.5}");
        println!("{grad:.5?}");

        for (i, par) in ["l", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = fcsaft.0;
            let h = params[i + 1] * 1e-7;
            params[i + 1] += h;
            let fcsaft_h = FcSaftPure(params);
            let (p_h, _) =
                PhaseEquilibrium::pure_t(&fcsaft_h, temperature, None, Default::default())?;
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
    fn test_boiling_temperature_derivatives_fit() -> FeosResult<()> {
        let fcsaft = fcsaft_non_assoc();
        let fcsaft_ad =
            FcSaftPure::<f64, 5>::seed_derivatives(&fcsaft.0, ["l", "sigma", "epsilon_k"]);
        let pressure = BAR;
        let t = BoilingTemperature(pressure).evaluate(&fcsaft_ad)?;
        let t = t.convert_into(KELVIN);
        let (t, grad) = (t.re, t.eps.unwrap_generic(U3, U1));

        println!("{t:.5}");
        println!("{grad:.5?}");

        let (t_check, _) = PhaseEquilibrium::pure_p(
            &fcsaft_ad,
            Pressure::from_inner(&pressure),
            None,
            Default::default(),
        )?;
        let t_check = t_check.convert_into(KELVIN);
        let (t_check, grad_check) = (t_check.re, t_check.eps.unwrap_generic(U3, U1));
        println!("{t_check:.5}");
        println!("{grad_check:.5?}");
        assert_relative_eq!(t, t_check, max_relative = 1e-15);
        assert_relative_eq!(grad, grad_check, max_relative = 1e-15);

        for (i, par) in ["l", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = fcsaft.0;
            let h = params[i + 1] * 1e-8;
            params[i + 1] += h;
            let fcsaft_h = FcSaftPure(params);
            let (t_h, _) = PhaseEquilibrium::pure_p(&fcsaft_h, pressure, None, Default::default())?;
            let dt_h = (t_h.convert_into(KELVIN) - t) / h;
            let dt = grad[i];
            println!(
                "{par:12}: {:11.5} {:11.5} {:.3e}",
                dt_h,
                dt,
                ((dt_h - dt) / dt).abs()
            );
            assert_relative_eq!(dt, dt_h, max_relative = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_equilibrium_liquid_density_derivatives_fit() -> FeosResult<()> {
        let fcsaft = fcsaft_non_assoc();
        let fcsaft_ad =
            FcSaftPure::<f64, 5>::seed_derivatives(&fcsaft.0, ["l", "sigma", "epsilon_k"]);
        let temperature = 200.0 * KELVIN;
        let rho = EquilibriumLiquidDensity(temperature).evaluate(&fcsaft_ad)?;
        let rho = rho.convert_into(MOL / LITER);
        let (rho, rho_grad) = (rho.re, rho.eps.unwrap_generic(U3, U1));

        println!("{rho:.5}");
        println!("{rho_grad:.5?}");

        for (i, par) in ["l", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = fcsaft.0;
            let h = params[i + 1] * 1e-7;
            params[i + 1] += h;
            let fcsaft_h = FcSaftPure(params);
            let (_, [_, rho_h]) =
                PhaseEquilibrium::pure_t(&fcsaft_h, temperature, None, Default::default())?;
            let drho_h = (rho_h.convert_into(MOL / LITER) - rho) / h;
            let drho = rho_grad[i];
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
    fn test_liquid_density_derivatives_fit() -> FeosResult<()> {
        let fcsaft = fcsaft_non_assoc();
        let fcsaft_ad =
            FcSaftPure::<f64, 5>::seed_derivatives(&fcsaft.0, ["l", "sigma", "epsilon_k"]);
        let temperature = 200.0 * KELVIN;
        let pressure = BAR;
        let rho = LiquidDensity(temperature, pressure).evaluate(&fcsaft_ad)?;
        let rho = rho.convert_into(MOL / LITER);
        let (rho, grad) = (rho.re, rho.eps.unwrap_generic(U3, U1));

        println!("{rho:.5}");
        println!("{grad:.5?}");

        for (i, par) in ["l", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = fcsaft.0;
            let h = params[i + 1] * 1e-7;
            params[i + 1] += h;
            let fcsaft_h = FcSaftPure(params);
            let rho_h = State::new_npt(&fcsaft_h, temperature, pressure, (), Some(Liquid))?.density;
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
}
