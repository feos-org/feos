use feos_core::{
    DensityInitialization::Liquid, FeosResult, PhaseEquilibrium, ReferenceSystem, Residual,
    density_iteration,
};
use nalgebra::{Const, SVector, U1};
use num_dual::{DualStruct, DualVec};
use quantity::{Density, Pressure, Temperature};

mod parallel;
pub use parallel::{BinaryModel, PureModel};

type Gradient<const P: usize> = DualVec<f64, f64, Const<P>>;

// impl<R: Residual<U1, Gradient<P>>, const P: usize> HelmholtzEnergyWrapper<'_, R, Gradient<P>, 1> {
//     pub fn vapor_pressure(&self, temperature: Temperature) -> FeosResult<Pressure<Gradient<P>>> {
//         let eos_f64 = self.eos.wrap();
//         let (_, [vapor_density, liquid_density]) =
//             PhaseEquilibriumAD::pure_t(&eos_f64, temperature, None, Default::default())?;

//         // implicit differentiation is implemented here instead of just calling pure_t with dual
//         // numbers, because for the first derivative, we can avoid calculating density derivatives.
//         let v1 = 1.0 / liquid_density.to_reduced();
//         let v2 = 1.0 / vapor_density.to_reduced();
//         let t = temperature.into_reduced();
//         let (a1, a2) = {
//             let t = Gradient::from(t);
//             let v1 = Gradient::from(v1);
//             let v2 = Gradient::from(v2);
//             let x = SVector::from([Gradient::from(1.0)]);

//             let a1 = R::residual_molar_helmholtz_energy(self.parameters, t, v1, &x);
//             let a2 = R::residual_molar_helmholtz_energy(self.parameters, t, v2, &x);
//             (a1, a2)
//         };

//         let p = -(a1 - a2 + t * (v2 / v1).ln()) / (v1 - v2);
//         Ok(Pressure::from_reduced(p))
//     }

//     pub fn equilibrium_liquid_density(
//         &self,
//         temperature: Temperature,
//     ) -> FeosResult<Density<Gradient<P>>> {
//         let t = Temperature::from_inner(&temperature);
//         PhaseEquilibriumAD::pure_t(self, t, None, Default::default()).map(|(_, [_, rho])| rho)
//     }

//     pub fn liquid_density(
//         &self,
//         temperature: Temperature,
//         pressure: Pressure,
//     ) -> FeosResult<Density<Gradient<P>>> {
//         let x = Self::pure_molefracs();
//         let t = Temperature::from_inner(&temperature);
//         let p = Pressure::from_inner(&pressure);
//         density_iteration(self, t, p, &x, Some(Liquid))
//     }
// }

// impl<R: ResidualHelmholtzEnergy<2>, const P: usize> HelmholtzEnergyWrapper<'_, R, Gradient<P>, 2> {
//     pub fn bubble_point_pressure(
//         &self,
//         temperature: Temperature,
//         pressure: Option<Pressure>,
//         liquid_molefracs: SVector<f64, 2>,
//     ) -> FeosResult<Pressure<Gradient<P>>> {
//         let eos_f64 = self.eos.wrap();
//         let vle = PhaseEquilibriumAD::bubble_point(
//             &eos_f64,
//             temperature,
//             &liquid_molefracs,
//             pressure,
//             None,
//             Default::default(),
//         )?;

//         let v_l = 1.0 / vle.liquid().density.to_reduced();
//         let v_v = 1.0 / vle.vapor().density.to_reduced();
//         let y = &vle.vapor().molefracs;
//         let y: SVector<_, 2> = SVector::from_fn(|i, _| y[i]);
//         let t = temperature.into_reduced();
//         let (a_l, a_v, v_l, v_v) = {
//             let t = Gradient::from(t);
//             let v_l = Gradient::from(v_l);
//             let v_v = Gradient::from(v_v);
//             let y = y.map(Gradient::from);
//             let x = liquid_molefracs.map(Gradient::from);

//             let a_v = R::residual_molar_helmholtz_energy(self.parameters, t, v_v, &y);
//             let (p_l, mu_res_l, dp_l, dmu_l) = self.dmu_dv(t, v_l, &x);
//             let vi_l = dmu_l / dp_l;
//             let v_l = vi_l.dot(&y);
//             let a_l = (mu_res_l - vi_l * p_l).dot(&y);
//             (a_l, a_v, v_l, v_v)
//         };
//         let rho_l = vle.liquid().partial_density.to_reduced();
//         let rho_l = [rho_l[0], rho_l[1]];
//         let rho_v = vle.vapor().partial_density.to_reduced();
//         let rho_v = [rho_v[0], rho_v[1]];
//         let p = -(a_v - a_l
//             + t * (y[0] * (rho_v[0] / rho_l[0]).ln() + y[1] * (rho_v[1] / rho_l[1]).ln() - 1.0))
//             / (v_v - v_l);
//         Ok(Pressure::from_reduced(p))
//     }

//     pub fn dew_point_pressure(
//         &self,
//         temperature: Temperature,
//         pressure: Option<Pressure>,
//         vapor_molefracs: SVector<f64, 2>,
//     ) -> FeosResult<Pressure<Gradient<P>>> {
//         let eos_f64 = self.eos.wrap();
//         let vle = PhaseEquilibriumAD::dew_point(
//             &eos_f64,
//             temperature,
//             &vapor_molefracs,
//             pressure,
//             None,
//             Default::default(),
//         )?;

//         let v_l = 1.0 / vle.liquid().density.to_reduced();
//         let v_v = 1.0 / vle.vapor().density.to_reduced();
//         let x = &vle.liquid().molefracs;
//         let x: SVector<_, 2> = SVector::from_fn(|i, _| x[i]);
//         let t = temperature.into_reduced();
//         let (a_l, a_v, v_l, v_v) = {
//             let t = Gradient::from(t);
//             let v_l = Gradient::from(v_l);
//             let v_v = Gradient::from(v_v);
//             let x = x.map(Gradient::from);
//             let y = vapor_molefracs.map(Gradient::from);

//             let a_l = R::residual_molar_helmholtz_energy(self.parameters, t, v_l, &x);
//             let (p_v, mu_res_v, dp_v, dmu_v) = self.dmu_dv(t, v_v, &y);
//             let vi_v = dmu_v / dp_v;
//             let v_v = vi_v.dot(&x);
//             let a_v = (mu_res_v - vi_v * p_v).dot(&x);
//             (a_l, a_v, v_l, v_v)
//         };
//         let rho_l = vle.liquid().partial_density.to_reduced();
//         let rho_l = [rho_l[0], rho_l[1]];
//         let rho_v = vle.vapor().partial_density.to_reduced();
//         let rho_v = [rho_v[0], rho_v[1]];
//         let p = -(a_l - a_v
//             + t * (x[0] * (rho_l[0] / rho_v[0]).ln() + x[1] * (rho_l[1] / rho_v[1]).ln() - 1.0))
//             / (v_l - v_v);
//         Ok(Pressure::from_reduced(p))
//     }
// }

#[cfg(test)]
#[cfg(feature = "pcsaft")]
mod test {
    use super::*;
    use crate::eos::pcsaft::test::{pcsaft, pcsaft_binary, pcsaft_non_assoc};
    use crate::eos::{PcSaftBinary, PcSaftPure};
    use crate::{NamedParameters, ParametersAD, PhaseEquilibriumAD, StateAD};
    use approx::assert_relative_eq;
    use feos_core::Contributions;
    use nalgebra::U1;
    use quantity::{BAR, KELVIN, LITER, MOL, PASCAL};

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
        let pcsaft_ad = pcsaft.derivatives(&pcsaft_ad);
        let temperature = 250.0 * KELVIN;
        let p = pcsaft_ad.vapor_pressure(temperature)?;
        let p = p.convert_into(PASCAL);
        let (p, grad) = (p.re, p.eps.unwrap_generic(Const::<8>, U1));

        println!("{p:.5}");
        println!("{grad:.5?}");

        for (i, par) in pcsaft_params.into_iter().enumerate() {
            let mut params = pcsaft.0;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = PcSaftPure(params);
            let pcsaft_h = pcsaft_h.wrap();
            let p_h = PhaseEquilibriumAD::pure_t(&pcsaft_h, temperature, None, Default::default())?
                .vapor()
                .pressure(Contributions::Total);
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
        let (pcsaft, _) = pcsaft_non_assoc()?;
        let pcsaft_ad = pcsaft.named_derivatives(["m", "sigma", "epsilon_k"]);
        let pcsaft_ad = pcsaft.derivatives(&pcsaft_ad);
        let temperature = 150.0 * KELVIN;
        let p = pcsaft_ad.vapor_pressure(temperature)?;
        let p = p.convert_into(PASCAL);
        let (p, grad) = (p.re, p.eps.unwrap_generic(Const::<3>, U1));

        println!("{p:.5}");
        println!("{grad:.5?}");

        for (i, par) in ["m", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = pcsaft.0;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = PcSaftPure(params);
            let pcsaft_h = pcsaft_h.wrap();
            let p_h = PhaseEquilibriumAD::pure_t(&pcsaft_h, temperature, None, Default::default())?
                .vapor()
                .pressure(Contributions::Total);
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
        let (pcsaft, _) = pcsaft_non_assoc()?;
        let pcsaft_ad = pcsaft.named_derivatives(["m", "sigma", "epsilon_k"]);
        let pcsaft_ad = pcsaft.derivatives(&pcsaft_ad);
        let temperature = 150.0 * KELVIN;
        let (p, rho) = pcsaft_ad.equilibrium_liquid_density(temperature)?;
        let p = p.convert_into(PASCAL);
        let rho = rho.convert_into(MOL / LITER);
        let (p, p_grad) = (p.re, p.eps.unwrap_generic(Const::<3>, U1));
        let (rho, rho_grad) = (rho.re, rho.eps.unwrap_generic(Const::<3>, U1));

        println!("{p:.5} {rho:.5}");
        println!("{p_grad:.5?}");
        println!("{rho_grad:.5?}");

        for (i, par) in ["m", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = *pcsaft;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = PcSaftPure(params);
            let vle = PhaseEquilibriumAD::pure_t(
                &pcsaft_h.wrap(),
                temperature,
                None,
                Default::default(),
            )?;
            let rho_h = vle.liquid().density;
            let p_h = vle.vapor().pressure(Contributions::Total);
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
        let (pcsaft, _) = pcsaft_non_assoc()?;
        let pcsaft_ad = pcsaft.named_derivatives(["m", "sigma", "epsilon_k"]);
        let pcsaft_ad = pcsaft.derivatives(&pcsaft_ad);
        let temperature = 150.0 * KELVIN;
        let pressure = BAR;
        let rho = pcsaft_ad.liquid_density(temperature, pressure)?;
        let rho = rho.convert_into(MOL / LITER);
        let (rho, grad) = (rho.re, rho.eps.unwrap_generic(Const::<3>, U1));

        println!("{rho:.5}");
        println!("{grad:.5?}");

        for (i, par) in ["m", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = *pcsaft;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = PcSaftPure(params);
            let rho_h = StateAD::new_xpt(
                &pcsaft_h.wrap(),
                temperature,
                pressure,
                &SVector::from([1.0]),
                Liquid,
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
        let pcsaft_ad = pcsaft.derivatives(&pcsaft_ad);
        let temperature = 500.0 * KELVIN;
        let x = SVector::from([0.5, 0.5]);
        let p = pcsaft_ad.bubble_point_pressure(temperature, None, x)?;
        let p = p.convert_into(BAR);
        let (p, [[grad]]) = (p.re, p.eps.unwrap_generic(U1, U1).data.0);

        println!("{p:.5}");
        println!("{grad:.5?}");

        let (params, mut kij) = *pcsaft;
        let h = 1e-7;
        kij += h;
        let pcsaft_h = PcSaftBinary::new(params, kij);
        let p_h = PhaseEquilibriumAD::bubble_point(
            &pcsaft_h.wrap(),
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
        let pcsaft_ad = pcsaft.derivatives(&pcsaft_ad);
        let temperature = 500.0 * KELVIN;
        let y = SVector::from([0.5, 0.5]);
        let p = pcsaft_ad.dew_point_pressure(temperature, None, y)?;
        let p = p.convert_into(BAR);
        let (p, [[grad]]) = (p.re, p.eps.unwrap_generic(U1, U1).data.0);

        println!("{p:.5}");
        println!("{grad:.5?}");

        let (params, mut kij) = *pcsaft;
        let h = 1e-7;
        kij += h;
        let pcsaft_h = PcSaftBinary::new(params, kij);
        let p_h = PhaseEquilibriumAD::dew_point(
            &pcsaft_h.wrap(),
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
}
