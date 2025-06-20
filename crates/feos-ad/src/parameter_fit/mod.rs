use crate::{HelmholtzEnergyWrapper, ResidualHelmholtzEnergy};
use feos_core::{
    DensityInitialization::Liquid, EosResult, PhaseEquilibrium, ReferenceSystem, State,
};
use nalgebra::{Const, SVector};
use ndarray::arr1;
use num_dual::DualVec;
use quantity::{Density, Moles, Pressure, Temperature};

type Gradient<const P: usize> = DualVec<f64, f64, Const<P>>;

pub fn vapor_pressure<R: ResidualHelmholtzEnergy<1>, const P: usize>(
    eos: &HelmholtzEnergyWrapper<R, Gradient<P>, 1>,
    temperature: Temperature,
) -> EosResult<Pressure<Gradient<P>>> {
    let vle = PhaseEquilibrium::pure(&eos.eos, temperature, None, Default::default())?;

    let v1 = 1.0 / vle.liquid().density.to_reduced();
    let v2 = 1.0 / vle.vapor().density.to_reduced();
    let t = temperature.into_reduced();
    let (a1, a2) = {
        let t = Gradient::from(t);
        let v1 = Gradient::from(v1);
        let v2 = Gradient::from(v2);
        let x = SVector::from([Gradient::from(1.0)]);

        let a1 = R::residual_molar_helmholtz_energy(&eos.parameters, t, v1, &x);
        let a2 = R::residual_molar_helmholtz_energy(&eos.parameters, t, v2, &x);
        (a1, a2)
    };

    let p = -(a1 - a2 + t * (v2 / v1).ln()) / (v1 - v2);
    Ok(Pressure::from_reduced(p))
}

pub fn equilibrium_liquid_density<R: ResidualHelmholtzEnergy<1>, const P: usize>(
    eos: &HelmholtzEnergyWrapper<R, Gradient<P>, 1>,
    temperature: Temperature,
) -> EosResult<(Pressure<Gradient<P>>, Density<Gradient<P>>)> {
    let vle = PhaseEquilibrium::pure(&eos.eos, temperature, None, Default::default())?;

    let v_l = 1.0 / vle.liquid().density.to_reduced();
    let v_v = 1.0 / vle.vapor().density.to_reduced();
    let t = temperature.into_reduced();
    let (f_l, p_l, dp_l, a_v) = {
        let t = Gradient::from(temperature.into_reduced());
        let v_l = Gradient::from(v_l);
        let v_v = Gradient::from(v_v);
        let x = SVector::from([Gradient::from(1.0)]);

        let (f_l, p_l, dp_l) = R::dp_drho(&eos.parameters, t, v_l, &x);
        let a_v = R::residual_molar_helmholtz_energy(&eos.parameters, t, v_v, &x);
        (f_l, p_l, dp_l, a_v)
    };

    let p = -(f_l * v_l - a_v + t * (v_v / v_l).ln()) / (v_l - v_v);
    let rho = (p - p_l) / dp_l + 1.0 / v_l;
    Ok((Pressure::from_reduced(p), Density::from_reduced(rho)))
}

pub fn liquid_density<R: ResidualHelmholtzEnergy<1>, const P: usize>(
    eos: &HelmholtzEnergyWrapper<R, Gradient<P>, 1>,
    temperature: Temperature,
    pressure: Pressure,
) -> EosResult<Density<Gradient<P>>> {
    let moles = Moles::from_reduced(arr1(&[1.0]));
    let state = State::new_npt(&eos.eos, temperature, pressure, &moles, Liquid)?;

    let t = temperature.into_reduced();
    let v = 1.0 / state.density.to_reduced();
    let p0 = pressure.into_reduced();
    let (p, dp) = {
        let t = Gradient::from(t);
        let v = Gradient::from(v);
        let x = SVector::from([Gradient::from(1.0)]);
        let (_, p, dp) = R::dp_drho(&eos.parameters, t, v, &x);

        (p, dp)
    };

    let rho = -(p - p0) / dp + 1.0 / v;
    Ok(Density::from_reduced(rho))
}

pub fn bubble_point_pressure<R: ResidualHelmholtzEnergy<2>, const P: usize>(
    eos: &HelmholtzEnergyWrapper<R, Gradient<P>, 2>,
    temperature: Temperature,
    pressure: Option<Pressure>,
    liquid_molefracs: SVector<f64, 2>,
) -> EosResult<Pressure<Gradient<P>>> {
    let x = arr1(liquid_molefracs.as_slice());
    let vle = PhaseEquilibrium::bubble_point(
        &eos.eos,
        temperature,
        &x,
        pressure,
        None,
        Default::default(),
    )?;

    let v_l = 1.0 / vle.liquid().density.to_reduced();
    let v_v = 1.0 / vle.vapor().density.to_reduced();
    let y = &vle.vapor().molefracs;
    let y: SVector<_, 2> = SVector::from_fn(|i, _| y[i]);
    let t = temperature.into_reduced();
    let (a_l, a_v, v_l, v_v) = {
        let t = Gradient::from(t);
        let v_l = Gradient::from(v_l);
        let v_v = Gradient::from(v_v);
        let y = y.map(Gradient::from);
        let x = liquid_molefracs.map(Gradient::from);

        let a_v = R::residual_molar_helmholtz_energy(&eos.parameters, t, v_v, &y);
        let (p_l, mu_res_l, dp_l, dmu_l) = R::dmu_dv(&eos.parameters, t, v_l, &x);
        let vi_l = dmu_l / dp_l;
        let v_l = vi_l.dot(&y);
        let a_l = (mu_res_l - vi_l * p_l).dot(&y);
        (a_l, a_v, v_l, v_v)
    };
    let rho_l = vle.liquid().partial_density.to_reduced();
    let rho_l = [rho_l[0], rho_l[1]];
    let rho_v = vle.vapor().partial_density.to_reduced();
    let rho_v = [rho_v[0], rho_v[1]];
    let p = -(a_v - a_l
        + t * (y[0] * (rho_v[0] / rho_l[0]).ln() + y[1] * (rho_v[1] / rho_l[1]).ln() - 1.0))
        / (v_v - v_l);
    Ok(Pressure::from_reduced(p))
}

pub fn dew_point_pressure<R: ResidualHelmholtzEnergy<2>, const P: usize>(
    eos: &HelmholtzEnergyWrapper<R, Gradient<P>, 2>,
    temperature: Temperature,
    pressure: Option<Pressure>,
    vapor_molefracs: SVector<f64, 2>,
) -> EosResult<Pressure<Gradient<P>>> {
    let y = arr1(vapor_molefracs.as_slice());
    let vle = PhaseEquilibrium::dew_point(
        &eos.eos,
        temperature,
        &y,
        pressure,
        None,
        Default::default(),
    )?;

    let v_l = 1.0 / vle.liquid().density.to_reduced();
    let v_v = 1.0 / vle.vapor().density.to_reduced();
    let x = &vle.liquid().molefracs;
    let x: SVector<_, 2> = SVector::from_fn(|i, _| x[i]);
    let t = temperature.into_reduced();
    let (a_l, a_v, v_l, v_v) = {
        let t = Gradient::from(t);
        let v_l = Gradient::from(v_l);
        let v_v = Gradient::from(v_v);
        let x = x.map(Gradient::from);
        let y = vapor_molefracs.map(Gradient::from);

        let a_l = R::residual_molar_helmholtz_energy(&eos.parameters, t, v_l, &x);
        let (p_v, mu_res_v, dp_v, dmu_v) = R::dmu_dv(&eos.parameters, t, v_v, &y);
        let vi_v = dmu_v / dp_v;
        let v_v = vi_v.dot(&x);
        let a_v = (mu_res_v - vi_v * p_v).dot(&x);
        (a_l, a_v, v_l, v_v)
    };
    let rho_l = vle.liquid().partial_density.to_reduced();
    let rho_l = [rho_l[0], rho_l[1]];
    let rho_v = vle.vapor().partial_density.to_reduced();
    let rho_v = [rho_v[0], rho_v[1]];
    let p = -(a_l - a_v
        + t * (x[0] * (rho_l[0] / rho_v[0]).ln() + x[1] * (rho_l[1] / rho_v[1]).ln() - 1.0))
        / (v_l - v_v);
    Ok(Pressure::from_reduced(p))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::eos::pcsaft::test::{pcsaft, pcsaft_binary, pcsaft_non_assoc};
    use crate::eos::{PcSaftBinary, PcSaftPure};
    use crate::{ParametersAD, PhaseEquilibriumAD, StateAD};
    use approx::assert_relative_eq;
    use nalgebra::U1;
    use quantity::{BAR, KELVIN, LITER, MOL, PASCAL};

    #[test]
    fn test_vapor_pressure_derivatives() -> EosResult<()> {
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
        let pcsaft = pcsaft.wrap();
        let pcsaft_ad = pcsaft.named_derivatives(pcsaft_params);
        let temperature = 250.0 * KELVIN;
        let p = vapor_pressure(&pcsaft_ad, temperature)?;
        let p = p.convert_into(PASCAL);
        let (p, grad) = (p.re, p.eps.unwrap_generic(Const::<8>, U1));

        println!("{:.5}", p);
        println!("{:.5?}", grad);

        for (i, par) in pcsaft_params.into_iter().enumerate() {
            let mut params = pcsaft.parameters;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = PcSaftPure(params).wrap();
            let (_, p_h) = PhaseEquilibriumAD::new_t(&pcsaft_h, temperature)?;
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
    fn test_vapor_pressure_derivatives_fit() -> EosResult<()> {
        let (pcsaft, _) = pcsaft_non_assoc()?;
        let pcsaft = pcsaft.wrap();
        let pcsaft_ad = pcsaft.named_derivatives(["m", "sigma", "epsilon_k"]);
        let temperature = 150.0 * KELVIN;
        let p = vapor_pressure(&pcsaft_ad, temperature)?;
        let p = p.convert_into(PASCAL);
        let (p, grad) = (p.re, p.eps.unwrap_generic(Const::<3>, U1));

        println!("{:.5}", p);
        println!("{:.5?}", grad);

        for (i, par) in ["m", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = pcsaft.parameters;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = PcSaftPure(params).wrap();
            let (_, p_h) = PhaseEquilibriumAD::new_t(&pcsaft_h, temperature)?;
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
    fn test_equilibrium_liquid_density_derivatives_fit() -> EosResult<()> {
        let (pcsaft, _) = pcsaft_non_assoc()?;
        let pcsaft = pcsaft.wrap();
        let pcsaft_ad = pcsaft.named_derivatives(["m", "sigma", "epsilon_k"]);
        let temperature = 150.0 * KELVIN;
        let (p, rho) = equilibrium_liquid_density(&pcsaft_ad, temperature)?;
        let p = p.convert_into(PASCAL);
        let rho = rho.convert_into(MOL / LITER);
        let (p, p_grad) = (p.re, p.eps.unwrap_generic(Const::<3>, U1));
        let (rho, rho_grad) = (rho.re, rho.eps.unwrap_generic(Const::<3>, U1));

        println!("{:.5} {:.5}", p, rho);
        println!("{:.5?}", p_grad);
        println!("{:.5?}", rho_grad);

        for (i, par) in ["m", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = pcsaft.parameters;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = PcSaftPure(params).wrap();
            let (vle, p_h) = PhaseEquilibriumAD::new_t(&pcsaft_h, temperature)?;
            let v_h = vle.liquid.molar_volume;
            let dp_h = (p_h.convert_into(PASCAL) - p) / h;
            let drho_h = (v_h.convert_into(LITER / MOL).recip() - rho) / h;
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
    fn test_liquid_density_derivatives_fit() -> EosResult<()> {
        let (pcsaft, _) = pcsaft_non_assoc()?;
        let pcsaft = pcsaft.wrap();
        let pcsaft_ad = pcsaft.named_derivatives(["m", "sigma", "epsilon_k"]);
        let temperature = 150.0 * KELVIN;
        let pressure = BAR;
        let rho = liquid_density(&pcsaft_ad, temperature, pressure)?;
        let rho = rho.convert_into(MOL / LITER);
        let (rho, grad) = (rho.re, rho.eps.unwrap_generic(Const::<3>, U1));

        println!("{:.5}", rho);
        println!("{:.5?}", grad);

        for (i, par) in ["m", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = pcsaft.parameters;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = PcSaftPure(params).wrap();
            let v_h = StateAD::new_tp(
                &pcsaft_h,
                temperature,
                pressure,
                SVector::from([1.0]),
                Liquid,
            )?
            .molar_volume;
            let drho_h = (v_h.convert_into(LITER / MOL).recip() - rho) / h;
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
    fn test_bubble_point_pressure() -> EosResult<()> {
        let (pcsaft, _) = pcsaft_binary()?;
        let pcsaft = pcsaft.wrap();
        let pcsaft_ad = pcsaft.named_derivatives(["k_ij"]);
        let temperature = 500.0 * KELVIN;
        let x = SVector::from([0.5, 0.5]);
        let p = bubble_point_pressure(&pcsaft_ad, temperature, None, x)?;
        let p = p.convert_into(BAR);
        let (p, [[grad]]) = (p.re, p.eps.unwrap_generic(U1, U1).data.0);

        println!("{:.5}", p);
        println!("{:.5?}", grad);

        let (params, mut kij) = pcsaft.parameters;
        let h = 1e-7;
        kij += h;
        let pcsaft_h = PcSaftBinary::new(params, kij).wrap();
        let (_, p_h) = PhaseEquilibriumAD::bubble_point(&pcsaft_h, temperature, x)?;
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
    fn test_dew_point_pressure() -> EosResult<()> {
        let (pcsaft, _) = pcsaft_binary()?;
        let pcsaft = pcsaft.wrap();
        let pcsaft_ad = pcsaft.named_derivatives(["k_ij"]);
        let temperature = 500.0 * KELVIN;
        let y = SVector::from([0.5, 0.5]);
        let p = dew_point_pressure(&pcsaft_ad, temperature, None, y)?;
        let p = p.convert_into(BAR);
        let (p, [[grad]]) = (p.re, p.eps.unwrap_generic(U1, U1).data.0);

        println!("{:.5}", p);
        println!("{:.5?}", grad);

        let (params, mut kij) = pcsaft.parameters;
        let h = 1e-7;
        kij += h;
        let pcsaft_h = PcSaftBinary::new(params, kij).wrap();
        let (_, p_h) = PhaseEquilibriumAD::dew_point(&pcsaft_h, temperature, y)?;
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
