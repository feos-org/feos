use super::{HelmholtzEnergyWrapper, ParametersAD, ResidualHelmholtzEnergy, StateAD};
use feos_core::{Contributions, FeosResult, PhaseEquilibrium, ReferenceSystem};
use nalgebra::SVector;
use ndarray::{arr1, Array};
use num_dual::{linalg::LU, DualNum};
use quantity::{Dimensionless, Moles, Pressure, Temperature};

impl<'a, R: ResidualHelmholtzEnergy<N>, D: DualNum<f64> + Copy, const N: usize>
    StateAD<'a, R, D, N>
{
    /// Perform a Tp-flash calculation. Returns the [PhaseEquilibriumAD] and the vapor fraction.
    pub fn tp_flash(&self) -> FeosResult<(PhaseEquilibriumAD<'a, R, D, N>, Dimensionless<D>)> {
        let pressure = self.pressure();
        let feed = Moles::from_reduced(arr1(&self.molefracs.data.0[0].map(|x| x.re())));
        let vle = PhaseEquilibrium::tp_flash(
            &self.eos.eos,
            self.temperature.re(),
            pressure.re(),
            &feed,
            None,
            Default::default(),
            None,
        )?;
        let rho_l = vle.liquid().partial_density.to_reduced();
        let mut rho_l = SVector::from_fn(|i, _| D::from(rho_l[i]));
        let rho_v = vle.vapor().partial_density.to_reduced();
        let mut rho_v = SVector::from_fn(|i, _| D::from(rho_v[i]));
        let mut v_l = D::from(vle.liquid().volume.to_reduced());
        let mut v_v = D::from(vle.vapor().volume.to_reduced());
        let t = self.reduced_temperature;
        let p = pressure.into_reduced();

        for _ in 0..D::NDERIV {
            let (p_l, mu_res_l, dp_l, dmu_l) = R::dmu_drho(&self.eos.parameters, t, &rho_l);
            let (p_v, mu_res_v, dp_v, dmu_v) = R::dmu_drho(&self.eos.parameters, t, &rho_v);

            let f = Array::from_shape_fn((2 * N + 2,), |i| {
                if i < N {
                    mu_res_l[i] - mu_res_v[i] + (rho_l[i] / rho_v[i]).ln() * t
                } else if i < 2 * N {
                    rho_l[i - N] * v_l + rho_v[i - N] * v_v - self.molefracs[i - N]
                } else if i == 2 * N {
                    p_l - p
                } else if i == 2 * N + 1 {
                    p_v - p
                } else {
                    unreachable!()
                }
            });
            let jac = Array::from_shape_fn((2 * N + 2, 2 * N + 2), |(i, j)| {
                if i < N {
                    if j < N {
                        dmu_l[(i, j)]
                    } else if j < 2 * N {
                        -dmu_v[(i, j - N)]
                    } else {
                        D::zero()
                    }
                } else if i < 2 * N {
                    if j + N == i {
                        v_l
                    } else if j == i {
                        v_v
                    } else if j == 2 * N {
                        rho_l[i - N]
                    } else if j == 2 * N + 1 {
                        rho_v[i - N]
                    } else {
                        D::zero()
                    }
                } else if i == 2 * N && j < N {
                    dp_l[j]
                } else if i == 2 * N + 1 && N <= j && j < 2 * N {
                    dp_v[j - N]
                } else {
                    D::zero()
                }
            });

            let dx = LU::new(jac).unwrap().solve(&(-f));
            let drho_l = SVector::from_fn(|i, _| dx[i]);
            let drho_v = SVector::from_fn(|i, _| dx[i + N]);
            let dv_l = dx[2 * N];
            let dv_v = dx[2 * N + 1];

            rho_l += drho_l;
            rho_v += drho_v;
            v_l += dv_l;
            v_v += dv_v;
        }
        let molar_volume_l = rho_l.sum().recip();
        let molar_volume_v = rho_v.sum().recip();
        let molefracs_l = rho_l * molar_volume_l;
        let molefracs_v = rho_v * molar_volume_v;
        Ok((
            PhaseEquilibriumAD {
                liquid: StateAD::new(self.eos, t, molar_volume_l, molefracs_l),
                vapor: StateAD::new(self.eos, t, molar_volume_v, molefracs_v),
            },
            Dimensionless::from_reduced(v_v / molar_volume_v / self.molefracs.sum()),
        ))
    }
}

/// An equilibrium state consisting of a vapor and a liquid phase.
pub struct PhaseEquilibriumAD<'a, E: ParametersAD, D: DualNum<f64> + Copy, const N: usize> {
    pub liquid: StateAD<'a, E, D, N>,
    pub vapor: StateAD<'a, E, D, N>,
}

impl<'a, R: ResidualHelmholtzEnergy<1>, D: DualNum<f64> + Copy> PhaseEquilibriumAD<'a, R, D, 1> {
    /// Calculate a phase equilibrium of a pure component for a given temperature.
    /// Returns the phase equilibrium and the vapor pressure.
    pub fn new_t(
        eos: &'a HelmholtzEnergyWrapper<R, D, 1>,
        temperature: Temperature<D>,
    ) -> FeosResult<(Self, Pressure<D>)> {
        let vle = PhaseEquilibrium::pure(&eos.eos, temperature.re(), None, Default::default())?;
        let mut density1 = D::from(vle.liquid().density.to_reduced());
        let mut density2 = D::from(vle.vapor().density.to_reduced());
        let molefracs = SVector::from([D::one()]);
        let t = temperature.into_reduced();
        let mut p = D::from(vle.vapor().pressure(Contributions::Total).to_reduced());
        for _ in 0..D::NDERIV {
            let (f1, p1, dp_drho1) = R::dp_drho(&eos.parameters, t, density1.recip(), &molefracs);
            let (f2, p2, dp_drho2) = R::dp_drho(&eos.parameters, t, density2.recip(), &molefracs);
            p = -(density2 * f1 - density1 * f2
                + density1 * density2 * t * (density1 / density2).ln())
                / (density2 - density1);
            density1 -= (p1 - p) / dp_drho1;
            density2 -= (p2 - p) / dp_drho2;
        }
        Ok((
            Self {
                liquid: StateAD::new(eos, t, density1.recip(), molefracs),
                vapor: StateAD::new(eos, t, density2.recip(), molefracs),
            },
            Pressure::from_reduced(p),
        ))
    }
}

impl<'a, R: ResidualHelmholtzEnergy<N>, D: DualNum<f64> + Copy, const N: usize>
    PhaseEquilibriumAD<'a, R, D, N>
{
    /// Calculate a bubble point of a mixture for a given temperature.
    /// Returns the phase equilibrium and the bubble point pressure.
    pub fn bubble_point(
        eos: &'a HelmholtzEnergyWrapper<R, D, N>,
        temperature: Temperature<D>,
        liquid_molefracs: SVector<D, N>,
    ) -> FeosResult<(Self, Pressure<D>)> {
        let x = arr1(liquid_molefracs.map(|x| x.re()).as_slice());
        let vle = PhaseEquilibrium::bubble_point(
            &eos.eos,
            temperature.re(),
            &x,
            None,
            None,
            Default::default(),
        )?;
        let rho_v = vle.vapor().partial_density.to_reduced();
        let (liquid, vapor, pressure) = Self::bubble_dew_point(
            eos,
            temperature,
            liquid_molefracs,
            vle.liquid().pressure(Contributions::Total).to_reduced(),
            vle.liquid().density.to_reduced(),
            SVector::from_fn(|i, _| rho_v[i]),
        )?;
        Ok((Self { liquid, vapor }, pressure))
    }

    /// Calculate a dew point of a mixture for a given temperature.
    /// Returns the phase equilibrium and the dew point pressure.
    pub fn dew_point(
        eos: &'a HelmholtzEnergyWrapper<R, D, N>,
        temperature: Temperature<D>,
        vapor_molefracs: SVector<D, N>,
    ) -> FeosResult<(Self, Pressure<D>)> {
        let y = arr1(vapor_molefracs.map(|y| y.re()).as_slice());
        let vle = PhaseEquilibrium::dew_point(
            &eos.eos,
            temperature.re(),
            &y,
            None,
            None,
            Default::default(),
        )?;
        let rho_l = vle.liquid().partial_density.to_reduced();
        let (vapor, liquid, pressure) = Self::bubble_dew_point(
            eos,
            temperature,
            vapor_molefracs,
            vle.vapor().pressure(Contributions::Total).to_reduced(),
            vle.vapor().density.to_reduced(),
            SVector::from_fn(|i, _| rho_l[i]),
        )?;
        Ok((Self { liquid, vapor }, pressure))
    }

    #[expect(clippy::type_complexity)]
    fn bubble_dew_point(
        eos: &'a HelmholtzEnergyWrapper<R, D, N>,
        temperature: Temperature<D>,
        molefracs: SVector<D, N>,
        pressure: f64,
        density: f64,
        partial_density_other_phase: SVector<f64, N>,
    ) -> FeosResult<(StateAD<'a, R, D, N>, StateAD<'a, R, D, N>, Pressure<D>)> {
        let mut rho = SVector::from_fn(|i, _| D::from(partial_density_other_phase[i]));
        let mut v = D::from(density.recip());
        let t = temperature.into_reduced();
        let mut p = D::from(pressure);
        for _ in 0..D::NDERIV {
            let (p_1, mu_res_1, dp_1, dmu_1) = R::dmu_drho(&eos.parameters, t, &rho);
            let (p_2, mu_res_2, dp_2, dmu_2) = R::dmu_dv(&eos.parameters, t, v, &molefracs);

            let f = Array::from_shape_fn((N + 2,), |i| {
                if i == N {
                    p_1 - p
                } else if i == N + 1 {
                    p_2 - p
                } else {
                    mu_res_1[i] - mu_res_2[i] + (rho[i] * v / molefracs[i]).ln() * t
                }
            });
            let jac = Array::from_shape_fn((N + 2, N + 2), |(i, j)| {
                if i < N && j < N {
                    dmu_1[(i, j)]
                } else if i < N && j == N {
                    -dmu_2[i]
                } else if i == N && j < N {
                    dp_1[j]
                } else if i == N + 1 && j == N {
                    dp_2
                } else if i >= N && j == N + 1 {
                    -D::one()
                } else {
                    D::zero()
                }
            });

            let dx = LU::new(jac).unwrap().solve(&(-f));
            let drho = SVector::from_fn(|i, _| dx[i]);
            let dv = dx[N];
            let dp = dx[N + 1];

            rho += drho;
            v += dv;
            p += dp;
        }
        let v_o = rho.sum().recip();
        let molefracs_other_phase = rho * v_o;
        Ok((
            StateAD::new(eos, t, v, molefracs),
            StateAD::new(eos, t, v_o, molefracs_other_phase),
            Pressure::from_reduced(p),
        ))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::eos::pcsaft::test::pcsaft;
    use crate::eos::{GcPcSaft, GcPcSaftParameters, Joback};
    use crate::EquationOfStateAD;
    use approx::assert_relative_eq;
    use feos_core::{Contributions, DensityInitialization, FeosResult, PhaseEquilibrium};
    use num_dual::{Dual, Dual64};
    use quantity::KELVIN;
    use std::collections::HashMap;

    #[test]
    fn test_phase_equilibrium() -> FeosResult<()> {
        let (pcsaft, eos) = pcsaft()?;
        let pcsaft = pcsaft.wrap();
        let temperature = 250.0 * KELVIN;
        let (vle, p) = PhaseEquilibriumAD::new_t(&pcsaft, temperature)?;
        let rho_l = 1.0 / vle.liquid.molar_volume;
        let rho_v = 1.0 / vle.vapor.molar_volume;
        let vle_feos = PhaseEquilibrium::pure(&eos, temperature, None, Default::default())?;
        let p_feos = vle_feos.vapor().pressure(Contributions::Total);
        println!("{:.5} {:.5} {:.5}", rho_l, rho_v, p);
        println!(
            "{:.5} {:.5} {:.5}",
            vle_feos.liquid().density,
            vle_feos.vapor().density,
            p_feos
        );
        println!("{} {}", vle.liquid.pressure(), vle.vapor.pressure());
        assert_relative_eq!(rho_l, vle_feos.liquid().density, max_relative = 1e-10);
        assert_relative_eq!(rho_v, vle_feos.vapor().density, max_relative = 1e-10);
        assert_relative_eq!(p, p_feos, max_relative = 1e-10);
        Ok(())
    }

    #[test]
    fn test_phase_equilibrium_derivative() -> FeosResult<()> {
        let (pcsaft, eos) = pcsaft()?;
        let eos_ad = pcsaft.wrap().derivatives(pcsaft.params());
        let t: Temperature<Dual64> = Temperature::from_reduced(Dual::from(250.0).derivative());
        let (vle, p) = PhaseEquilibriumAD::new_t(&eos_ad, t)?;
        let rho_l = vle.liquid.reduced_molar_volume.recip();
        let rho_v = vle.vapor.reduced_molar_volume.recip();
        let p = p.into_reduced();
        let vle_feos = PhaseEquilibrium::pure(&eos, 250.0 * KELVIN, None, Default::default())?;
        let h = 1e-5 * KELVIN;
        let vle_feos_h =
            PhaseEquilibrium::pure(&eos, 250.0 * KELVIN + h, None, Default::default())?;
        let drho_l = ((vle_feos_h.liquid().density - vle_feos.liquid().density) / h).to_reduced();
        let drho_v = ((vle_feos_h.vapor().density - vle_feos.vapor().density) / h).to_reduced();
        let dp = ((vle_feos_h.vapor().pressure(Contributions::Total)
            - vle_feos.vapor().pressure(Contributions::Total))
            / h)
            .to_reduced();
        println!("{:11.5e} {:11.5e} {:11.5e}", rho_l.eps, rho_v.eps, p.eps);
        println!("{:11.5e} {:11.5e} {:11.5e}", drho_l, drho_v, dp,);
        println!(
            "{} {}",
            vle.liquid.pressure().into_reduced(),
            vle.vapor.pressure().into_reduced()
        );
        assert_relative_eq!(rho_l.eps, drho_l, max_relative = 1e-5);
        assert_relative_eq!(rho_v.eps, drho_v, max_relative = 1e-5);
        assert_relative_eq!(p.eps, dp, max_relative = 1e-5);
        Ok(())
    }

    fn acetone_pentane_parameters() -> GcPcSaftParameters<f64, 2> {
        let mut groups1 = HashMap::new();
        groups1.insert("CH3", 2.0);
        groups1.insert(">C=O", 1.0);
        let mut bonds1 = HashMap::new();
        bonds1.insert(["CH3", ">C=O"], 2.0);
        let mut groups2 = HashMap::new();
        groups2.insert("CH3", 2.0);
        groups2.insert("CH2", 3.0);
        let mut bonds2 = HashMap::new();
        bonds2.insert(["CH3", "CH2"], 2.0);
        bonds2.insert(["CH2", "CH2"], 2.0);

        GcPcSaftParameters::from_groups([&groups1, &groups2], [&bonds1, &bonds2])
    }

    fn acetone_groups() -> HashMap<&'static str, f64> {
        let mut groups = HashMap::new();
        groups.insert("CH3", 2.0);
        groups.insert(">C=O", 1.0);
        groups
    }

    fn pentane_groups() -> HashMap<&'static str, f64> {
        let mut groups = HashMap::new();
        groups.insert("CH3", 2.0);
        groups.insert("CH2", 3.0);
        groups
    }

    #[test]
    fn test_dew_point() -> FeosResult<()> {
        let params = GcPcSaft(acetone_pentane_parameters());
        let joback = [
            Joback(Joback::from_group_counts(&acetone_groups())),
            Joback(Joback::from_group_counts(&pentane_groups())),
        ];

        let mut params_dual = params.params::<Dual64>();
        params_dual.groups[0].eps = 1.0;
        let joback_dual = joback.map(|j| j.params());

        let mut params_h = GcPcSaft(acetone_pentane_parameters());
        let h = 1e-7;
        params_h.0.groups[0] += h;

        let eos = EquationOfStateAD::new(joback, params).wrap();
        let eos_h = EquationOfStateAD::new(joback, params_h).wrap();
        let eos_dual = eos.derivatives((joback_dual, params_dual));

        let temperature = Temperature::from_reduced(Dual::from(300.0));
        let vapor_molefracs = SVector::from([Dual::from(0.5); 2]);
        let (vle, p_dew) = PhaseEquilibriumAD::dew_point(
            &eos,
            Temperature::from_reduced(300.0),
            SVector::from([0.5, 0.5]),
        )?;
        let (vle_h, p_dew_h) = PhaseEquilibriumAD::dew_point(
            &eos_h,
            Temperature::from_reduced(300.0),
            SVector::from([0.5, 0.5]),
        )?;
        let (vle_dual, p_dew_dual) =
            PhaseEquilibriumAD::dew_point(&eos_dual, temperature, vapor_molefracs)?;

        println!(
            "{:.6} + {:.6}ε   {:.6} + {:.6}ε   {:.6} + {:.6}ε",
            vle.vapor.reduced_molar_volume,
            (vle_h.vapor.reduced_molar_volume - vle.vapor.reduced_molar_volume) / h,
            vle.liquid.reduced_molar_volume,
            (vle_h.liquid.reduced_molar_volume - vle.liquid.reduced_molar_volume) / h,
            p_dew.into_reduced(),
            (p_dew_h.into_reduced() - p_dew.into_reduced()) / h
        );
        println!(
            "{:.6} + {:.6}ε   {:.6} + {:.6}ε   {:.6} + {:.6}ε",
            vle_dual.vapor.reduced_molar_volume.re,
            vle_dual.vapor.reduced_molar_volume.eps,
            vle_dual.liquid.reduced_molar_volume.re,
            vle_dual.liquid.reduced_molar_volume.eps,
            p_dew_dual.into_reduced().re,
            p_dew_dual.into_reduced().eps,
        );

        println!(
            "{:.6} + {:.6}ε   {:.6} + {:.6}ε",
            vle.liquid.molefracs[0],
            (vle_h.liquid.molefracs[0] - vle.liquid.molefracs[0]) / h,
            vle.liquid.molefracs[1],
            (vle_h.liquid.molefracs[1] - vle.liquid.molefracs[1]) / h,
        );
        println!(
            "{:.6} + {:.6}ε   {:.6} + {:.6}ε",
            vle_dual.liquid.molefracs[0].re,
            vle_dual.liquid.molefracs[0].eps,
            vle_dual.liquid.molefracs[1].re,
            vle_dual.liquid.molefracs[1].eps
        );

        let dx = (vle_h.liquid.molefracs[0] - vle.liquid.molefracs[0]) / h;
        assert_relative_eq!(vle_dual.liquid.molefracs[0].eps, dx, max_relative = 1e-6);
        let dp = (p_dew_h.into_reduced() - p_dew.into_reduced()) / h;
        assert_relative_eq!(p_dew_dual.into_reduced().eps, dp, max_relative = 1e-6);

        Ok(())
    }

    #[test]
    fn test_bubble_point() -> FeosResult<()> {
        let params = GcPcSaft(acetone_pentane_parameters());
        let joback = [
            Joback(Joback::from_group_counts(&acetone_groups())),
            Joback(Joback::from_group_counts(&pentane_groups())),
        ];

        let mut params_dual = params.params::<Dual64>();
        params_dual.groups[0].eps = 1.0;
        let joback_dual = joback.map(|j| j.params());

        let mut params_h = GcPcSaft(acetone_pentane_parameters());
        let h = 1e-7;
        params_h.0.groups[0] += h;

        let eos = EquationOfStateAD::new(joback, params).wrap();
        let eos_h = EquationOfStateAD::new(joback, params_h).wrap();
        let eos_dual = eos.derivatives((joback_dual, params_dual));

        let temperature = Temperature::from_reduced(Dual::from(300.0));
        let liquid_molefracs = SVector::from([Dual::from(0.5); 2]);
        let (vle, p_bubble) = PhaseEquilibriumAD::bubble_point(
            &eos,
            Temperature::from_reduced(300.0),
            SVector::from([0.5, 0.5]),
        )?;
        let (vle_h, p_bubble_h) = PhaseEquilibriumAD::bubble_point(
            &eos_h,
            Temperature::from_reduced(300.0),
            SVector::from([0.5, 0.5]),
        )?;
        let (vle_dual, p_bubble_dual) =
            PhaseEquilibriumAD::bubble_point(&eos_dual, temperature, liquid_molefracs)?;

        println!(
            "{:.6} + {:.6}ε   {:.6} + {:.6}ε   {:.6} + {:.6}ε",
            vle.vapor.reduced_molar_volume,
            (vle_h.vapor.reduced_molar_volume - vle.vapor.reduced_molar_volume) / h,
            vle.liquid.reduced_molar_volume,
            (vle_h.liquid.reduced_molar_volume - vle.liquid.reduced_molar_volume) / h,
            p_bubble.into_reduced(),
            (p_bubble_h.into_reduced() - p_bubble.into_reduced()) / h
        );
        println!(
            "{:.6} + {:.6}ε   {:.6} + {:.6}ε   {:.6} + {:.6}ε",
            vle_dual.vapor.reduced_molar_volume.re,
            vle_dual.vapor.reduced_molar_volume.eps,
            vle_dual.liquid.reduced_molar_volume.re,
            vle_dual.liquid.reduced_molar_volume.eps,
            p_bubble_dual.into_reduced().re,
            p_bubble_dual.into_reduced().eps,
        );

        println!(
            "{:.6} + {:.6}ε   {:.6} + {:.6}ε",
            vle.vapor.molefracs[0],
            (vle_h.vapor.molefracs[0] - vle.vapor.molefracs[0]) / h,
            vle.vapor.molefracs[1],
            (vle_h.vapor.molefracs[1] - vle.vapor.molefracs[1]) / h,
        );
        println!(
            "{:.6} + {:.6}ε   {:.6} + {:.6}ε",
            vle_dual.vapor.molefracs[0].re,
            vle_dual.vapor.molefracs[0].eps,
            vle_dual.vapor.molefracs[1].re,
            vle_dual.vapor.molefracs[1].eps
        );

        let dx = (vle_h.vapor.molefracs[0] - vle.vapor.molefracs[0]) / h;
        assert_relative_eq!(vle_dual.vapor.molefracs[0].eps, dx, max_relative = 1e-6);
        let dp = (p_bubble_h.into_reduced() - p_bubble.into_reduced()) / h;
        assert_relative_eq!(p_bubble_dual.into_reduced().eps, dp, max_relative = 1e-3);

        Ok(())
    }

    #[test]
    fn test_tp_flash() -> FeosResult<()> {
        let params = GcPcSaft(acetone_pentane_parameters());
        let joback = [
            Joback(Joback::from_group_counts(&acetone_groups())),
            Joback(Joback::from_group_counts(&pentane_groups())),
        ];

        let mut params_dual = params.params::<Dual64>();
        params_dual.groups[0].eps = 1.0;
        let joback_dual = joback.map(|j| j.params());

        let mut params_h = GcPcSaft(acetone_pentane_parameters());
        let h = 1e-5;
        params_h.0.groups[0] += h;

        let eos = EquationOfStateAD::new(joback, params).wrap();
        let eos_h = EquationOfStateAD::new(joback, params_h).wrap();
        let eos_dual = eos.derivatives((joback_dual, params_dual));

        let temperature = Temperature::from_reduced(Dual::from(300.0));
        let pressure = Pressure::from_reduced(Dual::from(0.005));
        let molefracs = SVector::from([Dual::from(0.5); 2]);
        let (vle_dual, phi_dual) = StateAD::new_tp(
            &eos_dual,
            temperature,
            pressure,
            molefracs,
            DensityInitialization::None,
        )?
        .tp_flash()?;
        let (vle, phi) = StateAD::new_tp(
            &eos,
            Temperature::from_reduced(300.0),
            Pressure::from_reduced(0.005),
            SVector::from([0.5, 0.5]),
            DensityInitialization::None,
        )?
        .tp_flash()?;
        let (vle_h, phi_h) = StateAD::new_tp(
            &eos_h,
            Temperature::from_reduced(300.0),
            Pressure::from_reduced(0.005),
            SVector::from([0.5, 0.5]),
            DensityInitialization::None,
        )?
        .tp_flash()?;

        println!(
            "{:.6} + {:.6}ε   {:.6} + {:.6}ε   {:.6} + {:.6}ε",
            vle.vapor.reduced_molar_volume,
            (vle_h.vapor.reduced_molar_volume - vle.vapor.reduced_molar_volume) / h,
            vle.liquid.reduced_molar_volume,
            (vle_h.liquid.reduced_molar_volume - vle.liquid.reduced_molar_volume) / h,
            phi.to_reduced(),
            (phi_h - phi).to_reduced() / h
        );
        println!(
            "{:.6} + {:.6}ε   {:.6} + {:.6}ε   {:.6} + {:.6}ε",
            vle_dual.vapor.reduced_molar_volume.re,
            vle_dual.vapor.reduced_molar_volume.eps,
            vle_dual.liquid.reduced_molar_volume.re,
            vle_dual.liquid.reduced_molar_volume.eps,
            phi_dual.into_reduced().re,
            phi_dual.into_reduced().eps
        );

        println!(
            "{:.6} + {:.6}ε   {:.6} + {:.6}ε",
            vle.vapor.molefracs[0],
            (vle_h.vapor.molefracs[0] - vle.vapor.molefracs[0]) / h,
            vle.vapor.molefracs[1],
            (vle_h.vapor.molefracs[1] - vle.vapor.molefracs[1]) / h,
        );
        println!(
            "{:.6} + {:.6}ε   {:.6} + {:.6}ε",
            vle_dual.vapor.molefracs[0].re,
            vle_dual.vapor.molefracs[0].eps,
            vle_dual.vapor.molefracs[1].re,
            vle_dual.vapor.molefracs[1].eps
        );

        println!(
            "{:.6} + {:.6}ε   {:.6} + {:.6}ε",
            vle.liquid.molefracs[0],
            (vle_h.liquid.molefracs[0] - vle.liquid.molefracs[0]) / h,
            vle.liquid.molefracs[1],
            (vle_h.liquid.molefracs[1] - vle.liquid.molefracs[1]) / h,
        );
        println!(
            "{:.6} + {:.6}ε   {:.6} + {:.6}ε",
            vle_dual.liquid.molefracs[0].re,
            vle_dual.liquid.molefracs[0].eps,
            vle_dual.liquid.molefracs[1].re,
            vle_dual.liquid.molefracs[1].eps
        );

        let dx = (vle_h.vapor.molefracs[0] - vle.vapor.molefracs[0]) / h;
        assert_relative_eq!(vle_dual.vapor.molefracs[0].eps, dx, max_relative = 1e-4);
        assert_relative_eq!(
            phi_dual.into_reduced().eps,
            (phi_h - phi).into_reduced() / h,
            max_relative = 1e-4
        );

        Ok(())
    }
}
