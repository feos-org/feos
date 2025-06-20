use super::{
    FeOsWrapper, HelmholtzEnergyWrapper, ParametersAD, ResidualHelmholtzEnergy,
    TotalHelmholtzEnergy,
};
use feos_core::{DensityInitialization, FeosResult, ReferenceSystem, State};
use nalgebra::{Const, SMatrix, SVector};
use ndarray::arr1;
use num_dual::{hessian, jacobian, third_derivative, Dual2Vec, Dual3, DualNum, DualVec};
use quantity::{MolarEnergy, MolarEntropy, MolarVolume, Moles, Pressure, Temperature};

/// An (intensive) thermodynamic state representing a single phase.
pub struct StateAD<'a, E: ParametersAD, D: DualNum<f64> + Copy, const N: usize> {
    pub eos: &'a HelmholtzEnergyWrapper<E, D, N>,
    pub temperature: Temperature<D>,
    pub molar_volume: MolarVolume<D>,
    pub molefracs: SVector<D, N>,
    pub reduced_temperature: D,
    pub reduced_molar_volume: D,
}

impl<'a, E: ParametersAD, D: DualNum<f64> + Copy, const N: usize> StateAD<'a, E, D, N> {
    /// Crate a state from its thermodynamic variables (temperature, molar volume, composition)
    pub fn new(
        eos: &'a HelmholtzEnergyWrapper<E, D, N>,
        temperature: D,
        molar_volume: D,
        molefracs: SVector<D, N>,
    ) -> Self {
        Self {
            eos,
            temperature: Temperature::from_reduced(temperature),
            molar_volume: MolarVolume::from_reduced(molar_volume),
            molefracs,
            reduced_temperature: temperature,
            reduced_molar_volume: molar_volume,
        }
    }

    fn from_state<
        F: Fn(
            &E::Parameters<DualVec<D, f64, Const<2>>>,
            DualVec<D, f64, Const<2>>,
            DualVec<D, f64, Const<2>>,
            &SVector<DualVec<D, f64, Const<2>>, N>,
        ) -> SVector<DualVec<D, f64, Const<2>>, 2>,
    >(
        eos: &'a HelmholtzEnergyWrapper<E, D, N>,
        state: State<FeOsWrapper<E, N>>,
        molefracs: SVector<D, N>,
        f: F,
        rhs: SVector<D, 2>,
    ) -> Self {
        let mut vars = SVector::from([
            D::from(state.temperature.to_reduced()),
            D::from(state.density.to_reduced().recip()),
        ]);
        let x = molefracs.map(DualVec::from_re);
        let params = E::params_from_inner(&eos.parameters);
        for _ in 0..D::NDERIV {
            let (mut f, jac) = jacobian(|vars| f(&params, vars[0], vars[1], &x), vars);
            f -= rhs;
            let det = (jac[(0, 0)] * jac[(1, 1)] - jac[(0, 1)] * jac[(1, 0)]).recip();
            vars[0] -= det * (jac[(1, 1)] * f[0] - jac[(0, 1)] * f[1]);
            vars[1] -= det * (jac[(0, 0)] * f[1] - jac[(1, 0)] * f[0]);
        }
        let [temperature, molar_volume] = vars.data.0[0];
        Self::new(eos, temperature, molar_volume, molefracs)
    }
}

impl<'a, E: ResidualHelmholtzEnergy<N>, D: DualNum<f64> + Copy, const N: usize>
    StateAD<'a, E, D, N>
{
    /// Calculate a state from given temperature, pressure and composition.
    pub fn new_tp(
        eos: &'a HelmholtzEnergyWrapper<E, D, N>,
        temperature: Temperature<D>,
        pressure: Pressure<D>,
        molefracs: SVector<D, N>,
        density_initialization: DensityInitialization,
    ) -> FeosResult<Self> {
        let t = temperature.re();
        let p = pressure.re();
        let moles = Moles::from_reduced(arr1(&molefracs.data.0[0].map(|x| x.re())));
        let state = State::new_npt(&eos.eos, t, p, &moles, density_initialization)?;
        let mut density = D::from(state.density.to_reduced());
        let t = temperature.into_reduced();
        for _ in 0..D::NDERIV {
            let (_, p, dp_drho) = E::dp_drho(&eos.parameters, t, density.recip(), &molefracs);
            density -= (p - pressure.into_reduced()) / dp_drho;
        }
        Ok(Self::new(eos, t, density.recip(), molefracs))
    }

    pub fn pressure(&self) -> Pressure<D> {
        Pressure::from_reduced(E::pressure(
            &self.eos.parameters,
            self.reduced_temperature,
            self.reduced_molar_volume,
            &self.molefracs,
        ))
    }
}

impl<'a, E: TotalHelmholtzEnergy<N>, D: DualNum<f64> + Copy, const N: usize> StateAD<'a, E, D, N> {
    /// Calculate a state from given pressure, molar entropy and composition.
    pub fn new_ps(
        eos: &'a HelmholtzEnergyWrapper<E, D, N>,
        pressure: Pressure<D>,
        molar_entropy: MolarEntropy<D>,
        molefracs: SVector<D, N>,
        density_initialization: DensityInitialization,
        initial_temperature: Option<Temperature>,
    ) -> FeosResult<Self> {
        let moles = Moles::from_reduced(arr1(&molefracs.data.0[0].map(|x| x.re())));
        let state = State::new_nps(
            &eos.eos,
            pressure.re(),
            molar_entropy.re(),
            &moles,
            density_initialization,
            initial_temperature,
        )?;
        Ok(Self::from_state(
            eos,
            state,
            molefracs,
            E::pressure_entropy,
            SVector::from([pressure.into_reduced(), molar_entropy.into_reduced()]),
        ))
    }

    /// Calculate a state from given pressure, molar enthalpy and composition.
    pub fn new_ph(
        eos: &'a HelmholtzEnergyWrapper<E, D, N>,
        pressure: Pressure<D>,
        molar_enthalpy: MolarEnergy<D>,
        molefracs: SVector<D, N>,
        density_initialization: DensityInitialization,
        initial_temperature: Option<Temperature>,
    ) -> FeosResult<Self> {
        let moles = Moles::from_reduced(arr1(&molefracs.data.0[0].map(|x| x.re())));
        let state = State::new_nph(
            &eos.eos,
            pressure.re(),
            molar_enthalpy.re(),
            &moles,
            density_initialization,
            initial_temperature,
        )?;
        Ok(Self::from_state(
            eos,
            state,
            molefracs,
            E::pressure_enthalpy,
            SVector::from([pressure.into_reduced(), molar_enthalpy.into_reduced()]),
        ))
    }

    pub fn molar_entropy(&self) -> MolarEntropy<D> {
        MolarEntropy::from_reduced(E::molar_entropy(
            &self.eos.parameters,
            self.reduced_temperature,
            self.reduced_molar_volume,
            &self.molefracs,
        ))
    }

    pub fn molar_enthalpy(&self) -> MolarEnergy<D> {
        MolarEnergy::from_reduced(E::molar_enthalpy(
            &self.eos.parameters,
            self.reduced_temperature,
            self.reduced_molar_volume,
            &self.molefracs,
        ))
    }

    pub fn molar_isochoric_heat_capacity(&self) -> MolarEntropy<D> {
        MolarEntropy::from_reduced(E::molar_isochoric_heat_capacity(
            &self.eos.parameters,
            self.reduced_temperature,
            self.reduced_molar_volume,
            &self.molefracs,
        ))
    }

    pub fn molar_isobaric_heat_capacity(&self) -> MolarEntropy<D> {
        MolarEntropy::from_reduced(E::molar_isobaric_heat_capacity(
            &self.eos.parameters,
            self.reduced_temperature,
            self.reduced_molar_volume,
            &self.molefracs,
        ))
    }
}

impl<'a, E: ResidualHelmholtzEnergy<1>, D: DualNum<f64> + Copy> StateAD<'a, E, D, 1> {
    /// Calculate the critical point of a pure component.
    pub fn critical_point_pure(eos: &'a HelmholtzEnergyWrapper<E, D, 1>) -> FeosResult<Self> {
        Self::critical_point(eos, SVector::from([D::one()]))
    }
}

impl<'a, E: ResidualHelmholtzEnergy<N>, D: DualNum<f64> + Copy, const N: usize>
    StateAD<'a, E, D, N>
{
    /// Calculate the critical point of a mixture with given composition.
    pub fn critical_point(
        eos: &'a HelmholtzEnergyWrapper<E, D, N>,
        molefracs: SVector<D, N>,
    ) -> FeosResult<Self>
    where
        Const<N>: Eigen<N>,
    {
        let moles = Moles::from_reduced(arr1(molefracs.map(|x| x.re()).as_slice()));
        let state = State::critical_point(&eos.eos, Some(&moles), None, Default::default())?;
        Ok(Self::from_state(
            eos,
            state,
            molefracs,
            Self::criticality_conditions,
            SVector::from([D::from(0.0); 2]),
        ))
    }
}

impl<E: ResidualHelmholtzEnergy<N>, D: DualNum<f64> + Copy, const N: usize> StateAD<'_, E, D, N> {
    fn criticality_conditions(
        parameters: &E::Parameters<DualVec<D, f64, Const<2>>>,
        temperature: DualVec<D, f64, Const<2>>,
        molar_volume: DualVec<D, f64, Const<2>>,
        molefracs: &SVector<DualVec<D, f64, Const<2>>, N>,
    ) -> SVector<DualVec<D, f64, Const<2>>, 2>
    where
        Const<N>: Eigen<N>,
    {
        // calculate M
        let sqrt_z = molefracs.map(|z| z.sqrt());
        let z_mix = sqrt_z * sqrt_z.transpose();
        let (_, _, m) = hessian(
            |x| {
                let params = E::params_from_inner(parameters);
                let t = Dual2Vec::from_re(temperature);
                let v = Dual2Vec::from_re(molar_volume);
                E::residual_molar_helmholtz_energy(&params, t, v, &x)
            },
            *molefracs,
        );
        let m = m.component_mul(&z_mix) / temperature + SMatrix::identity();

        // calculate smallest eigenvalue and corresponding eigenvector
        let (l, u) = <Const<N> as Eigen<N>>::eigen(m);

        let (_, _, _, c2) = third_derivative(
            |s| {
                let x = molefracs.map(Dual3::from_re);
                let x = x + sqrt_z.component_mul(&u).map(Dual3::from_re) * s;
                let params = E::params_from_inner(parameters);
                let t = Dual3::from_re(temperature);
                let v = Dual3::from_re(molar_volume);
                let ig = x.component_mul(&x.map(|x| (x / v).ln() - 1.0)).sum();
                E::residual_molar_helmholtz_energy(&params, t, v, &x) / t + ig
            },
            DualVec::from_re(D::zero()),
        );

        SVector::from([l, c2])
    }
}

pub trait Eigen<const N: usize> {
    fn eigen<D: DualNum<f64> + Copy>(matrix: SMatrix<D, N, N>) -> (D, SVector<D, N>);
}

impl Eigen<1> for Const<1> {
    fn eigen<D: DualNum<f64> + Copy>(matrix: SMatrix<D, 1, 1>) -> (D, SVector<D, 1>) {
        let [[l]] = matrix.data.0;
        (l, SVector::from([D::one()]))
    }
}

impl Eigen<2> for Const<2> {
    fn eigen<D: DualNum<f64> + Copy>(matrix: SMatrix<D, 2, 2>) -> (D, SVector<D, 2>) {
        let [[a, b], [_, c]] = matrix.data.0;
        let l = (a + c - ((a - c).powi(2) + b * b * 4.0).sqrt()) * 0.5;
        let u = SVector::from([D::one(), (l - a) / b]);
        let u = u / (u[0] * u[0] + u[1] * u[1]).sqrt();
        (l, u)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::eos::ideal_gas::test::joback;
    use crate::eos::pcsaft::test::pcsaft;
    use crate::eos::PcSaftPure;
    use crate::EquationOfStateAD;
    use approx::assert_relative_eq;
    use feos_core::{Contributions, EquationOfState, FeosResult, PhaseEquilibrium};
    use num_dual::{Dual, Dual64};
    use quantity::{BAR, JOULE, KELVIN, MOL, PASCAL};
    use std::sync::Arc;

    #[test]
    fn test_critical_point() -> FeosResult<()> {
        let (pcsaft, eos) = pcsaft()?;
        let mut params = pcsaft.params();
        let mut params_dual = pcsaft.params::<Dual64>();
        params_dual[0] = params_dual[0].derivative();
        let pcsaft = pcsaft.wrap().derivatives(params_dual);
        let cp = State::critical_point(&eos, None, None, Default::default())?;
        let state = StateAD::critical_point_pure(&pcsaft)?;
        let t = state.temperature.re();
        let rho = 1.0 / state.molar_volume.re();
        println!("{:.5} {:.5}", t, rho);
        println!("{:.5} {:.5}", cp.temperature, cp.density);
        assert_relative_eq!(t, cp.temperature, max_relative = 1e-10);
        assert_relative_eq!(rho, cp.density, max_relative = 1e-10);

        let h = 1e-8;
        params[0] += h;
        let eos = PcSaftPure(params).wrap();
        let cp_h = State::critical_point(&eos.eos, None, None, Default::default())?;
        let dt = (cp_h.temperature - cp.temperature).to_reduced() / h;
        let drho = (cp_h.density - cp.density).to_reduced() / h;

        println!(
            "{:.5e} {:.5e}",
            state.reduced_temperature.eps,
            state.reduced_molar_volume.recip().eps
        );
        println!("{:.5e} {:.5e}", dt, drho);
        assert_relative_eq!(state.reduced_temperature.eps, dt, max_relative = 1e-6);
        assert_relative_eq!(
            state.reduced_molar_volume.recip().eps,
            drho,
            max_relative = 1e-6
        );
        Ok(())
    }

    #[test]
    fn test_state_tp() -> FeosResult<()> {
        let (pcsaft, eos) = pcsaft()?;
        let pcsaft = pcsaft.wrap();
        let p = BAR;
        let t = 300.0 * KELVIN;
        let state_feos = State::new_npt(
            &eos,
            t,
            p,
            &(arr1(&[1.0]) * MOL),
            DensityInitialization::Liquid,
        )?;
        let state = StateAD::new_tp(
            &pcsaft,
            t,
            p,
            SVector::from([1.0]),
            DensityInitialization::Liquid,
        )?;
        let density = 1.0 / state.molar_volume;
        println!("{:.5}", density);
        println!("{:.5}", state_feos.density);
        assert_relative_eq!(density, state_feos.density, max_relative = 1e-10);
        Ok(())
    }

    #[test]
    fn test_state_tp_derivative() -> FeosResult<()> {
        let (pcsaft, residual) = pcsaft()?;
        let p = BAR;
        let t = 300.0 * KELVIN;
        let state_feos = State::new_npt(
            &residual,
            t,
            p,
            &(arr1(&[1.0]) * MOL),
            DensityInitialization::Liquid,
        )?;
        let h = 1e2 * PASCAL;
        let state_h = State::new_npt(
            &residual,
            t,
            p + h,
            &(arr1(&[1.0]) * MOL),
            DensityInitialization::Liquid,
        )?;
        let params: [Dual64; 8] = pcsaft.params();
        let eos_ad = pcsaft.wrap().derivatives(params);
        let t = Temperature::from_reduced(Dual::from(t.to_reduced()));
        let p = Pressure::from_reduced(Dual::from(p.to_reduced()).derivative());
        let state = StateAD::new_tp(
            &eos_ad,
            t,
            p,
            SVector::from([Dual::from(1.0)]),
            DensityInitialization::Liquid,
        )?;
        let density = state.molar_volume.into_reduced().recip();
        println!("{:.5} {:.5}", density.re, density.eps);
        let density_h = ((state_h.density - state_feos.density) / h).into_reduced();
        println!("{:.5} {:.5}", state_feos.density.into_reduced(), density_h);
        assert_relative_eq!(density.eps, density_h, max_relative = 1e-6);
        Ok(())
    }

    #[test]
    fn test_state_ps() -> FeosResult<()> {
        let (joback, ideal_gas) = joback()?;
        let (pcsaft, residual) = pcsaft()?;
        let eos_ad = EquationOfStateAD::new([joback], pcsaft).wrap();
        let eos = Arc::new(EquationOfState::new(ideal_gas, residual));
        let vle = PhaseEquilibrium::pure(&eos, 250.0 * KELVIN, None, Default::default())?;
        let p = vle.liquid().pressure(Contributions::Total);
        let s = vle.liquid().molar_entropy(Contributions::Total);
        let t = vle.liquid().temperature;
        let state = StateAD::new_ps(
            &eos_ad,
            p,
            s,
            SVector::from([1.0]),
            DensityInitialization::Liquid,
            Some(t),
        )?;
        let density = 1.0 / state.molar_volume;
        println!("{:.5} {:.5}", state.temperature, density);
        println!(
            "{:.5} {:.5}",
            vle.liquid().temperature,
            vle.liquid().density,
        );
        assert_relative_eq!(
            state.temperature,
            vle.liquid().temperature,
            max_relative = 1e-10
        );
        assert_relative_eq!(density, vle.liquid().density, max_relative = 1e-10);
        Ok(())
    }

    #[test]
    fn test_state_ps_derivative() -> FeosResult<()> {
        let (joback, ideal_gas) = joback()?;
        let (pcsaft, residual) = pcsaft()?;
        let eos = Arc::new(EquationOfState::new(ideal_gas, residual));
        let vle = PhaseEquilibrium::pure(&eos, 250.0 * KELVIN, None, Default::default())?;
        let h = 1e-3 * JOULE / KELVIN / MOL;
        let state_h = State::new_nps(
            &eos,
            vle.liquid().pressure(Contributions::Total),
            vle.liquid().molar_entropy(Contributions::Total) + h,
            &vle.liquid().moles,
            DensityInitialization::Liquid,
            Some(vle.liquid().temperature),
        )?;
        let p = vle.liquid().pressure(Contributions::Total).to_reduced();
        let s = vle
            .liquid()
            .molar_entropy(Contributions::Total)
            .to_reduced();
        let t = vle.liquid().temperature;
        let eos_ad = EquationOfStateAD::new([joback], pcsaft)
            .wrap()
            .derivatives(([joback.params()], pcsaft.params()));
        let p: Pressure<Dual64> = Pressure::from_reduced(Dual::from(p));
        let s = MolarEntropy::from_reduced(Dual::from(s).derivative());
        let state = StateAD::new_ps(
            &eos_ad,
            p,
            s,
            SVector::from([Dual::from(1.0)]),
            DensityInitialization::Liquid,
            Some(t),
        )?;
        println!(
            "{:.5e} {:.5e}",
            state.reduced_temperature.eps,
            state.reduced_molar_volume.recip().eps
        );
        println!(
            "{:.5e} {:.5e}",
            ((state_h.temperature - vle.liquid().temperature) / h).to_reduced(),
            ((state_h.density - vle.liquid().density) / h).to_reduced(),
        );
        assert_relative_eq!(
            state.reduced_temperature.eps,
            ((state_h.temperature - vle.liquid().temperature) / h).to_reduced(),
            max_relative = 1e-6
        );
        assert_relative_eq!(
            state.reduced_molar_volume.recip().eps,
            ((state_h.density - vle.liquid().density) / h).to_reduced(),
            max_relative = 1e-6
        );
        Ok(())
    }

    #[test]
    fn test_state_ph() -> FeosResult<()> {
        let (joback, ideal_gas) = joback()?;
        let (pcsaft, residual) = pcsaft()?;
        let eos_ad = EquationOfStateAD::new([joback], pcsaft).wrap();
        let eos = Arc::new(EquationOfState::new(ideal_gas, residual));
        let vle = PhaseEquilibrium::pure(&eos, 250.0 * KELVIN, None, Default::default())?;
        let p = vle.liquid().pressure(Contributions::Total);
        let h = vle.liquid().molar_enthalpy(Contributions::Total);
        let t = vle.liquid().temperature;
        let state = StateAD::new_ph(
            &eos_ad,
            p,
            h,
            SVector::from([1.0]),
            DensityInitialization::Liquid,
            Some(t),
        )?;
        let density = 1.0 / state.molar_volume;
        println!("{:.5} {:.5}", state.temperature, density);
        println!(
            "{:.5} {:.5}",
            vle.liquid().temperature,
            vle.liquid().density,
        );
        assert_relative_eq!(
            state.temperature,
            vle.liquid().temperature,
            max_relative = 1e-10
        );
        assert_relative_eq!(density, vle.liquid().density, max_relative = 1e-10);
        Ok(())
    }

    #[test]
    fn test_state_ph_derivative() -> FeosResult<()> {
        let (joback, ideal_gas) = joback()?;
        let (pcsaft, residual) = pcsaft()?;
        let eos = Arc::new(EquationOfState::new(ideal_gas, residual));
        let vle = PhaseEquilibrium::pure(&eos, 250.0 * KELVIN, None, Default::default())?;
        let delta = 1e-1 * JOULE / MOL;
        let state_h = State::new_nph(
            &eos,
            vle.liquid().pressure(Contributions::Total),
            vle.liquid().molar_enthalpy(Contributions::Total) + delta,
            &vle.liquid().moles,
            DensityInitialization::Liquid,
            Some(vle.liquid().temperature),
        )?;
        let p = vle.liquid().pressure(Contributions::Total).to_reduced();
        let h = vle
            .liquid()
            .molar_enthalpy(Contributions::Total)
            .to_reduced();
        let t = vle.liquid().temperature;
        let eos_ad = EquationOfStateAD::new([joback], pcsaft)
            .wrap()
            .derivatives(([joback.params()], pcsaft.params()));
        let p: Pressure<Dual64> = Pressure::from_reduced(Dual::from(p));
        let h = MolarEnergy::from_reduced(Dual::from(h).derivative());
        let state = StateAD::new_ph(
            &eos_ad,
            p,
            h,
            SVector::from([Dual::from(1.0)]),
            DensityInitialization::Liquid,
            Some(t),
        )?;
        println!(
            "{:.5e} {:.5e}",
            state.reduced_temperature.eps,
            state.reduced_molar_volume.recip().eps
        );
        println!(
            "{:.5e} {:.5e}",
            ((state_h.temperature - vle.liquid().temperature) / delta).to_reduced(),
            ((state_h.density - vle.liquid().density) / delta).to_reduced(),
        );
        assert_relative_eq!(
            state.reduced_temperature.eps,
            ((state_h.temperature - vle.liquid().temperature) / delta).to_reduced(),
            max_relative = 1e-6
        );
        assert_relative_eq!(
            state.reduced_molar_volume.recip().eps,
            ((state_h.density - vle.liquid().density) / delta).to_reduced(),
            max_relative = 1e-6
        );
        Ok(())
    }

    #[test]
    fn test_heat_capacities() -> FeosResult<()> {
        let (joback, ideal_gas) = joback()?;
        let (pcsaft, residual) = pcsaft()?;
        let eos = Arc::new(EquationOfState::new(ideal_gas, residual));
        let eos_ad = EquationOfStateAD::new([joback], pcsaft)
            .wrap()
            .derivatives(([joback.params()], pcsaft.params()));

        let temperature = 300.0 * KELVIN;
        let pressure = 5.0 * BAR;

        let state = State::new_npt(
            &eos,
            temperature,
            pressure,
            &(arr1(&[1.0]) * MOL),
            DensityInitialization::None,
        )?;
        let state_ad = StateAD::new_tp(
            &eos_ad,
            temperature,
            pressure,
            SVector::from([1.0]),
            DensityInitialization::None,
        )?;

        let c_v = state.molar_isochoric_heat_capacity(Contributions::Total);
        let c_p = state.molar_isobaric_heat_capacity(Contributions::Total);
        let c_v_ad = state_ad.molar_isochoric_heat_capacity();
        let c_p_ad = state_ad.molar_isobaric_heat_capacity();

        println!("{c_v} {c_p}");
        println!("{c_v_ad} {c_p_ad}");

        assert_relative_eq!(c_v, c_v_ad, max_relative = 1e-10);
        assert_relative_eq!(c_p, c_p_ad, max_relative = 1e-10);

        Ok(())
    }
}
