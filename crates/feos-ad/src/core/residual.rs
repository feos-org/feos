use super::{FeOsWrapper, HelmholtzEnergyWrapper};
use nalgebra::{SMatrix, SVector};
use num_dual::{
    first_derivative, gradient, hessian, partial_hessian, second_derivative, Dual, Dual2, Dual2Vec,
    DualNum, DualVec, HyperDualVec,
};
use std::sync::Arc;

/// A model that can be evaluated with derivatives of its parameters.
pub trait ParametersAD: Send + Sync + Sized {
    /// The type of the structure that stores the parameters internally.
    type Parameters<D: DualNum<f64> + Copy>: Clone;

    /// Return the parameters in the given data type.
    fn params<D: DualNum<f64> + Copy>(&self) -> Self::Parameters<D>;

    /// Lift the parameters to the given type of dual number.
    fn params_from_inner<D: DualNum<f64> + Copy, D2: DualNum<f64, Inner = D> + Copy>(
        parameters: &Self::Parameters<D>,
    ) -> Self::Parameters<D2>;

    /// Wraps the model in the [HelmholtzEnergyWrapper] struct, so that it can be used
    /// as an argument to [StateAD](crate::StateAD) and [PhaseEquilibriumAD](crate::PhaseEquilibriumAD) constructors.
    fn wrap<const N: usize>(self) -> HelmholtzEnergyWrapper<Self, f64, N> {
        let parameters = self.params();
        HelmholtzEnergyWrapper {
            eos: Arc::new(FeOsWrapper(self)),
            parameters,
        }
    }
}

/// Implementation of a residual Helmholtz energy model.
pub trait ResidualHelmholtzEnergy<const N: usize>: ParametersAD {
    /// The name of the model.
    const RESIDUAL: &str;

    /// Return a density (in reduced units) that corresponds to a dense liquid phase.
    fn compute_max_density(&self, molefracs: &SVector<f64, N>) -> f64;

    fn residual_helmholtz_energy_density<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        partial_density: &SVector<D, N>,
    ) -> D;

    fn residual_molar_helmholtz_energy<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        molar_volume: D,
        molefracs: &SVector<D, N>,
    ) -> D {
        let partial_density = molefracs / molar_volume;
        Self::residual_helmholtz_energy_density(parameters, temperature, &partial_density)
            * molar_volume
    }

    fn residual_chemical_potential<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        molar_volume: D,
        molefracs: &SVector<D, N>,
    ) -> SVector<D, N> {
        let params = Self::params_from_inner(parameters);
        let temperature = DualVec::from_re(temperature);
        let molar_volume = DualVec::from_re(molar_volume);
        let (_, mu) = gradient(
            |molefracs| {
                Self::residual_molar_helmholtz_energy(
                    &params,
                    temperature,
                    molar_volume,
                    &molefracs,
                )
            },
            *molefracs,
        );
        mu
    }

    fn pressure<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        molar_volume: D,
        molefracs: &SVector<D, N>,
    ) -> D {
        let params = Self::params_from_inner(parameters);
        let t = Dual::from_re(temperature);
        let molefracs = molefracs.map(Dual::from_re);
        let (_, dadv) = first_derivative(
            |v| Self::residual_molar_helmholtz_energy(&params, t, v, &molefracs),
            molar_volume,
        );
        -dadv + temperature / molar_volume
    }

    fn residual_molar_entropy<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        molar_volume: D,
        molefracs: &SVector<D, N>,
    ) -> D {
        let params = Self::params_from_inner(parameters);
        let molar_volume = Dual::from_re(molar_volume);
        let molefracs = molefracs.map(Dual::from_re);
        let (_, da_dt) = first_derivative(
            |temperature| {
                Self::residual_molar_helmholtz_energy(
                    &params,
                    temperature,
                    molar_volume,
                    &molefracs,
                )
            },
            temperature,
        );
        -da_dt
    }

    fn residual_molar_enthalpy<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        molar_volume: D,
        molefracs: &SVector<D, N>,
    ) -> D {
        let params = Self::params_from_inner(parameters);
        let molefracs = molefracs.map(DualVec::from_re);
        let (a, da) = gradient(
            |x| {
                let [temperature, molar_volume] = x.data.0[0];
                Self::residual_molar_helmholtz_energy(
                    &params,
                    temperature,
                    molar_volume,
                    &molefracs,
                )
            },
            SVector::from([temperature, molar_volume]),
        );
        let [da_dt, da_dv] = da.data.0[0];
        a - temperature * da_dt - molar_volume * da_dv
    }

    fn dp_drho<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        molar_volume: D,
        molefracs: &SVector<D, N>,
    ) -> (D, D, D) {
        let params = Self::params_from_inner(parameters);
        let t = Dual2::from_re(temperature);
        let x = molefracs.map(Dual2::from_re);
        let (a, da, d2a) = second_derivative(
            |molar_volume| Self::residual_molar_helmholtz_energy(&params, t, molar_volume, &x),
            molar_volume,
        );
        let density = molar_volume.recip();
        (
            a * density,
            -da + temperature * density,
            molar_volume * molar_volume * d2a + temperature,
        )
    }

    /// calculates p, mu_res, dp_drho, dmu_drho
    fn dmu_drho<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        partial_density: &SVector<D, N>,
    ) -> (D, SVector<D, N>, SVector<D, N>, SMatrix<D, N, N>) {
        let params = Self::params_from_inner(parameters);
        let t = Dual2Vec::from_re(temperature);
        let (f_res, mu_res, dmu_res) = hessian(
            |rho| Self::residual_helmholtz_energy_density(&params, t, &rho),
            *partial_density,
        );
        let p = mu_res.dot(partial_density) - f_res + temperature * partial_density.sum();
        let dmu = dmu_res + SMatrix::from_diagonal(&partial_density.map(|d| temperature / d));
        let dp = dmu * partial_density;
        (p, mu_res, dp, dmu)
    }

    /// calculates p, mu_res, dp_dv, dmu_dv
    fn dmu_dv<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        molar_volume: D,
        molefracs: &SVector<D, N>,
    ) -> (D, SVector<D, N>, D, SVector<D, N>) {
        let params = Self::params_from_inner(parameters);
        let t = HyperDualVec::from_re(temperature);
        let (_, mu_res, a_res_v, mu_res_v) = partial_hessian(
            |x, v| Self::residual_molar_helmholtz_energy(&params, t, v[0], &x),
            *molefracs,
            SVector::from([molar_volume]),
        );
        let p = (-a_res_v)[0] + temperature / molar_volume;
        let mu_v = mu_res_v.map(|m| m - temperature / molar_volume);
        let p_v = mu_v.dot(molefracs) / molar_volume;
        (p, mu_res, p_v, mu_v)
    }
}
