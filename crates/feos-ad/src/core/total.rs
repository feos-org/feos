use super::{ParametersAD, ResidualHelmholtzEnergy};
use nalgebra::SVector;
use num_dual::{
    first_derivative, gradient, hessian, second_derivative, Dual, Dual2, Dual2Vec, DualNum, DualVec,
};

/// Implementation of an ideal gas Helmholtz energy contribution.
pub trait IdealGasAD: ParametersAD {
    /// The name of the model.
    const IDEAL_GAS: &str;

    /// The logarithmic cubed thermal de Broglie wavelength for the given temperature.
    fn ln_lambda3_dual<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
    ) -> D;
}

/// An equation of state consisting of a residual model and an ideal gas model.
pub struct EquationOfStateAD<I, R, const N: usize> {
    ideal_gas: [I; N],
    residual: R,
}

impl<I, R, const N: usize> EquationOfStateAD<I, R, N> {
    pub fn new(ideal_gas: [I; N], residual: R) -> Self {
        Self {
            ideal_gas,
            residual,
        }
    }
}

impl<I: ParametersAD, R: ParametersAD, const N: usize> ParametersAD for EquationOfStateAD<I, R, N> {
    type Parameters<D: DualNum<f64> + Copy> = ([I::Parameters<D>; N], R::Parameters<D>);

    fn params<D: DualNum<f64> + Copy>(&self) -> Self::Parameters<D> {
        (
            self.ideal_gas.each_ref().map(I::params),
            self.residual.params(),
        )
    }

    fn params_from_inner<D: DualNum<f64> + Copy, D2: DualNum<f64, Inner = D> + Copy>(
        (ideal_gas, residual): &([I::Parameters<D>; N], R::Parameters<D>),
    ) -> Self::Parameters<D2> {
        (
            ideal_gas.each_ref().map(|ig| I::params_from_inner(ig)),
            R::params_from_inner(residual),
        )
    }
}

impl<I: ParametersAD, R: ResidualHelmholtzEnergy<N>, const N: usize> ResidualHelmholtzEnergy<N>
    for EquationOfStateAD<I, R, N>
{
    const RESIDUAL: &str = R::RESIDUAL;

    fn compute_max_density(&self, molefracs: &SVector<f64, N>) -> f64 {
        self.residual.compute_max_density(molefracs)
    }

    fn residual_helmholtz_energy_density<D: DualNum<f64> + Copy>(
        (_, residual): &([I::Parameters<D>; N], R::Parameters<D>),
        temperature: D,
        partial_density: &SVector<D, N>,
    ) -> D {
        R::residual_helmholtz_energy_density(residual, temperature, partial_density)
    }
}

/// Methods of [EquationOfStateAD] extracted in a trait for genericness.
pub trait TotalHelmholtzEnergy<const N: usize>: ResidualHelmholtzEnergy<N> {
    const IDEAL_GAS: &str;

    fn ln_lambda3<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
    ) -> SVector<D, N>;

    fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        partial_density: &SVector<D, N>,
    ) -> D {
        let ln_lambda_3 = Self::ln_lambda3(parameters, temperature);
        let ig = partial_density
            .component_mul(
                &(partial_density.map(|d| d.ln()) + ln_lambda_3 - SVector::from([D::from(1.0); N])),
            )
            .sum()
            * temperature;
        Self::residual_helmholtz_energy_density(parameters, temperature, partial_density) + ig
    }

    fn molar_helmholtz_energy<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        molar_volume: D,
        molefracs: &SVector<D, N>,
    ) -> D {
        let partial_density = molefracs / molar_volume;
        Self::helmholtz_energy_density(parameters, temperature, &partial_density) * molar_volume
    }

    fn chemical_potential<D: DualNum<f64> + Copy>(
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
                Self::molar_helmholtz_energy(&params, temperature, molar_volume, &molefracs)
            },
            *molefracs,
        );
        mu
    }

    fn molar_entropy<D: DualNum<f64> + Copy>(
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
                Self::molar_helmholtz_energy(&params, temperature, molar_volume, &molefracs)
            },
            temperature,
        );
        -da_dt
    }

    fn molar_enthalpy<D: DualNum<f64> + Copy>(
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
                Self::molar_helmholtz_energy(&params, temperature, molar_volume, &molefracs)
            },
            SVector::from([temperature, molar_volume]),
        );
        let [da_dt, da_dv] = da.data.0[0];
        a - temperature * da_dt - molar_volume * da_dv
    }

    fn molar_isochoric_heat_capacity<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        molar_volume: D,
        molefracs: &SVector<D, N>,
    ) -> D {
        let params = Self::params_from_inner(parameters);
        let molar_volume = Dual2::from_re(molar_volume);
        let molefracs = molefracs.map(Dual2::from_re);
        let (_, _, d2a) = second_derivative(
            |temperature| {
                Self::molar_helmholtz_energy(&params, temperature, molar_volume, &molefracs)
            },
            temperature,
        );
        -temperature * d2a
    }

    fn molar_isobaric_heat_capacity<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        molar_volume: D,
        molefracs: &SVector<D, N>,
    ) -> D {
        let params = Self::params_from_inner(parameters);
        let molefracs = molefracs.map(Dual2Vec::from_re);
        let (_, _, d2a) = hessian(
            |x| {
                let [temperature, molar_volume] = x.data.0[0];
                Self::molar_helmholtz_energy(&params, temperature, molar_volume, &molefracs)
            },
            SVector::from([temperature, molar_volume]),
        );
        let [[a_tt, a_tv], [_, a_vv]] = d2a.data.0;
        temperature * (a_tv * a_tv / a_vv - a_tt)
    }

    fn pressure_entropy<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        molar_volume: D,
        molefracs: &SVector<D, N>,
    ) -> SVector<D, 2> {
        let params = Self::params_from_inner(parameters);
        let molefracs = molefracs.map(DualVec::from_re);
        gradient(
            |x| {
                let [molar_volume, temperature] = x.data.0[0];
                -Self::molar_helmholtz_energy(&params, temperature, molar_volume, &molefracs)
            },
            SVector::from([molar_volume, temperature]),
        )
        .1
    }

    fn pressure_enthalpy<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        molar_volume: D,
        molefracs: &SVector<D, N>,
    ) -> SVector<D, 2> {
        let params = Self::params_from_inner(parameters);
        let molefracs = molefracs.map(DualVec::from_re);
        let (a, da) = gradient(
            |x| {
                let [temperature, molar_volume] = x.data.0[0];
                Self::molar_helmholtz_energy(&params, temperature, molar_volume, &molefracs)
            },
            SVector::from([temperature, molar_volume]),
        );
        let [da_dt, da_dv] = da.data.0[0];
        let h = a - temperature * da_dt - molar_volume * da_dv;
        let p = -da_dv;
        SVector::from([p, h])
    }
}

impl<I: IdealGasAD, R: ResidualHelmholtzEnergy<N>, const N: usize> TotalHelmholtzEnergy<N>
    for EquationOfStateAD<I, R, N>
{
    const IDEAL_GAS: &str = I::IDEAL_GAS;

    fn ln_lambda3<D: DualNum<f64> + Copy>(
        (ideal_gas, _): &([I::Parameters<D>; N], R::Parameters<D>),
        temperature: D,
    ) -> SVector<D, N> {
        SVector::from(
            ideal_gas
                .each_ref()
                .map(|ig| I::ln_lambda3_dual(ig, temperature)),
        )
    }
}
