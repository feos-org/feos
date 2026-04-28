#[cfg(feature = "ndarray")]
pub mod parameter_optimization;

use crate::DensityInitialization::Liquid;
use crate::density_iteration::density_iteration;
use crate::{Composition, FeosResult, PhaseEquilibrium, ReferenceSystem, Residual, State};
use nalgebra::{Const, SVector, U1, U2};
#[cfg(feature = "ndarray")]
use ndarray::{Array1, Array2, ArrayView2, Zip};
use num_dual::{Derivative, DualNum, DualSVec, DualStruct, first_derivative, partial2};
use quantity::{Density, MolarEnergy, MolarEntropy, Pressure, Temperature};
#[cfg(feature = "ndarray")]
use quantity::{JOULE, KELVIN, KILO, METER, MOL, PASCAL};

type Gradient<const P: usize> = DualSVec<f64, f64, P>;

/// A model that can be evaluated with derivatives of its parameters.
pub trait ParametersAD<const N: usize>: Residual<Const<N>> {
    /// Build the model by requesting each parameter by name.
    ///
    /// Call `f(name, differentiable)` for each parameter. The order of calls
    /// defines the canonical parameter order. This is the single source of
    /// truth for parameter names, ordering, differentiability, and mapping
    /// to the internal model structure.
    ///
    /// Set `differentiable` to `false` for discrete or structurally fixed
    /// parameters (e.g. critical parameters in cubics or association site counts).
    fn build<D: DualNum<f64, Inner = f64> + Copy>(
        f: impl FnMut(&'static str, bool) -> D,
    ) -> Self::Lifted<D>;

    /// Canonical parameter names in the order defined by [`build`](Self::build).
    fn parameter_names() -> Vec<&'static str> {
        let mut names = Vec::new();
        let _ = Self::build(|name, _| {
            names.push(name);
            0.0_f64
        });
        names
    }

    /// Parameter names that can be differentiated, in canonical order.
    fn differentiable_parameters() -> Vec<&'static str> {
        let mut names = Vec::new();
        let _ = Self::build(|name, differentiable| {
            if differentiable {
                names.push(name);
            }
            0.0_f64
        });
        names
    }

    /// Construct the model with derivative seeds for the `P` named parameters.
    ///
    /// - `parameter_values`: all parameter values in the canonical order
    ///   defined by [`build`](Self::build).
    /// - `derivative_names`: names of the parameters to differentiate with
    ///   respect to. Gradient component `i` corresponds to
    ///   `derivative_names[i]`.
    ///
    fn seed_derivatives<const P: usize>(
        parameter_values: &[f64],
        derivative_names: [&str; P],
    ) -> Self::Lifted<Gradient<P>> {
        let mut idx = 0;
        Self::build(|name, _differentiable| {
            let i = idx;
            idx += 1;
            let mut d = Gradient::<P>::from(parameter_values[i]);
            if let Some(seed_idx) = derivative_names.iter().position(|&n| n == name) {
                d.eps = Derivative::derivative_generic(Const::<P>, U1, seed_idx);
            }
            d
        })
    }
}

/// Properties that can be evaluated with derivatives of model parameters.
pub trait PropertiesAD {
    fn vapor_pressure<const P: usize>(
        &self,
        temperature: Temperature,
    ) -> FeosResult<Pressure<Gradient<P>>>
    where
        Self: Residual<U1, Gradient<P>>,
    {
        let eos_f64 = self.re();
        let (_, [vapor_density, liquid_density]) =
            PhaseEquilibrium::pure_t(&eos_f64, temperature, None, Default::default())?;

        // implicit differentiation is implemented here instead of just calling pure_t with dual
        // numbers, because for the first derivative, we can avoid calculating density derivatives.
        let v1 = 1.0 / liquid_density.to_reduced();
        let v2 = 1.0 / vapor_density.to_reduced();
        let t = temperature.into_reduced();
        let (a1, a2) = {
            let t = Gradient::from(t);
            let v1 = Gradient::from(v1);
            let v2 = Gradient::from(v2);
            let x = Self::pure_molefracs();

            let a1 = self.residual_helmholtz_energy(t, v1, &x);
            let a2 = self.residual_helmholtz_energy(t, v2, &x);
            (a1, a2)
        };

        let p = -(a1 - a2 + t * (v2 / v1).ln()) / (v1 - v2);
        Ok(Pressure::from_reduced(p))
    }

    fn boiling_temperature<const P: usize>(
        &self,
        pressure: Pressure,
    ) -> FeosResult<Temperature<Gradient<P>>>
    where
        Self: Residual<U1, Gradient<P>>,
    {
        let eos_f64 = self.re();
        let (temperature, [vapor_density, liquid_density]) =
            PhaseEquilibrium::pure_p(&eos_f64, pressure, None, Default::default())?;

        // implicit differentiation is implemented here instead of just calling pure_t with dual
        // numbers, because for the first derivative, we can avoid calculating density derivatives.
        let t = temperature.into_reduced();
        let v1 = 1.0 / liquid_density.to_reduced();
        let v2 = 1.0 / vapor_density.to_reduced();
        let p = pressure.into_reduced();
        let t = Gradient::from(t);
        let t = t + {
            let v1 = Gradient::from(v1);
            let v2 = Gradient::from(v2);
            let p = Gradient::from(p);
            let x = Self::pure_molefracs();

            let residual_entropy = |v| {
                let (a, s) = first_derivative(
                    partial2(
                        |t, &v, x| self.lift().residual_helmholtz_energy(t, v, x),
                        &v,
                        &x,
                    ),
                    t,
                );
                (a, -s)
            };
            let (a1, s1) = residual_entropy(v1);
            let (a2, s2) = residual_entropy(v2);

            let ln_rho = (v1 / v2).ln();
            (p * (v2 - v1) + (a2 - a1 + t * ln_rho)) / (s2 - s1 - ln_rho)
        };
        Ok(Temperature::from_reduced(t))
    }

    fn equilibrium_liquid_density<const P: usize>(
        &self,
        temperature: Temperature,
    ) -> FeosResult<(Pressure<Gradient<P>>, Density<Gradient<P>>)>
    where
        Self: Residual<U1, Gradient<P>>,
    {
        let t = Temperature::from_inner(&temperature);
        PhaseEquilibrium::pure_t(self, t, None, Default::default()).map(|(p, [_, rho])| (p, rho))
    }

    fn liquid_density<const P: usize>(
        &self,
        temperature: Temperature,
        pressure: Pressure,
    ) -> FeosResult<Density<Gradient<P>>>
    where
        Self: Residual<U1, Gradient<P>>,
    {
        let x = Self::pure_molefracs();
        let t = Temperature::from_inner(&temperature);
        let p = Pressure::from_inner(&pressure);
        density_iteration(self, t, p, &x, Some(Liquid))
    }

    /// Residual isobaric molar heat capacity of the liquid phase at the given
    /// temperature and pressure.
    fn residual_isobaric_heat_capacity<const P: usize>(
        &self,
        temperature: Temperature,
        pressure: Pressure,
    ) -> FeosResult<MolarEntropy<Gradient<P>>>
    where
        Self: Residual<U1, Gradient<P>>,
    {
        let x = Self::pure_molefracs();
        let t = Temperature::from_inner(&temperature);
        let p = Pressure::from_inner(&pressure);
        let density = density_iteration(self, t, p, &x, Some(Liquid))?;
        let state = State::new_pure(self, t, density)?;
        Ok(state.residual_molar_isobaric_heat_capacity())
    }

    fn enthalpy_of_vaporization<const P: usize>(
        &self,
        temperature: Temperature,
    ) -> FeosResult<MolarEnergy<Gradient<P>>>
    where
        Self: Residual<U1, Gradient<P>>,
    {
        let t = Temperature::from_inner(&temperature);
        let (_, [vapor_density, liquid_density]) =
            PhaseEquilibrium::pure_t(self, t, None, Default::default())?;

        let v1 = liquid_density.into_reduced().recip();
        let v2 = vapor_density.into_reduced().recip();
        let x = Self::pure_molefracs();
        let t = t.into_reduced();
        let residual_entropy = |v| {
            let (_a, s) = first_derivative(
                partial2(
                    |t, &v, x| self.lift().residual_helmholtz_energy(t, v, x),
                    &v,
                    &x,
                ),
                t,
            );
            -s
        };

        let s1 = residual_entropy(v1);
        let s2 = residual_entropy(v2);

        let dh = t * ((v2 / v1).ln() + s2 - s1);
        Ok(MolarEnergy::from_reduced(dh))
    }

    #[cfg(feature = "ndarray")]
    fn vapor_pressure_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>)
    where
        Self: ParametersAD<1>,
    {
        vectorize::<_, Self, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &Self::Lifted<Gradient<P>>, inp| {
                eos.vapor_pressure(inp[0] * KELVIN)
                    .map(|p| p.convert_into(PASCAL))
            },
        )
    }

    #[cfg(feature = "ndarray")]
    fn boiling_temperature_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>)
    where
        Self: ParametersAD<1>,
    {
        vectorize::<_, Self, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &Self::Lifted<Gradient<P>>, inp| {
                eos.boiling_temperature(inp[0] * PASCAL)
                    .map(|p| p.convert_into(KELVIN))
            },
        )
    }

    #[cfg(feature = "ndarray")]
    fn liquid_density_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>)
    where
        Self: ParametersAD<1>,
    {
        vectorize::<_, Self, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &Self::Lifted<Gradient<P>>, inp| {
                eos.liquid_density(inp[0] * KELVIN, inp[1] * PASCAL)
                    .map(|d| d.convert_into(KILO * MOL / (METER * METER * METER)))
            },
        )
    }

    #[cfg(feature = "ndarray")]
    fn residual_isobaric_heat_capacity_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>)
    where
        Self: ParametersAD<1>,
    {
        vectorize::<_, Self, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &Self::Lifted<Gradient<P>>, inp| {
                eos.residual_isobaric_heat_capacity(inp[0] * KELVIN, inp[1] * PASCAL)
                    .map(|cp| cp.convert_into(JOULE / (MOL * KELVIN)))
            },
        )
    }

    #[cfg(feature = "ndarray")]
    fn enthalpy_of_vaporization_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>)
    where
        Self: ParametersAD<1>,
    {
        vectorize::<_, Self, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &Self::Lifted<Gradient<P>>, inp| {
                eos.enthalpy_of_vaporization(inp[0] * KELVIN)
                    .map(|dh| dh.convert_into(JOULE / MOL))
            },
        )
    }

    #[cfg(feature = "ndarray")]
    fn equilibrium_liquid_density_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>)
    where
        Self: ParametersAD<1>,
    {
        vectorize::<_, Self, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &Self::Lifted<Gradient<P>>, inp| {
                eos.equilibrium_liquid_density(inp[0] * KELVIN)
                    .map(|(_, d)| d.convert_into(KILO * MOL / (METER * METER * METER)))
            },
        )
    }

    fn bubble_point_pressure<const P: usize, X: Composition<f64, U2>>(
        &self,
        temperature: Temperature,
        pressure: Option<Pressure>,
        liquid_molefracs: X,
    ) -> FeosResult<Pressure<Gradient<P>>>
    where
        Self: Residual<U2, Gradient<P>>,
    {
        let eos_f64 = self.re();
        let (liquid_molefracs, _) = liquid_molefracs.into_molefracs(&eos_f64)?;
        let vle = PhaseEquilibrium::bubble_point(
            &eos_f64,
            temperature,
            liquid_molefracs,
            pressure,
            None,
            Default::default(),
        )?;

        // implicit differentiation is implemented here instead of just calling bubble_point with dual
        // numbers, because for the first derivative, we can avoid calculating density derivatives.
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

            let a_v = self.residual_helmholtz_energy(t, v_v, &y);
            let (p_l, mu_res_l, dp_l, dmu_l) = self.dmu_dv(t, v_l, &x);
            let vi_l = dmu_l / dp_l;
            let v_l = vi_l.dot(&y);
            let a_l = (mu_res_l - vi_l * p_l).dot(&y);
            (a_l, a_v, v_l, v_v)
        };
        let rho_l = vle.liquid().partial_density().to_reduced();
        let rho_l = [rho_l[0], rho_l[1]];
        let rho_v = vle.vapor().partial_density().to_reduced();
        let rho_v = [rho_v[0], rho_v[1]];
        let p = -(a_v - a_l
            + t * (y[0] * (rho_v[0] / rho_l[0]).ln() + y[1] * (rho_v[1] / rho_l[1]).ln() - 1.0))
            / (v_v - v_l);
        Ok(Pressure::from_reduced(p))
    }

    fn dew_point_pressure<const P: usize, X: Composition<f64, U2>>(
        &self,
        temperature: Temperature,
        pressure: Option<Pressure>,
        vapor_molefracs: X,
    ) -> FeosResult<Pressure<Gradient<P>>>
    where
        Self: Residual<U2, Gradient<P>>,
    {
        let eos_f64 = self.re();
        let (vapor_molefracs, _) = vapor_molefracs.into_molefracs(&eos_f64)?;
        let vle = PhaseEquilibrium::dew_point(
            &eos_f64,
            temperature,
            vapor_molefracs,
            pressure,
            None,
            Default::default(),
        )?;

        // implicit differentiation is implemented here instead of just calling dew_point with dual
        // numbers, because for the first derivative, we can avoid calculating density derivatives.
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

            let a_l = self.residual_helmholtz_energy(t, v_l, &x);
            let (p_v, mu_res_v, dp_v, dmu_v) = self.dmu_dv(t, v_v, &y);
            let vi_v = dmu_v / dp_v;
            let v_v = vi_v.dot(&x);
            let a_v = (mu_res_v - vi_v * p_v).dot(&x);
            (a_l, a_v, v_l, v_v)
        };
        let rho_l = vle.liquid().partial_density().to_reduced();
        let rho_l = [rho_l[0], rho_l[1]];
        let rho_v = vle.vapor().partial_density().to_reduced();
        let rho_v = [rho_v[0], rho_v[1]];
        let p = -(a_l - a_v
            + t * (x[0] * (rho_l[0] / rho_v[0]).ln() + x[1] * (rho_l[1] / rho_v[1]).ln() - 1.0))
            / (v_l - v_v);
        Ok(Pressure::from_reduced(p))
    }

    #[cfg(feature = "ndarray")]
    fn bubble_point_pressure_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>)
    where
        Self: ParametersAD<2>,
    {
        vectorize::<_, Self, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &Self::Lifted<Gradient<P>>, inp| {
                eos.bubble_point_pressure(inp[0] * KELVIN, Some(inp[2] * PASCAL), inp[1])
                    .map(|p| p.convert_into(PASCAL))
            },
        )
    }

    #[cfg(feature = "ndarray")]
    fn dew_point_pressure_parallel<const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>)
    where
        Self: ParametersAD<2>,
    {
        vectorize::<_, Self, _, _>(
            parameter_names,
            parameters,
            input,
            |eos: &Self::Lifted<Gradient<P>>, inp| {
                eos.dew_point_pressure(inp[0] * KELVIN, Some(inp[2] * PASCAL), inp[1])
                    .map(|p| p.convert_into(PASCAL))
            },
        )
    }
}

impl<T> PropertiesAD for T {}

#[cfg(feature = "ndarray")]
fn vectorize<F, E: ParametersAD<N>, const N: usize, const P: usize>(
    parameter_names: [String; P],
    parameters: ArrayView2<f64>,
    input: ArrayView2<f64>,
    f: F,
) -> (Array1<f64>, Array2<f64>, Array1<bool>)
where
    F: Fn(&E::Lifted<Gradient<P>>, &[f64]) -> FeosResult<Gradient<P>> + Sync,
{
    let parameter_names = parameter_names.each_ref().map(|s| s as &str);

    #[cfg(feature = "rayon")]
    let value_dual = Zip::from(parameters.rows())
        .and(input.rows())
        .par_map_collect(|par, inp| {
            let par = par.as_slice().expect("Parameter array is not contiguous!");
            let inp = inp.as_slice().expect("Input array is not contiguous!");
            let eos = E::seed_derivatives(par, parameter_names);
            f(&eos, inp)
        });

    #[cfg(not(feature = "rayon"))]
    let value_dual = Zip::from(parameters.rows())
        .and(input.rows())
        .map_collect(|par, inp| {
            let par = par.as_slice().expect("Parameter array is not contiguous!");
            let inp = inp.as_slice().expect("Input array is not contiguous!");
            let eos = E::seed_derivatives(par, parameter_names);
            f(&eos, inp)
        });

    let n = parameters.nrows();
    let status = value_dual.iter().map(|p| p.is_ok()).collect();
    let mut value = Array1::from_elem(n, f64::NAN);
    let mut grad = Array2::zeros([n, P]);
    for (i, result) in value_dual.into_iter().enumerate() {
        if let Ok(p_dual) = result {
            value[i] = p_dual.re;
            let eps = p_dual.eps.unwrap_generic(Const::<P>, U1);
            for (g, &e) in grad.row_mut(i).iter_mut().zip(eps.data.0[0].iter()) {
                *g = e;
            }
        }
    }
    (value, grad, status)
}
