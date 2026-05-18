pub mod parameter_optimization;
pub mod properties;

use crate::{FeosResult, Residual};
use nalgebra::{Const, U1};
use ndarray::{Array1, Array2, ArrayView2, Zip};
use num_dual::{Derivative, DualNum, DualSVec};

pub(crate) type Gradient<const P: usize> = DualSVec<f64, f64, P>;

/// A model that can be evaluated with derivatives of its parameters.
pub trait ParametersAD<const N: usize>: Residual<Const<N>> {
    /// Build the model by requesting each parameter by name.
    ///
    /// Call `f(name, differentiable)` for each parameter. The order of calls
    /// defines the canonical parameter order.
    ///
    /// Set `differentiable` to `false` for fixed parameters.
    fn build<D: DualNum<f64, Inner = f64> + Copy>(
        f: impl FnMut(&'static str, bool) -> D,
    ) -> Self::Lifted<D>;

    /// Canonical parameter names in the order defined by [`build`](Self::build).
    fn parameter_names() -> Vec<&'static str> {
        let mut names = Vec::new();
        let _ = Self::build(|name, _| {
            names.push(name);
            0.0
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
            0.0
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

/// Evaluate a function and its gradients for a batch of parameters and inputs.
pub(crate) fn vectorize_ad<F, E: ParametersAD<N>, const N: usize, const P: usize>(
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

/// Evaluate a function for a batch of inputs using the same parameters for each sample.
pub(crate) fn vectorize<F, E>(eos: &E, input: ArrayView2<f64>, f: F) -> (Array1<f64>, Array1<bool>)
where
    E: Sync,
    F: Fn(&E, &[f64]) -> FeosResult<f64> + Sync,
{
    #[cfg(feature = "rayon")]
    let values = Zip::from(input.rows()).par_map_collect(|inp| {
        let inp = inp.as_slice().expect("Input array is not contiguous!");
        f(eos, inp)
    });

    #[cfg(not(feature = "rayon"))]
    let values = Zip::from(input.rows()).map_collect(|inp| {
        let inp = inp.as_slice().expect("Input array is not contiguous!");
        f(eos, inp)
    });

    let n = input.nrows();
    let status: Array1<bool> = values.iter().map(|r| r.is_ok()).collect();
    let mut value = Array1::from_elem(n, f64::NAN);
    for (i, result) in values.into_iter().enumerate() {
        if let Ok(v) = result {
            value[i] = v;
        }
    }
    (value, status)
}
