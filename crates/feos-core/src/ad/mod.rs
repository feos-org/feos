use crate::Residual;
use nalgebra::{Const, DefaultAllocator, Dim, U1, allocator::Allocator};
use num_dual::{Derivative, DualNum, DualSVec};

#[cfg(feature = "ndarray")]
mod dataset;
mod properties;
#[cfg(feature = "ndarray")]
pub use dataset::*;
pub use properties::*;

pub(crate) type Gradient<const P: usize> = DualSVec<f64, f64, P>;

/// A model that can be evaluated with derivatives of its parameters.
pub trait ParametersAD<N: Dim>: Residual<N>
where
    DefaultAllocator: Allocator<N>,
{
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
                d.eps =
                    Derivative::<_, _, Const<P>, _>::derivative_generic(Const::<P>, U1, seed_idx);
            }
            d
        })
    }
}
