use nalgebra::{Const, U1};
use num_dual::{Derivative, DualSVec};

mod properties;
pub use properties::{BinaryModel, PureModel};

type Gradient<const P: usize> = DualSVec<f64, f64, P>;

/// A model that can be evaluated with derivatives of its parameters.
pub trait ParametersAD<const P: usize>: for<'a> From<&'a [f64]> {
    /// Return a mutable reference to the parameter named by `index` from the parameter set.
    fn index_parameters_mut<'a>(&'a mut self, index: &str) -> &'a mut Gradient<P>;

    /// Return the parameters with the appropriate derivatives.
    fn named_derivatives(parameters: &[f64], parameter_names: [&str; P]) -> Self {
        let mut eos = Self::from(parameters);
        for (i, p) in parameter_names.into_iter().enumerate() {
            eos.index_parameters_mut(p).eps = Derivative::derivative_generic(Const::<P>, U1, i)
        }
        eos
    }
}
