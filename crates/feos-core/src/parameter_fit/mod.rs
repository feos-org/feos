use nalgebra::{Const, U1};
use num_dual::{Derivative, DualSVec};

mod properties;
pub use properties::{BinaryModel, PureModel};

use crate::Residual;

type Gradient<const P: usize> = DualSVec<f64, f64, P>;

/// A model that can be evaluated with derivatives of its parameters.
pub trait ParametersAD<const N: usize> {
    type Eos<const P: usize>: Residual<Const<N>, Gradient<P>> + for<'a> From<&'a [f64]>;

    /// Return a mutable reference to the parameter named by `index` from the parameter set.
    fn index_parameters_mut<'a, const P: usize>(
        eos: &'a mut Self::Eos<P>,
        index: &str,
    ) -> &'a mut Gradient<P>;

    /// Return the parameters with the appropriate derivatives.
    fn from_parameter_slice<const P: usize>(
        parameters: &[f64],
        parameter_names: [&str; P],
    ) -> Self::Eos<P> {
        let mut eos = Self::Eos::from(parameters);
        for (i, p) in parameter_names.into_iter().enumerate() {
            Self::index_parameters_mut(&mut eos, p).eps =
                Derivative::derivative_generic(Const::<P>, U1, i)
        }
        eos
    }

    /// Return the parameters with the appropriate derivatives.
    fn named_derivatives<const P: usize>(&self, parameter_names: [&str; P]) -> Self::Eos<P>
    where
        Self: Residual<Const<N>, f64, Lifted<Gradient<P>> = Self::Eos<P>>,
    {
        let mut eos = self.lift::<Gradient<P>>();
        for (i, p) in parameter_names.into_iter().enumerate() {
            Self::index_parameters_mut(&mut eos, p).eps =
                Derivative::derivative_generic(Const::<P>, U1, i)
        }
        eos
    }
}
