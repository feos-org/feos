use super::Gradient;
use crate::{FeosResult, Residual};
use nalgebra::{DefaultAllocator, Dim, allocator::Allocator};
#[cfg(feature = "ndarray")]
use ndarray::{Array1, Array2, ArrayView2};
use num_dual::DualNum;
use quantity::Quantity;

mod boiling_temperature;
mod bubble_point_pressure;
mod dew_point_pressure;
mod enthalpy_of_vaporization;
mod equilibrium_liquid_density;
mod liquid_density;
mod residual_isobaric_heat_capacity;
mod vapor_pressure;

pub use boiling_temperature::BoilingTemperature;
pub use bubble_point_pressure::BubblePointPressure;
pub use dew_point_pressure::DewPointPressure;
pub use enthalpy_of_vaporization::EnthalpyOfVaporization;
pub use equilibrium_liquid_density::EquilibriumLiquidDensity;
pub use liquid_density::LiquidDensity;
pub use residual_isobaric_heat_capacity::ResidualIsobaricHeatCapacity;
pub use vapor_pressure::VaporPressure;

/// Properties that can be rapidly evaluated in parallel together with
/// their gradients with respect to model parameters
pub trait PropertyAD<N: Dim>: for<'a> From<&'a [f64]>
where
    DefaultAllocator: Allocator<N>,
{
    type Unit;
    const REFERENCE: Quantity<f64, Self::Unit>;

    /// Evaluate the property for an arbitrary derivative.
    fn evaluate<E: Residual<N, D>, D: DualNum<f64, Inner = f64> + Copy>(
        &self,
        eos: &E,
    ) -> FeosResult<Quantity<D, Self::Unit>>;

    /// Evaluate the property for the first derivative w.r.t. model parameters.
    ///
    /// This can be overridden if there is a more performant implementation than
    /// the general implementation in `evaluate`.
    fn evaluate_gradient<E: Residual<N, Gradient<P>>, const P: usize>(
        &self,
        eos: &E,
    ) -> FeosResult<Quantity<Gradient<P>, Self::Unit>> {
        self.evaluate(eos)
    }

    /// Evaluate the property for all inputs in parallel.
    ///
    /// Return the property values and the success of the calculations.
    #[cfg(feature = "ndarray")]
    fn evaluate_parallel<E: Residual<N> + Sync>(
        eos: &E,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array1<bool>) {
        #[cfg(feature = "rayon")]
        let values = ndarray::Zip::from(input.rows()).par_map_collect(|inp| {
            let inp = inp.as_slice().expect("Input array is not contiguous!");
            Self::from(inp)
                .evaluate(eos)
                .map(|d| d.convert_into(Self::REFERENCE))
        });

        #[cfg(not(feature = "rayon"))]
        let values = ndarray::Zip::from(input.rows()).map_collect(|inp| {
            let inp = inp.as_slice().expect("Input array is not contiguous!");
            Self::from(inp)
                .evaluate(eos)
                .map(|d| d.convert_into(Self::REFERENCE))
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

    /// Evaluate the property and its gradients for all inputs in parallel.
    ///
    /// Return the property values, the gradients, and the success of the calculations.  
    #[cfg(feature = "ndarray")]
    fn evaluate_parallel_ad<E: super::ParametersAD<N>, const P: usize>(
        parameter_names: [String; P],
        parameters: ArrayView2<f64>,
        input: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        let parameter_names = parameter_names.each_ref().map(|s| s as &str);

        #[cfg(feature = "rayon")]
        let value_dual = ndarray::Zip::from(parameters.rows())
            .and(input.rows())
            .par_map_collect(|par, inp| {
                let par = par.as_slice().expect("Parameter array is not contiguous!");
                let inp = inp.as_slice().expect("Input array is not contiguous!");
                let eos = E::seed_derivatives(par, parameter_names);
                Self::from(inp)
                    .evaluate_gradient(&eos)
                    .map(|d| d.convert_into(Self::REFERENCE))
            });

        #[cfg(not(feature = "rayon"))]
        let value_dual = ndarray::Zip::from(parameters.rows())
            .and(input.rows())
            .map_collect(|par, inp| {
                let par = par.as_slice().expect("Parameter array is not contiguous!");
                let inp = inp.as_slice().expect("Input array is not contiguous!");
                let eos = E::seed_derivatives(par, parameter_names);
                Self::from(inp)
                    .evaluate_gradient(&eos)
                    .map(|d| d.convert_into(Self::REFERENCE))
            });

        let n = parameters.nrows();
        let status = value_dual.iter().map(|p| p.is_ok()).collect();
        let mut value = Array1::from_elem(n, f64::NAN);
        let mut grad = Array2::zeros([n, P]);
        for (i, result) in value_dual.into_iter().enumerate() {
            if let Ok(p_dual) = result {
                value[i] = p_dual.re;
                let eps = p_dual
                    .eps
                    .unwrap_generic(nalgebra::Const::<P>, nalgebra::U1);
                for (g, &e) in grad.row_mut(i).iter_mut().zip(eps.data.0[0].iter()) {
                    *g = e;
                }
            }
        }
        (value, grad, status)
    }
}
