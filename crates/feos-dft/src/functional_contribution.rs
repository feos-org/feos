use crate::weight_functions::WeightFunctionInfo;
use feos_core::{FeosResult, StateHD};
use ndarray::prelude::*;
use ndarray::{RemoveAxis, ScalarOperand};
use num_dual::*;
use num_traits::Zero;

/// Individual functional contribution that can be evaluated using generalized (hyper) dual numbers.
pub trait FunctionalContribution: Sync + Send {
    /// Return the name of the contribution.
    fn name(&self) -> &'static str;

    /// Return the weight functions required in this contribution.
    fn weight_functions<N: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        temperature: N,
    ) -> WeightFunctionInfo<N>;

    /// Overwrite this if the weight functions in pDGT are different than for DFT.
    fn weight_functions_pdgt<N: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        temperature: N,
    ) -> WeightFunctionInfo<N> {
        self.weight_functions(temperature)
    }

    /// Return the Helmholtz energy density for the given temperature and weighted densities.
    fn helmholtz_energy_density<N: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> FeosResult<Array1<N>>;

    fn helmholtz_energy<N: DualNum<f64> + Copy + ScalarOperand>(&self, state: &StateHD<N>) -> N {
        // calculate weight functions
        let weight_functions = self.weight_functions(state.temperature);

        // calculate segment density
        let density = weight_functions
            .component_index
            .mapv(|c| state.partial_density[c]);

        // calculate weighted density and Helmholtz energy
        let weight_constants = weight_functions.weight_constants(Zero::zero(), 0);
        let weighted_densities = weight_constants.dot(&density).insert_axis(Axis(1));
        self.helmholtz_energy_density(state.temperature, weighted_densities.view())
            .unwrap()[0]
            * state.volume
    }

    fn first_partial_derivatives<N: DualNum<f64> + Copy>(
        &self,
        temperature: N,
        weighted_densities: Array2<N>,
        mut helmholtz_energy_density: ArrayViewMut1<N>,
        mut first_partial_derivative: ArrayViewMut2<N>,
    ) -> FeosResult<()> {
        let mut wd = weighted_densities.mapv(Dual::from_re);
        let t = Dual::from_re(temperature);
        let mut phi = Array::zeros(weighted_densities.raw_dim().remove_axis(Axis(0)));

        for i in 0..wd.shape()[0] {
            wd.index_axis_mut(Axis(0), i)
                .map_inplace(|x| x.eps = N::one());
            phi = self.helmholtz_energy_density(t, wd.view())?;
            first_partial_derivative
                .index_axis_mut(Axis(0), i)
                .assign(&phi.mapv(|p| p.eps));
            wd.index_axis_mut(Axis(0), i)
                .map_inplace(|x| x.eps = N::zero());
        }
        helmholtz_energy_density.assign(&phi.mapv(|p| p.re));
        Ok(())
    }

    fn second_partial_derivatives(
        &self,
        temperature: f64,
        weighted_densities: ArrayView2<f64>,
        mut helmholtz_energy_density: ArrayViewMut1<f64>,
        mut first_partial_derivative: ArrayViewMut2<f64>,
        mut second_partial_derivative: ArrayViewMut3<f64>,
    ) -> FeosResult<()> {
        let mut wd = weighted_densities.mapv(HyperDual64::from);
        let t = HyperDual64::from(temperature);
        let mut phi = Array::zeros(weighted_densities.raw_dim().remove_axis(Axis(0)));

        for i in 0..wd.shape()[0] {
            wd.index_axis_mut(Axis(0), i).map_inplace(|x| x.eps1 = 1.0);
            for j in 0..=i {
                wd.index_axis_mut(Axis(0), j).map_inplace(|x| x.eps2 = 1.0);
                phi = self.helmholtz_energy_density(t, wd.view())?;
                let p = phi.mapv(|p| p.eps1eps2);
                second_partial_derivative
                    .index_axis_mut(Axis(0), i)
                    .index_axis_mut(Axis(0), j)
                    .assign(&p);
                if i != j {
                    second_partial_derivative
                        .index_axis_mut(Axis(0), j)
                        .index_axis_mut(Axis(0), i)
                        .assign(&p);
                }
                wd.index_axis_mut(Axis(0), j).map_inplace(|x| x.eps2 = 0.0);
            }
            first_partial_derivative
                .index_axis_mut(Axis(0), i)
                .assign(&phi.mapv(|p| p.eps1));
            wd.index_axis_mut(Axis(0), i).map_inplace(|x| x.eps1 = 0.0);
        }
        helmholtz_energy_density.assign(&phi.mapv(|p| p.re));
        Ok(())
    }
}
