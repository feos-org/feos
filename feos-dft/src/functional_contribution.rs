use crate::weight_functions::WeightFunctionInfo;
use feos_core::{EosResult, HelmholtzEnergyDual, StateHD};
use ndarray::prelude::*;
use ndarray::RemoveAxis;
use num_dual::*;
use num_traits::Zero;
use std::fmt::Display;

macro_rules! impl_helmholtz_energy {
    ($number:ty) => {
        impl HelmholtzEnergyDual<$number> for Box<dyn FunctionalContribution> {
            fn helmholtz_energy(&self, state: &StateHD<$number>) -> $number {
                // calculate weight functions
                let weight_functions = self.weight_functions(state.temperature);

                // calculate segment density
                let density = weight_functions
                    .component_index
                    .mapv(|c| state.partial_density[c]);

                // calculate weighted density and Helmholtz energy
                let weight_constants = weight_functions.weight_constants(Zero::zero(), 0);
                let weighted_densities = weight_constants.dot(&density).insert_axis(Axis(1));
                self.calculate_helmholtz_energy_density(
                    state.temperature,
                    weighted_densities.view(),
                )
                .unwrap()[0]
                    * state.volume
            }
        }
    };
}

impl_helmholtz_energy!(f64);
impl_helmholtz_energy!(Dual64);
impl_helmholtz_energy!(Dual<DualVec64<3>, f64>);
impl_helmholtz_energy!(HyperDual64);
impl_helmholtz_energy!(Dual3_64);
impl_helmholtz_energy!(HyperDual<Dual64, f64>);
impl_helmholtz_energy!(HyperDual<DualVec64<2>, f64>);
impl_helmholtz_energy!(HyperDual<DualVec64<3>, f64>);
impl_helmholtz_energy!(Dual3<Dual64, f64>);
impl_helmholtz_energy!(Dual3<DualVec64<2>, f64>);
impl_helmholtz_energy!(Dual3<DualVec64<3>, f64>);

/// Individual functional contribution that can
/// be evaluated using generalized (hyper) dual numbers.
///
/// This trait needs to be implemented generically or for
/// the specific types in the supertraits of [FunctionalContribution]
/// so that the implementor can be used as a functional
/// contribution in the Helmholtz energy functional.
pub trait FunctionalContributionDual<N: DualNum<f64>>: Display {
    /// Return the weight functions required in this contribution.
    fn weight_functions(&self, temperature: N) -> WeightFunctionInfo<N>;
    /// Overwrite this if the weight functions in pDGT are different than for DFT.
    fn weight_functions_pdgt(&self, temperature: N) -> WeightFunctionInfo<N> {
        self.weight_functions(temperature)
    }

    /// Return the Helmholtz energy density for the given temperature and weighted densities.
    fn calculate_helmholtz_energy_density(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> EosResult<Array1<N>>;
}

/// Object safe version of the [FunctionalContributionDual] trait.
///
/// The trait is implemented automatically for every struct that implements
/// the supertraits.
pub trait FunctionalContribution:
    FunctionalContributionDual<f64>
    + FunctionalContributionDual<Dual64>
    + FunctionalContributionDual<Dual<DualVec64<3>, f64>>
    + FunctionalContributionDual<HyperDual64>
    + FunctionalContributionDual<Dual3_64>
    + FunctionalContributionDual<HyperDual<Dual64, f64>>
    + FunctionalContributionDual<HyperDual<DualVec64<2>, f64>>
    + FunctionalContributionDual<HyperDual<DualVec64<3>, f64>>
    + FunctionalContributionDual<Dual3<Dual64, f64>>
    + FunctionalContributionDual<Dual3<DualVec64<2>, f64>>
    + FunctionalContributionDual<Dual3<DualVec64<3>, f64>>
    + Display
{
    fn first_partial_derivatives(
        &self,
        temperature: f64,
        weighted_densities: Array2<f64>,
        mut helmholtz_energy_density: ArrayViewMut1<f64>,
        mut first_partial_derivative: ArrayViewMut2<f64>,
    ) -> EosResult<()> {
        let mut wd = weighted_densities.mapv(Dual64::from);
        let t = Dual64::from(temperature);
        let mut phi = Array::zeros(weighted_densities.raw_dim().remove_axis(Axis(0)));

        for i in 0..wd.shape()[0] {
            wd.index_axis_mut(Axis(0), i)
                .map_inplace(|x| x.eps[0] = 1.0);
            phi = self.calculate_helmholtz_energy_density(t, wd.view())?;
            first_partial_derivative
                .index_axis_mut(Axis(0), i)
                .assign(&phi.mapv(|p| p.eps[0]));
            wd.index_axis_mut(Axis(0), i)
                .map_inplace(|x| x.eps[0] = 0.0);
        }
        helmholtz_energy_density.assign(&phi.mapv(|p| p.re));
        Ok(())
    }

    fn second_partial_derivatives(
        &self,
        temperature: f64,
        weighted_densities: Array2<f64>,
        mut helmholtz_energy_density: ArrayViewMut1<f64>,
        mut first_partial_derivative: ArrayViewMut2<f64>,
        mut second_partial_derivative: ArrayViewMut3<f64>,
    ) -> EosResult<()> {
        let mut wd = weighted_densities.mapv(HyperDual64::from);
        let t = HyperDual64::from(temperature);
        let mut phi = Array::zeros(weighted_densities.raw_dim().remove_axis(Axis(0)));

        for i in 0..wd.shape()[0] {
            wd.index_axis_mut(Axis(0), i)
                .map_inplace(|x| x.eps1[0] = 1.0);
            for j in 0..=i {
                wd.index_axis_mut(Axis(0), j)
                    .map_inplace(|x| x.eps2[0] = 1.0);
                phi = self.calculate_helmholtz_energy_density(t, wd.view())?;
                let p = phi.mapv(|p| p.eps1eps2[(0, 0)]);
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
                wd.index_axis_mut(Axis(0), j)
                    .map_inplace(|x| x.eps2[0] = 0.0);
            }
            first_partial_derivative
                .index_axis_mut(Axis(0), i)
                .assign(&phi.mapv(|p| p.eps1[0]));
            wd.index_axis_mut(Axis(0), i)
                .map_inplace(|x| x.eps1[0] = 0.0);
        }
        helmholtz_energy_density.assign(&phi.mapv(|p| p.re));
        Ok(())
    }
}

impl<T> FunctionalContribution for T where
    T: FunctionalContributionDual<f64>
        + FunctionalContributionDual<Dual64>
        + FunctionalContributionDual<Dual<DualVec64<3>, f64>>
        + FunctionalContributionDual<HyperDual64>
        + FunctionalContributionDual<Dual3_64>
        + FunctionalContributionDual<HyperDual<Dual64, f64>>
        + FunctionalContributionDual<HyperDual<DualVec64<2>, f64>>
        + FunctionalContributionDual<HyperDual<DualVec64<3>, f64>>
        + FunctionalContributionDual<Dual3<Dual64, f64>>
        + FunctionalContributionDual<Dual3<DualVec64<2>, f64>>
        + FunctionalContributionDual<Dual3<DualVec64<3>, f64>>
        + Display
{
}
