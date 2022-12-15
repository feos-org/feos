use super::PcSaftParameters;
use crate::hard_sphere::HardSphereProperties;
use feos_core::EosResult;
use feos_derive::FunctionalContribution;
use feos_dft::{
    FunctionalContribution, FunctionalContributionDual, PartialDerivativesDual, WeightFunction,
    WeightFunctionInfo, WeightFunctionShape,
};
use ndarray::*;
use num_dual::*;
use std::fmt;
use std::sync::Arc;

#[derive(FunctionalContribution)]
#[max_size(12)]
pub struct ChainFunctional {
    parameters: Arc<PcSaftParameters>,
}

impl ChainFunctional {
    pub fn new(parameters: Arc<PcSaftParameters>) -> Self {
        Self { parameters }
    }
}

impl<N: DualNum<f64> + ScalarOperand> FunctionalContributionDual<N> for ChainFunctional {
    fn weight_functions(&self, temperature: N) -> WeightFunctionInfo<N> {
        let p = &self.parameters;
        let d = p.hs_diameter(temperature);
        WeightFunctionInfo::new(p.component_index().into_owned(), true)
            .add(
                WeightFunction {
                    prefactor: p.m.mapv(|m| m.into()) / (&d * 8.0),
                    kernel_radius: d.clone(),
                    shape: WeightFunctionShape::Theta,
                },
                true,
            )
            .add(
                WeightFunction {
                    prefactor: p.m.mapv(|m| (m / 8.0).into()),
                    kernel_radius: d.clone(),
                    shape: WeightFunctionShape::Theta,
                },
                true,
            )
            .add(
                WeightFunction::new_scaled(d, WeightFunctionShape::Delta),
                false,
            )
    }

    fn calculate_helmholtz_energy_density(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> EosResult<Array1<N>> {
        let p = &self.parameters;
        // number of segments
        let n = (weighted_densities.shape()[0] - 2) / 2;

        // weighted densities
        let rho = weighted_densities.slice_axis(Axis(0), Slice::new(0, Some(n as isize), 1));
        // negative lambdas lead to nan, therefore the absolute value is used
        let lambda = weighted_densities
            .slice_axis(Axis(0), Slice::new(n as isize, Some(2 * n as isize), 1))
            .mapv(|l| if l.re() < 0.0 { -l } else { l } + N::from(f64::EPSILON));
        let zeta2 = weighted_densities.index_axis(Axis(0), 2 * n);
        let zeta3 = weighted_densities.index_axis(Axis(0), 2 * n + 1);

        // temperature dependent segment diameter
        let d = p.hs_diameter(temperature);

        let z3i = zeta3.mapv(|z3| (-z3 + 1.0).recip());
        let mut phi = Array::zeros(zeta2.raw_dim());
        for (i, (lambdai, rhoi)) in lambda.outer_iter().zip(rho.outer_iter()).enumerate() {
            // cavity correlation
            let z2d = zeta2.mapv(|z2| z2 * d[i]);
            let yi = &z2d * &z3i * &z3i * (z2d * &z3i * 0.5 + 1.5) + &z3i;

            // Helmholtz energy density
            phi = phi - (yi * lambdai).mapv(|x| x.ln() - 1.0) * rhoi * (p.m[i] - 1.0);
        }
        Ok(phi)
    }
}

impl fmt::Display for ChainFunctional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hard chain functional")
    }
}
