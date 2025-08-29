use crate::hard_sphere::HardSphereProperties;
use crate::pcsaft::parameters::PcSaftPars;
use feos_core::FeosError;
use feos_dft::{FunctionalContribution, WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use ndarray::*;
use num_dual::DualNum;

#[derive(Clone)]
pub(crate) struct ChainFunctional<'a> {
    parameters: &'a PcSaftPars,
}

impl<'a> ChainFunctional<'a> {
    pub(crate) fn new(parameters: &'a PcSaftPars) -> Self {
        Self { parameters }
    }
}

impl<'a> FunctionalContribution for ChainFunctional<'a> {
    fn name(&self) -> &'static str {
        "Hard chain functional"
    }

    fn weight_functions<N: DualNum<f64> + Copy>(&self, temperature: N) -> WeightFunctionInfo<N> {
        let p = &self.parameters;
        let d = p.hs_diameter(temperature);
        WeightFunctionInfo::new(p.component_index().into_owned(), true)
            .add(
                WeightFunction {
                    prefactor: p.m.map(|m| (m / 8.0).into()).component_div(&d),
                    kernel_radius: d.clone(),
                    shape: WeightFunctionShape::Theta,
                },
                true,
            )
            .add(
                WeightFunction {
                    prefactor: p.m.map(|m| (m / 8.0).into()),
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

    fn helmholtz_energy_density<N: DualNum<f64> + Copy>(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> Result<Array1<N>, FeosError> {
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
