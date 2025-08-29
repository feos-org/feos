use super::GcPcSaftFunctionalParameters;
use crate::hard_sphere::HardSphereProperties;
use feos_core::FeosError;
use feos_dft::{FunctionalContribution, WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use ndarray::*;
use num_dual::DualNum;
use petgraph::visit::EdgeRef;

#[derive(Clone)]
pub struct ChainFunctional<'a> {
    parameters: &'a GcPcSaftFunctionalParameters,
}

impl<'a> ChainFunctional<'a> {
    pub fn new(parameters: &'a GcPcSaftFunctionalParameters) -> Self {
        Self { parameters }
    }
}

impl<'a> FunctionalContribution for ChainFunctional<'a> {
    fn name(&self) -> &'static str {
        "Hard chain functional (GC)"
    }

    fn weight_functions<N: DualNum<f64> + Copy>(
        &self,
        temperature: N,
    ) -> WeightFunctionInfo<N> {
        let p = &self.parameters;
        let d = p.hs_diameter(temperature);
        WeightFunctionInfo::new(p.component_index.clone(), true)
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
                    kernel_radius: d,
                    shape: WeightFunctionShape::Theta,
                },
                true,
            )
    }

    fn helmholtz_energy_density<N: DualNum<f64> + Copy>(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> Result<Array1<N>, FeosError> {
        let p = &self.parameters;
        // number of segments
        let segments = weighted_densities.shape()[0] - 2;

        // weighted densities
        let rho = weighted_densities.slice_axis(Axis(0), Slice::new(0, Some(segments as isize), 1));
        let zeta2 = weighted_densities.index_axis(Axis(0), segments);
        let zeta3 = weighted_densities.index_axis(Axis(0), segments + 1);

        // temperature dependent segment diameter
        let d = p.hs_diameter(temperature);

        // Helmholtz energy
        let frac_1mz3 = zeta3.mapv(|z3| (-z3 + 1.0).recip());
        let c = &zeta2 * &frac_1mz3 * &frac_1mz3;

        let mut phi = Array::zeros(zeta2.raw_dim());
        for i in p.bonds.node_indices() {
            let rho_i = rho.index_axis(Axis(0), i.index());
            let edges = p.bonds.edges(i);
            let y = edges
                .map(|e| {
                    let di = d[e.source().index()];
                    let dj = d[e.target().index()];
                    let cdij = &c * di * dj / (di + dj);
                    &frac_1mz3 + &cdij * 3.0 - &cdij * &cdij * (&zeta3 - 1.0) * 2.0
                })
                .reduce(|acc, y| acc * y);
            if let Some(y) = y {
                phi -= &(y.map(N::ln) * rho_i * 0.5);
            }
        }

        Ok(phi)
    }
}
