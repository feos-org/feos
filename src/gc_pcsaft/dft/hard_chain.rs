use super::GcPcSaftFunctionalParameters;
use crate::hard_sphere::HardSphereProperties;
use feos_core::EosError;
use feos_dft::{
    FunctionalContributionDual, WeightFunction, WeightFunctionInfo, WeightFunctionShape,
};
use ndarray::*;
use num_dual::DualNum;
use petgraph::visit::EdgeRef;
use std::fmt;
use std::sync::Arc;

#[derive(Clone)]
pub struct ChainFunctional {
    parameters: Arc<GcPcSaftFunctionalParameters>,
}

impl ChainFunctional {
    pub fn new(parameters: &Arc<GcPcSaftFunctionalParameters>) -> Self {
        Self {
            parameters: parameters.clone(),
        }
    }
}

impl<N: DualNum<f64> + Copy + ScalarOperand> FunctionalContributionDual<N> for ChainFunctional {
    fn weight_functions(&self, temperature: N) -> WeightFunctionInfo<N> {
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

    fn calculate_helmholtz_energy_density(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> Result<Array1<N>, EosError> {
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

impl fmt::Display for ChainFunctional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hard chain functional (GC)")
    }
}
