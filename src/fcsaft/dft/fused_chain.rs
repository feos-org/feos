use super::FcSaftParameters;
use crate::hard_sphere::HardSphereProperties;
use feos_core::EosResult;
use feos_dft::{FunctionalContribution, WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use ndarray::*;
use num_dual::DualNum;
use petgraph::visit::EdgeRef;
use std::fmt;
use std::sync::Arc;

pub struct FusedHardChainFunctional {
    parameters: Arc<FcSaftParameters>,
}

impl FusedHardChainFunctional {
    pub fn new(parameters: &Arc<FcSaftParameters>) -> Self {
        Self {
            parameters: parameters.clone(),
        }
    }
}

impl FunctionalContribution for FusedHardChainFunctional {
    fn weight_functions<N: DualNum<f64> + Copy>(&self, temperature: N) -> WeightFunctionInfo<N> {
        let d = self.parameters.hs_diameter(temperature);
        let [a, v] = self.parameters.geometry_coefficients(&d);
        WeightFunctionInfo::new(self.parameters.component_index.clone(), true)
            .add(
                WeightFunction {
                    prefactor: a / (&d * 8.0),
                    kernel_radius: d.clone(),
                    shape: WeightFunctionShape::Theta,
                },
                true,
            )
            .add(
                WeightFunction {
                    prefactor: v / 8.0,
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
    ) -> EosResult<Array1<N>> {
        // number of segments
        let n = weighted_densities.shape()[0] - 2;

        // temperature dependent segment diameter
        let diameter = self.parameters.hs_diameter(temperature);

        // weighted densities
        let rho = weighted_densities.slice_axis(Axis(0), Slice::new(0, Some(n as isize), 1));
        let zeta2 = weighted_densities.index_axis(Axis(0), n);
        let zeta3 = weighted_densities.index_axis(Axis(0), n + 1);

        let z3i = zeta3.mapv(|z3| (-z3 + 1.0).recip());

        let mut phi = Array1::zeros(zeta2.raw_dim());
        for (rho_i, i) in rho
            .axis_iter(Axis(0))
            .zip(self.parameters.bonds.node_indices())
        {
            let edges = self.parameters.bonds.edges(i);
            let y = edges
                .map(|e| {
                    let l = e.weight();
                    let s1 = diameter[e.source().index()];
                    let s2 = diameter[e.target().index()];
                    let b = -((s1 - s2).powi(2) - 4.0 * l.powi(2)) / (4.0 * l);
                    let z2l = zeta2.mapv(|z2| z2 * b);
                    &z2l * &z3i * &z3i * (z2l * &z3i * 0.5 + 1.5) + &z3i
                })
                .reduce(|acc, y| acc * y);
            if let Some(y) = y {
                phi -= &(y.map(N::ln) * rho_i * 0.5);
            }
        }

        Ok(phi)
    }
}

impl fmt::Display for FusedHardChainFunctional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fused hard chain functional")
    }
}
