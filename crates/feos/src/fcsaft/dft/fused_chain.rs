use crate::fcsaft::parameters::FcSaftPars;
use crate::hard_sphere::HardSphereProperties;
use feos_core::FeosResult;
use feos_dft::{FunctionalContribution, WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use ndarray::*;
use num_dual::DualNum;
use petgraph::visit::EdgeRef;

#[derive(Clone)]
pub struct FusedChainFunctional<'a> {
    parameters: &'a FcSaftPars,
}

impl<'a> FusedChainFunctional<'a> {
    pub fn new(parameters: &'a FcSaftPars) -> Self {
        Self { parameters }
    }
}

impl<'a> FunctionalContribution for FusedChainFunctional<'a> {
    fn name(&self) -> &'static str {
        "Fused hard chain functional"
    }

    fn weight_functions<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
    ) -> WeightFunctionInfo<N> {
        let p = &self.parameters;
        let d = p.hs_diameter(temperature);
        let [a, v] = p.fused_sphere_coefficients(&d);
        WeightFunctionInfo::new(p.component_index.clone(), true)
            .add(
                WeightFunction {
                    prefactor: a.component_div(&(&d * N::from(8.0))),
                    kernel_radius: d.clone(),
                    shape: WeightFunctionShape::Theta,
                },
                true,
            )
            .add(
                WeightFunction {
                    prefactor: v / N::from(8.0),
                    kernel_radius: d,
                    shape: WeightFunctionShape::Theta,
                },
                true,
            )
    }

    fn helmholtz_energy_density<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> FeosResult<Array1<N>> {
        let p = &self.parameters;

        // number of segments
        let n = weighted_densities.shape()[0] - 2;

        // temperature dependent segment diameter
        let diameter = p.hs_diameter(temperature);

        // weighted densities
        let rho = weighted_densities.slice_axis(Axis(0), Slice::new(0, Some(n as isize), 1));
        let zeta2 = weighted_densities.index_axis(Axis(0), n);
        let zeta3 = weighted_densities.index_axis(Axis(0), n + 1);

        let z3i = zeta3.mapv(|z3| (-z3 + 1.0).recip());

        let mut phi = Array1::zeros(zeta2.raw_dim());
        for i in p.bonds.node_indices() {
            let rho_i = rho.index_axis(Axis(0), i.index());
            let y = p
                .bonds
                .edges(i)
                .map(|e| {
                    let l = *e.weight();
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
