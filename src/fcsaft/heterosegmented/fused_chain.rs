use super::FcSaftParameters;
use crate::hard_sphere::{HardSphereProperties, MonomerShape};
use feos_core::StateHD;
use ndarray::*;
use num_dual::DualNum;
use petgraph::visit::EdgeRef;
use std::fmt;
use std::sync::Arc;

impl FcSaftParameters {
    pub fn geometry_coefficients<D: DualNum<f64> + Copy>(
        &self,
        diameter: &Array1<D>,
    ) -> [Array1<D>; 2] {
        let mut a = Array1::ones(diameter.len());
        let mut v = Array1::ones(diameter.len());
        for n in self.bonds.node_indices() {
            let d1 = diameter[n.index()];
            for e in self.bonds.edges(n) {
                let d2 = diameter[e.target().index()];
                let l12 = e.weight();
                let delta12 = (d1.powi(2) - d2.powi(2) + 4.0 * l12.powi(2)) / (8.0 * l12);
                a[n.index()] -= (-delta12 * 2.0 / d1 + 1.0) * 0.5;
                v[n.index()] -= (-delta12 * 3.0 / d1 + (delta12 / d1).powi(3) * 4.0 + 1.0) * 0.5;
            }
        }
        [a, v]
    }
}

impl HardSphereProperties for FcSaftParameters {
    fn monomer_shape<N: DualNum<f64> + Copy>(&self, temperature: N) -> MonomerShape<N> {
        let [a, v] = self.geometry_coefficients(&self.hs_diameter(temperature));
        MonomerShape::Heterosegmented(
            [Array1::ones(a.len()), a.clone(), a, v],
            &self.component_index,
        )
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D> {
        let ti = temperature.recip() * -3.0;
        Array1::from_shape_fn(self.sigma.len(), |i| {
            -((ti * self.epsilon_k[i]).exp() * 0.12 - 1.0) * self.sigma[i]
        })
    }
}

#[derive(Clone)]
pub struct FusedChain {
    pub parameters: Arc<FcSaftParameters>,
}

impl FusedChain {
    pub fn new(parameters: &Arc<FcSaftParameters>) -> Self {
        Self {
            parameters: parameters.clone(),
        }
    }

    pub fn helmholtz_energy<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> D {
        let diameter = self.parameters.hs_diameter(state.temperature);
        let [z2, z3] = self
            .parameters
            .zeta(state.temperature, &state.partial_density, [2, 3]);
        let frac_1mz3 = -(z3 - 1.0).recip();

        let mut helmholtz_energy_density = D::zero();
        for (j, i) in self.parameters.bonds.node_indices().enumerate() {
            let edges = self.parameters.bonds.edges(i);
            let y = edges
                .map(|e| {
                    let l = e.weight();
                    let s1 = diameter[e.source().index()];
                    let s2 = diameter[e.target().index()];
                    let b = -((s1 - s2).powi(2) - 4.0 * l.powi(2)) / (4.0 * l);
                    let z2l = z2 * b;
                    z2l * frac_1mz3 * frac_1mz3 * (z2l * frac_1mz3 * 0.5 + 1.5) + frac_1mz3
                })
                .reduce(|acc, y| acc * y);
            if let Some(y) = y {
                helmholtz_energy_density -=
                    y.ln() * state.partial_density[self.parameters.component_index[j]] * 0.5;
            }
        }
        helmholtz_energy_density * state.volume
    }
}

impl fmt::Display for FusedChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fused-sphere chain")
    }
}
