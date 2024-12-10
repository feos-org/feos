use super::FcSaftHomoParameters;
use crate::hard_sphere::HardSphereProperties;
use feos_core::StateHD;
use ndarray::Array1;
use num_dual::*;
use std::fmt;
use std::sync::Arc;

pub struct HardChain {
    pub parameters: Arc<FcSaftHomoParameters>,
}

impl HardChain {
    pub fn new(parameters: &Arc<FcSaftHomoParameters>) -> Self {
        Self {
            parameters: parameters.clone(),
        }
    }

    pub fn helmholtz_energy<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> D {
        let p = &self.parameters;
        let [z2, z3] = p.zeta(state.temperature, &state.partial_density, [2, 3]);
        let frac_1mz3 = -(z3 - 1.0).recip();

        // chain contribution
        let c = z2 * frac_1mz3 * frac_1mz3;
        let g_hs =
            p.l.mapv(|l| frac_1mz3 + c * l * 1.5 - c.powi(2) * l.powi(2) * (z3 - 1.0) * 0.5);
        Array1::from_shape_fn(p.s.len(), |i| {
            state.partial_density[i] * (1.0 - p.s[i]) * g_hs[i].ln()
        })
        .sum()
            * state.volume
    }
}

impl fmt::Display for HardChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hard Chain")
    }
}
