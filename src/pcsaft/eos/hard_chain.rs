use super::PcSaftParameters;
use crate::hard_sphere::HardSphereProperties;
use feos_core::{HelmholtzEnergyDual, StateHD};
use ndarray::Array;
use num_dual::*;
use std::fmt;
use std::sync::Arc;

pub struct HardChain {
    pub parameters: Arc<PcSaftParameters>,
}

impl<D: DualNum<f64> + Copy> HelmholtzEnergyDual<D> for HardChain {
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let p = &self.parameters;
        let d = self.parameters.hs_diameter(state.temperature);
        let [zeta2, zeta3] = p.zeta(state.temperature, &state.partial_density, [2, 3]);
        let frac_1mz3 = -(zeta3 - 1.0).recip();
        let c = zeta2 * frac_1mz3 * frac_1mz3;
        let g_hs =
            d.mapv(|d| frac_1mz3 + d * c * 1.5 - d.powi(2) * c.powi(2) * (zeta3 - 1.0) * 0.5);
        Array::from_shape_fn(self.parameters.m.len(), |i| {
            state.partial_density[i] * (1.0 - self.parameters.m[i]) * g_hs[i].ln()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcsaft::parameters::utils::{
        butane_parameters, propane_butane_parameters, propane_parameters,
    };
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn helmholtz_energy() {
        let hc = HardChain {
            parameters: propane_parameters(),
        };
        let t = 250.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a_rust = hc.helmholtz_energy(&s);
        assert_relative_eq!(a_rust, -0.12402626171926148, epsilon = 1e-10);
    }

    #[test]
    fn mix() {
        let c1 = HardChain {
            parameters: propane_parameters(),
        };
        let c2 = HardChain {
            parameters: butane_parameters(),
        };
        let c12 = HardChain {
            parameters: propane_butane_parameters(),
        };
        let t = 250.0;
        let v = 2.5e28;
        let n = 1.0;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a1 = c1.helmholtz_energy(&s);
        let a2 = c2.helmholtz_energy(&s);
        let s1m = StateHD::new(t, v, arr1(&[n, 0.0]));
        let a1m = c12.helmholtz_energy(&s1m);
        let s2m = StateHD::new(t, v, arr1(&[0.0, n]));
        let a2m = c12.helmholtz_energy(&s2m);
        assert_relative_eq!(a1, a1m, epsilon = 1e-14);
        assert_relative_eq!(a2, a2m, epsilon = 1e-14);
    }
}
