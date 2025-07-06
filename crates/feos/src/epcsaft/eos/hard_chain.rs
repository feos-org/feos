use crate::epcsaft::parameters::ElectrolytePcSaftPars;
use crate::hard_sphere::HardSphereProperties;
use feos_core::StateHD;
use ndarray::Array;
use num_dual::*;

pub struct HardChain;

impl HardChain {
    #[inline]
    pub fn helmholtz_energy<D: DualNum<f64> + Copy>(
        &self,
        parameters: &ElectrolytePcSaftPars,
        state: &StateHD<D>,
    ) -> D {
        let p = parameters;
        let d = p.hs_diameter(state.temperature);
        let [zeta2, zeta3] = p.zeta(state.temperature, &state.partial_density, [2, 3]);
        let frac_1mz3 = -(zeta3 - 1.0).recip();
        let c = zeta2 * frac_1mz3 * frac_1mz3;
        let g_hs =
            d.mapv(|d| frac_1mz3 + d * c * 1.5 - d.powi(2) * c.powi(2) * (zeta3 - 1.0) * 0.5);
        Array::from_shape_fn(p.m.len(), |i| {
            state.partial_density[i] * (1.0 - p.m[i]) * g_hs[i].ln()
        })
        .sum()
            * state.volume
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epcsaft::parameters::utils::{
        butane_parameters, propane_butane_parameters, propane_parameters,
    };
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn helmholtz_energy() {
        let p = ElectrolytePcSaftPars::new(&propane_parameters()).unwrap();
        let t = 250.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a_rust = HardChain.helmholtz_energy(&p, &s);
        assert_relative_eq!(a_rust, -0.12402626171926148, epsilon = 1e-10);
    }

    #[test]
    fn mix() {
        let p1 = ElectrolytePcSaftPars::new(&propane_parameters()).unwrap();
        let p2 = ElectrolytePcSaftPars::new(&butane_parameters()).unwrap();
        let p12 = ElectrolytePcSaftPars::new(&propane_butane_parameters()).unwrap();
        let t = 250.0;
        let v = 2.5e28;
        let n = 1.0;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a1 = HardChain.helmholtz_energy(&p1, &s);
        let a2 = HardChain.helmholtz_energy(&p2, &s);
        let s1m = StateHD::new(t, v, arr1(&[n, 0.0]));
        let a1m = HardChain.helmholtz_energy(&p12, &s1m);
        let s2m = StateHD::new(t, v, arr1(&[0.0, n]));
        let a2m = HardChain.helmholtz_energy(&p12, &s2m);
        assert_relative_eq!(a1, a1m, epsilon = 1e-14);
        assert_relative_eq!(a2, a2m, epsilon = 1e-14);
    }
}
