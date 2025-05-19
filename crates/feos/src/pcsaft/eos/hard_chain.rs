use super::PcSaftParameters;
use feos_core::StateHD;
use ndarray::{Array, Array1};
use num_dual::*;

pub struct HardChain {
    m: Array1<f64>,
}

impl HardChain {
    pub fn new(parameters: &PcSaftParameters) -> Self {
        let [m] = parameters.collate(|pr| [pr.m]);
        Self { m }
    }
    #[inline]
    pub fn helmholtz_energy<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
        diameter: &Array1<D>,
        zeta2: D,
        zeta3: D,
    ) -> D {
        let frac_1mz3 = -(zeta3 - 1.0).recip();
        let c = zeta2 * frac_1mz3 * frac_1mz3;
        let g_hs = diameter
            .mapv(|d| frac_1mz3 + d * c * 1.5 - d.powi(2) * c.powi(2) * (zeta3 - 1.0) * 0.5);
        Array::from_shape_fn(self.m.len(), |i| {
            state.partial_density[i] * (1.0 - self.m[i]) * g_hs[i].ln()
        })
        .sum()
            * state.volume
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hard_sphere::HardSphere;
    use crate::pcsaft::PcSaft;
    use crate::pcsaft::parameters::utils::{
        butane_parameters, propane_butane_parameters, propane_parameters,
    };
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn helmholtz_energy() {
        let parameters = propane_parameters();
        let hc = HardChain::new(&parameters);
        let eos = PcSaft::new(parameters);
        let t = 250.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, arr1(&[n]));
        let (_, [_, _, zeta2, zeta3], d) = HardSphere.helmholtz_energy_and_properties(&eos, &s);
        let a_rust = hc.helmholtz_energy(&s, &d, zeta2, zeta3);
        assert_relative_eq!(a_rust, -0.12402626171926148, epsilon = 1e-10);
    }

    #[test]
    fn mix() {
        let propane = propane_parameters();
        let butane = butane_parameters();
        let mix = propane_butane_parameters();
        let c1 = HardChain::new(&propane);
        let c2 = HardChain::new(&butane);
        let c12 = HardChain::new(&mix);
        let propane = PcSaft::new(propane);
        let butane = PcSaft::new(butane);
        let mix = PcSaft::new(mix);
        let t = 250.0;
        let v = 2.5e28;
        let n = 1.0;
        let s = StateHD::new(t, v, arr1(&[n]));
        let (_, [_, _, zeta2, zeta3], d) = HardSphere.helmholtz_energy_and_properties(&propane, &s);
        let a1 = c1.helmholtz_energy(&s, &d, zeta2, zeta3);
        let (_, [_, _, zeta2, zeta3], d) = HardSphere.helmholtz_energy_and_properties(&butane, &s);
        let a2 = c2.helmholtz_energy(&s, &d, zeta2, zeta3);
        let s1m = StateHD::new(t, v, arr1(&[n, 0.0]));
        let (_, [_, _, zeta2, zeta3], d) = HardSphere.helmholtz_energy_and_properties(&mix, &s1m);
        let a1m = c12.helmholtz_energy(&s1m, &d, zeta2, zeta3);
        let s2m = StateHD::new(t, v, arr1(&[0.0, n]));
        let (_, [_, _, zeta2, zeta3], d) = HardSphere.helmholtz_energy_and_properties(&mix, &s2m);
        let a2m = c12.helmholtz_energy(&s2m, &d, zeta2, zeta3);
        assert_relative_eq!(a1, a1m, epsilon = 1e-14);
        assert_relative_eq!(a2, a2m, epsilon = 1e-14);
    }
}
