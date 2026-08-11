use crate::pcsaft::parameters::PcSaftPars;
use feos_core::StateHD;
use nalgebra::DVector;
use num_dual::*;

pub struct HardChain;

impl HardChain {
    /// `diameter` and `zeta` are the hard-sphere diameters and packing
    /// fractions, shared with the other contributions.
    #[inline]
    pub fn helmholtz_energy_density<D: DualNum<Primitive = f64> + Copy>(
        &self,
        parameters: &PcSaftPars,
        state: &StateHD<D>,
        diameter: &DVector<D>,
        zeta: [D; 4],
    ) -> D {
        let [_, _, zeta2, zeta3] = zeta;
        let frac_1mz3 = -(zeta3 - 1.0).recip();
        let c = zeta2 * frac_1mz3 * frac_1mz3;
        let mut a = D::zero();
        for (i, &d) in diameter.iter().enumerate() {
            let g_hs = frac_1mz3 + d * c * 1.5 - d.powi(2) * c.powi(2) * (zeta3 - 1.0) * 0.5;
            a -= state.partial_density[i] * g_hs.ln() * (parameters.m[i] - 1.0);
        }
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hard_sphere::HardSphere;
    use crate::pcsaft::parameters::utils::{
        butane_parameters, propane_butane_parameters, propane_parameters,
    };
    use approx::assert_relative_eq;
    use nalgebra::dvector;

    fn a_hc(p: &PcSaftPars, s: &StateHD<f64>) -> f64 {
        let (_, zeta, d) = HardSphere.helmholtz_energy_density_and_properties(p, s);
        HardChain.helmholtz_energy_density(p, s, &d, zeta)
    }

    #[test]
    fn helmholtz_energy() {
        let t = 250.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, &dvector![n]);
        let a_rust = a_hc(&propane_parameters().params, &s) * v;
        assert_relative_eq!(a_rust, -0.12402626171926148, epsilon = 1e-10);
    }

    #[test]
    fn mix() {
        let t = 250.0;
        let v = 2.5;
        let n = 1.0;
        let s = StateHD::new(t, v, &dvector![n]);
        let a1 = a_hc(&propane_parameters().params, &s);
        let a2 = a_hc(&butane_parameters().params, &s);
        let s1m = StateHD::new(t, v, &dvector![n, 0.0]);
        let a1m = a_hc(&propane_butane_parameters().params, &s1m);
        let s2m = StateHD::new(t, v, &dvector![0.0, n]);
        let a2m = a_hc(&propane_butane_parameters().params, &s2m);
        assert_relative_eq!(a1, a1m, epsilon = 1e-14);
        assert_relative_eq!(a2, a2m, epsilon = 1e-14);
    }
}
