use super::PcSaftPars;
use feos_core::StateHD;
use ndarray::Array1;
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_3, PI};

pub const A0: [f64; 7] = [
    0.91056314451539,
    0.63612814494991,
    2.68613478913903,
    -26.5473624914884,
    97.7592087835073,
    -159.591540865600,
    91.2977740839123,
];
pub const A1: [f64; 7] = [
    -0.30840169182720,
    0.18605311591713,
    -2.50300472586548,
    21.4197936296668,
    -65.2558853303492,
    83.3186804808856,
    -33.7469229297323,
];
pub const A2: [f64; 7] = [
    -0.09061483509767,
    0.45278428063920,
    0.59627007280101,
    -1.72418291311787,
    -4.13021125311661,
    13.7766318697211,
    -8.67284703679646,
];
pub const B0: [f64; 7] = [
    0.72409469413165,
    2.23827918609380,
    -4.00258494846342,
    -21.00357681484648,
    26.8556413626615,
    206.5513384066188,
    -355.60235612207947,
];
pub const B1: [f64; 7] = [
    -0.57554980753450,
    0.69950955214436,
    3.89256733895307,
    -17.21547164777212,
    192.6722644652495,
    -161.8264616487648,
    -165.2076934555607,
];
pub const B2: [f64; 7] = [
    0.09768831158356,
    -0.25575749816100,
    -9.15585615297321,
    20.64207597439724,
    -38.80443005206285,
    93.6267740770146,
    -29.66690558514725,
];

pub struct Dispersion;

impl Dispersion {
    pub fn helmholtz_energy<D: DualNum<f64> + Copy>(
        &self,
        parameters: &PcSaftPars,
        state: &StateHD<D>,
        diameter: &Array1<D>,
    ) -> D {
        // auxiliary variables
        let n = parameters.m.len();
        let p = parameters;
        let rho = &state.partial_density;

        // temperature dependent segment radius
        let r = diameter * 0.5;

        // packing fraction
        let eta = (rho * &p.m * &r * &r * &r).sum() * 4.0 * FRAC_PI_3;

        // mean segment number
        let m = (&state.molefracs * &p.m).sum();

        // inverse temperature
        let t_inv = state.temperature.recip();

        // mixture densities, crosswise interactions of all segments on all chains
        let mut rho1mix = D::zero();
        let mut rho2mix = D::zero();
        for i in 0..n {
            for j in 0..n {
                let eps_ij = t_inv * p.epsilon_k_ij[(i, j)];
                let sigma_ij = p.sigma_ij[[i, j]].powi(3);
                rho1mix += rho[i] * rho[j] * p.m[i] * p.m[j] * eps_ij * sigma_ij;
                rho2mix += rho[i] * rho[j] * p.m[i] * p.m[j] * eps_ij * eps_ij * sigma_ij;
            }
        }

        // I1, I2 and C1
        let mut i1 = D::zero();
        let mut i2 = D::zero();
        let mut eta_i = D::one();
        let m1_m = (m - 1.0) / m;
        let m2_m = (m - 2.0) / m;
        for i in 0..=6 {
            i1 += (m1_m * (m2_m * A2[i] + A1[i]) + A0[i]) * eta_i;
            i2 += (m1_m * (m2_m * B2[i] + B1[i]) + B0[i]) * eta_i;
            eta_i *= eta;
        }
        let c1 = (m * (eta * 8.0 - eta.powi(2) * 2.0) / (eta - 1.0).powi(4)
            + (D::one() - m)
                * (eta * 20.0 - eta.powi(2) * 27.0 + eta.powi(3) * 12.0 - eta.powi(4) * 2.0)
                / ((eta - 1.0) * (eta - 2.0)).powi(2)
            + 1.0)
            .recip();

        // Helmholtz energy
        (-rho1mix * i1 * 2.0 - rho2mix * m * c1 * i2) * PI * state.volume
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hard_sphere::HardSphereProperties;
    use crate::pcsaft::parameters::utils::{
        butane_parameters, propane_butane_parameters, propane_parameters,
    };
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn helmholtz_energy() {
        let params = &propane_parameters().params;
        let t = 250.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, arr1(&[n]));
        let d = params.hs_diameter(t);
        let a_rust = Dispersion.helmholtz_energy(params, &s, &d);
        assert_relative_eq!(a_rust, -1.0622531100351962, epsilon = 1e-10);
    }

    #[test]
    fn mix() {
        let propane = &propane_parameters().params;
        let butane = &butane_parameters().params;
        let mix = &propane_butane_parameters().params;
        let t = 250.0;
        let v = 2.5e28;
        let n = 1.0;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a1 = Dispersion.helmholtz_energy(propane, &s, &propane.hs_diameter(t));
        let a2 = Dispersion.helmholtz_energy(butane, &s, &butane.hs_diameter(t));
        let s1m = StateHD::new(t, v, arr1(&[n, 0.0]));
        let a1m = Dispersion.helmholtz_energy(mix, &s1m, &mix.hs_diameter(t));
        let s2m = StateHD::new(t, v, arr1(&[0.0, n]));
        let a2m = Dispersion.helmholtz_energy(mix, &s2m, &mix.hs_diameter(t));
        assert_relative_eq!(a1, a1m, epsilon = 1e-14);
        assert_relative_eq!(a2, a2m, epsilon = 1e-14);
    }
}
