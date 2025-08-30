use crate::epcsaft::parameters::ElectrolytePcSaftPars;
use feos_core::StateHD;
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_6, PI};

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

pub const T_REF: f64 = 298.15;

impl ElectrolytePcSaftPars {
    pub fn k_ij_t<D: DualNum<f64>>(&self, temperature: D) -> DMatrix<f64> {
        let k_ij = &self.k_ij;
        let n = self.m.len();

        let mut k_ij_t = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                // Calculate k_ij(T)
                k_ij_t[(i, j)] = (temperature.re() - T_REF) * k_ij[(i, j)][1]
                    + (temperature.re() - T_REF).powi(2) * k_ij[(i, j)][2]
                    + (temperature.re() - T_REF).powi(3) * k_ij[(i, j)][3]
                    + k_ij[(i, j)][0];
            }
        }
        //println!("k_ij_t: {}", k_ij_t);
        k_ij_t
    }

    pub fn epsilon_k_ij_t<D: DualNum<f64>>(&self, temperature: D) -> DMatrix<f64> {
        let k_ij_t = self.k_ij_t(temperature);
        let n = self.m.len();

        let mut epsilon_k_ij_t = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                epsilon_k_ij_t[(i, j)] = (1.0 - k_ij_t[(i, j)]) * self.e_k_ij[(i, j)];
            }
        }
        epsilon_k_ij_t
    }
}

pub struct Dispersion;

impl Dispersion {
    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        parameters: &ElectrolytePcSaftPars,
        state: &StateHD<D>,
        diameter: &DVector<D>,
    ) -> D {
        // auxiliary variables
        let n = parameters.m.len();
        let p = parameters;
        let rho = &state.partial_density;

        // convert sigma_ij
        let sigma_ij_t = p.sigma_ij_t(state.temperature);

        // convert epsilon_k_ij
        let epsilon_k_ij_t = p.epsilon_k_ij_t(state.temperature);

        // packing fraction
        let eta = rho.dot(&diameter.zip_map(&p.m, |d, m| d.powi(3) * m)) * FRAC_PI_6;

        // mean segment number
        let m = state.molefracs.dot(&p.m.map(D::from));

        // mixture densities, crosswise interactions of all segments on all chains
        let mut rho1mix = D::zero();
        let mut rho2mix = D::zero();
        for i in 0..n {
            for j in 0..n {
                let eps_ij = state.temperature.recip() * epsilon_k_ij_t[(i, j)];
                let sigma_ij = sigma_ij_t[[i, j]].powi(3);
                rho1mix += rho[i] * rho[j] * p.m[i] * p.m[j] * eps_ij * sigma_ij;
                rho2mix += rho[i] * rho[j] * p.m[i] * p.m[j] * eps_ij * eps_ij * sigma_ij;
            }
        }

        // I1, I2 and C1
        let mut i1 = D::zero();
        let mut i2 = D::zero();
        let mut eta_i = D::one();
        for i in 0..=6 {
            i1 += ((m - 1.0) / m * ((m - 2.0) / m * A2[i] + A1[i]) + A0[i]) * eta_i;
            i2 += ((m - 1.0) / m * ((m - 2.0) / m * B2[i] + B1[i]) + B0[i]) * eta_i;
            eta_i *= eta;
        }
        let c1 = (m * (eta * 8.0 - eta.powi(2) * 2.0) / (eta - 1.0).powi(4)
            + (D::one() - m)
                * (eta * 20.0 - eta.powi(2) * 27.0 + eta.powi(3) * 12.0 - eta.powi(4) * 2.0)
                / ((eta - 1.0) * (eta - 2.0)).powi(2)
            + 1.0)
            .recip();

        // Helmholtz energy
        (-rho1mix * i1 * 2.0 - rho2mix * m * c1 * i2) * PI
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epcsaft::parameters::utils::{
        butane_parameters, propane_butane_parameters, propane_parameters,
    };
    use crate::hard_sphere::HardSphereProperties;
    use approx::assert_relative_eq;
    use nalgebra::dvector;

    #[test]
    fn helmholtz_energy() {
        let p = ElectrolytePcSaftPars::new(&propane_parameters()).unwrap();
        let t = 250.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, &dvector![n]);
        let d = p.hs_diameter(t);
        let a_rust = Dispersion.helmholtz_energy_density(&p, &s, &d) * v;
        assert_relative_eq!(a_rust, -1.0622531100351962, epsilon = 1e-10);
    }

    #[test]
    fn mix() {
        let p1 = ElectrolytePcSaftPars::new(&propane_parameters()).unwrap();
        let p2 = ElectrolytePcSaftPars::new(&butane_parameters()).unwrap();
        let p12 = ElectrolytePcSaftPars::new(&propane_butane_parameters()).unwrap();
        let t = 250.0;
        let v = 2.5e28;
        let n = 1.0;
        let s = StateHD::new(t, v, &dvector![n]);
        let a1 = Dispersion.helmholtz_energy_density(&p1, &s, &p1.hs_diameter(t));
        let a2 = Dispersion.helmholtz_energy_density(&p2, &s, &p2.hs_diameter(t));
        let s1m = StateHD::new(t, v, &dvector![n, 0.0]);
        let a1m = Dispersion.helmholtz_energy_density(&p12, &s1m, &p12.hs_diameter(t));
        let s2m = StateHD::new(t, v, &dvector![0.0, n]);
        let a2m = Dispersion.helmholtz_energy_density(&p12, &s2m, &p12.hs_diameter(t));
        assert_relative_eq!(a1, a1m, epsilon = 1e-14);
        assert_relative_eq!(a2, a2m, epsilon = 1e-14);
    }
}
