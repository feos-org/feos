use crate::hard_sphere::HardSphereProperties;
use crate::pets::parameters::PetsParameters;
use feos_core::{HelmholtzEnergyDual, StateHD};
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_3, PI};
use std::fmt;
use std::sync::Arc;

pub const A: [f64; 7] = [
    0.690603404,
    1.189317012,
    1.265604153,
    -24.34554201,
    93.67300357,
    -157.8773415,
    96.93736697,
];
pub const B: [f64; 7] = [
    0.664852128,
    2.10733079,
    -9.597951213,
    -17.37871193,
    30.17506222,
    209.3942909,
    -353.2743581,
];

#[derive(Debug, Clone)]
pub struct Dispersion {
    pub parameters: Arc<PetsParameters>,
}

impl<D: DualNum<f64> + Copy> HelmholtzEnergyDual<D> for Dispersion {
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        // auxiliary variables
        let n = self.parameters.sigma.len();
        let p = &self.parameters;
        let rho = &state.partial_density;

        // temperature dependent segment radius
        let r = p.hs_diameter(state.temperature) * 0.5;

        // packing fraction
        let eta = (rho * &r * &r * &r).sum() * 4.0 * FRAC_PI_3;

        // mixture densities, crosswise interactions of all segments on all chains
        let mut rho1mix = D::zero();
        let mut rho2mix = D::zero();
        for i in 0..n {
            for j in 0..n {
                let eps_ij = state.temperature.recip() * p.epsilon_k_ij[(i, j)];
                let sigma_ij = p.sigma_ij[[i, j]].powi(3);
                rho1mix += rho[i] * rho[j] * eps_ij * sigma_ij;
                rho2mix += rho[i] * rho[j] * eps_ij * eps_ij * sigma_ij;
            }
        }

        // I1, I2 and C1
        let mut i1 = D::zero();
        let mut i2 = D::zero();
        let mut eta_i = D::one();
        for i in 0..=6 {
            i1 += eta_i * A[i];
            i2 += eta_i * B[i];
            eta_i *= eta;
        }
        let c1 = ((eta * 8.0 - eta.powi(2) * 2.0) / (eta - 1.0).powi(4) + 1.0).recip();

        // Helmholtz energy
        (-rho1mix * i1 * 2.0 - rho2mix * c1 * i2) * PI * state.volume
    }
}

impl fmt::Display for Dispersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dispersion")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pets::parameters::utils::{
        argon_krypton_parameters, argon_parameters, krypton_parameters,
    };
    use approx::assert_relative_eq;
    use ndarray::arr1;

    // #[test]
    // fn helmholtz_energy() {
    //     let disp = Dispersion {
    //         parameters: argon_parameters(),
    //     };
    //     let t = 250.0;
    //     let v = 1000.0;
    //     let n = 1.0;
    //     let s = StateHD::new(t, v, arr1(&[n]));
    //     let a_rust = disp.helmholtz_energy(&s);
    //     assert_relative_eq!(a_rust, -1.0622531100351962, epsilon = 1e-10);
    // }

    #[test]
    fn mix() {
        let c1 = Dispersion {
            parameters: argon_parameters(),
        };
        let c2 = Dispersion {
            parameters: krypton_parameters(),
        };
        let c12 = Dispersion {
            parameters: argon_krypton_parameters(),
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
