use crate::uvtheory::parameters::UVParameters;
use feos_core::{HelmholtzEnergyDual, StateHD};
use lazy_static::lazy_static;
use ndarray::prelude::*;
use num_dual::DualNum;
use std::f64::consts::PI;
use std::fmt;
use std::rc::Rc;

lazy_static! {
    static ref WCA_CONSTANTS_ETA_B: Array2<f64> = arr2(&[
        [-0.883143456, -0.618156214],
        [-0.589914255, -3.015264636],
        [-2.152046477, 4.7038689542],
    ]);
    static ref WCA_CONSTANTS_ETA_A: Array2<f64> = arr2(&[
        [-0.888512176, 0.265207151, -0.851803291, -1.380304110],
        [-0.395548410, -0.626398537, -1.484059291, -3.041216688],
        [-2.905719617, -1.778798984, -1.556827067, -4.308085347],
        [0.429154871, 20.765871545, 9.341250676, -33.787719418],
    ]);
    pub static ref WCA_CONSTANTS_Q: Array2<f64> = arr2(&[
        [1.92840364363978, 4.43165896265079E-01, 0.0, 0.0],
        [
            5.20120816141761E-01,
            1.82526759234412E-01,
            1.10319989659929E-02,
            -7.97813995328348E-05
        ],
        [
            0.0,
            1.29885156087242E-02,
            6.41039871789327E-03,
            1.85866741090323E-05
        ],
    ]);
}

#[derive(Debug, Clone)]
pub struct HardSphereWCA {
    pub parameters: Rc<UVParameters>,
}

impl<D: DualNum<f64>> HelmholtzEnergyDual<D> for HardSphereWCA {
    /// Helmholtz energy for hard spheres, eq. 19 (check Volume)
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let d = diameter_wca(&self.parameters, state.temperature);
        let zeta = zeta(&state.partial_density, &d);
        let frac_1mz3 = -(zeta[3] - 1.0).recip();
        let zeta_23 = zeta_23(&state.molefracs, &d);
        state.volume * 6.0 / std::f64::consts::PI
            * (zeta[1] * zeta[2] * frac_1mz3 * 3.0
                + zeta[2].powi(2) * frac_1mz3.powi(2) * zeta_23
                + (zeta[2] * zeta_23.powi(2) - zeta[0]) * (zeta[3] * (-1.0)).ln_1p())
    }
}

impl fmt::Display for HardSphereWCA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hard Sphere")
    }
}

/// Dimensionless Hard-sphere diameter according to Weeks-Chandler-Andersen division.
pub fn diameter_wca<D: DualNum<f64>>(parameters: &UVParameters, temperature: D) -> Array1<D> {
    parameters
        .sigma
        .iter()
        .enumerate()
        .map(|(i, _b)| {
            let t = temperature / parameters.epsilon_k[i];
            let rm = (parameters.rep[i] / parameters.att[i])
                .powf(1.0 / (parameters.rep[i] - parameters.att[i]));
            let c = (parameters.rep[i] / 6.0)
                .powf(-parameters.rep[i] / (12.0 - 2.0 * parameters.rep[i]))
                - 1.0;

            (((t.sqrt() * c + 1.0).powf(2.0 / parameters.rep[i])).recip() * rm)
                * parameters.sigma[i]
        })
        .collect()
}

// Hard sphere diameter q for WCA division from eq. (S28)
// pub fn diameter_q_wca<D: DualNum<f64>>(parameters: &UVParameters, temperature: D) -> Array1<D> {
//     parameters
//         .sigma
//         .iter()
//         .enumerate()
//         .map(|(i, _c)| {
//             let nu = parameters.rep[i];
//             let n = parameters.att[i];
//             let t = temperature / parameters.epsilon_k[i];
//             let rs = (nu / n).powf(1.0 / (nu - n));
//             let coeffs = arr1(&[
//                 (nu * 2.0 * PI / n).sqrt(),
//                 WCA_CONSTANTS_Q[[0, 0]] + WCA_CONSTANTS_Q[[0, 1]] * (nu - 7.0),
//                 WCA_CONSTANTS_Q[[1, 0]]
//                     + WCA_CONSTANTS_Q[[1, 1]] * (nu - 7.0)
//                     + WCA_CONSTANTS_Q[[1, 2]] * (nu - 7.0).powi(2)
//                     + WCA_CONSTANTS_Q[[1, 3]] * (nu - 7.0).powi(3),
//                 WCA_CONSTANTS_Q[[2, 0]]
//                     + WCA_CONSTANTS_Q[[2, 1]] * (nu - 7.0)
//                     + WCA_CONSTANTS_Q[[2, 2]] * (nu - 7.0).powi(2)
//                     + WCA_CONSTANTS_Q[[2, 3]] * (nu - 7.0).powi(3),
//             ]);

//             (t.powf(2.0) * coeffs[3]
//                 + t.powf(3.0 / 2.0) * coeffs[2]
//                 + t * coeffs[1]
//                 + t.powf(1.0 / 2.0) * coeffs[0]
//                 + 1.0)
//                 .powf(-1.0 / (2.0 * nu))
//                 * rs
//                 * parameters.sigma[i]
//         })
//         .collect()
// }

//

pub fn dimensionless_diameter_q_wca<D: DualNum<f64>>(t_x: D, rep_x: D, att_x: D) -> D {
    let nu = rep_x;
    let n = att_x;
    let rs = (nu / n).powd((nu - n).recip());
    let coeffs = arr1(&[
        (nu * 2.0 * PI / n).sqrt(),
        (nu - 7.0) * WCA_CONSTANTS_Q[[0, 1]] + WCA_CONSTANTS_Q[[0, 0]],
        (nu - 7.0) * WCA_CONSTANTS_Q[[1, 1]]
            + (nu - 7.0).powi(2) * WCA_CONSTANTS_Q[[1, 2]]
            + (nu - 7.0).powi(3) * WCA_CONSTANTS_Q[[1, 3]]
            + WCA_CONSTANTS_Q[[1, 0]],
        (nu - 7.0) * WCA_CONSTANTS_Q[[2, 1]]
            + (nu - 7.0).powi(2) * WCA_CONSTANTS_Q[[2, 2]]
            + (nu - 7.0).powi(3) * WCA_CONSTANTS_Q[[2, 3]]
            + WCA_CONSTANTS_Q[[2, 0]],
    ]);

    (t_x.powf(2.0) * coeffs[3]
        + t_x.powf(3.0 / 2.0) * coeffs[2]
        + t_x * coeffs[1]
        + t_x.powf(1.0 / 2.0) * coeffs[0]
        + 1.0)
        .powd(-(nu * 2.0).recip())
        * rs
}

pub fn zeta<D: DualNum<f64>>(partial_density: &Array1<D>, diameter: &Array1<D>) -> [D; 4] {
    let mut zeta: [D; 4] = [D::zero(), D::zero(), D::zero(), D::zero()];
    for i in 0..partial_density.len() {
        for k in 0..4 {
            zeta[k] +=
                partial_density[i] * diameter[i].powi(k as i32) * (std::f64::consts::PI / 6.0);
        }
    }
    zeta
}

pub fn packing_fraction<D: DualNum<f64>>(partial_density: &Array1<D>, diameter: &Array1<D>) -> D {
    (0..partial_density.len()).fold(D::zero(), |acc, i| {
        acc + partial_density[i] * diameter[i].powi(3) * (std::f64::consts::PI / 6.0)
    })
}

pub fn zeta_23<D: DualNum<f64>>(molefracs: &Array1<D>, diameter: &Array1<D>) -> D {
    let mut zeta: [D; 2] = [D::zero(), D::zero()];
    for i in 0..molefracs.len() {
        for k in 0..2 {
            zeta[k] += molefracs[i] * diameter[i].powi((k + 2) as i32);
        }
    }
    zeta[0] / zeta[1]
}

#[inline]
pub fn dimensionless_length_scale<D: DualNum<f64>>(
    parameters: &UVParameters,
    temperature: D,
) -> Array1<D> {
    parameters
        .sigma
        .iter()
        .enumerate()
        .map(|(i, _c)| {
            let rs = (parameters.rep[i] / parameters.att[i])
                .powf(1.0 / (parameters.rep[i] - parameters.att[i]));
            -diameter_wca(parameters, temperature)[i] + rs * parameters.sigma[i]
            // parameters.sigma[i]
        })
        .collect()
}

#[inline]

pub fn packing_fraction_b<D: DualNum<f64>>(
    parameters: &UVParameters,
    eta: D,
    temperature: D,
) -> Array2<D> {
    let n = parameters.att.len();
    let dimensionless_lengths = dimensionless_length_scale(parameters, temperature);
    Array2::from_shape_fn((n, n), |(i, j)| {
        let tau = (dimensionless_lengths[i] + dimensionless_lengths[j])
            / parameters.sigma_ij[[i, j]]
            * 0.5; //dimensionless
        let tau2 = tau * tau;

        let c = arr1(&[
            tau * WCA_CONSTANTS_ETA_B[[0, 0]] + tau2 * WCA_CONSTANTS_ETA_B[[0, 1]],
            tau * WCA_CONSTANTS_ETA_B[[1, 0]] + tau2 * WCA_CONSTANTS_ETA_B[[1, 1]],
            tau * WCA_CONSTANTS_ETA_B[[2, 0]] + tau2 * WCA_CONSTANTS_ETA_B[[2, 1]],
        ]);
        eta + eta * c[0] + eta * eta * c[1] + eta.powi(3) * c[2]
    })
}

pub fn packing_fraction_a<D: DualNum<f64>>(
    parameters: &UVParameters,
    eta: D,
    temperature: D,
) -> Array2<D> {
    let dimensionless_lengths = dimensionless_length_scale(parameters, temperature);
    let n = parameters.att.len();
    Array2::from_shape_fn((n, n), |(i, j)| {
        let tau = (dimensionless_lengths[i] + dimensionless_lengths[j])
            / parameters.sigma_ij[[i, j]]
            * 0.5; //dimensionless

        let tau2 = tau * tau;
        let rep_inv = 1.0 / parameters.rep_ij[[i, j]];
        let c = arr1(&[
            tau * (WCA_CONSTANTS_ETA_A[[0, 0]] + WCA_CONSTANTS_ETA_A[[0, 1]] * rep_inv)
                + tau2 * (WCA_CONSTANTS_ETA_A[[0, 2]] + WCA_CONSTANTS_ETA_A[[0, 3]] * rep_inv),
            tau * (WCA_CONSTANTS_ETA_A[[1, 0]] + WCA_CONSTANTS_ETA_A[[1, 1]] * rep_inv)
                + tau2 * (WCA_CONSTANTS_ETA_A[[1, 2]] + WCA_CONSTANTS_ETA_A[[1, 3]] * rep_inv),
            tau * (WCA_CONSTANTS_ETA_A[[2, 0]] + WCA_CONSTANTS_ETA_A[[2, 1]] * rep_inv)
                + tau2 * (WCA_CONSTANTS_ETA_A[[2, 2]] + WCA_CONSTANTS_ETA_A[[2, 3]] * rep_inv),
            tau * (WCA_CONSTANTS_ETA_A[[3, 0]] + WCA_CONSTANTS_ETA_A[[3, 1]] * rep_inv)
                + tau2 * (WCA_CONSTANTS_ETA_A[[3, 2]] + WCA_CONSTANTS_ETA_A[[3, 3]] * rep_inv),
        ]);
        eta + eta * c[0] + eta * eta * c[1] + eta.powi(3) * c[2] + eta.powi(4) * c[3]
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::uvtheory::parameters::utils::{
        methane_parameters, test_parameters, test_parameters_mixture,
    };
    use approx::assert_relative_eq;
    #[test]
    fn test_wca_diameter() {
        let p = test_parameters(24.0, 6.0, 2.0, 1.0);
        let temp = 4.0;
        assert_eq!(diameter_wca(&p, temp)[0] / p.sigma[0], 0.9614325601663462);

        // Methane
        let p = methane_parameters(24.0, 6.0);

        assert_eq!(
            diameter_wca(&p, 4.0 * p.epsilon_k[0])[0] / p.sigma[0],
            0.9614325601663462
        );
        assert_eq!(
            diameter_wca(&p, 4.0 * p.epsilon_k[0])[0] / p.sigma[0],
            0.9614325601663462
        );

        assert_relative_eq!(
            dimensionless_diameter_q_wca(temp, p.rep[0], p.att[0]),
            0.9751576149023506,
            epsilon = 1e-8
        );

        assert_relative_eq!(
            dimensionless_length_scale(&p, 4.0 * p.epsilon_k[0])[0] / p.sigma[0],
            0.11862717872596029,
            epsilon = 1e-8
        );
    }

    #[test]
    fn test_hard_sphere_wca_mixture() {
        let moles = arr1(&[0.40000000000000002, 0.59999999999999998]);
        let reduced_temperature = 1.0;
        let reduced_density = 0.90000000000000002;
        let reduced_volume = (moles[0] + moles[1]) / reduced_density;

        let p = test_parameters_mixture(
            arr1(&[12.0, 12.0]),
            arr1(&[6.0, 6.0]),
            arr1(&[1.0, 1.0]),
            arr1(&[1.0, 0.5]),
        );

        let pt = HardSphereWCA {
            parameters: Rc::new(p),
        };
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());
        let a = pt.helmholtz_energy(&state) / (moles[0] + moles[1]);
        assert_relative_eq!(a, 3.8636904888563084, epsilon = 1e-10);
    }
}
