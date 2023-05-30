use crate::saftvrqmie::parameters::SaftVRQMieParameters;
use feos_core::{HelmholtzEnergyDual, StateHD};
use ndarray::{Array1, Array2};
use num_dual::DualNum;
use std::f64::consts::FRAC_PI_6;
use std::fmt;
use std::sync::Arc;

const LAM_COEFF: [[f64; 4]; 4] = [
    [0.81096, 1.7888, -37.578, 92.284],
    [1.0205, -19.341, 151.26, -463.50],
    [-1.9057, 22.845, -228.14, 973.92],
    [1.0885, -6.1962, 106.98, -677.64],
];

const PHI: [[f64; 7]; 6] = [
    [
        7.5365557, -37.60463, 71.745953, -46.83552, -2.467982, -0.50272, 8.0956883,
    ],
    [-359.44, 1825.6, -3168.0, 1884.2, -0.82376, -3.1935, 3.709],
    [1550.9, -5070.1, 6534.6, -3288.7, -2.7171, 2.0883, 0.0],
    [
        -1.19932, 9.063632, -17.9482, 11.34027, 20.52142, -56.6377, 40.53683,
    ],
    [
        -1911.28, 21390.175, -51320.7, 37064.54, 1103.742, -3264.61, 2556.181,
    ],
    [
        9236.9, -129430.0, 357230.0, -315530.0, 1390.2, -4518.2, 4241.6,
    ],
];

pub struct Alpha<D: DualNum<f64>> {
    alpha_ij: Array2<D>,
}

impl<D: DualNum<f64> + Copy> Alpha<D> {
    pub fn new(
        parameters: &SaftVRQMieParameters,
        sigma_eff_ij: &Array2<D>,
        epsilon_k_eff_ij: &Array2<D>,
        temperature: D,
    ) -> Self {
        let p = parameters;
        let nc = sigma_eff_ij.shape()[0];
        let mut alpha_ij: Array2<D> = Array2::zeros((nc, nc));

        for i in 0..nc {
            for j in 0..nc {
                let sigma_ratio = D::one() * p.sigma_ij[[i, j]] / sigma_eff_ij[[i, j]];
                let eps_ratio = D::one() * p.epsilon_k_ij[[i, j]] / epsilon_k_eff_ij[[i, j]];
                let la = p.lambda_a_ij[[i, j]];
                let lr = p.lambda_r_ij[[i, j]];
                let sigma_ratio_a = sigma_ratio.powf(la);
                let sigma_ratio_r = sigma_ratio.powf(lr);
                let dmt = p.quantum_d_ij(i, j, temperature) / p.sigma_ij[[i, j]].powi(2);
                let ma = sigma_ratio_a / (la - 3.0);
                let mr = sigma_ratio_r / (lr - 3.0);
                let q1a = sigma_ratio_a * sigma_ratio.powi(2) * la * (la - 1.0) / (la - 1.0);
                let q1r = sigma_ratio_r * sigma_ratio.powi(2) * lr * (lr - 1.0) / (lr - 1.0);
                alpha_ij[[i, j]] = (dmt * (q1a - q1r) + ma - mr) * p.c_ij[[i, j]] * eps_ratio;
            }
        }
        Self { alpha_ij }
    }

    fn f(&self, k: usize, i: usize, j: usize) -> D {
        let a_ij = self.alpha_ij[[i, j]];
        let phi = PHI[k];
        (a_ij * phi[1] + a_ij.powi(2) * phi[2] + a_ij.powi(3) * phi[3] + phi[0])
            / (a_ij * phi[4] + a_ij.powi(2) * phi[5] + a_ij.powi(3) * phi[6] + 1.0)
    }
}

pub struct Dispersion {
    pub parameters: Arc<SaftVRQMieParameters>,
}

impl<D: DualNum<f64> + Copy> HelmholtzEnergyDual<D> for Dispersion {
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        // auxiliary variables
        let n = self.parameters.m.len();
        let p = &self.parameters;
        let rho = &state.partial_density;
        // temperature dependent segment radius
        let s_eff_ij = Array2::from_shape_fn((n, n), |(i, j)| -> D {
            p.calc_sigma_eff_ij(i, j, state.temperature)
        });

        // temperature dependent segment radius
        let d_hs_ij = Array2::from_shape_fn((n, n), |(i, j)| -> D {
            p.hs_diameter_ij(i, j, state.temperature, s_eff_ij[[i, j]])
        });

        // temperature dependent well depth
        let epsilon_k_eff_ij = Array2::from_shape_fn((n, n), |(i, j)| -> D {
            p.calc_epsilon_k_eff_ij(i, j, state.temperature)
        });

        // temperature dependent well depth
        let dq_ij = Array2::from_shape_fn((n, n), |(i, j)| -> D {
            p.quantum_d_ij(i, j, state.temperature)
        });

        // segment fractions
        let mut x_s = Array1::from_shape_fn(n, |i| -> D { state.molefracs[i] * p.m[i] });
        let inv_x_s_sum = x_s.sum().recip();
        for i in 0..n {
            x_s[i] *= inv_x_s_sum;
        }
        // Segment density
        let mut rho_s = D::zero();
        for i in 0..n {
            rho_s += rho[i] * p.m[i];
        }
        // packing fractions
        let zeta = zeta_saft_vrq_mie(&p.m, &x_s, &d_hs_ij, rho_s);
        let zeta_bar = zeta_saft_vrq_mie(&p.m, &x_s, &s_eff_ij, rho_s);

        // alphas ....
        let alpha = Alpha::new(p, &s_eff_ij, &epsilon_k_eff_ij, state.temperature);

        let a1 = first_order_perturbation(p, &x_s, zeta, rho_s, &d_hs_ij, &s_eff_ij, &dq_ij);
        let a2 = second_order_perturbation(
            p, &alpha, &x_s, zeta, zeta_bar, rho_s, &d_hs_ij, &s_eff_ij, &dq_ij,
        );
        let a3 = third_order_perturbation(p, &alpha, &x_s, zeta_bar, &epsilon_k_eff_ij);

        let mut n_s = D::zero();
        for i in 0..n {
            n_s += state.moles[i] * p.m[i];
        }
        let inv_t = state.temperature.recip();
        n_s * (a1 * inv_t + a2 * inv_t.powi(2) + a3 * inv_t.powi(3))
    }
}

#[cfg(feature = "dft")]
pub fn dispersion_energy_density<D: DualNum<f64> + Copy>(
    parameters: &SaftVRQMieParameters,
    d_hs_ij: &Array2<D>,
    s_eff_ij: &Array2<D>,
    epsilon_k_eff_ij: &Array2<D>,
    dq_ij: &Array2<D>,
    alpha: &Alpha<D>,
    rho: &Array1<D>,
    temperature: D,
) -> D {
    // auxiliary variables
    let n = rho.len();
    let p = &parameters;

    // Segment density
    let mut rho_s = D::zero();
    for i in 0..n {
        rho_s += rho[i] * p.m[i];
    }
    // segment fractions
    let x_s = Array1::from_shape_fn(n, |i| -> D { rho[i] * p.m[i] * rho_s.recip() });

    // packing fractions
    let zeta = zeta_saft_vrq_mie(&p.m, &x_s, d_hs_ij, rho_s);
    let zeta_bar = zeta_saft_vrq_mie(&p.m, &x_s, s_eff_ij, rho_s);

    let a1 = first_order_perturbation(p, &x_s, zeta, rho_s, d_hs_ij, s_eff_ij, dq_ij);
    let a2 = second_order_perturbation(
        p, alpha, &x_s, zeta, zeta_bar, rho_s, d_hs_ij, s_eff_ij, dq_ij,
    );
    let a3 = third_order_perturbation(p, alpha, &x_s, zeta_bar, epsilon_k_eff_ij);

    let inv_t = temperature.recip();
    rho_s * (a1 * inv_t + a2 * inv_t.powi(2) + a3 * inv_t.powi(3))
}

fn zeta_saft_vrq_mie<D: DualNum<f64> + Copy>(
    m: &Array1<f64>,
    x_s: &Array1<D>,
    diameter: &Array2<D>,
    rho_s: D,
) -> D {
    let mut zeta = D::zero();
    for i in 0..m.len() {
        for j in 0..m.len() {
            zeta += x_s[i] * x_s[j] * diameter[[i, j]].powi(3);
        }
    }
    zeta * FRAC_PI_6 * rho_s
}

fn first_order_perturbation<D: DualNum<f64> + Copy>(
    parameters: &SaftVRQMieParameters,
    x_s: &Array1<D>,
    zeta: D,
    rho_s: D,
    d_hs_ij: &Array2<D>,
    s_eff_ij: &Array2<D>,
    dq_ij: &Array2<D>,
) -> D {
    let n = parameters.sigma.len();
    let mut a1 = D::zero();
    for i in 0..n {
        for j in 0..n {
            let x0 = d_hs_ij[[i, j]].recip() * parameters.sigma_ij[[i, j]];
            let x0_eff = s_eff_ij[[i, j]] / d_hs_ij[[i, j]];
            let dq_div_sigma_2 = dq_ij[[i, j]] / parameters.sigma_ij[[i, j]].powi(2);
            a1 += x_s[i]
                * x_s[j]
                * FRAC_PI_6
                * rho_s
                * d_hs_ij[[i, j]].powi(3)
                * first_order_perturbation_ij(
                    parameters.lambda_a_ij[[i, j]],
                    parameters.lambda_r_ij[[i, j]],
                    parameters.epsilon_k_ij[[i, j]],
                    zeta,
                    x0,
                    x0_eff,
                    parameters.c_ij[[i, j]],
                    dq_div_sigma_2,
                )
        }
    }
    a1
}

fn first_order_perturbation_ij<D: DualNum<f64> + Copy>(
    lambda_a: f64,
    lambda_r: f64,
    epsilon_k: f64,
    zeta: D,
    x0: D,
    x0_eff: D,
    c: f64,
    dq_div_sigma_2: D,
) -> D {
    let int_a = combine_sutherland_and_b(lambda_a, epsilon_k, zeta, x0, x0_eff);
    let int_r = combine_sutherland_and_b(lambda_r, epsilon_k, zeta, x0, x0_eff);
    // Quantum correction
    let int_qa = combine_sutherland_and_b(lambda_a + 2.0, epsilon_k, zeta, x0, x0_eff);
    let int_qr = combine_sutherland_and_b(lambda_r + 2.0, epsilon_k, zeta, x0, x0_eff);
    let qa1 = dq_div_sigma_2 * quantum_prefactor(lambda_a);
    let qr1 = dq_div_sigma_2 * quantum_prefactor(lambda_r);

    (int_qa * qa1 - int_qr * qr1 + int_a - int_r) * c
}

fn eta_eff<D: DualNum<f64> + Copy>(lambda: f64, zeta: D) -> D {
    let inv_lambda = Array1::from(vec![
        1.0,
        1.0 / lambda,
        1.0 / lambda.powi(2),
        1.0 / lambda.powi(3),
    ]);
    let c = Array1::from_shape_fn(4, |i| {
        inv_lambda[0] * LAM_COEFF[i][0]
            + inv_lambda[1] * LAM_COEFF[i][1]
            + inv_lambda[2] * LAM_COEFF[i][2]
            + inv_lambda[3] * LAM_COEFF[i][3]
    });
    zeta * (zeta * (zeta * (zeta * c[3] + c[2]) + c[1]) + c[0])
}

fn sutherland<D: DualNum<f64> + Copy>(lambda: f64, epsilon_k: f64, zeta: D, x0: D) -> D {
    let ef = eta_eff(lambda, zeta);
    (-ef * 0.5 + 1.0) * -12.0 * x0.powf(lambda) * epsilon_k / (lambda - 3.0) / (-ef + 1.0).powi(3)
}

fn ilambda<D: DualNum<f64>>(lambda: f64, x0: D) -> D {
    -(x0.powf(3.0 - lambda) - 1.0) / (lambda - 3.0)
}

fn jlambda<D: DualNum<f64>>(lambda: f64, x0: D) -> D {
    -(x0.powf(4.0 - lambda) * (lambda - 3.0) - x0.powf(3.0 - lambda) * (lambda - 4.0) - 1.0)
        / ((lambda - 3.0) * (lambda - 4.0))
}

/// Calculate equation 33 of Lafitte 2013
/// B is divided by the packing fraction
///
/// \author Morten Hammer, February 2018
fn b<D: DualNum<f64> + Copy>(lambda: f64, epsilon_k: f64, zeta: D, x0: D, x0_eff: D) -> D {
    let ilambda = ilambda(lambda, x0_eff);
    let jlambda = jlambda(lambda, x0_eff);
    let denum = (-zeta + 1.0).powi(3);
    x0.powf(lambda)
        * ((-zeta + 2.0) / denum * ilambda + -zeta * 9.0 * (zeta + 1.0) / denum * jlambda)
        * 6.0
        * epsilon_k
}

#[inline]
fn combine_sutherland_and_b<D: DualNum<f64> + Copy>(
    lambda: f64,
    epsilon_k: f64,
    zeta: D,
    x0: D,
    x0_eff: D,
) -> D {
    let int_as = sutherland(lambda, epsilon_k, zeta, x0);
    let int_b = b(lambda, epsilon_k, zeta, x0, x0_eff);
    int_as + int_b
}

fn second_order_perturbation<D: DualNum<f64> + Copy>(
    parameters: &SaftVRQMieParameters,
    alpha: &Alpha<D>,
    x_s: &Array1<D>,
    zeta: D,
    zeta_bar: D,
    rho_s: D,
    d_hs_ij: &Array2<D>,
    s_eff_ij: &Array2<D>,
    dq_ij: &Array2<D>,
) -> D {
    let n = parameters.sigma.len();
    let mut a2 = D::zero();

    // Calculate isothermal hard sphere compressibillity factor
    let k = (-zeta + 1.0).powi(4)
        / (zeta * 4.0 + zeta.powi(2) * 4.0 - zeta.powi(3) * 4.0 + zeta.powi(4) + 1.0);

    for i in 0..n {
        for j in 0..n {
            let chi = alpha.f(0, i, j) * zeta_bar
                + alpha.f(1, i, j) * zeta_bar.powi(5)
                + alpha.f(2, i, j) * zeta_bar.powi(8);
            let x0 = d_hs_ij[[i, j]].recip() * parameters.sigma_ij[[i, j]];
            let x0_eff = s_eff_ij[[i, j]] / d_hs_ij[[i, j]];
            let dq_div_sigma_2 = dq_ij[[i, j]] / parameters.sigma_ij[[i, j]].powi(2);
            a2 += x_s[i]
                * x_s[j]
                * FRAC_PI_6
                * rho_s
                * d_hs_ij[[i, j]].powi(3)
                * (chi + 1.0)
                * second_order_perturbation_ij(
                    parameters.lambda_a_ij[[i, j]],
                    parameters.lambda_r_ij[[i, j]],
                    parameters.epsilon_k_ij[[i, j]],
                    zeta,
                    x0,
                    x0_eff,
                    parameters.c_ij[[i, j]],
                    dq_div_sigma_2,
                )
        }
    }
    a2 * k
}

#[inline]
fn quantum_prefactor(lambda: f64) -> f64 {
    lambda * (lambda - 1.0)
}

fn second_order_perturbation_ij<D: DualNum<f64> + Copy>(
    lambda_a: f64,
    lambda_r: f64,
    epsilon_k: f64,
    zeta: D,
    x0: D,
    x0_eff: D,
    c: f64,
    dq_div_sigma_2: D,
) -> D {
    let lambda_2r = 2.0 * lambda_r;
    let lambda_2a = 2.0 * lambda_a;
    let lambda_ar = lambda_a + lambda_r;
    // Quantum contributions
    let qa1 = dq_div_sigma_2 * quantum_prefactor(lambda_a);
    let qr1 = dq_div_sigma_2 * quantum_prefactor(lambda_r);

    let mut a2_ij = D::zero();
    let mut afac = D::one();
    let mut rfac = D::one();
    let mut arfac = -D::one() * 2.0;
    // Loop all contributions
    // 0: Mie contribution
    // 1,2: Quantum corrections
    for q in 0..3 {
        let int_a =
            combine_sutherland_and_b(lambda_2a + 2.0 * q as f64, epsilon_k, zeta, x0, x0_eff);
        let int_r =
            combine_sutherland_and_b(lambda_2r + 2.0 * q as f64, epsilon_k, zeta, x0, x0_eff);
        let int_ar =
            combine_sutherland_and_b(lambda_ar + 2.0 * q as f64, epsilon_k, zeta, x0, x0_eff);
        if q == 1 {
            rfac = qr1 * 2.0;
            afac = qa1 * 2.0;
            arfac = -(rfac + afac);
        } else if q == 2 {
            rfac = qr1 * qr1;
            afac = qa1 * qa1;
            arfac = -qa1 * qr1 * 2.0;
        }
        a2_ij += int_a * afac + int_ar * arfac + int_r * rfac;
    }
    a2_ij * 0.5 * epsilon_k * c.powi(2)
}

fn third_order_perturbation<D: DualNum<f64> + Copy>(
    parameters: &SaftVRQMieParameters,
    alpha: &Alpha<D>,
    x_s: &Array1<D>,
    zeta_bar: D,
    epsilon_k_eff_ij: &Array2<D>,
) -> D {
    let n = parameters.sigma.len();
    let mut a3 = D::zero();
    for i in 0..n {
        for j in 0..n {
            a3 += x_s[i]
                * x_s[j]
                * third_order_perturbation_ij(i, j, epsilon_k_eff_ij[[i, j]], alpha, zeta_bar)
        }
    }
    a3
}

fn third_order_perturbation_ij<D: DualNum<f64> + Copy>(
    i: usize,
    j: usize,
    epsilon_k_eff: D,
    alpha: &Alpha<D>,
    zeta_bar: D,
) -> D {
    -epsilon_k_eff.powi(3)
        * alpha.f(3, i, j)
        * zeta_bar
        * (alpha.f(4, i, j) * zeta_bar + alpha.f(5, i, j) * zeta_bar.powi(2)).exp()
}

impl fmt::Display for Dispersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dispersion")
    }
}

#[allow(clippy::excessive_precision)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::saftvrqmie::parameters::utils::h2_ne_fh1;
    use crate::saftvrqmie::parameters::utils::hydrogen_fh1;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_eta_eff() {
        let lambda_6 = 6.0;
        let lambda_12 = 12.0;
        let mut zeta = 0.45;
        let e_eff_6_045 = eta_eff(lambda_6, zeta);
        let e_eff_12_045 = eta_eff(lambda_12, zeta);
        zeta = 0.1;
        let e_eff_6_01 = eta_eff(lambda_6, zeta);
        let e_eff_12_01 = eta_eff(lambda_12, zeta);
        assert_relative_eq!(e_eff_6_045, 0.19401806958333320, epsilon = 1e-7);
        assert_relative_eq!(e_eff_12_045, 0.32193464807291666, epsilon = 1e-7);
        assert_relative_eq!(e_eff_6_01, 4.7840898518518506E-002, epsilon = 1e-7);
        assert_relative_eq!(e_eff_12_01, 7.6226364537037058E-002, epsilon = 1e-7);
    }

    #[test]
    fn test_alpha() {
        let temperature = 26.7060;
        let parameters = hydrogen_fh1();
        let n = 1;
        let s_eff_ij = Array2::from_shape_fn((n, n), |(i, j)| {
            parameters.calc_sigma_eff_ij(i, j, temperature)
        });
        let epsilon_k_eff_ij = Array2::from_shape_fn((n, n), |(i, j)| {
            parameters.calc_epsilon_k_eff_ij(i, j, temperature)
        });
        let alpha = Alpha::new(&parameters, &s_eff_ij, &epsilon_k_eff_ij, temperature);
        assert_relative_eq!(alpha.alpha_ij[[0, 0]], 1.0239374984636636, epsilon = 5e-8);
    }

    #[test]
    fn test_sutherland() {
        let x0 = 1.1;
        let zeta = 0.333;
        let lambda = 13.77;
        let eps_div_k = 13.88;
        let asa = sutherland(lambda, eps_div_k, zeta, x0);
        assert_relative_eq!(asa, -122.12017536923423, epsilon = 1e-12);
    }

    #[test]
    fn test_b() {
        let x0 = 1.1;
        let zeta = 0.333;
        let lambda = 13.77;
        let eps_div_k = 13.88;
        let ba = b(lambda, eps_div_k, zeta, x0, x0);
        assert_relative_eq!(ba, 93.436438943866293, epsilon = 1e-12);
    }

    #[test]
    fn test_quantum_d_ij() {
        let p = hydrogen_fh1();
        let temperature = 26.7060;
        let dq_ij = p.quantum_d_ij(0, 0, temperature);
        assert_relative_eq!(dq_ij, 7.5092605940987542e-2, epsilon = 5e-8);
    }

    #[test]
    fn test_first_order_perturbation_ij() {
        let p = hydrogen_fh1();
        let temperature = 26.7060;
        let zeta = 0.333;
        let dq_div_s2 = p.quantum_d_ij(0, 0, temperature) / p.sigma_ij[[0, 0]].powi(2);
        let s_eff = p.calc_sigma_eff_ij(0, 0, temperature);
        let d_hs = p.hs_diameter_ij(0, 0, temperature, s_eff);
        let x0 = d_hs.recip() * p.sigma_ij[[0, 0]];
        let x0_eff = s_eff / d_hs;

        let a1_ij = first_order_perturbation_ij(
            p.lambda_a_ij[[0, 0]],
            p.lambda_r_ij[[0, 0]],
            p.epsilon_k_ij[[0, 0]],
            zeta,
            x0,
            x0_eff,
            p.c_ij[[0, 0]],
            dq_div_s2,
        );
        let rel_err = (a1_ij + 332.00915966785539) / 332.00915966785539;
        assert_relative_eq!(rel_err, 0.0, epsilon = 1e-7);
    }

    #[test]
    fn test_second_order_perturbation_ij() {
        let p = hydrogen_fh1();
        let temperature = 26.7060;
        let zeta = 0.333;
        let dq_div_s2 = p.quantum_d_ij(0, 0, temperature) / p.sigma_ij[[0, 0]].powi(2);
        let s_eff = p.calc_sigma_eff_ij(0, 0, temperature);
        let d_hs = p.hs_diameter_ij(0, 0, temperature, s_eff);
        let x0 = d_hs.recip() * p.sigma_ij[[0, 0]];
        let x0_eff = s_eff / d_hs;

        let a2_ij = second_order_perturbation_ij(
            p.lambda_a_ij[[0, 0]],
            p.lambda_r_ij[[0, 0]],
            p.epsilon_k_ij[[0, 0]],
            zeta,
            x0,
            x0_eff,
            p.c_ij[[0, 0]],
            dq_div_s2,
        );
        let rel_err = (a2_ij + 1907.5055256805874) / 1907.5055256805874;
        assert_relative_eq!(rel_err, 0.0, epsilon = 1e-7);
    }

    #[test]
    fn test_third_order_perturbation_ij() {
        let p = hydrogen_fh1();
        let temperature = 26.7060;
        let zeta_bar = 0.333;
        let n = 1;
        let s_eff_ij =
            Array2::from_shape_fn((n, n), |(i, j)| p.calc_sigma_eff_ij(i, j, temperature));
        let epsilon_k_eff_ij =
            Array2::from_shape_fn((n, n), |(i, j)| p.calc_epsilon_k_eff_ij(i, j, temperature));
        let alpha = Alpha::new(&p, &s_eff_ij, &epsilon_k_eff_ij, temperature);

        let a3_ij = third_order_perturbation_ij(0, 0, epsilon_k_eff_ij[[0, 0]], &alpha, zeta_bar);

        let rel_err = (a3_ij + 25.807966819127916) / 25.807966819127916;
        assert_relative_eq!(rel_err, 0.0, epsilon = 5e-7);
    }

    #[test]
    fn test_zeta_saft_vrq_mie() {
        let p = hydrogen_fh1();
        let t = 26.7060;
        let v = 1.0e26;
        let n = 6.02214076e23;
        let state = StateHD::new(t, v, arr1(&[n]));
        let nc = 1;
        // temperature dependent sigma
        let s_eff_ij = Array2::from_shape_fn((nc, nc), |(i, j)| {
            p.calc_sigma_eff_ij(i, j, state.temperature)
        });
        // temperature dependent segment diameter
        let d_hs_ij = Array2::from_shape_fn((nc, nc), |(i, j)| {
            p.hs_diameter_ij(i, j, state.temperature, s_eff_ij[[i, j]])
        });

        // segment fractions
        let mut x_s = Array1::from_shape_fn(nc, |i| state.molefracs[i] * p.m[i]);
        let inv_x_s_sum = x_s.sum().recip();
        for i in 0..nc {
            x_s[i] *= inv_x_s_sum;
        }
        // Segment density
        let mut rho_s = 0.0;
        for i in 0..nc {
            rho_s += state.partial_density[i] * p.m[i];
        }
        // packing fractions
        let zeta = zeta_saft_vrq_mie(&p.m, &x_s, &d_hs_ij, rho_s);
        let zeta_bar = zeta_saft_vrq_mie(&p.m, &x_s, &s_eff_ij, rho_s);
        assert_relative_eq!(zeta, 9.7717457994590765E-002, epsilon = 5e-9);
        assert_relative_eq!(zeta_bar, 0.10864364645845238, epsilon = 5e-9);
    }

    #[test]
    fn test_perturbation_terms() {
        let p = hydrogen_fh1();
        let t = 26.7060;
        let v = 1.0e26;
        let n = 6.02214076e23;
        let state = StateHD::new(t, v, arr1(&[n]));
        let nc = 1;
        // temperature dependent sigma
        let s_eff_ij = Array2::from_shape_fn((nc, nc), |(i, j)| {
            p.calc_sigma_eff_ij(i, j, state.temperature)
        });
        // temperature dependent segment diameter
        let d_hs_ij = Array2::from_shape_fn((nc, nc), |(i, j)| {
            p.hs_diameter_ij(i, j, state.temperature, s_eff_ij[[i, j]])
        });

        // segment fractions
        let mut x_s = Array1::from_shape_fn(nc, |i| state.molefracs[i] * p.m[i]);
        let inv_x_s_sum = x_s.sum().recip();
        for i in 0..nc {
            x_s[i] *= inv_x_s_sum;
        }

        // Segment density
        let mut rho_s = 0.0;
        for i in 0..nc {
            rho_s += state.partial_density[i] * p.m[i];
        }

        // packing fractions
        let zeta = zeta_saft_vrq_mie(&p.m, &x_s, &d_hs_ij, rho_s);
        let zeta_bar = zeta_saft_vrq_mie(&p.m, &x_s, &s_eff_ij, rho_s);

        // temperature dependent well depth
        let epsilon_k_eff_ij = Array2::from_shape_fn((nc, nc), |(i, j)| {
            p.calc_epsilon_k_eff_ij(i, j, state.temperature)
        });

        // alphas ....
        let alpha = Alpha::new(&p, &s_eff_ij, &epsilon_k_eff_ij, state.temperature);

        // temperature dependent well depth
        let dq_ij =
            Array2::from_shape_fn((nc, nc), |(i, j)| p.quantum_d_ij(i, j, state.temperature));

        let a1 = first_order_perturbation(&p, &x_s, zeta, rho_s, &d_hs_ij, &s_eff_ij, &dq_ij);
        let a2 = second_order_perturbation(
            &p, &alpha, &x_s, zeta, zeta_bar, rho_s, &d_hs_ij, &s_eff_ij, &dq_ij,
        );
        let a3 = third_order_perturbation(&p, &alpha, &x_s, zeta_bar, &epsilon_k_eff_ij);

        let rel_err_a1 = (a1 + 30.702499892515764) / 30.702499892515764;
        let rel_err_a2 = (a2 + 67.046957636607587) / 67.046957636607587;
        let rel_err_a3 = (a3 + 470.96241656623727) / 470.96241656623727;
        assert_relative_eq!(rel_err_a1, 0.0, epsilon = 5e-7);
        assert_relative_eq!(rel_err_a2, 0.0, epsilon = 5e-7);
        assert_relative_eq!(rel_err_a3, 0.0, epsilon = 5e-7);
    }

    #[test]
    fn test_dispersion() {
        let disp = Dispersion {
            parameters: hydrogen_fh1(),
        };
        let a_ref = [
            -1.2683816065838103,
            -0.61628364979962436,
            -0.40740884837300861,
            -0.30420171646199534,
            -0.24264152651314486,
        ];
        let na = 6.02214076e23;
        for (it, &a) in a_ref.iter().enumerate() {
            let t = 26.7060 * (it + 1) as f64;
            let v = 1.0e26;
            let state = StateHD::new(t, v, arr1(&[na]));
            let a_disp = disp.helmholtz_energy(&state) / na;
            assert_relative_eq!(a_disp, a, epsilon = 1e-7);
        }
        let t = 26.7060;
        let v = 1.0e26 * 2.0;
        let n = na * 2.0;
        let state = StateHD::new(t, v, arr1(&[n]));
        let a_disp = disp.helmholtz_energy(&state) / na;
        assert_relative_eq!(a_disp, a_ref[0] * 2.0, epsilon = 1e-7);
    }

    #[test]
    fn test_parameters_mix() {
        let disp = Dispersion {
            parameters: h2_ne_fh1(),
        };
        let p = disp.parameters;
        assert_relative_eq!(p.c_ij[[0, 1]], 4.7303195840057679, epsilon = 1e-7);
        assert_relative_eq!(p.epsilon_k_ij[[0, 1]], 28.246978839971383, epsilon = 1e-7);
        assert_relative_eq!(p.lambda_a_ij[[0, 1]], 6.0, epsilon = 1e-7);
        assert_relative_eq!(p.lambda_r_ij[[0, 1]], 10.745966692414834, epsilon = 1e-7);
    }

    #[test]
    fn test_dispersion_mix() {
        let disp = Dispersion {
            parameters: h2_ne_fh1(),
        };
        let a_ref = [
            -4.4340438372333235,
            -2.1563424617699911,
            -1.4211021562556054,
            -1.0581654195146963,
            -0.84210863940206726,
        ];
        let na = 6.02214076e23;
        let n = [1.1 * na, 1.0 * na];
        let v = 1.0e26;
        for (it, &a) in a_ref.iter().enumerate() {
            let t = 30.0 * (it + 1) as f64;
            let state = StateHD::new(t, v, arr1(&n));
            let a_disp = disp.helmholtz_energy(&state) / na;
            dbg!(it);
            assert_relative_eq!(a_disp, a, epsilon = 1e-7);
        }
        let t = 30.0;
        let v = 1.0e26 * 2.0;
        let n = [2.2 * na, 2.0 * na];
        let state = StateHD::new(t, v, arr1(&n));
        let a_disp = disp.helmholtz_energy(&state) / na;
        assert_relative_eq!(a_disp, a_ref[0] * 2.0, epsilon = 1e-7);
    }

    #[cfg(feature = "dft")]
    #[test]
    fn test_dispersion_energy_density() {
        let disp = Dispersion {
            parameters: hydrogen_fh1(),
        };
        let p = &disp.parameters;
        let n = p.m.len();
        let rho = Array1::from_shape_fn(n, |_i| 0.01);
        let t = 25.0;
        // temperature dependent segment radius // calc & store this in struct
        let s_eff_ij = Array2::from_shape_fn((n, n), |(i, j)| p.calc_sigma_eff_ij(i, j, t));

        // temperature dependent segment radius // calc & store this in struct
        let d_hs_ij =
            Array2::from_shape_fn((n, n), |(i, j)| p.hs_diameter_ij(i, j, t, s_eff_ij[[i, j]]));

        // temperature dependent well depth // calc & store this in struct
        let epsilon_k_eff_ij =
            Array2::from_shape_fn((n, n), |(i, j)| p.calc_epsilon_k_eff_ij(i, j, t));

        // temperature dependent well depth // calc & store this in struct
        let dq_ij = Array2::from_shape_fn((n, n), |(i, j)| p.quantum_d_ij(i, j, t));

        // alphas .... // calc & store this in struct
        let alpha = Alpha::new(p, &s_eff_ij, &epsilon_k_eff_ij, t);
        let a_disp = dispersion_energy_density(
            p,
            &d_hs_ij,
            &s_eff_ij,
            &epsilon_k_eff_ij,
            &dq_ij,
            &alpha,
            &rho,
            t,
        );

        dbg!(rho);
        dbg!(a_disp);
        assert_relative_eq!(a_disp, -0.022349175545184223, epsilon = 1e-7);
    }
}
