use std::f64::consts::{FRAC_PI_6, PI};

use feos_core::StateHD;
use itertools::Itertools;
use ndarray::{Array, Array1, Array2, ScalarOperand};
use num_dual::{Dual, Dual2, DualNum};
use num_traits::Zero;

use crate::hard_sphere::HardSphereProperties;

use super::SaftVRMieParameters;

const C: [[f64; 4]; 4] = [
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

/// Helper struct to calculate terms that are repeatedly used across multiple terms.
///
/// Stored is
/// x0^l * (a1s(l) + B(l))
///
/// the factor 2 * PI * rho_s * d_ij^3 * epsilon_k_ij is NOT included in these terms
struct AttractiveEnergyTerms<D> {
    lr: Array2<D>,
    la: Array2<D>,
    lr2: Array2<D>,
    la2: Array2<D>,
    lrla: Array2<D>,
}

impl<D: DualNum<f64> + Copy + ScalarOperand> AttractiveEnergyTerms<D> {
    fn new(n: usize, zeta_x: D, x0_ij: &Array2<D>, parameters: &SaftVRMieParameters) -> Self {
        // Function evaluations
        let mut lr = Array2::zeros((n, n));
        let mut la = Array2::zeros((n, n));
        let mut lr2 = Array2::zeros((n, n));
        let mut la2 = Array2::zeros((n, n));
        let mut lrla = Array2::zeros((n, n));

        for i in 0..n {
            for j in i..n {
                lr[[i, j]] = a1s_b_ij(zeta_x, x0_ij[[i, j]], parameters.lr_ij[[i, j]]);
                lr2[[i, j]] = a1s_b_ij(zeta_x, x0_ij[[i, j]], 2.0 * parameters.lr_ij[[i, j]]);
                la[[i, j]] = a1s_b_ij(zeta_x, x0_ij[[i, j]], parameters.la_ij[[i, j]]);
                la2[[i, j]] = a1s_b_ij(zeta_x, x0_ij[[i, j]], 2.0 * parameters.la_ij[[i, j]]);
                lrla[[i, j]] = a1s_b_ij(
                    zeta_x,
                    x0_ij[[i, j]],
                    parameters.lr_ij[[i, j]] + parameters.la_ij[[i, j]],
                );

                if i != j {
                    lr[[j, i]] = lr[[i, j]];
                    lr2[[j, i]] = lr2[[i, j]];
                    la[[j, i]] = la[[i, j]];
                    la2[[j, i]] = la2[[i, j]];
                    lrla[[j, i]] = lrla[[i, j]];
                }
            }
        }
        Self {
            lr,
            la,
            lr2,
            la2,
            lrla,
        }
    }

    fn new_derivatives(
        n: usize,
        zeta_x: Dual<D, f64>,
        x0_ij: &Array2<Dual<D, f64>>,
        parameters: &SaftVRMieParameters,
    ) -> (Self, Self) {
        // Function evaluations
        let mut lr = Array2::zeros((n, n));
        let mut la = Array2::zeros((n, n));
        let mut lr2 = Array2::zeros((n, n));
        let mut la2 = Array2::zeros((n, n));
        let mut lrla = Array2::zeros((n, n));

        // First derivative w.r.t. rho_s
        let mut lrd = Array2::zeros((n, n));
        let mut lad = Array2::zeros((n, n));
        let mut lr2d = Array2::zeros((n, n));
        let mut la2d = Array2::zeros((n, n));
        let mut lrlad = Array2::zeros((n, n));

        let mut d = Dual::<D, f64>::zero();
        for i in 0..n {
            for j in i..n {
                d = a1s_b_ij(zeta_x, x0_ij[[i, j]], parameters.lr_ij[[i, j]]);
                lr[[i, j]] = d.re;
                lrd[[i, j]] = d.eps;
                d = a1s_b_ij(zeta_x, x0_ij[[i, j]], 2.0 * parameters.lr_ij[[i, j]]);
                lr2[[i, j]] = d.re;
                lr2d[[i, j]] = d.eps;
                d = a1s_b_ij(zeta_x, x0_ij[[i, j]], parameters.la_ij[[i, j]]);
                la[[i, j]] = d.re;
                lad[[i, j]] = d.eps;
                d = a1s_b_ij(zeta_x, x0_ij[[i, j]], 2.0 * parameters.la_ij[[i, j]]);
                la2[[i, j]] = d.re;
                la2d[[i, j]] = d.eps;
                d = a1s_b_ij(
                    zeta_x,
                    x0_ij[[i, j]],
                    parameters.lr_ij[[i, j]] + parameters.la_ij[[i, j]],
                );
                lrla[[i, j]] = d.re;
                lrlad[[i, j]] = d.eps;

                if i != j {
                    lr[[j, i]] = lr[[i, j]];
                    lr2[[j, i]] = lr2[[i, j]];
                    la[[j, i]] = la[[i, j]];
                    la2[[j, i]] = la2[[i, j]];
                    lrla[[j, i]] = lrla[[i, j]];

                    lrd[[j, i]] = lrd[[i, j]];
                    lr2d[[j, i]] = lr2d[[i, j]];
                    lad[[j, i]] = lad[[i, j]];
                    la2d[[j, i]] = la2d[[i, j]];
                    lrlad[[j, i]] = lrlad[[i, j]];
                }
            }
        }
        (
            Self {
                lr,
                la,
                lr2,
                la2,
                lrla,
            },
            Self {
                lr: lrd,
                la: lad,
                lr2: lr2d,
                la2: la2d,
                lrla: lrlad,
            },
        )
    }
}

struct Properties<D> {
    segment_density: D,
    d_ij: Array2<D>,
    x0_ij: Array2<D>,
    zeta_x: D,
    attractive_energy_terms: AttractiveEnergyTerms<D>,
    // only needed for chain term
    derivatives: Option<AttractiveEnergyTerms<D>>,
}

impl<D: DualNum<f64> + Copy + Zero + ScalarOperand> Properties<D> {
    /// Initialize properties that can be used across different properties excluding
    /// derivatives for chain term.
    fn new(parameters: &SaftVRMieParameters, state: &StateHD<D>) -> Self {
        let n = parameters.m.len();
        let t = state.temperature;
        let x = &state.molefracs;
        let rho_i = &state.partial_density;

        let xs = x * &parameters.m / (x * &parameters.m).sum();

        // Set eps to one -> get partial derivatives w.r.t segment density
        let segment_density = rho_i.sum() * (x * &parameters.m).sum();

        // diameter
        let d = parameters.hs_diameter(t);
        let d_ij = Array2::from_shape_fn((n, n), |(i, j)| (d[i] + d[j]) * 0.5);
        let d3_ij = d_ij.mapv(|d| d.powi(3));

        // segment packing fraction
        let zeta_x = (0..n)
            .cartesian_product(0..n)
            .fold(D::zero(), |acc, (i, j)| acc + xs[i] * xs[j] * d3_ij[[i, j]])
            * FRAC_PI_6
            * segment_density;

        let x0_ij = Array2::from_shape_fn((n, n), |(i, j)| {
            d_ij[[i, j]].recip() * parameters.sigma_ij[[i, j]]
        });
        let attractive_energy_terms = AttractiveEnergyTerms::new(n, zeta_x, &x0_ij, parameters);
        Self {
            segment_density,
            d_ij,
            x0_ij,
            zeta_x,
            attractive_energy_terms,
            derivatives: None,
        }
    }

    /// Initialize properties that can be used across different properties including
    /// derivatives for chain term.
    fn new_derivative(parameters: &SaftVRMieParameters, state: &StateHD<D>) -> Self {
        let n = parameters.m.len();
        let t = Dual::from_re(state.temperature);
        let x = state.molefracs.mapv(|x| Dual::from_re(x));
        let rho_i = state.partial_density.mapv(|rho_i| Dual::from_re(rho_i));

        let xs = &x * &parameters.m / (&x * &parameters.m).sum();

        // Set eps to one -> get partial derivatives w.r.t segment density
        let segment_density = (rho_i.sum() * (&x * &parameters.m).sum()).derivative();

        // diameter
        let d = parameters.hs_diameter(t);
        let d_ij = Array2::from_shape_fn((n, n), |(i, j)| (d[i] + d[j]) * 0.5);
        let d3_ij = d_ij.mapv(|d| d.powi(3));

        // segment packing fraction
        let zeta_x = (0..n)
            .cartesian_product(0..n)
            .fold(Dual::<D, f64>::zero(), |acc, (i, j)| {
                acc + xs[i] * xs[j] * d3_ij[[i, j]]
            })
            * FRAC_PI_6
            * segment_density;

        let x0_ij = Array2::from_shape_fn((n, n), |(i, j)| {
            d_ij[[i, j]].recip() * parameters.sigma_ij[[i, j]]
        });
        let (attractive_energy_terms, derivatives) =
            AttractiveEnergyTerms::new_derivatives(n, zeta_x, &x0_ij, parameters);
        Self {
            segment_density: segment_density.re,
            d_ij: d_ij.mapv(|d| d.re),
            x0_ij: x0_ij.mapv(|x0| x0.re),
            zeta_x: zeta_x.re,
            attractive_energy_terms,
            derivatives: Some(derivatives),
        }
    }
}

fn zeta_eff<D: DualNum<f64> + Copy>(zeta: D, lambda: f64) -> D {
    let li = 1. / lambda;
    let li2 = li * li;
    let li3 = li * li2;
    let c = [
        li * C[0][1] + li2 * C[0][2] + li3 * C[0][3] + C[0][0],
        li * C[1][1] + li2 * C[1][2] + li3 * C[1][3] + C[1][0],
        li * C[2][1] + li2 * C[2][2] + li3 * C[2][3] + C[2][0],
        li * C[3][1] + li2 * C[3][2] + li3 * C[3][3] + C[3][0],
    ];
    zeta * c[0] + zeta.powi(2) * c[1] + zeta.powi(3) * c[2] + zeta.powi(4) * c[3]
}

fn f(k: usize, c: f64, lr: f64, la: f64) -> f64 {
    let alpha = c * (1.0 / (la - 3.0) - 1.0 / (lr - 3.0));
    let alpha2 = alpha * alpha;
    let alpha3 = alpha * alpha2;
    let phi = PHI[k];
    (alpha * phi[1] + alpha2 * phi[2] + alpha3 * phi[3] + phi[0])
        / (alpha * phi[4] + alpha2 * phi[5] + alpha3 * phi[6] + 1.0)
}

/// Sutherland potential for mixtures (Eq. A 16) divided by 2 PI rho_s d_ij^3 epsilon_k_ij
fn a1s_ij<D: DualNum<f64> + Copy>(zeta_x: D, lambda: f64) -> D {
    let zeta_eff = zeta_eff(zeta_x, lambda);
    -(-zeta_eff * 0.5 + 1.0) / ((-zeta_eff + 1.0).powi(3) * (lambda - 3.0))
}

/// Eq. A 12 of Lafitte divided by 2 PI rho_s d_ij^3 epsilon_k_ij
fn b_ij<D: DualNum<f64> + Copy>(zeta_x: D, x0: D, lambda: f64) -> D {
    let i = -(x0.powf(3.0 - lambda) - 1.0) / (lambda - 3.0);
    let j =
        -(x0.powf(4.0 - lambda) * (lambda - 3.0) - x0.powf(3.0 - lambda) * (lambda - 4.0) - 1.0)
            / ((lambda - 3.0) * (lambda - 4.0));
    ((-zeta_x * 0.5 + 1.0) * i - zeta_x * (zeta_x + 1.0) * 4.5 * j) / (-zeta_x + 1.0).powi(3)
}

/// Calculates x0^l (a1s_ij + b_ij) without prefactor C * rho_s * epsilon * d3
fn a1s_b_ij<D: DualNum<f64> + Copy>(zeta_x: D, x0: D, lambda: f64) -> D {
    x0.powf(lambda) * (a1s_ij(zeta_x, lambda) + b_ij(zeta_x, x0, lambda))
}

fn a1_k_ij<D: DualNum<f64> + Copy>(
    rho_s: D,
    zeta_x: D,
    x0: D,
    d3: D,
    c: f64,
    lr: f64,
    la: f64,
    epsilon_k: f64,
) -> D {
    (rho_s * 2.0 * PI * d3 * epsilon_k * c) * (a1s_b_ij(zeta_x, x0, la) - a1s_b_ij(zeta_x, x0, lr))
}

/// First-order mean-attractive contribution for mixtures
fn a1<D: DualNum<f64> + Copy + ScalarOperand>(
    parameters: &SaftVRMieParameters,
    xs: &Array1<D>,
    rho_s: D,
    zeta_x: D,
    x0_ij: &Array2<D>,
    d3_ij: &Array2<D>,
    state: &StateHD<D>,
) -> D {
    let n = parameters.m.len();

    let mut a_k = D::zero();
    for i in 0..n {
        for j in i..n {
            let mul = if i == j { 1.0 } else { 2.0 };
            a_k += xs[i]
                * xs[j]
                * mul
                * a1_k_ij(
                    rho_s,
                    zeta_x,
                    x0_ij[[i, j]],
                    d3_ij[[i, j]],
                    parameters.c_ij[[i, j]],
                    parameters.lr_ij[[i, j]],
                    parameters.la_ij[[i, j]],
                    parameters.epsilon_k_ij[[i, j]],
                )
        }
    }
    a_k * state.moles.sum() / state.temperature
}

fn a2_k2_ij<D: DualNum<f64> + Copy>(
    rho_s: D,
    zeta_x: D,
    zeta_x_bar: D,
    x0: D,
    d3: D,
    c: f64,
    lr: f64,
    la: f64,
    epsilon_k: f64,
) -> D {
    let xi = zeta_x_bar * f(0, c, lr, la)
        + zeta_x_bar.powi(5) * f(1, c, lr, la)
        + zeta_x_bar.powi(8) * f(2, c, lr, la);
    ((xi + 1.0) * rho_s * PI * 2.0 * d3 * epsilon_k.powi(2) * c.powi(2))
        * (x0.powf(2.0 * la) * (a1s_ij(zeta_x, 2.0 * la) + b_ij(zeta_x, x0, 2.0 * la))
            - x0.powf(lr + la) * 2.0 * (a1s_ij(zeta_x, lr + la) + b_ij(zeta_x, x0, lr + la))
            + x0.powf(2.0 * lr) * (a1s_ij(zeta_x, 2.0 * lr) + b_ij(zeta_x, x0, 2.0 * lr)))
}

fn a2<D: DualNum<f64> + Copy + ScalarOperand>(
    parameters: &SaftVRMieParameters,
    xs: &Array1<D>,
    rho_s: D,
    zeta_x: D,
    zeta_x_bar: D,
    x0_ij: &Array2<D>,
    d3_ij: &Array2<D>,
    state: &StateHD<D>,
) -> D {
    let n = parameters.m.len();

    // HS isothermal compressibility
    let k_hs = (zeta_x - 1.0).powi(4)
        / ((zeta_x + zeta_x.powi(2) - zeta_x.powi(3)) * 4.0 + zeta_x.powi(4) + 1.0);

    let mut a_k = D::zero();
    for i in 0..n {
        for j in i..n {
            let mul = if i == j { 1.0 } else { 2.0 };
            a_k += xs[i]
                * xs[j]
                * mul
                * a2_k2_ij(
                    rho_s,
                    zeta_x,
                    zeta_x_bar,
                    x0_ij[[i, j]],
                    d3_ij[[i, j]],
                    parameters.c_ij[[i, j]],
                    parameters.lr_ij[[i, j]],
                    parameters.la_ij[[i, j]],
                    parameters.epsilon_k_ij[[i, j]],
                )
        }
    }
    a_k * state.moles.sum() / state.temperature.powi(2) * 0.5 * k_hs
}

#[inline]
fn a3_k3_ij<D: DualNum<f64> + Copy>(zeta_x_bar: D, c: f64, lr: f64, la: f64, epsilon_k: f64) -> D {
    -zeta_x_bar
        * f(3, c, lr, la)
        * (zeta_x_bar * f(4, c, lr, la) + zeta_x_bar.powi(2) * f(5, c, lr, la)).exp()
        * epsilon_k.powi(3)
}

fn a3<D: DualNum<f64> + Copy + ScalarOperand>(
    parameters: &SaftVRMieParameters,
    xs: &Array1<D>,
    zeta_x_bar: D,
    state: &StateHD<D>,
) -> D {
    let n = parameters.m.len();
    let mut a_k = D::zero();
    for i in 0..n {
        for j in i..n {
            let mul = if i == j { 1.0 } else { 2.0 };
            a_k += xs[i]
                * xs[j]
                * mul
                * a3_k3_ij(
                    zeta_x_bar,
                    parameters.c_ij[[i, j]],
                    parameters.lr_ij[[i, j]],
                    parameters.la_ij[[i, j]],
                    parameters.epsilon_k_ij[[i, j]],
                )
        }
    }
    a_k * state.moles.sum() / state.temperature.powi(3)
}

fn a_monomer<D: DualNum<f64> + Copy + ScalarOperand>(
    parameters: &SaftVRMieParameters,
    state: &StateHD<D>,
) -> D {
    a_monomer_perturbations(parameters, state)
        .iter()
        .fold(D::zero(), |a, a_i| a + a_i)
}

fn a_monomer_perturbations<D: DualNum<f64> + Copy + ScalarOperand>(
    parameters: &SaftVRMieParameters,
    state: &StateHD<D>,
) -> [D; 3] {
    let n = parameters.m.len();
    let xs = &state.molefracs * &parameters.m / (&state.molefracs * &parameters.m).sum();
    let rho_s = state.moles.sum() / state.volume * (&state.molefracs * &parameters.m).sum();

    // diameter
    let d = parameters.hs_diameter(state.temperature);
    let d_ij = Array2::from_shape_fn((n, n), |(i, j)| (d[i] + d[j]) * 0.5);
    let d3_ij = d_ij.mapv(|d| d.powi(3));

    // segment packing fraction
    let zeta_x = (0..n)
        .cartesian_product(0..n)
        .fold(D::zero(), |acc, (i, j)| acc + xs[i] * xs[j] * d3_ij[[i, j]])
        * FRAC_PI_6
        * rho_s;

    let zeta_x_bar = (0..n)
        .cartesian_product(0..n)
        .fold(D::zero(), |acc, (i, j)| {
            acc + xs[i] * xs[j] * parameters.sigma_ij[[i, j]].powi(3)
        })
        * FRAC_PI_6
        * rho_s;

    let x0_ij = Array2::from_shape_fn((n, n), |(i, j)| {
        d_ij[[i, j]].recip() * parameters.sigma_ij[[i, j]]
    });

    let a1 = a1(parameters, &xs, rho_s, zeta_x, &x0_ij, &d3_ij, state);
    let a2 = a2(
        parameters, &xs, rho_s, zeta_x, zeta_x_bar, &x0_ij, &d3_ij, state,
    );
    let a3 = a3(parameters, &xs, zeta_x_bar, state);
    [a1, a2, a3]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::saftvrmie::ethane;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_a1() {
        let parameters = ethane();
        let moles = 1.0;
        let volume = 249.6620487495878;
        let state = StateHD::new(200.0, volume, Array1::from_vec(vec![moles]));
        let a1 = a_monomer_perturbations(&parameters, &state)[0];
        dbg!(a1);
        assert_relative_eq!(a1, -1.8523115205290075, epsilon = 1e-12);
    }

    #[test]
    fn test_a2() {
        let parameters = ethane();
        let moles = 1.0;
        let volume = 249.6620487495878;
        let state = StateHD::new(200.0, volume, Array1::from_vec(vec![moles]));
        let a2 = a_monomer_perturbations(&parameters, &state)[1];
        dbg!(a2);
        assert_relative_eq!(a2, -0.14758954314496384, epsilon = 1e-12);
    }

    #[test]
    fn test_a3() {
        let parameters = ethane();
        let moles = 1.0;
        let volume = 249.6620487495878;
        let state = StateHD::new(200.0, volume, Array1::from_vec(vec![moles]));
        let a3 = a_monomer_perturbations(&parameters, &state)[2];
        dbg!(a3);
        assert_relative_eq!(a3, -0.05051720427644026, epsilon = 1e-12);
    }
}
