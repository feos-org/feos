use crate::saftvrmie::parameters::SaftVRMiePars;
use feos_core::StateHD;
use ndarray::{Array1, Array2, ScalarOperand};
use num_dual::{Dual, DualNum};
use num_traits::Zero;
use std::f64::consts::{FRAC_PI_6, PI};

#[derive(Debug)]
pub struct Properties<D> {
    /// Temperature dependent diameter
    diameter: Array1<D>,
    /// total number density of segments
    pub segment_density: D,
    /// mole fraction of segments
    pub segment_molefracs: Array1<D>,
    /// mean segment number
    mean_segment_number: D,
    /// mixture packing fraction using d(T)
    zeta_x: D,
    /// mixture packing fraction using sigma
    pub zeta_x_bar: D,
    /// k-values for HS pair correlation fn
    k0: [D; 4],
}

impl<D: DualNum<f64> + Copy + Zero + ScalarOperand> Properties<D> {
    pub(super) fn new(
        parameters: &SaftVRMiePars,
        state: &StateHD<D>,
        diameter: &Array1<D>,
    ) -> Self {
        let n = parameters.m.len();
        let x = &state.molefracs;

        let mean_segment_number = (x * &parameters.m).sum();
        let xs = x * &parameters.m / mean_segment_number;

        // Set eps to one -> get partial derivatives w.r.t segment density
        let segment_density = (&state.partial_density * &parameters.m).sum();

        // diameter
        let d_ij = Array2::from_shape_fn((n, n), |(i, j)| (diameter[i] + diameter[j]) * 0.5);
        let d3_ij = d_ij.mapv(|d| d.powi(3));

        // segment packing fraction
        let mut zeta_x = D::zero();
        let mut zeta_x_bar = D::zero();
        for i in 0..n {
            zeta_x += xs[i].powi(2) * d3_ij[[i, i]];
            zeta_x_bar += xs[i].powi(2) * parameters.sigma_ij[[i, i]].powi(3);
            for j in i + 1..n {
                zeta_x += xs[i] * xs[j] * d3_ij[[i, j]] * 2.0;
                zeta_x_bar += xs[i] * xs[j] * parameters.sigma_ij[[i, j]].powi(3) * 2.0;
            }
        }
        zeta_x *= segment_density * FRAC_PI_6;
        zeta_x_bar *= segment_density * FRAC_PI_6;

        let frac_1mzeta3 = (-zeta_x + 1.0).powi(3).recip();
        let z = zeta_x;
        let z2 = z * z;
        let z3 = z2 * z;
        let k0 =
            -(-z + 1.0).ln() + z * (-z * 39.0 + z2 * 9.0 - z3 * 2.0 + 42.0) * frac_1mzeta3 / 6.0;
        let k1 = z * frac_1mzeta3 * 0.5 * (z3 + z * 6.0 - 12.0);
        let k2 = -z2 * 3.0 / 8.0 * frac_1mzeta3 * (-zeta_x + 1.0);
        let k3 = z * frac_1mzeta3 / 6.0 * (-z3 + z * 3.0 + 3.0);

        Self {
            diameter: diameter.clone(),
            segment_density,
            segment_molefracs: xs,
            mean_segment_number,
            zeta_x,
            zeta_x_bar,
            k0: [k0, k1, k2, k3],
        }
    }
}

pub(super) const PHI: [[f64; 7]; 6] = [
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

/// First, second and third order perturbations for dispersive interactions
pub fn a_disp<D: DualNum<f64> + Copy + ScalarOperand>(
    parameters: &SaftVRMiePars,
    properties: &Properties<D>,
    state: &StateHD<D>,
) -> D {
    let p = &parameters;
    let n = p.sigma.len();
    let xs = &properties.segment_molefracs;
    let t_inv = state.temperature.inv();
    let zeta_x = properties.zeta_x;
    let k_hs = (zeta_x - 1.0).powi(4)
        / ((zeta_x + zeta_x.powi(2) - zeta_x.powi(3)) * 4.0 + zeta_x.powi(4) + 1.0);
    let zeta_x_bar = properties.zeta_x_bar;
    let zx5 = zeta_x_bar.powi(5);
    let zx8 = zeta_x_bar.powi(8);

    let mut a1 = D::zero();
    let mut a2 = D::zero();
    let mut a3 = D::zero();

    for i in 0..n {
        // parameters
        let eps_k = p.epsilon_k[i];
        let sig = p.sigma[i];
        let la = p.la[i];
        let lr = p.lr[i];
        let c = p.c_ij[[i, i]];

        let di = properties.diameter[i];
        let d3 = di.powi(3);
        let x0 = di.recip() * sig;
        let pref = properties.segment_density * d3 * eps_k * 2.0 * PI * c;
        let a1s_b_la = a1s_b_ij(zeta_x, x0, la);
        let a1s_b_lr = a1s_b_ij(zeta_x, x0, lr);
        let a1s_b_2la = a1s_b_ij(zeta_x, x0, 2.0 * la);
        let a1s_b_lalr = a1s_b_ij(zeta_x, x0, la + lr);
        let a1s_b_2lr = a1s_b_ij(zeta_x, x0, 2.0 * lr);
        let a1_ii = pref * (a1s_b_la - a1s_b_lr);
        let a2_ii = pref * eps_k * c * k_hs * 0.5 * (a1s_b_2la - a1s_b_lalr * 2.0 + a1s_b_2lr);

        // note indices of f(i, alpha) are shifted due to 0-indexing.
        let alpha = parameters.alpha_ij[[i, i]];
        let a3_ii = -zeta_x_bar
            * f(3, alpha)
            * (zeta_x_bar * (zeta_x_bar * f(5, alpha) + f(4, alpha))).exp()
            * eps_k.powi(3);

        let xii = zeta_x_bar * f(0, alpha) + zx5 * f(1, alpha) + zx8 * f(2, alpha);

        // accumulate contributions
        let xs_ii2 = xs[i] * xs[i];
        a1 += a1_ii * xs_ii2;
        a2 += a2_ii * xs_ii2 * (xii + 1.0);
        a3 += a3_ii * xs_ii2;

        for j in i + 1..n {
            // parameters
            let eps_k = p.epsilon_k_ij[[i, j]];
            let sig = p.sigma_ij[[i, j]];
            let la = p.la_ij[[i, j]];
            let lr = p.lr_ij[[i, j]];
            let c = p.c_ij[[i, j]];

            let dij = (di + properties.diameter[j]) * 0.5;
            let d3 = dij.powi(3);
            let x0 = dij.recip() * sig;
            let pref = properties.segment_density * d3 * eps_k * 2.0 * PI * c;
            let a1s_b_la = a1s_b_ij(zeta_x, x0, la);
            let a1s_b_lr = a1s_b_ij(zeta_x, x0, lr);
            let a1s_b_2la = a1s_b_ij(zeta_x, x0, 2.0 * la);
            let a1s_b_lalr = a1s_b_ij(zeta_x, x0, la + lr);
            let a1s_b_2lr = a1s_b_ij(zeta_x, x0, 2.0 * lr);
            let a1_ij = pref * (a1s_b_la - a1s_b_lr);
            let a2_ij = pref * eps_k * c * k_hs * 0.5 * (a1s_b_2la - a1s_b_lalr * 2.0 + a1s_b_2lr);

            // note indices of f(i, alpha) are shifted due to 0-indexing.
            let alpha = parameters.alpha_ij[[i, j]];
            let a3_ij = -zeta_x_bar
                * f(3, alpha)
                * (zeta_x_bar * (zeta_x_bar * f(5, alpha) + f(4, alpha))).exp()
                * eps_k.powi(3);
            let xij = zeta_x_bar * f(0, alpha) + zx5 * f(1, alpha) + zx8 * f(2, alpha);

            let xs_ij = xs[i] * xs[j];
            // accumulate contributions
            a1 += a1_ij * xs_ij * 2.0;
            a2 += a2_ij * xs_ij * (xij + 1.0) * 2.0;
            a3 += a3_ij * xs_ij * 2.0;
        }
    }
    state.moles.sum()
        * properties.mean_segment_number
        * (a1 * t_inv + a2 * t_inv.powi(2) + a3 * t_inv.powi(3))
}

/// Combine dispersion and chain contributions
pub fn a_disp_chain<D: DualNum<f64> + Copy + ScalarOperand>(
    parameters: &SaftVRMiePars,
    properties: &Properties<D>,
    state: &StateHD<D>,
) -> D {
    let p = &parameters;
    let n = p.sigma.len();
    let k = &properties.k0;
    let xs = &properties.segment_molefracs;
    let t_inv = state.temperature.inv();

    // wrap rho_s in Dual to calculate da1/drho_s and da2/drho_s
    // for chain contribution on the fly.
    let rho_s_dual = Dual::from_re(properties.segment_density).derivative();
    let zeta_x = properties.zeta_x;
    let zeta_x_dual = if properties.segment_density.is_zero() {
        rho_s_dual * 0.0
    } else {
        Dual::from_re(zeta_x / properties.segment_density) * rho_s_dual
    };
    // let zeta_x_dual = Dual::from_re(zeta_x / properties.segment_density) * rho_s_dual;
    let k_hs_dual = (zeta_x_dual - 1.0).powi(4)
        / ((zeta_x_dual + zeta_x_dual.powi(2) - zeta_x_dual.powi(3)) * 4.0
            + zeta_x_dual.powi(4)
            + 1.0);

    // non-dual things
    let zeta_x_bar = properties.zeta_x_bar;
    let zx5 = zeta_x_bar.powi(5);
    let zx8 = zeta_x_bar.powi(8);
    let k_hs = k_hs_dual.re;

    let mut a1 = D::zero();
    let mut a2 = D::zero();
    let mut a3 = D::zero();
    let mut a_chain = D::zero();

    for i in 0..n {
        // parameters
        let m = p.m[i];
        let eps_k = p.epsilon_k[i];
        let sig = p.sigma[i];
        let la = p.la[i];
        let lr = p.lr[i];
        let c = p.c_ij[[i, i]];

        let di = properties.diameter[i];

        // calculate a1 and a2 using Dual(D)
        // this is only done in the outer loop
        let d3 = Dual::from_re(di.powi(3));
        let x0 = Dual::from_re(di.recip() * sig);
        let pref = rho_s_dual * d3 * eps_k * 2.0 * PI * c;
        let a1s_b_la = a1s_b_ij(zeta_x_dual, x0, la);
        let a1s_b_lr = a1s_b_ij(zeta_x_dual, x0, lr);
        let a1s_b_2la = a1s_b_ij(zeta_x_dual, x0, 2.0 * la);
        let a1s_b_lalr = a1s_b_ij(zeta_x_dual, x0, la + lr);
        let a1s_b_2lr = a1s_b_ij(zeta_x_dual, x0, 2.0 * lr);
        let a1_ii = pref * (a1s_b_la - a1s_b_lr);
        let a2_ii = pref * eps_k * c * k_hs_dual * 0.5 * (a1s_b_2la - a1s_b_lalr * 2.0 + a1s_b_2lr);

        // note indices of f(i, alpha) are shifted due to 0-indexing.
        let alpha = parameters.alpha_ij[[i, i]];
        let a3_ii = -zeta_x_bar
            * f(3, alpha)
            * (zeta_x_bar * (zeta_x_bar * f(5, alpha) + f(4, alpha))).exp()
            * eps_k.powi(3);

        // chi is not Dual(D), since it is not needed in chain
        let xii = zeta_x_bar * f(0, alpha) + zx5 * f(1, alpha) + zx8 * f(2, alpha);

        // accumulate contributions
        let xs_ii2 = xs[i] * xs[i];
        a1 += a1_ii.re * xs_ii2;
        a2 += a2_ii.re * xs_ii2 * (xii + 1.0);
        a3 += a3_ii * xs_ii2;

        // calculate chain using D
        // use dual-parts of a1_ii and a2_ii for derivatives
        // and real-parts of a1s_b-terms.
        let x0 = x0.re;
        let pref = d3.re * eps_k * 2.0 * PI;
        let g_hs = (k[0] + k[1] * x0 + k[2] * x0.powi(2) + k[3] * x0.powi(3)).exp();
        let g1 = a1_ii.eps * 3.0 / pref - (a1s_b_la.re * la - a1s_b_lr.re * lr) * c;
        let g2_mca = a2_ii.eps * 3.0 / pref / eps_k
            - (a1s_b_2lr.re * lr - a1s_b_lalr.re * (la + lr) + a1s_b_2la.re * la)
                * k_hs
                * c.powi(2);
        let beta_eps = t_inv * eps_k;
        let gamma = zeta_x_bar
            * beta_eps.exp_m1()
            * 10.0
            * (-(10.0 * (0.57 - alpha)).tanh() + 1.0)
            * (-zeta_x_bar * 6.7 - zeta_x_bar.powi(2) * 8.0).exp();
        let g2 = g2_mca * (gamma + 1.0);
        let ln_g_mie = g_hs.ln() + (beta_eps * g1 + beta_eps.powi(2) * g2) / g_hs;
        a_chain += -state.molefracs[i] * (m - 1.0) * ln_g_mie;

        for j in i + 1..n {
            // parameters
            let eps_k = p.epsilon_k_ij[[i, j]];
            let sig = p.sigma_ij[[i, j]];
            let la = p.la_ij[[i, j]];
            let lr = p.lr_ij[[i, j]];
            let c = p.c_ij[[i, j]];

            let dij = (di + properties.diameter[j]) * 0.5;
            let d3 = dij.powi(3);
            let x0 = dij.recip() * sig;
            let pref = properties.segment_density * d3 * eps_k * 2.0 * PI * c;
            let a1s_b_la = a1s_b_ij(zeta_x, x0, la);
            let a1s_b_lr = a1s_b_ij(zeta_x, x0, lr);
            let a1s_b_2la = a1s_b_ij(zeta_x, x0, 2.0 * la);
            let a1s_b_lalr = a1s_b_ij(zeta_x, x0, la + lr);
            let a1s_b_2lr = a1s_b_ij(zeta_x, x0, 2.0 * lr);
            let a1_ij = pref * (a1s_b_la - a1s_b_lr);
            let a2_ij = pref * eps_k * c * k_hs * 0.5 * (a1s_b_2la - a1s_b_lalr * 2.0 + a1s_b_2lr);

            // note indices of f(i, alpha) are shifted due to 0-indexing.
            let alpha = parameters.alpha_ij[[i, j]];
            let a3_ij = -zeta_x_bar
                * f(3, alpha)
                * (zeta_x_bar * (zeta_x_bar * f(5, alpha) + f(4, alpha))).exp()
                * eps_k.powi(3);
            let xij = zeta_x_bar * f(0, alpha) + zx5 * f(1, alpha) + zx8 * f(2, alpha);

            // accumulate contributions
            let xs_ij = xs[i] * xs[j];
            a1 += a1_ij * xs_ij * 2.0;
            a2 += a2_ij * xs_ij * (xij + 1.0) * 2.0;
            a3 += a3_ij * xs_ij * 2.0;
        }
    }
    state.moles.sum()
        * (properties.mean_segment_number * (a1 * t_inv + a2 * t_inv.powi(2) + a3 * t_inv.powi(3))
            + a_chain)
}

#[inline]
pub(super) fn zeta_eff<D: DualNum<f64> + Copy>(zeta: D, lambda: f64) -> D {
    let li = 1. / lambda;
    let li2 = li * li;
    let li3 = li * li2;
    let c = [
        li * C[0][1] + li2 * C[0][2] + li3 * C[0][3] + C[0][0],
        li * C[1][1] + li2 * C[1][2] + li3 * C[1][3] + C[1][0],
        li * C[2][1] + li2 * C[2][2] + li3 * C[2][3] + C[2][0],
        li * C[3][1] + li2 * C[3][2] + li3 * C[3][3] + C[3][0],
    ];
    // zeta * c[0] + zeta.powi(2) * c[1] + zeta.powi(3) * c[2] + zeta.powi(4) * c[3]
    zeta * (zeta * (zeta * (zeta * c[3] + c[2]) + c[1]) + c[0])
}

/// Sutherland potential for mixtures (Eq. A 16) divided by 2 PI rho_s d_ij^3 epsilon_k_ij
#[inline]
fn a1s_ij<D: DualNum<f64> + Copy>(zeta_x: D, lambda: f64) -> D {
    let zeta_eff = zeta_eff(zeta_x, lambda);
    -(-zeta_eff * 0.5 + 1.0) / ((-zeta_eff + 1.0).powi(3) * (lambda - 3.0))
}

/// Eq. A 12 of Lafitte divided by 2 PI rho_s d_ij^3 epsilon_k_ij
#[inline]
fn b_ij<D: DualNum<f64> + Copy>(zeta_x: D, x0: D, lambda: f64) -> D {
    let x0_3ml = x0.powf(3.0 - lambda);
    let i = -(x0_3ml - 1.0) / (lambda - 3.0);
    let j = -(x0.powf(4.0 - lambda) * (lambda - 3.0) - x0_3ml * (lambda - 4.0) - 1.0)
        / ((lambda - 3.0) * (lambda - 4.0));
    ((-zeta_x * 0.5 + 1.0) * i - zeta_x * (zeta_x + 1.0) * 4.5 * j) * (-zeta_x + 1.0).powi(-3)
}

/// Calculates x0^l (a1s_ij + b_ij) without prefactor C
#[inline]
fn a1s_b_ij<D: DualNum<f64> + Copy>(zeta_x: D, x0: D, lambda: f64) -> D {
    x0.powf(lambda) * (a1s_ij(zeta_x, lambda) + b_ij(zeta_x, x0, lambda))
}

fn f(k: usize, alpha: f64) -> f64 {
    let alpha2 = alpha * alpha;
    let alpha3 = alpha * alpha2;
    let phi = PHI[k];
    (alpha * phi[1] + alpha2 * phi[2] + alpha3 * phi[3] + phi[0])
        / (alpha * phi[4] + alpha2 * phi[5] + alpha3 * phi[6] + 1.0)
}

const C: [[f64; 4]; 4] = [
    [0.81096, 1.7888, -37.578, 92.284],
    [1.0205, -19.341, 151.26, -463.50],
    [-1.9057, 22.845, -228.14, 973.92],
    [1.0885, -6.1962, 106.98, -677.64],
];
