use std::{collections::HashMap, f64::consts::TAU};

use super::parameters::{QuantumCorrection, UVCSPars};
use itertools::izip;
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use quantity::{GRAM, KILOGRAM, MOL, NAV};

const C: [[f64; 3]; 5] = [
    [-1.0577620680e+02, 2.7130516920e+01, -7.9885048560E-02],
    [-2.2875719760e+03, 5.1995559760e+02, 8.2111050000E-01],
    [1.7306319200e+03, -1.7476174200e+02, 3.2561549200e+01],
    [-9.0993639000e+01, 2.2823545260e+01, -2.8168749280e-02],
    [-1.4764571200e+03, 4.3447760860e+02, -1.1267359060e+01],
];

/// Parameters for effective sigma
const CS: [f64; 9] = [
    31.60144413,
    2.09612861,
    0.20116118,
    50.80286701,
    -4.28014515,
    0.20063555,
    25.11333946,
    1.35310377,
    0.20250069,
];

const CE: [f64; 9] = [
    10.60837315,
    1.58987996,
    0.19807347,
    17.92633472,
    -19.76493004,
    3.91514157,
    7.17199935,
    6.45489122,
    0.18773935,
];

/// Parameters according to corresponding states principle
#[derive(Debug)]
pub struct CorrespondingParameters<D> {
    pub ncomponents: usize,
    pub rep: DVector<D>,
    pub att: DVector<D>,
    pub sigma: DVector<D>,
    pub epsilon_k: DVector<D>,
    pub rep_ij: DMatrix<D>,
    pub att_ij: DMatrix<D>,
    pub sigma_ij: DMatrix<D>,
    pub eps_k_ij: DMatrix<D>,
}

impl<D: DualNum<f64> + Copy> CorrespondingParameters<D> {
    pub fn new(parameters: &UVCSPars, temperature: D) -> Self {
        let p = &parameters;
        let n = p.ncomponents;
        let mut molarweight = DVector::zeros(n);
        let mut rep = DVector::zeros(n);
        let mut att = DVector::zeros(n);
        let mut sigma = DVector::zeros(n);
        let mut epsilon_k = DVector::zeros(n);
        let mut mass = DVector::zeros(n);

        let to_mass_per_molecule = (GRAM / MOL / NAV / KILOGRAM).into_value();
        for i in 0..n {
            match p.quantum_correction[i] {
                None => {
                    rep[i] = D::one() * p.rep[i];
                    att[i] = D::one() * p.att[i];
                    sigma[i] = D::one() * p.sigma[i];
                    epsilon_k[i] = D::one() * p.epsilon_k[i];
                }
                Some(QuantumCorrection::FeynmanHibbs1 {
                    c_sigma,
                    c_epsilon_k,
                    c_rep: c_lr,
                }) => {
                    assert_eq!(p.att[i], 6.0);
                    mass[i] = p.molarweight[i] * to_mass_per_molecule;
                    let d_s2 = temperature.recip() / mass[i] * D_QM_PREFACTOR / p.sigma[i].powi(2);
                    // let [s, e, d_s2] = fh.effective_parameters(temperature, mass[i]);
                    let s = effective_sigma(d_s2, p.rep[i], c_sigma.as_ref());
                    let e = effective_epsilon_k(d_s2, p.rep[i], c_epsilon_k.as_ref());
                    let m = effective_repulsive_exponent_ratio(d_s2, p.rep[i], c_lr.as_ref());
                    rep[i] = m * p.rep[i];
                    att[i] = D::one() * p.att[i];
                    sigma[i] = s * p.sigma[i];
                    epsilon_k[i] = e * p.epsilon_k[i];
                }
            }
            molarweight[i] = p.molarweight[i];
        }

        let mut rep_ij = DMatrix::zeros(n, n);
        let mut att_ij = DMatrix::zeros(n, n);
        let mut sigma_ij = DMatrix::zeros(n, n);
        let mut eps_k_ij = DMatrix::zeros(n, n);
        for i in 0..n {
            rep_ij[(i, i)] = rep[i];
            att_ij[(i, i)] = att[i];
            sigma_ij[(i, i)] = sigma[i];
            eps_k_ij[(i, i)] = epsilon_k[i];
            for j in i + 1..n {
                rep_ij[(i, j)] = (rep[i] * rep[j]).sqrt();
                rep_ij[(j, i)] = rep_ij[(i, j)];
                att_ij[(i, j)] = (att[i] * att[j]).sqrt();
                att_ij[(j, i)] = att_ij[(i, j)];
                sigma_ij[(i, j)] = (sigma[i] + sigma[j]) * 0.5 * (1.0 - p.l_ij[(i, j)]);
                sigma_ij[(j, i)] = sigma_ij[(i, j)];
                eps_k_ij[(i, j)] = (epsilon_k[i] * epsilon_k[j]).sqrt() * (1.0 - p.k_ij[(i, j)]);
                eps_k_ij[(j, i)] = eps_k_ij[(i, j)];
            }
        }
        Self {
            ncomponents: n,
            rep,
            att,
            sigma,
            epsilon_k,
            rep_ij,
            att_ij,
            sigma_ij,
            eps_k_ij,
        }
    }
}

/// Calculate m_eff/m given D(T)/sig^2 and m.
fn effective_repulsive_exponent_ratio<D: DualNum<f64> + Copy>(
    qd: D,
    m: f64,
    c: Option<&[f64; 5]>,
) -> D {
    let c_scale = c.unwrap_or(&[1.0; 5]);
    let mut pref = [0.0; 5];
    for (pi, ci, c_scale_i) in izip!(&mut pref, &C, c_scale) {
        *pi = (ci[0] + m * (ci[1] + m * ci[2])) * c_scale_i;
    }
    (qd * (qd * ((qd * pref[2]) + pref[1]) + pref[0]) + 1.0)
        / (qd * ((qd * pref[4]) + pref[3]) + 1.0)
}

fn effective_sigma<D: DualNum<f64> + Copy>(qd: D, m: f64, c: Option<&[f64; 3]>) -> D {
    let c_scale = c.unwrap_or(&[1.0; 3]);
    let c0 = CS[0] + m * (CS[1] + (m * CS[2]));
    let c1 = CS[3] + m * (CS[4] + (m * CS[5]));
    let c2 = CS[6] + m * (CS[7] + (m * CS[8]));

    (qd * (qd * c1 * c_scale[1] + c0 * c_scale[0]) + 1.0) / (qd * c2 * c_scale[2] + 1.0)
}

fn effective_epsilon_k<D: DualNum<f64> + Copy>(qd: D, m: f64, c: Option<&[f64; 3]>) -> D {
    let c_scale = c.unwrap_or(&[1.0; 3]);
    let c0 = CE[0] + m * (CE[1] + (m * CE[2]));
    let c1 = CE[3] + m * (CE[4] + (m * CE[5]));
    let c2 = CE[6] + m * (CE[7] + (m * CE[8]));

    (qd * (qd * c1 * c_scale[1] + c0 * c_scale[0]) + 1.0) / (qd * c2 * c_scale[2] + 1.0)
}

const KB: f64 = 1.380649e-23;
const PLANCK: f64 = 6.62607015e-34;
const D_QM_PREFACTOR: f64 = PLANCK * PLANCK / (TAU * TAU) / 12.0 * 1e20 / KB;

pub struct MieFeynmanHibbs1 {
    pub sigma: f64,
    pub epsilon_k: f64,
    pub lr: f64,
    pub la: f64,
}

impl MieFeynmanHibbs1 {
    pub fn u_du_d2u2<D: DualNum<f64> + Copy>(&self, r: D, mass: f64, temperature: D) -> [D; 3] {
        let s = self.sigma;
        let eps = self.epsilon_k;
        let lr = self.lr;
        let la = self.la;
        let c = lr / (lr - la) * (lr / la).powf(la / (lr - la));

        let q1r = lr * (lr - 1.0);
        let q1a = la * (la - 1.0);
        let d = temperature.recip() / mass * D_QM_PREFACTOR;
        let mut u = r.powf(lr).recip() * s.powf(lr) - r.powf(la).recip() * s.powf(la);
        let mut u_r = -r.powf(lr + 1.0).recip() * lr * s.powf(lr)
            + r.powf(la + 1.0).recip() * la * s.powf(la);
        let mut u_rr = r.powf(lr + 2.0).recip() * lr * (lr + 1.0) * s.powf(lr)
            - r.powf(la + 2.0).recip() * la * (la + 1.0) * s.powf(la);

        // 1st order
        u += d
            * (r.powf(lr + 2.0).recip() * q1r * s.powf(lr)
                - r.powf(la + 2.0).recip() * q1a * s.powf(la));
        u_r += d
            * (r.powf(lr + 3.0).recip() * -q1r * (lr + 2.0) * s.powf(lr)
                + r.powf(la + 3.0).recip() * q1a * (la + 2.0) * s.powf(la));
        u_rr += d
            * (r.powf(lr + 4.0).recip() * q1r * (lr + 2.0) * (lr + 3.0) * s.powf(lr)
                - r.powf(la + 4.0).recip() * q1a * (la + 2.0) * (la + 3.0) * s.powf(la));

        u *= c * eps;
        u_r *= c * eps;
        u_rr *= c * eps;
        [u, u_r, u_rr]
    }

    // pub fn calc_sigma_eff_ij<D: DualNum<f64> + Copy>(
    //     &self,
    //     i: usize,
    //     j: usize,
    //     temperature: D,
    // ) -> D {
    //     let mut r = D::one() * self.sigma_ij[[i, j]];
    //     let mut u_vec = [D::zero(), D::zero(), D::zero()];
    //     for _k in 1..20 {
    //         u_vec = self.qmie_potential_ij(i, j, r, temperature);
    //         if u_vec[0].re().abs() < 1.0e-12 {
    //             break;
    //         }
    //         r += -u_vec[0] / u_vec[1];
    //     }
    //     if u_vec[0].re().abs() > 1.0e-12 {
    //         println!("calc_sigma_eff_ij calculation failed");
    //     }
    //     r
    // }

    pub fn effective_parameters<D: DualNum<f64> + Copy>(
        &self,
        temperature: D,
        mass: f64,
    ) -> [D; 3] {
        let mut r = D::one() * self.sigma;
        let mut u_vec = [D::zero(), D::zero(), D::zero()];

        // sigma
        for _k in 1..20 {
            u_vec = self.u_du_d2u2(r, mass, temperature);
            if u_vec[0].re().abs() < 1.0e-12 {
                break;
            }
            r += -u_vec[0] / u_vec[1];
        }
        if u_vec[0].re().abs() > 1.0e-12 {
            println!("calc_sigma_eff_ij calculation failed");
        }
        let sigma_eff = r;

        // epsilon
        let mut r = sigma_eff;
        for _k in 1..20 {
            u_vec = self.u_du_d2u2(r, mass, temperature);
            if u_vec[1].re().abs() < 1.0e-12 {
                break;
            }
            r += -u_vec[1] / u_vec[2];
        }
        if u_vec[1].re().abs() > 1.0e-12 {
            println!("calc_epsilon_k_eff_ij calculation failed");
        }

        // d
        let d = temperature.recip() / mass * D_QM_PREFACTOR;
        [sigma_eff, -u_vec[0], d / self.sigma.powi(2)]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_effective_parameters() {
        let fh = MieFeynmanHibbs1 {
            sigma: 2.7443,
            epsilon_k: 5.4195,
            lr: 9.0,
            la: 6.0,
        };
        let molarweight = 4.002601643881807;
        let to_mass_per_molecule = (GRAM / MOL / NAV / KILOGRAM).into_value();
        let [sigma, eps, d] = fh.effective_parameters(20.0, molarweight * to_mass_per_molecule);
        assert!((sigma - 2.92483368).abs() / 2.92483368 < 1e-5);
        assert!((eps - 4.50862751).abs() / 4.50862751 < 1e-5);
        assert!((d - 0.00670507).abs() / 0.00670507 < 1e-5)
    }

    #[test]
    fn test_effective_repulsive_exponent_ratio() {
        let m = effective_repulsive_exponent_ratio(10.0, 12.0, None);
        assert!(f64::abs(m - 22.18560064715075) / 22.18560064715075 < 1e-12);

        let m = effective_repulsive_exponent_ratio(1.234, 9.123, None);
        assert!(f64::abs(m - 3.7406349183375567) / 3.7406349183375567 < 1e-12);

        let m =
            effective_repulsive_exponent_ratio(1.234, 9.123, Some(&[1.01, 0.98, 0.99, 1.02, 2.0]));
        assert!(f64::abs(m - 1.8976602346883111) / 1.8976602346883111 < 1e-12)
    }

    #[test]
    fn test_effective_epsilon_k() {
        let e = effective_epsilon_k(1.234, 9.123, None);
        dbg!(e);
        assert!(f64::abs(e - 2.9592337917650835) / 2.9592337917650835 < 1e-12);

        let e = effective_epsilon_k(1.234, 9.123, Some(&[1.01, 0.98, 0.99]));
        dbg!(e);
        assert!(f64::abs(e - 2.94452907265039) / 2.94452907265039 < 1e-12)
    }

    #[test]
    fn test_effective_sigma() {
        let s = effective_sigma(1.234, 9.123, None);
        dbg!(s);
        assert!(f64::abs(s - 1.8756435576601465) / 1.8756435576601465 < 1e-12);

        let s = effective_sigma(1.234, 9.123, Some(&[1.01, 0.98, 0.99]));
        dbg!(s);
        assert!(f64::abs(s - 1.8938029821536044) / 1.8938029821536044 < 1e-12)
    }
}
