use super::GcPcSaftEosParameters;
use crate::hard_sphere::HardSphereProperties;
use feos_core::{HelmholtzEnergyDual, StateHD};
use ndarray::prelude::*;
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_3, PI};
use std::fmt;
use std::sync::Arc;

// Dipole parameters
pub const AD: [[f64; 3]; 5] = [
    [0.30435038064, 0.95346405973, -1.16100802773],
    [-0.13585877707, -1.83963831920, 4.52586067320],
    [1.44933285154, 2.01311801180, 0.97512223853],
    [0.35569769252, -7.37249576667, -12.2810377713],
    [-2.06533084541, 8.23741345333, 5.93975747420],
];

pub const BD: [[f64; 3]; 5] = [
    [0.21879385627, -0.58731641193, 3.48695755800],
    [-1.18964307357, 1.24891317047, -14.9159739347],
    [1.16268885692, -0.50852797392, 15.3720218600],
    [0.0; 3],
    [0.0; 3],
];

pub const CD: [[f64; 3]; 4] = [
    [-0.06467735252, -0.95208758351, -0.62609792333],
    [0.19758818347, 2.99242575222, 1.29246858189],
    [-0.80875619458, -2.38026356489, 1.65427830900],
    [0.69028490492, -0.27012609786, -3.43967436378],
];

pub const PI_SQ_43: f64 = 4.0 * PI * FRAC_PI_3;

fn pair_integral_ij<D: DualNum<f64> + Copy>(mij1: f64, mij2: f64, eta: D, eps_ij_t: D) -> D {
    let eta2 = eta * eta;
    let etas = [D::one(), eta, eta2, eta2 * eta, eta2 * eta2];
    (0..AD.len())
        .map(|i| {
            etas[i]
                * (eps_ij_t * (BD[i][0] + mij1 * BD[i][1] + mij2 * BD[i][2])
                    + (AD[i][0] + mij1 * AD[i][1] + mij2 * AD[i][2]))
        })
        .sum()
}

fn triplet_integral_ijk<D: DualNum<f64> + Copy>(mijk1: f64, mijk2: f64, eta: D) -> D {
    let eta2 = eta * eta;
    let etas = [D::one(), eta, eta2, eta2 * eta];
    (0..CD.len())
        .map(|i| etas[i] * (CD[i][0] + mijk1 * CD[i][1] + mijk2 * CD[i][2]))
        .sum()
}

pub struct Dipole {
    parameters: Arc<GcPcSaftEosParameters>,
    mij1: Array2<f64>,
    mij2: Array2<f64>,
    mijk1: Array3<f64>,
    mijk2: Array3<f64>,
    f2_term: Array2<f64>,
    f3_term: Array3<f64>,
}

impl Dipole {
    pub fn new(parameters: &Arc<GcPcSaftEosParameters>) -> Self {
        let ndipole = parameters.dipole_comp.len();

        let f2_term = Array2::from_shape_fn([ndipole; 2], |(i, j)| {
            parameters.mu2[i] * parameters.mu2[j] / parameters.s_ij[[i, j]].powi(3)
        });

        let f3_term = Array3::from_shape_fn([ndipole; 3], |(i, j, k)| {
            parameters.mu2[i] * parameters.mu2[j] * parameters.mu2[k]
                / (parameters.s_ij[[i, j]] * parameters.s_ij[[i, k]] * parameters.s_ij[[j, k]])
        });

        let mut mij1 = Array2::zeros((ndipole, ndipole));
        let mut mij2 = Array2::zeros((ndipole, ndipole));
        let mut mijk1 = Array3::zeros((ndipole, ndipole, ndipole));
        let mut mijk2 = Array3::zeros((ndipole, ndipole, ndipole));
        for i in 0..ndipole {
            let mi = parameters.m_mix[i].min(2.0);
            mij1[[i, i]] = (mi - 1.0) / mi;
            mij2[[i, i]] = mij1[[i, i]] * (mi - 2.0) / mi;

            mijk1[[i, i, i]] = mij1[[i, i]];
            mijk2[[i, i, i]] = mij2[[i, i]];
            for j in i + 1..ndipole {
                let mj = parameters.m_mix[j].min(2.0);
                let mij = (mi * mj).sqrt();
                mij1[[i, j]] = (mij - 1.0) / mij;
                mij2[[i, j]] = mij1[[i, j]] * (mij - 2.0) / mij;
                let mijk = (mi * mi * mj).cbrt();
                mijk1[[i, i, j]] = (mijk - 1.0) / mijk;
                mijk2[[i, i, j]] = mijk1[[i, i, j]] * (mijk - 2.0) / mijk;
                let mijk = (mi * mj * mj).cbrt();
                mijk1[[i, j, j]] = (mijk - 1.0) / mijk;
                mijk2[[i, j, j]] = mijk1[[i, j, j]] * (mijk - 2.0) / mijk;
                for k in j + 1..ndipole {
                    let mk = parameters.m_mix[k].min(2.0);
                    let mijk = (mi * mj * mk).cbrt();
                    mijk1[[i, j, k]] = (mijk - 1.0) / mijk;
                    mijk2[[i, j, k]] = mijk1[[i, j, k]] * (mijk - 2.0) / mijk;
                }
            }
        }
        Self {
            parameters: parameters.clone(),
            mij1,
            mij2,
            mijk1,
            mijk2,
            f2_term,
            f3_term,
        }
    }
}

impl<D: DualNum<f64> + Copy> HelmholtzEnergyDual<D> for Dipole {
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let p = &self.parameters;
        let ndipole = p.dipole_comp.len();

        let t_inv = state.temperature.inv();
        let eps_ij_t = Array2::from_shape_fn([ndipole; 2], |(i, j)| t_inv * p.e_k_ij[[i, j]]);

        let rho = &state.partial_density;
        let eta = p.zeta(state.temperature, &state.partial_density, [3])[0];

        let mut phi2 = D::zero();
        let mut phi3 = D::zero();
        for i in 0..ndipole {
            let di = p.dipole_comp[i];
            phi2 -= (rho[di] * rho[di] * self.f2_term[[i, i]])
                * pair_integral_ij(self.mij1[[i, i]], self.mij2[[i, i]], eta, eps_ij_t[[i, i]]);
            phi3 -= (rho[di] * rho[di] * rho[di] * self.f3_term[[i, i, i]])
                * triplet_integral_ijk(self.mijk1[[i, i, i]], self.mijk2[[i, i, i]], eta);
            for j in (i + 1)..ndipole {
                let dj = p.dipole_comp[j];
                phi2 -= (rho[di] * rho[dj] * self.f2_term[[i, j]])
                    * pair_integral_ij(self.mij1[[i, j]], self.mij2[[i, j]], eta, eps_ij_t[[i, j]])
                    * 2.0;
                phi3 -= (rho[di] * rho[di] * rho[dj] * self.f3_term[[i, i, j]])
                    * triplet_integral_ijk(self.mijk1[[i, i, j]], self.mijk2[[i, i, j]], eta)
                    * 3.0;
                phi3 -= (rho[di] * rho[dj] * rho[dj] * self.f3_term[[i, j, j]])
                    * triplet_integral_ijk(self.mijk1[[i, j, j]], self.mijk2[[i, j, j]], eta)
                    * 3.0;
                for k in (j + 1)..ndipole {
                    let dk = p.dipole_comp[k];
                    phi3 -= (rho[di] * rho[dj] * rho[dk] * self.f3_term[[i, j, k]])
                        * triplet_integral_ijk(self.mijk1[[i, j, k]], self.mijk2[[i, j, k]], eta)
                        * 6.0;
                }
            }
        }
        phi2 *= t_inv * t_inv * PI;
        phi3 *= t_inv.powi(3) * PI_SQ_43;
        let mut result = phi2 * phi2 / (phi2 - phi3) * state.volume;
        if result.re().is_nan() {
            result = phi2 * state.volume
        }
        result
    }
}

impl fmt::Display for Dipole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dipole")
    }
}
