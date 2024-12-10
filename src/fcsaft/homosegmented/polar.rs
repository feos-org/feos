use super::FcSaftHomoParameters;
use crate::hard_sphere::HardSphereProperties;
use feos_core::StateHD;
use ndarray::prelude::*;
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_3, FRAC_PI_6, PI};
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

pub(crate) const PI_SQ_43: f64 = 4.0 * PI * FRAC_PI_3;

pub struct MeanSegmentNumbers<D> {
    pub mij1: Array2<D>,
    pub mij2: Array2<D>,
    pub mijk1: Array3<D>,
    pub mijk2: Array3<D>,
}

impl<D: DualNum<f64> + Copy> MeanSegmentNumbers<D> {
    pub fn new(parameters: &FcSaftHomoParameters, m: &Array1<D>) -> Self {
        let (npoles, comp) = (parameters.ndipole, &parameters.dipole_comp);

        let mut mij1 = Array2::zeros((npoles, npoles));
        let mut mij2 = Array2::zeros((npoles, npoles));
        let mut mijk1 = Array3::zeros((npoles, npoles, npoles));
        let mut mijk2 = Array3::zeros((npoles, npoles, npoles));
        let clamp = |m: D| if m.re() > 2.0 { D::from(2.0) } else { m };
        for i in 0..npoles {
            let mi = clamp(m[comp[i]]);
            for j in i..npoles {
                let mj = clamp(m[comp[j]]);
                let mij = (mi * mj).sqrt();
                mij1[[i, j]] = (mij - 1.0) / mij;
                mij2[[i, j]] = mij1[[i, j]] * (mij - 2.0) / mij;
                for k in j..npoles {
                    let mk = clamp(m[comp[k]]);
                    let mijk = (mi * mj * mk).cbrt();
                    mijk1[[i, j, k]] = (mijk - 1.0) / mijk;
                    mijk2[[i, j, k]] = mijk1[[i, j, k]] * (mijk - 2.0) / mijk;
                }
            }
        }
        Self {
            mij1,
            mij2,
            mijk1,
            mijk2,
        }
    }
}

fn pair_integral_ij<D: DualNum<f64> + Copy>(
    mij1: D,
    mij2: D,
    etas: &[D],
    a: &[[f64; 3]],
    b: &[[f64; 3]],
    eps_ij_t: D,
) -> D {
    (0..a.len())
        .map(|i| {
            etas[i]
                * (eps_ij_t * (mij2 * b[i][2] + mij1 * b[i][1] + b[i][0])
                    + (mij2 * a[i][2] + mij1 * a[i][1] + a[i][0]))
        })
        .sum()
}

fn triplet_integral_ijk<D: DualNum<f64> + Copy>(
    mijk1: D,
    mijk2: D,
    etas: &[D],
    c: &[[f64; 3]],
) -> D {
    (0..c.len())
        .map(|i| etas[i] * (mijk2 * c[i][2] + mijk1 * c[i][1] + c[i][0]))
        .sum()
}

pub struct Dipole {
    pub parameters: Arc<FcSaftHomoParameters>,
}

impl Dipole {
    pub fn new(parameters: &Arc<FcSaftHomoParameters>) -> Self {
        Self {
            parameters: parameters.clone(),
        }
    }

    pub fn helmholtz_energy<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> D {
        let p = &self.parameters;

        let t_inv = state.temperature.inv();
        let eps_ij_t = p.e_k_ij.mapv(|v| t_inv * v);
        let sig_ij_3 = p.sigma_ij.mapv(|v| v.powi(3));
        let mu2_term: Array1<D> = p
            .dipole_comp
            .iter()
            .map(|&i| t_inv * sig_ij_3[[i, i]] * p.epsilon_k[i] * p.mu2[i])
            .collect();

        let rho = &state.partial_density;
        let d = p.hs_diameter(state.temperature);
        let [m, m_star] = p.m_values(&d);
        let m = MeanSegmentNumbers::new(&self.parameters, &m);
        let eta = (rho * &m_star * &d * &d * &d).sum() * FRAC_PI_6;
        let eta2 = eta * eta;
        let etas = [D::one(), eta, eta2, eta2 * eta, eta2 * eta2];

        let mut phi2 = D::zero();
        let mut phi3 = D::zero();
        for i in 0..p.ndipole {
            let di = p.dipole_comp[i];
            for j in i..p.ndipole {
                let dj = p.dipole_comp[j];
                let c = if i == j { 1.0 } else { 2.0 };
                phi2 -= rho[di]
                    * rho[dj]
                    * mu2_term[i]
                    * mu2_term[j]
                    * pair_integral_ij(
                        m.mij1[[i, j]],
                        m.mij2[[i, j]],
                        &etas,
                        &AD,
                        &BD,
                        eps_ij_t[[di, dj]],
                    )
                    / sig_ij_3[[di, dj]]
                    * c;
                for k in j..p.ndipole {
                    let dk = p.dipole_comp[k];
                    let c = if i == k {
                        1.0
                    } else if i == j || j == k {
                        3.0
                    } else {
                        6.0
                    };
                    phi3 -= rho[di] * rho[dj] * rho[dk] * mu2_term[i] * mu2_term[j] * mu2_term[k]
                        / (p.sigma_ij[[di, dj]] * p.sigma_ij[[di, dk]] * p.sigma_ij[[dj, dk]])
                        * triplet_integral_ijk(m.mijk1[[i, j, k]], m.mijk2[[i, j, k]], &etas, &CD)
                        * c;
                }
            }
        }
        phi2 *= PI;
        phi3 *= PI_SQ_43;
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
