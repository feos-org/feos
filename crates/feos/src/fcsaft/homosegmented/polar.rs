use super::FcSaftHomoPars;
use crate::hard_sphere::HardSphereProperties;
use feos_core::StateHD;
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_3, FRAC_PI_6, PI};

// Dipole parameters
pub(super) const AD: [[f64; 3]; 5] = [
    [0.30435038064, 0.95346405973, -1.16100802773],
    [-0.13585877707, -1.83963831920, 4.52586067320],
    [1.44933285154, 2.01311801180, 0.97512223853],
    [0.35569769252, -7.37249576667, -12.2810377713],
    [-2.06533084541, 8.23741345333, 5.93975747420],
];

pub(super) const BD: [[f64; 3]; 5] = [
    [0.21879385627, -0.58731641193, 3.48695755800],
    [-1.18964307357, 1.24891317047, -14.9159739347],
    [1.16268885692, -0.50852797392, 15.3720218600],
    [0.0; 3],
    [0.0; 3],
];

pub(super) const CD: [[f64; 3]; 4] = [
    [-0.06467735252, -0.95208758351, -0.62609792333],
    [0.19758818347, 2.99242575222, 1.29246858189],
    [-0.80875619458, -2.38026356489, 1.65427830900],
    [0.69028490492, -0.27012609786, -3.43967436378],
];

const PI_SQ_43: f64 = 4.0 * PI * FRAC_PI_3;

fn pair_integral_ij<D: DualNum<Primitive = f64> + Copy>(
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

fn triplet_integral_ijk<D: DualNum<Primitive = f64> + Copy>(
    mijk1: D,
    mijk2: D,
    etas: &[D],
    c: &[[f64; 3]],
) -> D {
    (0..c.len())
        .map(|i| etas[i] * (mijk2 * c[i][2] + mijk1 * c[i][1] + c[i][0]))
        .sum()
}

pub(super) struct Dipole;

impl Dipole {
    pub(super) fn helmholtz_energy_density<D: DualNum<Primitive = f64> + Copy>(
        &self,
        parameters: &FcSaftHomoPars,
        state: &StateHD<D>,
    ) -> D {
        let p = parameters;
        let ndipole = p.dipole_comp.len();

        let t_inv = state.temperature.inv();
        let eps_ij_t = p.e_k_ij.map(|v| t_inv * v);
        let sig_ij_3 = p.sigma_ij.map(|v| v.powi(3));
        let mu2_term: Vec<D> = p
            .dipole_comp
            .iter()
            .map(|&i| t_inv * sig_ij_3[(i, i)] * p.epsilon_k[i] * p.mu2[i])
            .collect();

        let rho = &state.partial_density;
        let d = p.hs_diameter(state.temperature);
        let [m, m_star] = p.m_values(&d);
        let eta = (0..d.len())
            .map(|i| rho[i] * m_star[i] * d[i].powi(3))
            .sum::<D>()
            * FRAC_PI_6;
        let eta2 = eta * eta;
        let etas = [D::one(), eta, eta2, eta2 * eta, eta2 * eta2];

        // mean segment numbers (limited to m=2)
        let clamp = |m: D| if m.re() > 2.0 { D::from(2.0) } else { m };
        let m: Vec<_> = p.dipole_comp.iter().map(|&i| clamp(m[i])).collect();

        let mut phi2 = D::zero();
        let mut phi3 = D::zero();
        for i in 0..ndipole {
            let di = p.dipole_comp[i];
            for j in i..ndipole {
                let dj = p.dipole_comp[j];
                let mij = (m[i] * m[j]).sqrt();
                let mij1 = (mij - 1.0) / mij;
                let mij2 = mij1 * (mij - 2.0) / mij;
                let c = if i == j { 1.0 } else { 2.0 };
                phi2 -= rho[di]
                    * rho[dj]
                    * mu2_term[i]
                    * mu2_term[j]
                    * pair_integral_ij(mij1, mij2, &etas, &AD, &BD, eps_ij_t[(di, dj)])
                    / sig_ij_3[(di, dj)]
                    * c;
                for k in j..ndipole {
                    let dk = p.dipole_comp[k];
                    let mijk = (m[i] * m[j] * m[k]).cbrt();
                    let mijk1 = (mijk - 1.0) / mijk;
                    let mijk2 = mijk1 * (mijk - 2.0) / mijk;
                    let c = if i == k {
                        1.0
                    } else if i == j || j == k {
                        3.0
                    } else {
                        6.0
                    };
                    phi3 -= rho[di] * rho[dj] * rho[dk] * mu2_term[i] * mu2_term[j] * mu2_term[k]
                        / (p.sigma_ij[(di, dj)] * p.sigma_ij[(di, dk)] * p.sigma_ij[(dj, dk)])
                        * triplet_integral_ijk(mijk1, mijk2, &etas, &CD)
                        * c;
                }
            }
        }
        phi2 *= PI;
        phi3 *= PI_SQ_43;
        let mut result = phi2 * phi2 / (phi2 - phi3);
        if result.re().is_nan() {
            result = phi2
        }
        result
    }
}
