use super::PcSaftPars;
use feos_core::StateHD;
use nalgebra::DMatrix;
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_3, PI};

pub const ALPHA: f64 = 1.1937350;

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

// Quadrupole parameters
pub const AQ: [[f64; 3]; 5] = [
    [1.237830788, 1.285410878, 1.794295401],
    [2.435503144, -11.46561451, 0.769510293],
    [1.633090469, 22.08689285, 7.264792255],
    [-1.611815241, 7.46913832, 94.48669892],
    [6.977118504, -17.19777208, -77.1484579],
];

pub const BQ: [[f64; 3]; 5] = [
    [0.454271755, -0.813734006, 6.868267516],
    [-4.501626435, 10.06402986, -5.173223765],
    [3.585886783, -10.87663092, -17.2402066],
    [0.0; 3],
    [0.0; 3],
];

pub const CQ: [[f64; 3]; 4] = [
    [-0.500043713, 2.000209381, 3.135827145],
    [6.531869153, -6.78386584, 7.247588801],
    [-16.01477983, 20.38324603, 3.075947834],
    [14.42597018, -10.89598394, 0.0],
];

// Dipole-Quadrupole parameters
pub const ADQ: [[f64; 3]; 4] = [
    [0.697094963, -0.673459279, 0.670340770],
    [-0.633554144, -1.425899106, -4.338471826],
    [2.945509028, 4.19441392, 7.234168360],
    [-1.467027314, 1.0266216, 0.0],
];

pub const BDQ: [[f64; 3]; 4] = [
    [-0.484038322, 0.67651011, -1.167560146],
    [1.970405465, -3.013867512, 2.13488432],
    [-2.118572671, 0.46742656, 0.0],
    [0.0; 3],
];

pub const CDQ: [[f64; 2]; 3] = [
    [0.795009692, -2.099579397],
    [3.386863396, -5.941376392],
    [0.475106328, -0.178820384],
];

pub const PI_SQ_43: f64 = 4.0 * PI * FRAC_PI_3;

pub struct MeanSegmentNumbers {
    pub mij1: DMatrix<f64>,
    pub mij2: DMatrix<f64>,
    pub mijk1: Vec<DMatrix<f64>>,
    pub mijk2: Vec<DMatrix<f64>>,
}

pub enum Multipole {
    Dipole,
    Quadrupole,
}

impl MeanSegmentNumbers {
    pub fn new(parameters: &PcSaftPars, polarity: Multipole) -> Self {
        let (npoles, comp) = match polarity {
            Multipole::Dipole => (parameters.ndipole, &parameters.dipole_comp),
            Multipole::Quadrupole => (parameters.nquadpole, &parameters.quadpole_comp),
        };

        let mut mij1 = DMatrix::zeros(npoles, npoles);
        let mut mij2 = DMatrix::zeros(npoles, npoles);
        let mut mijk1 = vec![DMatrix::zeros(npoles, npoles); npoles];
        let mut mijk2 = vec![DMatrix::zeros(npoles, npoles); npoles];
        for i in 0..npoles {
            let mi = parameters.m[comp[i]].min(2.0);
            for j in i..npoles {
                let mj = parameters.m[comp[j]].min(2.0);
                let mij = (mi * mj).sqrt();
                mij1[(i, j)] = (mij - 1.0) / mij;
                mij2[(i, j)] = mij1[(i, j)] * (mij - 2.0) / mij;
                for k in j..npoles {
                    let mk = parameters.m[comp[k]].min(2.0);
                    let mijk = (mi * mj * mk).cbrt();
                    mijk1[i][(j, k)] = (mijk - 1.0) / mijk;
                    mijk2[i][(j, k)] = mijk1[i][(j, k)] * (mijk - 2.0) / mijk;
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
    mij1: f64,
    mij2: f64,
    etas: &[D],
    a: &[[f64; 3]],
    b: &[[f64; 3]],
    eps_ij_t: D,
) -> D {
    (0..a.len())
        .map(|i| {
            etas[i]
                * (eps_ij_t * (b[i][0] + mij1 * b[i][1] + mij2 * b[i][2])
                    + (a[i][0] + mij1 * a[i][1] + mij2 * a[i][2]))
        })
        .sum()
}

fn triplet_integral_ijk<D: DualNum<f64> + Copy>(
    mijk1: f64,
    mijk2: f64,
    etas: &[D],
    c: &[[f64; 3]],
) -> D {
    (0..c.len())
        .map(|i| etas[i] * (c[i][0] + mijk1 * c[i][1] + mijk2 * c[i][2]))
        .sum()
}

fn triplet_integral_ijk_dq<D: DualNum<f64> + Copy>(mijk: f64, etas: &[D], c: &[[f64; 2]]) -> D {
    (0..c.len())
        .map(|i| etas[i] * (c[i][0] + mijk * c[i][1]))
        .sum()
}

/// Combinatorial factor of a pair term in the sum over $i \le j$.
fn pair_factor(i: usize, j: usize) -> f64 {
    if i == j { 1.0 } else { 2.0 }
}

/// Combinatorial factor of a triplet term in the sum over $i \le j \le k$.
fn triplet_factor(i: usize, j: usize, k: usize) -> f64 {
    if i == k {
        1.0
    } else if i == j || j == k {
        3.0
    } else {
        6.0
    }
}

/// Dipole contribution. Everything that only depends on the parameters is
/// precomputed in [Dipole::new].
pub struct Dipole {
    m: MeanSegmentNumbers,
    /// $c_{ij}\tilde\mu_i^2\tilde\mu_j^2/\sigma_{ij}^3$ for $i \le j$ (times $1/T^2$ at runtime)
    f2: DMatrix<f64>,
    /// $c_{ijk}\tilde\mu_i^2\tilde\mu_j^2\tilde\mu_k^2/(\sigma_{ij}\sigma_{ik}\sigma_{jk})$ for $i \le j \le k$ (times $1/T^3$ at runtime)
    f3: Vec<DMatrix<f64>>,
}

impl Dipole {
    pub fn new(parameters: &PcSaftPars) -> Self {
        let p = parameters;
        let n = p.ndipole;
        let mu2: Vec<f64> = p
            .dipole_comp
            .iter()
            .map(|&i| p.sigma[i].powi(3) * p.epsilon_k[i] * p.mu2[i])
            .collect();
        let mut f2 = DMatrix::zeros(n, n);
        let mut f3 = vec![DMatrix::zeros(n, n); n];
        for i in 0..n {
            let di = p.dipole_comp[i];
            for j in i..n {
                let dj = p.dipole_comp[j];
                f2[(i, j)] = pair_factor(i, j) * mu2[i] * mu2[j] / p.sigma_ij[(di, dj)].powi(3);
                for k in j..n {
                    let dk = p.dipole_comp[k];
                    f3[i][(j, k)] = triplet_factor(i, j, k) * mu2[i] * mu2[j] * mu2[k]
                        / (p.sigma_ij[(di, dj)] * p.sigma_ij[(di, dk)] * p.sigma_ij[(dj, dk)]);
                }
            }
        }
        Self {
            m: MeanSegmentNumbers::new(p, Multipole::Dipole),
            f2,
            f3,
        }
    }

    /// `eta` is the packing fraction $\zeta_3$, shared with the other contributions.
    #[inline]
    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        parameters: &PcSaftPars,
        state: &StateHD<D>,
        eta: D,
    ) -> D {
        let p = parameters;
        let t_inv = state.temperature.inv();
        let rho = &state.partial_density;
        let eta2 = eta * eta;
        let etas = [D::one(), eta, eta2, eta2 * eta, eta2 * eta2];

        let mut phi2 = D::zero();
        let mut phi3 = D::zero();
        for i in 0..p.ndipole {
            let di = p.dipole_comp[i];
            for j in i..p.ndipole {
                let dj = p.dipole_comp[j];
                phi2 -= rho[di]
                    * rho[dj]
                    * self.f2[(i, j)]
                    * pair_integral_ij(
                        self.m.mij1[(i, j)],
                        self.m.mij2[(i, j)],
                        &etas,
                        &AD,
                        &BD,
                        t_inv * p.e_k_ij[(di, dj)],
                    );
                for k in j..p.ndipole {
                    let dk = p.dipole_comp[k];
                    phi3 -= rho[di]
                        * rho[dj]
                        * rho[dk]
                        * self.f3[i][(j, k)]
                        * triplet_integral_ijk(
                            self.m.mijk1[i][(j, k)],
                            self.m.mijk2[i][(j, k)],
                            &etas,
                            &CD,
                        );
                }
            }
        }
        let t_inv2 = t_inv * t_inv;
        phi2 *= t_inv2 * PI;
        phi3 *= t_inv2 * t_inv * PI_SQ_43;
        let mut result = phi2 * phi2 / (phi2 - phi3);
        if result.re().is_nan() {
            result = phi2
        }
        result
    }
}

/// Quadrupole contribution. Everything that only depends on the parameters is
/// precomputed in [Quadrupole::new].
pub struct Quadrupole {
    m: MeanSegmentNumbers,
    /// $c_{ij}\tilde Q_i^2\tilde Q_j^2/\sigma_{ij}^7$ for $i \le j$ (times $1/T^2$ at runtime)
    f2: DMatrix<f64>,
    /// $c_{ijk}\tilde Q_i^2\tilde Q_j^2\tilde Q_k^2/(\sigma_{ij}\sigma_{ik}\sigma_{jk})^3$ for $i \le j \le k$ (times $1/T^3$ at runtime)
    f3: Vec<DMatrix<f64>>,
}

impl Quadrupole {
    pub fn new(parameters: &PcSaftPars) -> Self {
        let p = parameters;
        let n = p.nquadpole;
        let q2: Vec<f64> = p
            .quadpole_comp
            .iter()
            .map(|&i| p.sigma[i].powi(5) * p.epsilon_k[i] * p.q2[i])
            .collect();
        let mut f2 = DMatrix::zeros(n, n);
        let mut f3 = vec![DMatrix::zeros(n, n); n];
        for i in 0..n {
            let di = p.quadpole_comp[i];
            for j in i..n {
                let dj = p.quadpole_comp[j];
                f2[(i, j)] = pair_factor(i, j) * q2[i] * q2[j] / p.sigma_ij[(di, dj)].powi(7);
                for k in j..n {
                    let dk = p.quadpole_comp[k];
                    f3[i][(j, k)] = triplet_factor(i, j, k) * q2[i] * q2[j] * q2[k]
                        / (p.sigma_ij[(di, dj)] * p.sigma_ij[(di, dk)] * p.sigma_ij[(dj, dk)])
                            .powi(3);
                }
            }
        }
        Self {
            m: MeanSegmentNumbers::new(p, Multipole::Quadrupole),
            f2,
            f3,
        }
    }

    /// `eta` is the packing fraction $\zeta_3$, shared with the other contributions.
    #[inline]
    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        parameters: &PcSaftPars,
        state: &StateHD<D>,
        eta: D,
    ) -> D {
        let p = parameters;
        let t_inv = state.temperature.inv();
        let rho = &state.partial_density;
        let eta2 = eta * eta;
        let etas = [D::one(), eta, eta2, eta2 * eta, eta2 * eta2];

        let mut phi2 = D::zero();
        let mut phi3 = D::zero();
        for i in 0..p.nquadpole {
            let di = p.quadpole_comp[i];
            for j in i..p.nquadpole {
                let dj = p.quadpole_comp[j];
                phi2 -= rho[di]
                    * rho[dj]
                    * self.f2[(i, j)]
                    * pair_integral_ij(
                        self.m.mij1[(i, j)],
                        self.m.mij2[(i, j)],
                        &etas,
                        &AQ,
                        &BQ,
                        t_inv * p.e_k_ij[(di, dj)],
                    );
                for k in j..p.nquadpole {
                    let dk = p.quadpole_comp[k];
                    phi3 += rho[di]
                        * rho[dj]
                        * rho[dk]
                        * self.f3[i][(j, k)]
                        * triplet_integral_ijk(
                            self.m.mijk1[i][(j, k)],
                            self.m.mijk2[i][(j, k)],
                            &etas,
                            &CQ,
                        );
                }
            }
        }
        let t_inv2 = t_inv * t_inv;
        phi2 *= t_inv2 * (PI * 0.5625);
        phi3 *= t_inv2 * t_inv * (PI * PI * 0.5625);
        let mut result = phi2 * phi2 / (phi2 - phi3);
        if result.re().is_nan() {
            result = phi2
        }
        result
    }
}

/// Different combination rules used in the dipole-quadrupole contribution.
#[derive(Clone, Copy, PartialEq)]
pub enum DQVariants {
    DQ35,
    DQ44,
}

/// Dipole-quadrupole cross contribution. Everything that only depends on the
/// parameters (and the combination rule) is precomputed in [DipoleQuadrupole::new].
pub struct DipoleQuadrupole {
    mdq1: DMatrix<f64>,
    mdq2: DMatrix<f64>,
    mdqd: Vec<DMatrix<f64>>,
    mdqq: Vec<DMatrix<f64>>,
    /// pair prefactor for dipole $i$ and quadrupole $j$ (times $1/T^2$ at runtime)
    f2: DMatrix<f64>,
    /// triplet prefactor for dipole $i$, quadrupole $j$, dipole $k$ (times $1/T^3$ at runtime)
    f3d: Vec<DMatrix<f64>>,
    /// triplet prefactor for dipole $i$, quadrupoles $j$ and $k$, including $\alpha$ (times $1/T^3$ at runtime)
    f3q: Vec<DMatrix<f64>>,
}

impl DipoleQuadrupole {
    pub fn new(parameters: &PcSaftPars, variant: DQVariants) -> Self {
        let p = parameters;
        let (nd, nq) = (p.ndipole, p.nquadpole);
        let q2: Vec<f64> = p
            .quadpole_comp
            .iter()
            .map(|&i| p.sigma[i].powi(4) * p.epsilon_k[i] * p.q2[i])
            .collect();
        let mu2: Vec<f64> = p
            .dipole_comp
            .iter()
            .map(|&i| p.sigma[i].powi(4) * p.epsilon_k[i] * p.mu2[i])
            .collect();

        // mean segment numbers and prefactors
        let mut mdq1 = DMatrix::zeros(nd, nq);
        let mut mdq2 = DMatrix::zeros(nd, nq);
        let mut mdqd = vec![DMatrix::zeros(nq, nd); nd];
        let mut mdqq = vec![DMatrix::zeros(nq, nq); nd];
        let mut f2 = DMatrix::zeros(nd, nq);
        let mut f3d = vec![DMatrix::zeros(nq, nd); nd];
        let mut f3q = vec![DMatrix::zeros(nq, nq); nd];
        for i in 0..nd {
            let di = p.dipole_comp[i];
            let mi = p.m[di].min(2.0);
            for j in 0..nq {
                let qj = p.quadpole_comp[j];
                let mj = p.m[qj].min(2.0);
                let m = (mi * mj).sqrt();
                mdq1[(i, j)] = (m - 1.0) / m;
                mdq2[(i, j)] = mdq1[(i, j)] * (m - 2.0) / m;
                let mu2_q2 = match variant {
                    DQVariants::DQ35 => mu2[i] / p.sigma[di] * q2[j] * p.sigma[qj],
                    DQVariants::DQ44 => mu2[i] * q2[j],
                };
                f2[(i, j)] = mu2_q2 / p.sigma_ij[(di, qj)].powi(5);
                for k in 0..nd {
                    let dk = p.dipole_comp[k];
                    let mk = p.m[dk].min(2.0);
                    let m = (mi * mj * mk).cbrt();
                    mdqd[i][(j, k)] = (m - 1.0) / m;
                    f3d[i][(j, k)] = mu2[i] * q2[j] * mu2[k]
                        / (p.sigma_ij[(di, qj)] * p.sigma_ij[(di, dk)] * p.sigma_ij[(qj, dk)])
                            .powi(2);
                }
                for k in 0..nq {
                    let qk = p.quadpole_comp[k];
                    let mk = p.m[qk].min(2.0);
                    let m = (mi * mj * mk).cbrt();
                    mdqq[i][(j, k)] = (m - 1.0) / m;
                    f3q[i][(j, k)] = mu2[i] * q2[j] * q2[k]
                        / (p.sigma_ij[(di, qj)] * p.sigma_ij[(di, qk)] * p.sigma_ij[(qj, qk)])
                            .powi(2)
                        * ALPHA;
                }
            }
        }
        Self {
            mdq1,
            mdq2,
            mdqd,
            mdqq,
            f2,
            f3d,
            f3q,
        }
    }

    /// `eta` is the packing fraction $\zeta_3$, shared with the other contributions.
    #[inline]
    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        parameters: &PcSaftPars,
        state: &StateHD<D>,
        eta: D,
    ) -> D {
        let p = parameters;
        let t_inv = state.temperature.inv();
        let rho = &state.partial_density;
        let eta2 = eta * eta;
        let etas = [D::one(), eta, eta2, eta2 * eta, eta2 * eta2];

        let mut phi2 = D::zero();
        let mut phi3 = D::zero();
        for i in 0..p.ndipole {
            let di = p.dipole_comp[i];
            for j in 0..p.nquadpole {
                let qj = p.quadpole_comp[j];
                phi2 -= rho[di]
                    * rho[qj]
                    * self.f2[(i, j)]
                    * pair_integral_ij(
                        self.mdq1[(i, j)],
                        self.mdq2[(i, j)],
                        &etas,
                        &ADQ,
                        &BDQ,
                        t_inv * p.e_k_ij[(di, qj)],
                    );
                for k in 0..p.ndipole {
                    let dk = p.dipole_comp[k];
                    phi3 += rho[di]
                        * rho[qj]
                        * rho[dk]
                        * self.f3d[i][(j, k)]
                        * triplet_integral_ijk_dq(self.mdqd[i][(j, k)], &etas, &CDQ);
                }
                for k in 0..p.nquadpole {
                    let qk = p.quadpole_comp[k];
                    phi3 += rho[di]
                        * rho[qj]
                        * rho[qk]
                        * self.f3q[i][(j, k)]
                        * triplet_integral_ijk_dq(self.mdqq[i][(j, k)], &etas, &CDQ);
                }
            }
        }

        let t_inv2 = t_inv * t_inv;
        phi2 *= t_inv2 * (PI * 2.25);
        phi3 *= t_inv2 * t_inv * (PI * PI);
        let mut result = phi2 * phi2 / (phi2 - phi3);
        if result.re().is_nan() {
            result = phi2
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hard_sphere::HardSphere;
    use crate::pcsaft::PcSaftParameters;
    use crate::pcsaft::eos::dispersion::tests::a_disp;
    use crate::pcsaft::parameters::utils::{
        carbon_dioxide_parameters, dme_co2_parameters, dme_parameters,
    };
    use approx::assert_relative_eq;
    use feos_core::StateHD;
    use feos_core::parameter::IdentifierOption;
    use itertools::Itertools;
    use nalgebra::{DVector, dvector};

    fn eta(p: &PcSaftPars, s: &StateHD<f64>) -> f64 {
        HardSphere.helmholtz_energy_density_and_properties(p, s).1[3]
    }

    fn a_dipole(p: &PcSaftPars, s: &StateHD<f64>) -> f64 {
        Dipole::new(p).helmholtz_energy_density(p, s, eta(p, s))
    }

    fn a_quadrupole(p: &PcSaftPars, s: &StateHD<f64>) -> f64 {
        Quadrupole::new(p).helmholtz_energy_density(p, s, eta(p, s))
    }

    #[test]
    fn test_dipolar_contribution() {
        let dme = dme_parameters();
        let t = 350.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, &dvector![n]);
        let a = a_dipole(&dme, &s) * v;
        assert_relative_eq!(a, -1.40501033595417E-002, epsilon = 1e-6);
    }

    #[test]
    fn test_dipolar_contribution_mix() {
        let parameters = PcSaftPars::new(
            &PcSaftParameters::from_json(
                vec!["acetone", "butanal", "dimethyl ether"],
                "../../parameters/pcsaft/gross2006.json",
                None,
                IdentifierOption::Name,
            )
            .unwrap(),
        );
        let t = 350.0;
        let v = 1000.0;
        let s = StateHD::new(t, v, &dvector![1.0, 2.0, 3.0]);
        let a = a_dipole(&parameters, &s) * v;
        assert_relative_eq!(a, -1.4126308106201688, epsilon = 1e-10);
    }

    #[test]
    fn test_dipolar_contribution_order() {
        let a: Vec<_> = ["acetone", "butanal", "dimethyl ether"]
            .into_iter()
            .permutations(3)
            .zip([1.0, 2.0, 3.0].into_iter().permutations(3))
            .map(|(components, n)| {
                let parameters = PcSaftPars::new(
                    &PcSaftParameters::from_json(
                        components.clone(),
                        "../../parameters/pcsaft/gross2006.json",
                        None,
                        IdentifierOption::Name,
                    )
                    .unwrap(),
                );
                let t = 350.0;
                let v = 1000.0;
                let s = StateHD::new(t, v, &DVector::from(n));
                let a = a_dipole(&parameters, &s) * v;
                println!("{components:?}: {a}");
                a
            })
            .collect();
        for (a, b) in a.into_iter().tuple_windows() {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_quadrupolar_contribution() {
        let co2 = carbon_dioxide_parameters();
        let t = 350.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, &dvector![n]);
        let a = a_quadrupole(&co2, &s) * v;
        assert_relative_eq!(a, -4.38559558854186E-002, epsilon = 1e-6);
    }

    #[test]
    fn test_quadrupolar_contribution_mix() {
        let parameters = PcSaftPars::new(
            &PcSaftParameters::from_json(
                vec!["carbon dioxide", "chlorine", "ethylene"],
                "../../parameters/pcsaft/gross2005_literature.json",
                None,
                IdentifierOption::Name,
            )
            .unwrap(),
        );
        let t = 350.0;
        let v = 1000.0;
        let s = StateHD::new(t, v, &dvector![1.0, 2.0, 3.0]);
        let a = a_quadrupole(&parameters, &s) * v;
        assert_relative_eq!(a, -0.2748017468669352, epsilon = 1e-10);
    }

    #[test]
    fn test_quadrupolar_contribution_order() {
        let a: Vec<_> = ["carbon dioxide", "chlorine", "ethylene"]
            .into_iter()
            .permutations(3)
            .zip([1.0, 2.0, 3.0].into_iter().permutations(3))
            .map(|(components, n)| {
                let parameters = PcSaftPars::new(
                    &PcSaftParameters::from_json(
                        components.clone(),
                        "../../parameters/pcsaft/gross2005_literature.json",
                        None,
                        IdentifierOption::Name,
                    )
                    .unwrap(),
                );
                let t = 350.0;
                let v = 1000.0;
                let s = StateHD::new(t, v, &DVector::from(n));
                let a = a_quadrupole(&parameters, &s) * v;
                println!("{components:?}: {a}");
                a
            })
            .collect();
        for (a, b) in a.into_iter().tuple_windows() {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dipolar_quadrupolar_contribution() {
        let mix = dme_co2_parameters();
        // let dpqp = DipoleQuadrupole {
        //     parameters: Arc::new(dme_co2_parameters()),
        //     variant: DQVariants::DQ35,
        // };
        let t = 350.0;
        let v = 1000.0;
        let n_dme = 1.0;
        let n_co2 = 1.0;
        let n = n_dme + n_co2;
        let s = StateHD::new(t, v / n, &(dvector![n_dme, n_co2] / n));
        // let a_dpqp = dpqp.helmholtz_energy_density(&s)*v;
        let a_disp = a_disp(&mix, &s) * v;
        let a_qp = a_quadrupole(&mix, &s) * v;
        let a_dp = a_dipole(&mix, &s) * v;
        assert_relative_eq!(a_disp, -1.6283622072860044, epsilon = 1e-6);
        assert_relative_eq!(a_dp, -1.35361827881345E-002, epsilon = 1e-6);
        assert_relative_eq!(a_qp, -4.20168059082731E-002, epsilon = 1e-6);
        // assert_relative_eq!(a_dpqp, -2.2316252638709004E-002, epsilon = 1e-6);
    }
}
