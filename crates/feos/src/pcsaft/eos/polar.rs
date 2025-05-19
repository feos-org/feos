use super::PcSaftParameters;
use feos_core::StateHD;
use ndarray::prelude::*;
use num_dual::DualNum;
use quantity::{JOULE, KB, KELVIN};
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
    pub mij1: Array2<f64>,
    pub mij2: Array2<f64>,
    pub mijk1: Array3<f64>,
    pub mijk2: Array3<f64>,
}

impl MeanSegmentNumbers {
    pub fn new(m: &Array1<f64>, npoles: usize, comp: &Array1<usize>) -> Self {
        let mut mij1 = Array2::zeros((npoles, npoles));
        let mut mij2 = Array2::zeros((npoles, npoles));
        let mut mijk1 = Array3::zeros((npoles, npoles, npoles));
        let mut mijk2 = Array3::zeros((npoles, npoles, npoles));
        for i in 0..npoles {
            let mi = m[comp[i]].min(2.0);
            for j in i..npoles {
                let mj = m[comp[j]].min(2.0);
                let mij = (mi * mj).sqrt();
                mij1[[i, j]] = (mij - 1.0) / mij;
                mij2[[i, j]] = mij1[[i, j]] * (mij - 2.0) / mij;
                for k in j..npoles {
                    let mk = m[comp[k]].min(2.0);
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

pub struct Polar {
    m: Array1<f64>,
    sigma: Array1<f64>,
    sigma_ij: Array2<f64>,
    e_k_ij: Array2<f64>,
    ndipole: usize,
    dipole_comp: Array1<usize>,
    nquadpole: usize,
    quadpole_comp: Array1<usize>,
    mu2_term: Array1<f64>,
    q2_term: Array1<f64>,
    m_dipole: MeanSegmentNumbers,
    m_quadpole: MeanSegmentNumbers,
}

impl Polar {
    pub fn new(parameters: &PcSaftParameters) -> Self {
        // pure parameters
        let [m, sigma, epsilon_k, mu, q] =
            parameters.collate(|r| [r.m, r.sigma, r.epsilon_k, r.mu, r.q]);

        // dipoles and quadrupoles
        let mu2_term = &mu * &mu / &m * 1e-19 * (JOULE / KELVIN / KB).into_value();
        let q2_term = &q * &q / &m * 1e-19 * (JOULE / KELVIN / KB).into_value();
        let dipole_comp: Array1<usize> = mu2_term
            .iter()
            .enumerate()
            .filter_map(|(i, &mu2)| (mu2.abs() > 0.0).then_some(i))
            .collect();
        let ndipole = dipole_comp.len();
        let quadpole_comp: Array1<usize> = q2_term
            .iter()
            .enumerate()
            .filter_map(|(i, &q2)| (q2.abs() > 0.0).then_some(i))
            .collect();
        let nquadpole = quadpole_comp.len();

        // binary interaction parameters
        let n = parameters.pure_records.len();
        let mut sigma_ij = Array::zeros((n, n));
        let mut e_k_ij = Array::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                e_k_ij[[i, j]] = (epsilon_k[i] * epsilon_k[j]).sqrt();
                sigma_ij[[i, j]] = 0.5 * (sigma[i] + sigma[j]);
            }
        }

        // Mean segment numbers
        let m_dipole = MeanSegmentNumbers::new(&m, ndipole, &dipole_comp);
        let m_quadpole = MeanSegmentNumbers::new(&m, nquadpole, &quadpole_comp);

        Self {
            m,
            sigma,
            sigma_ij,
            e_k_ij,
            ndipole,
            dipole_comp,
            nquadpole,
            quadpole_comp,
            mu2_term,
            q2_term,
            m_dipole,
            m_quadpole,
        }
    }

    #[inline]
    pub fn helmholtz_energy_dd<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
        eta: D,
    ) -> Option<D> {
        if self.ndipole == 0 {
            return None;
        }

        let t_inv = state.temperature.inv();
        let eps_ij_t = self.e_k_ij.mapv(|v| t_inv * v);
        let sig_ij_3 = self.sigma_ij.mapv(|v| v.powi(3));
        let mu2_term: Array1<D> = self
            .dipole_comp
            .iter()
            .map(|&i| t_inv * self.mu2_term[i])
            .collect();

        let rho = &state.partial_density;
        let eta2 = eta * eta;
        let etas = [D::one(), eta, eta2, eta2 * eta, eta2 * eta2];

        let mut phi2 = D::zero();
        let mut phi3 = D::zero();
        for i in 0..self.ndipole {
            let di = self.dipole_comp[i];
            for j in i..self.ndipole {
                let dj = self.dipole_comp[j];
                let c = if i == j { 1.0 } else { 2.0 };
                phi2 -= rho[di]
                    * rho[dj]
                    * mu2_term[i]
                    * mu2_term[j]
                    * pair_integral_ij(
                        self.m_dipole.mij1[[i, j]],
                        self.m_dipole.mij2[[i, j]],
                        &etas,
                        &AD,
                        &BD,
                        eps_ij_t[[di, dj]],
                    )
                    / sig_ij_3[[di, dj]]
                    * c;
                for k in j..self.ndipole {
                    let dk = self.dipole_comp[k];
                    let c = if i == k {
                        1.0
                    } else if i == j || j == k {
                        3.0
                    } else {
                        6.0
                    };
                    phi3 -= rho[di] * rho[dj] * rho[dk] * mu2_term[i] * mu2_term[j] * mu2_term[k]
                        / (self.sigma_ij[[di, dj]]
                            * self.sigma_ij[[di, dk]]
                            * self.sigma_ij[[dj, dk]])
                        * triplet_integral_ijk(
                            self.m_dipole.mijk1[[i, j, k]],
                            self.m_dipole.mijk2[[i, j, k]],
                            &etas,
                            &CD,
                        )
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
        Some(result)
    }

    #[inline]
    pub fn helmholtz_energy_qq<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
        eta: D,
    ) -> Option<D> {
        if self.nquadpole == 0 {
            return None;
        }

        let t_inv = state.temperature.inv();
        let eps_ij_t = self.e_k_ij.mapv(|v| t_inv * v);
        let sig_ij_3 = self.sigma_ij.mapv(|v| v.powi(3));
        let q2_term: Array1<D> = self
            .quadpole_comp
            .iter()
            .map(|&i| t_inv * self.q2_term[i])
            .collect();

        let rho = &state.partial_density;
        let eta2 = eta * eta;
        let etas = [D::one(), eta, eta2, eta2 * eta, eta2 * eta2];

        let mut phi2 = D::zero();
        let mut phi3 = D::zero();
        for i in 0..self.nquadpole {
            let di = self.quadpole_comp[i];
            for j in i..self.nquadpole {
                let dj = self.quadpole_comp[j];
                let c = if i == j { 1.0 } else { 2.0 };
                phi2 -= (rho[di]
                    * rho[dj]
                    * q2_term[i]
                    * q2_term[j]
                    * pair_integral_ij(
                        self.m_quadpole.mij1[[i, j]],
                        self.m_quadpole.mij2[[i, j]],
                        &etas,
                        &AQ,
                        &BQ,
                        eps_ij_t[[di, dj]],
                    ))
                    / self.sigma_ij[[di, di]].powi(7)
                    * c;
                for k in j..self.nquadpole {
                    let dk = self.quadpole_comp[k];
                    let c = if i == k {
                        1.0
                    } else if i == j || j == k {
                        3.0
                    } else {
                        6.0
                    };
                    phi3 += rho[di] * rho[dj] * rho[dk] * q2_term[i] * q2_term[j] * q2_term[k]
                        / (sig_ij_3[[di, dj]] * sig_ij_3[[di, dk]] * sig_ij_3[[dj, dk]])
                        * triplet_integral_ijk(
                            self.m_quadpole.mijk1[[i, j, k]],
                            self.m_quadpole.mijk2[[i, j, k]],
                            &etas,
                            &CQ,
                        )
                        * c;
                }
            }
        }

        phi2 *= PI * 0.5625;
        phi3 *= PI * PI * 0.5625;
        let mut result = phi2 * phi2 / (phi2 - phi3) * state.volume;
        if result.re().is_nan() {
            result = phi2 * state.volume
        }
        Some(result)
    }
}

/// Different combination rules used in the dipole-quadrupole contribution.
#[derive(Clone, Copy, PartialEq)]
pub enum DQVariants {
    DQ35,
    DQ44,
}

impl Polar {
    #[inline]
    pub fn helmholtz_energy_dq<D: DualNum<f64> + Copy>(
        &self,
        variant: DQVariants,
        state: &StateHD<D>,
        eta: D,
    ) -> Option<D> {
        if self.ndipole * self.nquadpole == 0 {
            return None;
        }

        let t_inv = state.temperature.inv();
        let eps_ij_t = self.e_k_ij.mapv(|v| t_inv * v);

        let q2_term: Array1<D> = self
            .quadpole_comp
            .iter()
            .map(|&i| t_inv * self.sigma[i] * self.q2_term[i])
            .collect();
        let mu2_term: Array1<D> = self
            .dipole_comp
            .iter()
            .map(|&i| t_inv * self.mu2_term[i] / self.sigma[i])
            .collect();

        let rho = &state.partial_density;
        let eta2 = eta * eta;
        let etas = [D::one(), eta, eta2, eta2 * eta, eta2 * eta2];

        // mean segment number
        let mut mdq1 = Array2::zeros((self.ndipole, self.nquadpole));
        let mut mdq2 = Array2::zeros((self.ndipole, self.nquadpole));
        let mut mdqd = Array3::zeros((self.ndipole, self.nquadpole, self.ndipole));
        let mut mdqq = Array3::zeros((self.ndipole, self.nquadpole, self.nquadpole));
        for i in 0..self.ndipole {
            let di = self.dipole_comp[i];
            let mi = self.m[di].min(2.0);
            for j in 0..self.nquadpole {
                let qj = self.quadpole_comp[j];
                let mj = self.m[qj].min(2.0);
                let m = (mi * mj).sqrt();
                mdq1[[i, j]] = (m - 1.0) / m;
                mdq2[[i, j]] = mdq1[[i, j]] * (m - 2.0) / m;
                for k in 0..self.ndipole {
                    let dk = self.dipole_comp[k];
                    let mk = self.m[dk].min(2.0);
                    let m = (mi * mj * mk).cbrt();
                    mdqd[[i, j, k]] = (m - 1.0) / m;
                }
                for k in 0..self.nquadpole {
                    let qk = self.quadpole_comp[k];
                    let mk = self.m[qk].min(2.0);
                    let m = (mi * mj * mk).cbrt();
                    mdqq[[i, j, k]] = (m - 1.0) / m;
                }
            }
        }

        let mut phi2 = D::zero();
        let mut phi3 = D::zero();
        for i in 0..self.ndipole {
            for j in 0..self.nquadpole {
                let di = self.dipole_comp[i];
                let qj = self.quadpole_comp[j];
                let mu2_q2_term = match variant {
                    DQVariants::DQ35 => mu2_term[i] / self.sigma[di] * q2_term[j] * self.sigma[qj],
                    DQVariants::DQ44 => mu2_term[i] * q2_term[j],
                };
                phi2 -= rho[di] * rho[qj] * mu2_q2_term / self.sigma_ij[[di, qj]].powi(5)
                    * pair_integral_ij(
                        mdq1[[i, j]],
                        mdq2[[i, j]],
                        &etas,
                        &ADQ,
                        &BDQ,
                        eps_ij_t[[di, qj]],
                    );
                for k in 0..self.ndipole {
                    let dk = self.dipole_comp[k];
                    phi3 += rho[di] * rho[qj] * rho[dk] * mu2_term[i] * q2_term[j] * mu2_term[k]
                        / (self.sigma_ij[[di, qj]]
                            * self.sigma_ij[[di, dk]]
                            * self.sigma_ij[[qj, dk]])
                        .powi(2)
                        * triplet_integral_ijk_dq(mdqd[[i, j, k]], &etas, &CDQ);
                }
                for k in 0..self.nquadpole {
                    let qk = self.quadpole_comp[k];
                    phi3 += rho[di] * rho[qj] * rho[qk] * mu2_term[i] * q2_term[j] * q2_term[k]
                        / (self.sigma_ij[[di, qj]]
                            * self.sigma_ij[[di, qk]]
                            * self.sigma_ij[[qj, qk]])
                        .powi(2)
                        * ALPHA
                        * triplet_integral_ijk_dq(mdqq[[i, j, k]], &etas, &CDQ);
                }
            }
        }

        phi2 *= PI * 2.25;
        phi3 *= PI * PI;
        let mut result = phi2 * phi2 / (phi2 - phi3) * state.volume;
        if result.re().is_nan() {
            result = phi2 * state.volume
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hard_sphere::HardSphereProperties;
    use crate::pcsaft::PcSaft;
    use crate::pcsaft::eos::dispersion::Dispersion;
    use crate::pcsaft::parameters::utils::{
        carbon_dioxide_parameters, dme_co2_parameters, dme_parameters,
    };
    use approx::assert_relative_eq;
    use feos_core::StateHD;
    use feos_core::parameter::IdentifierOption;

    #[test]
    fn test_dipolar_contribution() {
        let parameters = dme_parameters();
        let polar = Polar::new(&parameters);
        let eos = PcSaft::new(parameters);
        let t = 350.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, arr1(&[n]));
        let [eta] = eos.zeta(t, &s.partial_density, [3]);
        let a = polar.helmholtz_energy_dd(&s, eta).unwrap();
        assert_relative_eq!(a, -1.40501033595417E-002, epsilon = 1e-6);
    }

    #[test]
    fn test_dipolar_contribution_mix() {
        let parameters = PcSaftParameters::from_json(
            vec!["acetone", "butanal", "dimethyl ether"],
            "../../parameters/pcsaft/gross2006.json",
            None,
            IdentifierOption::Name,
        )
        .unwrap();
        let polar = Polar::new(&parameters);
        let eos = PcSaft::new(parameters);
        let t = 350.0;
        let v = 1000.0;
        let n = [1.0, 2.0, 3.0];
        let s = StateHD::new(t, v, arr1(&n));
        let [eta] = eos.zeta(t, &s.partial_density, [3]);
        let a = polar.helmholtz_energy_dd(&s, eta).unwrap();
        assert_relative_eq!(a, -1.4126308106201688, epsilon = 1e-10);
    }

    #[test]
    fn test_quadrupolar_contribution() {
        let parameters = carbon_dioxide_parameters();
        let polar = Polar::new(&parameters);
        let eos = PcSaft::new(parameters);
        let t = 350.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, arr1(&[n]));
        let [eta] = eos.zeta(t, &s.partial_density, [3]);
        let a = polar.helmholtz_energy_qq(&s, eta).unwrap();
        assert_relative_eq!(a, -4.38559558854186E-002, epsilon = 1e-6);
    }

    #[test]
    fn test_quadrupolar_contribution_mix() {
        let parameters = PcSaftParameters::from_json(
            vec!["carbon dioxide", "chlorine", "ethylene"],
            "../../parameters/pcsaft/gross2005_literature.json",
            None,
            IdentifierOption::Name,
        )
        .unwrap();
        let polar = Polar::new(&parameters);
        let eos = PcSaft::new(parameters);
        let t = 350.0;
        let v = 1000.0;
        let n = [1.0, 2.0, 3.0];
        let s = StateHD::new(t, v, arr1(&n));
        let [eta] = eos.zeta(t, &s.partial_density, [3]);
        let a = polar.helmholtz_energy_qq(&s, eta).unwrap();
        assert_relative_eq!(a, -0.327493924806138, epsilon = 1e-10);
    }

    #[test]
    fn test_dipolar_quadrupolar_contribution() {
        let parameters = dme_co2_parameters();
        let dispersion = Dispersion::new(&parameters);
        let polar = Polar::new(&parameters);
        let eos = PcSaft::new(parameters);

        // let dpqp = DipoleQuadrupole {
        //     parameters: Arc::new(dme_co2_parameters()),
        //     variant: DQVariants::DQ35,
        // };
        let t = 350.0;
        let v = 1000.0;
        let n_dme = 1.0;
        let n_co2 = 1.0;
        let s = StateHD::new(t, v, arr1(&[n_dme, n_co2]));
        // let a_dpqp = dpqp.helmholtz_energy(&s);
        let d = eos.hs_diameter(t);
        let [eta] = eos.zeta(t, &s.partial_density, [3]);
        let a_disp = dispersion.helmholtz_energy(&s, &d);
        let a_qp = polar.helmholtz_energy_qq(&s, eta).unwrap();
        let a_dp = polar.helmholtz_energy_dd(&s, eta).unwrap();
        assert_relative_eq!(a_disp, -1.6283622072860044, epsilon = 1e-6);
        assert_relative_eq!(a_dp, -1.35361827881345E-002, epsilon = 1e-6);
        assert_relative_eq!(a_qp, -4.20168059082731E-002, epsilon = 1e-6);
        // assert_relative_eq!(a_dpqp, -2.2316252638709004E-002, epsilon = 1e-6);
    }
}
