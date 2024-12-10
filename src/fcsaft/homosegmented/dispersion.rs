use super::FcSaftHomoParameters;
use crate::fcsaft::heterosegmented::dispersion::{
    Dispersion as DispersionHetero, A0, A1, A2, B0, B1, B2,
};
use crate::hard_sphere::HardSphereProperties;
use feos_core::StateHD;
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_6, PI};
use std::fmt;
use std::sync::Arc;

pub struct Dispersion {
    pub parameters: Arc<FcSaftHomoParameters>,
}

impl Dispersion {
    pub fn new(parameters: &Arc<FcSaftHomoParameters>) -> Self {
        Self {
            parameters: parameters.clone(),
        }
    }

    pub fn helmholtz_energy<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> D {
        // auxiliary variables
        let n = self.parameters.s.len();
        let p = &self.parameters;
        let rho = &state.partial_density;

        // temperature dependent segment radius
        let d = p.hs_diameter(state.temperature);

        // packing fraction
        let [m, m_star] = self.parameters.m_values(&d);
        let eta = (rho * &m_star * &d * &d * &d).sum() * FRAC_PI_6;

        // mean segment number
        let m_bar = (&state.molefracs * &m).sum();

        // mixture densities, crosswise interactions of all segments on all chains
        let mut rho1mix = D::zero();
        let mut rho2mix = D::zero();
        for i in 0..n {
            for j in 0..n {
                let eps_ij = state.temperature.recip() * p.epsilon_k_ij[(i, j)];
                let sigma_ij = p.sigma_ij[[i, j]].powi(3);
                rho1mix += rho[i] * rho[j] * m[i] * m[j] * eps_ij * sigma_ij;
                rho2mix += rho[i] * rho[j] * m[i] * m[j] * eps_ij * eps_ij * sigma_ij;
            }
        }

        // I1, I2 and C1
        let mut i1 = D::zero();
        let mut i2 = D::zero();
        let mut eta_i = D::one();
        for i in 0..=6 {
            i1 += ((m_bar - 1.0) / m_bar * ((m_bar - 2.0) / m_bar * A2[i] + A1[i]) + A0[i]) * eta_i;
            i2 += ((m_bar - 1.0) / m_bar * ((m_bar - 2.0) / m_bar * B2[i] + B1[i]) + B0[i]) * eta_i;
            eta_i *= eta;
        }
        let c1 = DispersionHetero::compressibility_term(p.s[0] as usize, m[0], m_star[0], eta);

        // Helmholtz energy
        (-rho1mix * i1 * 2.0 - rho2mix * m_bar * c1 * i2) * PI * state.volume
    }
}

impl fmt::Display for Dispersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dispersion")
    }
}
