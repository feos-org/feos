use super::FcSaftParameters;
use crate::hard_sphere::HardSphereProperties;
use feos_core::StateHD;
use num_dual::{Dual2, DualNum};
use std::f64::consts::{FRAC_PI_6, PI};
use std::fmt;
use std::sync::Arc;

pub const A0: [f64; 7] = [
    0.91056314451539,
    0.63612814494991,
    2.68613478913903,
    -26.5473624914884,
    97.7592087835073,
    -159.591540865600,
    91.2977740839123,
];
pub const A1: [f64; 7] = [
    -0.3025480264245456,
    0.15339799779908292,
    -3.5029950096342177,
    22.47266594147346,
    -58.73033629483515,
    78.87949764791878,
    -37.12157652601552,
];
pub const A2: [f64; 7] = [
    -0.23759723527983204,
    0.8672841965637575,
    -0.4037132968839511,
    -2.7238773379761865,
    -3.1303065099444667,
    13.834146355250077,
    -9.672727731662953,
];
pub const B0: [f64; 7] = [
    0.72409469413165,
    2.23827918609380,
    -4.00258494846342,
    -21.00357681484648,
    26.8556413626615,
    206.5513384066188,
    -355.60235612207947,
];
pub const B1: [f64; 7] = [
    -0.6986072717026419,
    1.6995054516296422,
    4.89255775397313,
    -15.493956797798422,
    211.93936299481138,
    -145.64449979151843,
    -181.72427385912198,
];
pub const B2: [f64; 7] = [
    0.398396810657585,
    0.7442315721787379,
    -8.155881160387127,
    22.70620390381875,
    -34.92425648309291,
    102.98859758589056,
    -26.704728265535547,
];

pub struct Dispersion {
    parameters: Arc<FcSaftParameters>,
    a1: [f64; 7],
    a2: [f64; 7],
    b1: [f64; 7],
    b2: [f64; 7],
}

impl Dispersion {
    pub fn new(parameters: &Arc<FcSaftParameters>, model_params: Option<[[f64; 7]; 4]>) -> Self {
        let [a1, a2, b1, b2] = model_params.unwrap_or([A1, A2, B1, B2]);
        Self {
            parameters: parameters.clone(),
            a1,
            a2,
            b1,
            b2,
        }
    }

    pub fn helmholtz_energy<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> D {
        // auxiliary variables
        let p = &self.parameters;
        let n = p.sigma.len();
        let rho = &state.partial_density;

        // packing fraction
        let diameter = p.hs_diameter(state.temperature);
        let [a, v] = p.geometry_coefficients(&diameter);
        let eta: D = diameter
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                state.partial_density[p.component_index[i]] * FRAC_PI_6 * v[i] * d * d * d
            })
            .sum();

        // mean segment numbers
        let m = a.sum();
        let m1 = v.sum();

        // mixture densities, crosswise interactions of all segments on all chains
        let mut rho1mix = D::zero();
        let mut rho2mix = D::zero();
        for i in 0..n {
            for j in 0..n {
                let eps_ij = state.temperature.recip() * p.epsilon_k_ij[(i, j)];
                let sigma_ij = p.sigma_ij[[i, j]].powi(3);
                let rho_ij = rho[p.component_index[i]] * rho[p.component_index[j]];
                rho1mix += rho_ij * a[i] * a[j] * eps_ij * sigma_ij;
                rho2mix += rho_ij * a[i] * a[j] * eps_ij * eps_ij * sigma_ij;
            }
        }

        // I1, I2 and C1
        let mut i1 = D::zero();
        let mut i2 = D::zero();
        let mut eta_i = D::one();
        for i in 0..=6 {
            i1 += ((m - 1.0) / m * ((m - 2.0) / m * self.a2[i] + self.a1[i]) + A0[i]) * eta_i;
            i2 += ((m - 1.0) / m * ((m - 2.0) / m * self.b2[i] + self.b1[i]) + B0[i]) * eta_i;
            eta_i *= eta;
        }
        let c1 = Self::compressibility_term(n, m, m1, eta);

        // Helmholtz energy
        (-rho1mix * i1 * 2.0 - rho2mix * m * c1 * i2) * PI * state.volume
    }
}

impl fmt::Display for Dispersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dispersion")
    }
}

impl Dispersion {
    pub(crate) fn compressibility_term<D: DualNum<f64> + Copy>(s: usize, m: D, m1: D, eta: D) -> D {
        match s {
            1 => Self::compressibility_term_monomer(eta),
            _ => Self::compressibility_term_chain(D::from(s as f64), m, m1, eta),
        }
    }

    pub(crate) fn compressibility_term_chain<D: DualNum<f64> + Copy>(
        s: D,
        m: D,
        m1: D,
        eta: D,
    ) -> D {
        let eta_dual: Dual2<_, f64> = Dual2::from_re(eta).derivative();
        let s: Dual2<_, f64> = Dual2::from_re(s);
        let m: Dual2<_, f64> = Dual2::from_re(m);
        let m1: Dual2<_, f64> = Dual2::from_re(m1);
        let eta_m1 = -(eta_dual - 1.0).recip();
        let x = (m - 1.0) / (s - 1.0) * m / m1 * eta_dual * eta_m1;
        let a = eta_dual * eta_m1 * 3.0 * m * m / m1
            + eta_dual * eta_m1 * eta_m1 * m * m * m / (m1 * m1)
            + (m * m * m / (m1 * m1) - s) * (-eta_dual).ln_1p()
            - (eta_m1 * (x * x * 0.5 + x * 1.5 + 1.0)).ln() * (s - 1.0);
        (a.v2 * eta * eta + a.v1 * eta * 2.0 + 1.0).recip()
    }

    pub(crate) fn compressibility_term_monomer<D: DualNum<f64> + Copy>(eta: D) -> D {
        let eta_dual: Dual2<_, f64> = Dual2::from_re(eta).derivative();
        let eta_m1 = -(eta_dual - 1.0).recip();
        let a = eta_dual * eta_m1 * 3.0 + eta_dual * eta_m1 * eta_m1;
        (a.v2 * eta * eta + a.v1 * eta * 2.0 + 1.0).recip()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::fcsaft::FcSaft;
    use feos_core::parameter::Identifier;
    use feos_core::{EosResult, State};
    use ndarray::arr1;
    use quantity::{KELVIN, METER, MOL};
    use typenum::N3;

    #[test]
    fn test_compressibility_term() {
        let eta = 0.3;
        let m = 2.5;
        // PC-SAFT C1
        let c1_pcsaft = (m * (eta * 8.0 - eta.powi(2) * 2.0) / (eta - 1.0).powi(4)
            - (m - 1.0)
                * (eta * 20.0 - eta.powi(2) * 27.0 + eta.powi(3) * 12.0 - eta.powi(4) * 2.0)
                / ((eta - 1.0) * (eta - 2.0)).powi(2)
            + 1.0)
            .recip();
        let c1_fcsaft = Dispersion::compressibility_term_chain(m, m, m, eta);
        assert_eq!(c1_fcsaft, c1_pcsaft)
    }

    #[test]
    fn test_dispersion() -> EosResult<()> {
        let id = Identifier::default();
        let params = Arc::new(FcSaftParameters::new_pure_homosegmented(
            id, 2, 3.0, 150.0, 1.5, 20.0, None,
        )?);
        let mut model_params = [[0.0; 7]; 4];
        model_params[0][0] = 1e-8;
        let disp = Dispersion::new(&params, Some(model_params));
        let state = StateHD::new(100.0, 1000.0, arr1(&[1.0]));
        println!("{}", disp.helmholtz_energy(&state));
        let state = State::new_pure(
            &Arc::new(FcSaft::new(params)),
            100.0 * KELVIN,
            0.1 * MOL * METER.powi::<N3>(),
        )?;
        println!("{}", state.total_molar_weight());
        Ok(())
    }
}
