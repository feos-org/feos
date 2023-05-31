use super::attractive_perturbation_wca::one_fluid_properties;
use super::hard_sphere_wca::{
    diameter_wca, dimensionless_diameter_q_wca, WCA_CONSTANTS_ETA_A_UVB3, WCA_CONSTANTS_ETA_B_UVB3,
};
use crate::uvtheory::parameters::*;
use feos_core::{HelmholtzEnergyDual, StateHD};
use ndarray::Array1;
use num_dual::DualNum;
use std::{f64::consts::PI, fmt, sync::Arc};

const C_WCA: [[f64; 6]; 6] = [
    [
        -0.2622378162,
        0.6585817423,
        5.5318022309,
        0.6902354794,
        -3.6825190645,
        -1.7263213318,
    ],
    [
        -0.1899241690,
        -0.5555205158,
        9.1361398949,
        0.7966155658,
        -6.1413017045,
        4.9553415149,
    ],
    [
        0.1169786415,
        -0.2216804790,
        -2.0470861617,
        -0.3742261343,
        0.9568416381,
        10.1401796764,
    ],
    [
        0.5852642702,
        2.0795520346,
        19.0711829725,
        -2.3403594600,
        2.5833371420,
        432.3858674425,
    ],
    [
        -0.6084232211,
        -7.2376034572,
        19.0412933614,
        3.2388986513,
        75.4442555789,
        -588.3837110653,
    ],
    [
        0.0512327656,
        6.6667943569,
        47.1109947616,
        -0.5011125797,
        -34.8918383146,
        189.5498636006,
    ],
];

// Constants for delta B2 RSAP
const C_B2_RSAP: [[f64; 4]; 4] = [
    [-0.063550989, 6.206829830, -37.45829549, 40.72849774],
    [1.519053409, 13.14989643, 85.35058674, 374.1906360],
    [0.693456220, 9.459946180, -53.28984218, 315.8199084],
    [0.007492596, 0.546171170, 7.979562575, -119.6126395],
];

// // Constants for B3 Model for Mie nu-6 fluids
const K_LJ_B3: [f64; 16] = [
    -3.9806, 79.565, 0.5489, 5.3632, 1.4245, 57.292, 0.0, 1.0031, -39.755, -81.213, 0.6987, 30.156,
    -23.692, -85.006, 0.7762, 12.798,
];

const P_B3: [f64; 4] = [0.80844, -0.09541, 0.47525, -2.83283];
const L_B3: [f64; 4] = [4.9485, -21.3, 7.0, 3.2162];
const M_B3: [f64; 4] = [0.11853, 0.078556, -0.55039, 0.009163];

const CU_WCA: [f64; 8] = [26.454, 1.8045, 1.7997, 161.96, 11.605, 12., 0.4, 2.0];

#[derive(Debug, Clone)]
pub struct AttractivePerturbationUVB3 {
    pub parameters: Arc<UVParameters>,
}

impl fmt::Display for AttractivePerturbationUVB3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Attractive Perturbation")
    }
}

impl<D: DualNum<f64> + Copy> HelmholtzEnergyDual<D> for AttractivePerturbationUVB3 {
    /// Helmholtz energy for attractive perturbation
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let p = &self.parameters;
        let t = state.temperature;
        let density = state.partial_density.sum();
        let d = diameter_wca(p, t);
        let x = &state.molefracs;
        // vdw effective one fluid properties
        let (rep_x, att_x, sigma_x, weighted_sigma3_ij, epsilon_k_x, d_x) =
            one_fluid_properties(p, x, t);
        let t_x = state.temperature / epsilon_k_x;
        let rho_x = density * sigma_x.powi(3);
        let rm_x = (rep_x / att_x).powd((rep_x - att_x).recip());
        let mean_field_constant_x = mean_field_constant(rep_x, att_x, rm_x);
        let q_vdw = dimensionless_diameter_q_wca(t_x, rep_x, att_x);
        let i_wca =
            correlation_integral_wca(rho_x, mean_field_constant_x, rep_x, att_x, d_x, q_vdw, rm_x);

        let delta_a1u = state.partial_density.sum() / t_x * i_wca * 2.0 * PI * weighted_sigma3_ij;

        let u_fraction_wca = u_fraction_wca(
            rep_x,
            density * (x * &p.sigma.mapv(|s| s.powi(3))).sum(),
            t_x,
        );

        let b21u = delta_b12u(t_x, mean_field_constant_x, weighted_sigma3_ij, q_vdw, rm_x);
        let b2bar = residual_virial_coefficient(p, x, state.temperature);

        let b3bar = residual_third_virial_coefficient(p, x, state.temperature, &d);
        let db31u = delta_b31u(t_x, weighted_sigma3_ij, rm_x, rep_x, att_x, d_x);
        let alpha =
            (-rho_x * rep_x.recip() * CU_WCA[0] * (t_x.powi(2).recip() * CU_WCA[1] + 1.0)).exp();
        state.moles.sum()
            * (delta_a1u
                + (-u_fraction_wca + 1.0)
                    * ((b2bar - b21u) * density + density.powi(2) * alpha * (b3bar - db31u) * 0.5))
    }
}

fn delta_b12u<D: DualNum<f64> + Copy>(
    t_x: D,
    mean_field_constant_x: D,
    weighted_sigma3_ij: D,
    q_x: D,
    rm_x: D,
) -> D {
    (-mean_field_constant_x - (rm_x.powi(3) - q_x.powi(3)) * 1.0 / 3.0) / t_x
        * 2.0
        * PI
        * weighted_sigma3_ij
}

fn residual_virial_coefficient<D: DualNum<f64> + Copy>(p: &UVParameters, x: &Array1<D>, t: D) -> D {
    let mut delta_b2bar = D::zero();

    for i in 0..p.ncomponents {
        let xi = x[i];

        for j in 0..p.ncomponents {
            let t_ij = t / p.eps_k_ij[[i, j]];
            let rep_ij = p.rep_ij[[i, j]];
            let att_ij = p.att_ij[[i, j]];

            let q_ij = dimensionless_diameter_q_wca(t_ij, D::from(rep_ij), D::from(att_ij));

            delta_b2bar +=
                xi * x[j] * p.sigma_ij[[i, j]].powi(3) * delta_b2(t_ij, rep_ij, att_ij, q_ij);
        }
    }
    delta_b2bar
}
fn residual_third_virial_coefficient<D: DualNum<f64> + Copy>(
    p: &UVParameters,
    x: &Array1<D>,
    t: D,
    d: &Array1<D>,
) -> D {
    let mut delta_b3bar = D::zero();

    for i in 0..p.ncomponents {
        let xi = x[i];

        for j in 0..p.ncomponents {
            let t_ij = t / p.eps_k_ij[[i, j]];
            let rep_ij = p.rep_ij[[i, j]];
            let att_ij = p.att_ij[[i, j]];
            let q_ij = dimensionless_diameter_q_wca(t_ij, D::from(rep_ij), D::from(att_ij));

            // No mixing rule defined for B3 yet! The implemented rule is just taken from B2 and not correct!
            let rm_ij = (rep_ij / att_ij).powd((rep_ij - att_ij).recip());
            let d_ij = (d[i] / p.sigma[i] + d[j] / p.sigma[j]) * 0.5;
            // Recheck mixing rule!
            delta_b3bar += xi
                * x[j]
                * p.sigma_ij[[i, j]].powi(6)
                * delta_b3(t_ij, rm_ij, rep_ij, att_ij, d_ij, q_ij);
        }
    }
    delta_b3bar
}
fn correlation_integral_wca<D: DualNum<f64> + Copy>(
    rho_x: D,
    mean_field_constant_x: D,
    rep_x: D,
    att_x: D,
    d_x: D,
    q_x: D,
    rm_x: D,
) -> D {
    let c = coefficients_wca(rep_x, att_x, d_x);

    (q_x.powi(3) - rm_x.powi(3)) * 1.0 / 3.0 - mean_field_constant_x
        + mie_prefactor(rep_x, att_x) * (c[0] * rho_x + c[1] * rho_x.powi(2) + c[2] * rho_x.powi(3))
            / (c[3] * rho_x + c[4] * rho_x.powi(2) + c[5] * rho_x.powi(3) + 1.0)
}

/// U-fraction with low temperature correction omega
fn u_fraction_wca<D: DualNum<f64> + Copy>(rep_x: D, reduced_density: D, t_x: D) -> D {
    let omega = if t_x.re() < 175.0 {
        (-t_x * CU_WCA[5] * (reduced_density - CU_WCA[6]).powi(2)).exp()
            * ((t_x * CU_WCA[7]).tanh().recip() - 1.0).powi(2)
    } else {
        (-t_x * CU_WCA[5] * (reduced_density - CU_WCA[6]).powi(2)).exp() * 0.0
    };

    -(-(reduced_density.powi(2) * ((rep_x + CU_WCA[4]).recip() * CU_WCA[3] + CU_WCA[2]) + omega))
        .exp()
        + 1.0
}

// Coefficients for IWCA
fn coefficients_wca<D: DualNum<f64> + Copy>(rep: D, att: D, d: D) -> [D; 6] {
    let rep_inv = rep.recip();
    let rs_x = (rep / att).powd((rep - att).recip());
    let tau_x = -d + rs_x;
    let c1 = rep_inv.powi(2) * C_WCA[0][2]
        + C_WCA[0][0]
        + rep_inv * C_WCA[0][1]
        + (rep_inv.powi(2) * C_WCA[0][5] + rep_inv * C_WCA[0][4] + C_WCA[0][3]) * tau_x;
    let c2 = rep_inv.powi(2) * C_WCA[1][2]
        + C_WCA[1][0]
        + rep_inv * C_WCA[1][1]
        + (rep_inv.powi(2) * C_WCA[1][5] + rep_inv * C_WCA[1][4] + C_WCA[1][3]) * tau_x;
    let c3 = rep_inv.powi(2) * C_WCA[2][2]
        + C_WCA[2][0]
        + rep_inv * C_WCA[2][1]
        + (rep_inv.powi(2) * C_WCA[2][5] + rep_inv * C_WCA[2][4] + C_WCA[2][3]) * tau_x;
    let c4 = rep_inv.powi(2) * C_WCA[3][2]
        + C_WCA[3][0]
        + rep_inv * C_WCA[3][1]
        + (rep_inv.powi(2) * C_WCA[3][5] + rep_inv * C_WCA[3][4] + C_WCA[3][3]) * tau_x;
    let c5 = rep_inv.powi(2) * C_WCA[4][2]
        + C_WCA[4][0]
        + rep_inv * C_WCA[4][1]
        + (rep_inv.powi(2) * C_WCA[4][5] + rep_inv * C_WCA[4][4] + C_WCA[4][3]) * tau_x;
    let c6 = rep_inv.powi(2) * C_WCA[5][2]
        + C_WCA[5][0]
        + rep_inv * C_WCA[5][1]
        + (rep_inv.powi(2) * C_WCA[5][5] + rep_inv * C_WCA[5][4] + C_WCA[5][3]) * tau_x;

    [c1, c2, c3, c4, c5, c6]
}

// Residual second virial coefficient from Revised series approximation RSAP

fn factorial(num: u64) -> u64 {
    (1..=num).product()
}

fn delta_b2<D: DualNum<f64> + Copy>(reduced_temperature: D, rep: f64, att: f64, q: D) -> D {
    let rm = (rep / att).powd((rep - att).recip());
    let beta = reduced_temperature.recip();
    let b20 = q.powi(3) * 2.0 / 3.0 * PI;
    let y = beta.exp() - 1.0;

    let c1 = rep.recip() * C_B2_RSAP[0][1]
        + (rep.powi(2)).recip() * C_B2_RSAP[0][2]
        + (rep.powi(3)).recip() * C_B2_RSAP[0][3]
        + C_B2_RSAP[0][0];

    let c2 = rep.recip() * C_B2_RSAP[1][1]
        + (rep.powi(2)).recip() * C_B2_RSAP[1][2]
        + (rep.powi(3)).recip() * C_B2_RSAP[1][3]
        + C_B2_RSAP[1][0];
    let c3 = rep.recip() * C_B2_RSAP[2][1]
        + (rep.powi(2)).recip() * C_B2_RSAP[2][2]
        + (rep.powi(3)).recip() * C_B2_RSAP[2][3]
        + C_B2_RSAP[2][0];
    let c4 = rep.recip() * C_B2_RSAP[3][1]
        + (rep.powi(2)).recip() * C_B2_RSAP[3][2]
        + (rep.powi(3)).recip() * C_B2_RSAP[3][3]
        + C_B2_RSAP[3][0];

    let mut sum_beta = beta;

    for i in 2..16 {
        let k = factorial(i as u64) as f64 * i as f64;
        sum_beta += beta.powi(i) / k
    }

    (b20 - rm.powi(3) * 2.0 / 3.0 * PI - c1) * y - sum_beta * c2 - beta * c3 - beta.powi(2) * c4
}

fn delta_b31u<D: DualNum<f64> + Copy>(
    t_x: D,
    weighted_sigma3_ij: D,
    rm_x: D,
    rep_x: D,
    att_x: D,
    d_x: D,
) -> D {
    let nu = rep_x;
    let tau = rm_x - d_x;

    let k1 = nu.recip() * C_WCA[0][1]
        + (nu.powi(2)).recip() * C_WCA[0][2]
        + C_WCA[0][0]
        + (nu.recip() * C_WCA[0][4] + (nu.powi(2)).recip() * C_WCA[0][5] + C_WCA[0][3]) * tau;
    t_x.recip() * 4.0 * mie_prefactor(rep_x, att_x) * PI * k1 * weighted_sigma3_ij.powi(2)
}

fn delta_b3<D: DualNum<f64> + Copy>(
    t_x: D,
    rm_x: f64,
    rep_x: f64,
    _att_x: f64,
    d_x: D,
    q_x: D,
) -> D {
    let beta = t_x.recip();
    let b30 = (q_x.powi(3) * PI / 6.0).powi(2) * 10.0;

    let b31 = ((t_x + K_LJ_B3[2])
        .powf(((rep_x - 12.0) / (rep_x - 6.0) * M_B3[0] + 1.0) * K_LJ_B3[3]))
    .recip()
        * K_LJ_B3[1]
        * ((rep_x - 12.0) / (rep_x - L_B3[0]) * P_B3[0] + 1.0)
        + K_LJ_B3[0];

    let b32 = ((t_x + K_LJ_B3[6])
        .powf(((rep_x - 12.0) / (rep_x - 6.0) * M_B3[1] + 1.0) * K_LJ_B3[7]))
    .recip()
        * K_LJ_B3[5]
        * ((rep_x - 12.0) / (rep_x - L_B3[1]) * P_B3[1] + 1.0)
        + K_LJ_B3[4];

    let b33 = ((t_x + K_LJ_B3[10])
        .powf(((rep_x - 12.0) / (rep_x - 6.0) * M_B3[2] + 1.0) * K_LJ_B3[11]))
    .recip()
        * K_LJ_B3[9]
        * ((rep_x - 12.0) / (rep_x - L_B3[2]) * P_B3[2] + 1.0)
        + K_LJ_B3[8];

    let b34 = ((t_x + K_LJ_B3[14])
        .powf(((rep_x - 12.0) / (rep_x - 6.0) * M_B3[3] + 1.0) * K_LJ_B3[15]))
    .recip()
        * K_LJ_B3[13]
        * ((rep_x - 12.0) / (rep_x - L_B3[3]) * P_B3[3] + 1.0)
        + K_LJ_B3[12];

    let b3 = b30 + b31 * beta + b32 * beta.powi(2) + b33 * beta.powi(3) + b34 * beta.powi(4);

    // Watch out: Not defined for mixtures!
    let tau = -d_x + rm_x;
    let tau2 = tau * tau;
    let rep_inv = rep_x.recip();

    let c1_eta_a = tau
        * (rep_inv * WCA_CONSTANTS_ETA_A_UVB3[0][1] + WCA_CONSTANTS_ETA_A_UVB3[0][0])
        + tau2 * (rep_inv * WCA_CONSTANTS_ETA_A_UVB3[0][3] + WCA_CONSTANTS_ETA_A_UVB3[0][2]);

    let c1_eta_b = tau * WCA_CONSTANTS_ETA_B_UVB3[0][0] + tau2 * WCA_CONSTANTS_ETA_B_UVB3[0][1];

    let b30_uv = (d_x.powi(3) * PI / 6.0).powi(2) * 10.0
        - d_x.powi(3) * 5.0 / 9.0
            * PI.powi(2)
            * ((-q_x.powi(3) + rm_x.powi(3)) * (c1_eta_a + 1.0)
                - (-d_x.powi(3) + rm_x.powi(3)) * (c1_eta_b + 1.0));

    b3 - b30_uv
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::uvtheory::parameters::utils::methane_parameters;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_attractive_perturbation_uvb3() {
        let moles = arr1(&[2.0]);
        let reduced_temperature = 4.0;
        let reduced_density = 0.5;
        let reduced_volume = moles[0] / reduced_density;

        let p = methane_parameters(12.0, 6.0);
        let pt = AttractivePerturbationUVB3 {
            parameters: Arc::new(p.clone()),
        };
        let state = StateHD::new(
            reduced_temperature * p.epsilon_k[0],
            reduced_volume * p.sigma[0].powi(3),
            moles.clone(),
        );
        let x = &state.molefracs;
        let (rep_x, att_x, sigma_x, weighted_sigma3_ij, epsilon_k_x, d_x) =
            one_fluid_properties(&p, &state.molefracs, state.temperature);

        let t_x = state.temperature / epsilon_k_x;
        let rho_x = state.partial_density.sum() * sigma_x.powi(3);
        let rm_x = (rep_x / att_x).powd((rep_x - att_x).recip());
        let mean_field_constant_x = mean_field_constant(rep_x, att_x, rm_x);

        // Effective Diameters:
        let q_vdw = dimensionless_diameter_q_wca(t_x, rep_x, att_x);
        assert_relative_eq!(q_vdw.re(), 0.9606854684075393, epsilon = 1e-10);
        assert_relative_eq!(d_x.re(), 0.934655265184067, epsilon = 1e-10);

        //u-fraction:
        let u_fraction_wca = u_fraction_wca(
            rep_x,
            state.partial_density.sum() * (x * &p.sigma.mapv(|s| s.powi(3))).sum(),
            t_x,
        );
        assert_relative_eq!(u_fraction_wca.re(), 0.8852775506870431, epsilon = 1e-10);
        // delta a1u
        let i_wca =
            correlation_integral_wca(rho_x, mean_field_constant_x, rep_x, att_x, d_x, q_vdw, rm_x);
        let delta_a1u = state.partial_density.sum() / t_x * i_wca * 2.0 * PI * weighted_sigma3_ij;
        assert!(delta_a1u.re() == -0.8992910890819197);
        // Virial coeffecients:
        let b2bar = residual_virial_coefficient(&p, x, state.temperature) / p.sigma[0].powi(3);
        assert_relative_eq!(b2bar.re(), -1.6142316456384618, epsilon = 1e-12);

        let b21u = delta_b12u(t_x, mean_field_constant_x, weighted_sigma3_ij, q_vdw, rm_x)
            / p.sigma[0].powi(3);
        assert_relative_eq!(b21u.re(), -1.5103749286162982, epsilon = 1e-10);

        let db3 = delta_b3(t_x, rm_x, rep_x, att_x, d_x, q_vdw);
        assert_relative_eq!(db3.re(), -0.6591980196661884, epsilon = 1e-10);

        // Full attractive perturbation:
        let a = pt.helmholtz_energy(&state) / moles[0];

        assert_relative_eq!(-0.9027781694834115, a.re(), epsilon = 1e-5);
    }
}
