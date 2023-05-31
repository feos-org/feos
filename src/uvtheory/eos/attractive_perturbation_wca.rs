use super::hard_sphere_wca::{diameter_wca, dimensionless_diameter_q_wca};
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

/// Constants for WCA u-fraction.
const CU_WCA: [f64; 3] = [1.4419, 1.1169, 16.8810];

/// Constants for WCA effective inverse reduced temperature.
const C2: [[f64; 2]; 3] = [
    [1.45805207053190E-03, 3.57786067657446E-02],
    [1.25869266841313E-04, 1.79889086453277E-03],
    [0.0, 0.0],
];

#[derive(Debug, Clone)]
pub struct AttractivePerturbationWCA {
    pub parameters: Arc<UVParameters>,
}

impl fmt::Display for AttractivePerturbationWCA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Attractive Perturbation")
    }
}

impl<D: DualNum<f64> + Copy> HelmholtzEnergyDual<D> for AttractivePerturbationWCA {
    /// Helmholtz energy for attractive perturbation, eq. 52
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let p = &self.parameters;
        let x = &state.molefracs;
        let t = state.temperature;
        let density = state.partial_density.sum();

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

        //                 state.partial_density.sum() / t_x * i_wca * 2.0 * PI * weighted_sigma3_ij;
        let u_fraction_wca =
            u_fraction_wca(rep_x, density * (x * &p.sigma.mapv(|s| s.powi(3))).sum());

        let b21u = delta_b12u(t_x, mean_field_constant_x, weighted_sigma3_ij, q_vdw, rm_x);
        let b2bar = residual_virial_coefficient(p, x, state.temperature);

        state.moles.sum() * (delta_a1u + (-u_fraction_wca + 1.0) * (b2bar - b21u) * density)
    }
}

// (S43) & (S53)
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
            //let q_ij = (q[i] / p.sigma[i] + q[j] / p.sigma[j]) * 0.5;
            let t_ij = t / p.eps_k_ij[[i, j]];
            let rep_ij = p.rep_ij[[i, j]];
            let att_ij = p.att_ij[[i, j]];

            let q_ij = dimensionless_diameter_q_wca(t_ij, D::from(rep_ij), D::from(att_ij));

            // Recheck mixing rule!
            delta_b2bar +=
                xi * x[j] * p.sigma_ij[[i, j]].powi(3) * delta_b2(t_ij, rep_ij, att_ij, q_ij);
        }
    }
    delta_b2bar
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

/// U-fraction according to Barker-Henderson division.
/// Eq. 15
fn u_fraction_wca<D: DualNum<f64> + Copy>(rep_x: D, reduced_density: D) -> D {
    (reduced_density * CU_WCA[0]
        + reduced_density.powi(2) * (rep_x.recip() * CU_WCA[2] + CU_WCA[1]))
        .tanh()
}

pub(super) fn one_fluid_properties<D: DualNum<f64> + Copy>(
    p: &UVParameters,
    x: &Array1<D>,
    t: D,
) -> (D, D, D, D, D, D) {
    let d = diameter_wca(p, t);
    // &p.sigma;

    let mut epsilon_k = D::zero();
    let mut weighted_sigma3_ij = D::zero();
    let mut rep = D::zero();
    let mut att = D::zero();
    let mut d_x_3 = D::zero();

    for i in 0..p.ncomponents {
        let xi = x[i];

        d_x_3 += x[i] * d[i].powi(3);
        for j in 0..p.ncomponents {
            let _y = xi * x[j] * p.sigma_ij[[i, j]].powi(3);
            weighted_sigma3_ij += _y;
            epsilon_k += _y * p.eps_k_ij[[i, j]];

            rep += xi * x[j] * p.rep_ij[[i, j]];
            att += xi * x[j] * p.att_ij[[i, j]];
        }
    }

    //let dx = (x * &d.mapv(|v| v.powi(3))).sum().powf(1.0 / 3.0);
    let sigma_x = (x * &p.sigma.mapv(|v| v.powi(3))).sum().powf(1.0 / 3.0);
    let dx = d_x_3.powf(1.0 / 3.0) / sigma_x;

    (
        rep,
        att,
        sigma_x,
        weighted_sigma3_ij,
        epsilon_k / weighted_sigma3_ij,
        dx,
    )
}

// Coefficients for IWCA from eq. (S55)
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

fn delta_b2<D: DualNum<f64> + Copy>(reduced_temperature: D, rep: f64, att: f64, q: D) -> D {
    let rm = (rep / att).powf(1.0 / (rep - att)); // Check mixing rule!!
    let rc = 5.0;
    let alpha = mean_field_constant(rep, att, rc);
    let beta = reduced_temperature.recip();
    let y = beta.exp() - 1.0;
    let yeff = y_eff(reduced_temperature, rep, att);
    -(yeff * (rc.powi(3) - rm.powi(3)) / 3.0 + y * (-q.powi(3) + rm.powi(3)) / 3.0 + beta * alpha)
        * 2.0
        * PI
}

fn y_eff<D: DualNum<f64> + Copy>(reduced_temperature: D, rep: f64, att: f64) -> D {
    // optimize: move this part to parameter initialization
    let rc = 5.0;
    let rs = (rep / att).powf(1.0 / (rep - att));
    let c0 = 1.0
        - 3.0 * (mean_field_constant(rep, att, rs) - mean_field_constant(rep, att, rc))
            / (rc.powi(3) - rs.powi(3));
    let c1 = C2[0][0] + C2[0][1] / rep;
    let c2 = C2[1][0] + C2[1][1] / rep;
    let c3 = C2[2][0] + C2[2][1] / rep;

    //exponents
    let a = 1.05968091375869;
    let b = 3.41106168592999;
    let c = 0.0;
    // (S58)
    let beta = reduced_temperature.recip();
    let beta_eff = beta
        * (-(beta.powf(a) * c1 + beta.powf(b) * c2 + beta.powf(c) * c3 + 1.0).recip() * c0 + 1.0);
    beta_eff.exp() - 1.0
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::uvtheory::parameters::utils::{methane_parameters, test_parameters_mixture};
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_attractive_perturbation() {
        // m = 24, t = 4.0, rho = 1.0
        let moles = arr1(&[2.0]);
        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let reduced_volume = moles[0] / reduced_density;

        let p = methane_parameters(24.0, 6.0);
        let pt = AttractivePerturbationWCA {
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
        dbg!(epsilon_k_x);
        let t_x = state.temperature / epsilon_k_x;
        let rho_x = state.partial_density.sum() * sigma_x.powi(3);
        let rm_x = (rep_x / att_x).powd((rep_x - att_x).recip());
        let mean_field_constant_x = mean_field_constant(rep_x, att_x, rm_x);
        dbg!(t_x);
        let q_vdw = dimensionless_diameter_q_wca(t_x, rep_x, att_x);
        let b21u = delta_b12u(t_x, mean_field_constant_x, weighted_sigma3_ij, q_vdw, rm_x)
            / p.sigma[0].powi(3);
        assert_relative_eq!(b21u.re(), -1.02233215790525, epsilon = 1e-12);

        let i_wca =
            correlation_integral_wca(rho_x, mean_field_constant_x, rep_x, att_x, d_x, q_vdw, rm_x);

        let delta_a1u = state.partial_density.sum() / t_x * i_wca * 2.0 * PI * weighted_sigma3_ij;

        assert_relative_eq!(delta_a1u.re(), -1.52406840346272, epsilon = 1e-6);

        let u_fraction_wca = u_fraction_wca(
            rep_x,
            state.partial_density.sum() * (x * &p.sigma.mapv(|s| s.powi(3))).sum(),
        );

        let b2bar = residual_virial_coefficient(&p, x, state.temperature) / p.sigma[0].powi(3);
        dbg!(b2bar);
        assert_relative_eq!(b2bar.re(), -1.09102560732964, epsilon = 1e-12);
        dbg!(u_fraction_wca);

        assert_relative_eq!(u_fraction_wca.re(), 0.997069754340431, epsilon = 1e-5);

        let a_test = delta_a1u
            + (-u_fraction_wca + 1.0)
                * (b2bar - b21u)
                * p.sigma[0].powi(3)
                * state.partial_density.sum();
        dbg!(a_test);
        dbg!(state.moles.sum());
        let a = pt.helmholtz_energy(&state) / moles[0];
        dbg!(a.re());

        assert_relative_eq!(-1.5242697155023, a.re(), epsilon = 1e-5);
    }

    #[test]
    fn test_attractive_perturbation_wca_mixture() {
        let moles = arr1(&[0.40000000000000002, 0.59999999999999998]);
        let reduced_temperature = 1.0;
        let reduced_density = 0.90000000000000002;
        let reduced_volume = (moles[0] + moles[1]) / reduced_density;

        let p = test_parameters_mixture(
            arr1(&[12.0, 12.0]),
            arr1(&[6.0, 6.0]),
            arr1(&[1.0, 1.0]),
            arr1(&[1.0, 0.5]),
        );
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());
        let (rep_x, att_x, sigma_x, weighted_sigma3_ij, epsilon_k_x, d_x) =
            one_fluid_properties(&p, &state.molefracs, state.temperature);

        // u-fraction
        let phi_u = u_fraction_wca(rep_x, reduced_density);
        assert_relative_eq!(phi_u, 0.99750066585468078, epsilon = 1e-6);

        // Delta B21u
        let rm_x = (rep_x / att_x).powd((rep_x - att_x).recip());
        let mean_field_constant_x = mean_field_constant(rep_x, att_x, rm_x);
        let t_x = state.temperature / epsilon_k_x;

        dbg!(t_x.re());

        let q_vdw = dimensionless_diameter_q_wca(t_x, rep_x, att_x);
        dbg!(q_vdw.re());
        let delta_b21u = delta_b12u(t_x, mean_field_constant_x, weighted_sigma3_ij, q_vdw, rm_x);
        dbg!(delta_b21u);
        assert_relative_eq!(delta_b21u, -3.9309384983526585, epsilon = 1e-6);

        // delta a1u
        let rho_x = state.partial_density.sum() * sigma_x.powi(3);

        let i_wca =
            correlation_integral_wca(rho_x, mean_field_constant_x, rep_x, att_x, d_x, q_vdw, rm_x);

        let delta_a1u = state.partial_density.sum() / state.temperature
            * i_wca
            * 2.0
            * PI
            * weighted_sigma3_ij
            * epsilon_k_x;

        assert_relative_eq!(delta_a1u, -4.7678301069070645, epsilon = 1e-6);

        // Second virial coefficient

        let delta_b2 = residual_virial_coefficient(&p, &state.molefracs, state.temperature)
            / p.sigma[0].powi(3);

        dbg!(delta_b2);
        assert_relative_eq!(delta_b2, -4.7846399638747954, epsilon = 1e-6);
        // Full attractive contribution
        let pt = AttractivePerturbationWCA {
            parameters: Arc::new(p),
        };

        let a = pt.helmholtz_energy(&state) / (moles[0] + moles[1]);

        assert_relative_eq!(a, -4.7697504236074844, epsilon = 1e-5);
    }

    #[test]
    fn test_attractive_perturbation_wca_mixture_different_sigma() {
        let moles = arr1(&[0.40000000000000002, 0.59999999999999998]);
        let reduced_temperature = 1.5;
        let density = 0.10000000000000001;
        let volume = 1.0 / density;
        let p = test_parameters_mixture(
            arr1(&[12.0, 12.0]),
            arr1(&[6.0, 6.0]),
            arr1(&[1.0, 2.0]),
            arr1(&[1.0, 0.5]),
        );

        let state = StateHD::new(reduced_temperature, volume, moles.clone());
        let (rep_x, att_x, sigma_x, weighted_sigma3_ij, epsilon_k_x, d_x) =
            one_fluid_properties(&p, &state.molefracs, state.temperature);
        // u-fraction
        let density = state.partial_density.sum();
        let x = &state.molefracs;
        let phi_u = u_fraction_wca(rep_x, density * (x * &p.sigma.mapv(|s| s.powi(3))).sum());
        assert_relative_eq!(phi_u, 0.89210738762113795, epsilon = 1e-5);
        // delta b2

        let b2bar = residual_virial_coefficient(&p, x, state.temperature) / p.sigma[0].powi(3);
        assert_relative_eq!(b2bar.re(), -12.106977583257606, epsilon = 1e-12);

        //delta b21u
        let rm_x = (rep_x / att_x).powd((rep_x - att_x).recip());
        let mean_field_constant_x = mean_field_constant(rep_x, att_x, rm_x);
        let t_x = state.temperature / epsilon_k_x;
        let q_vdw = dimensionless_diameter_q_wca(t_x, rep_x, att_x);
        let delta_b21u = delta_b12u(t_x, mean_field_constant_x, weighted_sigma3_ij, q_vdw, rm_x);
        assert_relative_eq!(delta_b21u, -10.841841323394299, epsilon = 1e-6);

        let a_ufrac = (-phi_u + 1.0) * (b2bar - delta_b21u) * density;
        assert_relative_eq!(a_ufrac, -0.0136498856091876, epsilon = 1e-6);
        // delta b20
        dbg!(d_x.re());

        // delta a1u
        let rho_x = state.partial_density.sum() * sigma_x.powi(3);
        assert_relative_eq!(d_x, 0.95196953178057431, epsilon = 1e-6);

        let i_wca =
            correlation_integral_wca(rho_x, mean_field_constant_x, rep_x, att_x, d_x, q_vdw, rm_x);
        dbg!(weighted_sigma3_ij.re());
        dbg!(epsilon_k_x);
        let delta_a1u = state.partial_density.sum() / state.temperature
            * i_wca
            * 2.0
            * PI
            * weighted_sigma3_ij
            * epsilon_k_x;

        assert_relative_eq!(delta_a1u, -1.3182160310774731, epsilon = 1e-6);

        // Full attractive contribution
        let pt = AttractivePerturbationWCA {
            parameters: Arc::new(p),
        };
        let a = pt.helmholtz_energy(&state) / (moles[0] + moles[1]);
        assert_relative_eq!(a, -1.3318659166866607, epsilon = 1e-5);
    }
}
