//use super::attractive_perturbation_wca::one_fluid_properties;
use super::hard_sphere_bh::diameter_bh;
use crate::uvtheory::parameters::*;
use feos_core::{HelmholtzEnergyDual, StateHD};
use ndarray::Array1;
use num_dual::DualNum;
use std::{
    f64::consts::{FRAC_PI_3, PI},
    fmt,
    sync::Arc,
};

const C_BH: [[f64; 4]; 2] = [
    [
        0.168966996450507,
        -0.991545819144238,
        0.743142180601202,
        -4.32349593441145,
    ],
    [
        -0.532628162859638,
        2.66039013993583,
        -1.95070279905704,
        -0.000137219512394905,
    ],
];

/// Constants for BH u-fraction.
const CU_BH: [[f64; 2]; 4] = [
    [0.72188, 0.0],
    [-0.0059822, 2.4676],
    [2.2919, 14.9735],
    [5.1647, 2.4017],
];

/// Constants for BH effective inverse reduced temperature.
const C2: [[f64; 2]; 3] = [
    [1.50542979585173e-03, 3.90426109607451e-02],
    [3.23388827421376e-04, 1.29508541592689e-02],
    [5.25749466058948e-05, 5.26748277148572e-04],
];

#[derive(Debug, Clone)]
pub struct AttractivePerturbationBH {
    pub parameters: Arc<UVParameters>,
}

impl fmt::Display for AttractivePerturbationBH {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Attractive Perturbation")
    }
}

impl<D: DualNum<f64> + Copy> HelmholtzEnergyDual<D> for AttractivePerturbationBH {
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

        let mean_field_constant_x = mean_field_constant(rep_x, att_x, D::one());

        let i_bh = correlation_integral_bh(rho_x, mean_field_constant_x, rep_x, att_x, d_x);
        let delta_a1u = density / t_x * i_bh * 2.0 * PI * weighted_sigma3_ij;

        let u_fraction_bh = u_fraction_bh(
            rep_x,
            density * (x * &p.sigma.mapv(|s| s.powi(3))).sum(),
            t_x.recip(),
        );

        let b21u = delta_b12u(t_x, mean_field_constant_x, weighted_sigma3_ij);
        let b2bar = residual_virial_coefficient(p, x, state.temperature);

        state.moles.sum() * (delta_a1u + (-u_fraction_bh + 1.0) * (b2bar - b21u) * density)
    }
}

fn delta_b12u<D: DualNum<f64>>(t_x: D, mean_field_constant_x: D, weighted_sigma3_ij: D) -> D {
    -mean_field_constant_x / t_x * 2.0 * PI * weighted_sigma3_ij
}

fn residual_virial_coefficient<D: DualNum<f64> + Copy>(p: &UVParameters, x: &Array1<D>, t: D) -> D {
    let mut delta_b2bar = D::zero();
    for i in 0..p.ncomponents {
        let xi = x[i];
        for j in 0..p.ncomponents {
            delta_b2bar += xi
                * x[j]
                * p.sigma_ij[[i, j]].powi(3)
                * delta_b2(t / p.eps_k_ij[[i, j]], p.rep_ij[[i, j]], p.att_ij[[i, j]]);
        }
    }
    delta_b2bar
}

fn correlation_integral_bh<D: DualNum<f64> + Copy>(
    rho_x: D,
    mean_field_constant_x: D,
    rep_x: D,
    att_x: D,
    d_x: D,
) -> D {
    let c = coefficients_bh(rep_x, att_x, d_x);
    -mean_field_constant_x
        + mie_prefactor(rep_x, att_x) * (c[0] * rho_x + c[1] * rho_x.powi(2))
            / (c[2] * rho_x + 1.0).powi(2)
}

/// U-fraction according to Barker-Henderson division.
/// Eq. 15
fn u_fraction_bh<D: DualNum<f64> + Copy>(rep_x: D, reduced_density: D, one_fluid_beta: D) -> D {
    let mut c = [D::zero(); 4];
    let inv_rep = rep_x.recip();
    for i in 0..4 {
        c[i] = inv_rep * CU_BH[i][1] + CU_BH[i][0];
    }
    let a = 1.2187;
    let b = 4.2773;
    (activation(c[1], one_fluid_beta) * (-c[0] + 1.0) + c[0])
        * (reduced_density.powf(a) * c[2] + reduced_density.powf(b) * c[3]).tanh()
}

/// Activation function used for u-fraction according to Barker-Henderson division.
/// Eq. 16
fn activation<D: DualNum<f64> + Copy>(c: D, one_fluid_beta: D) -> D {
    one_fluid_beta * c.sqrt() / (one_fluid_beta.powi(2) * c + 1.0).sqrt()
}

fn one_fluid_properties<D: DualNum<f64> + Copy>(
    p: &UVParameters,
    x: &Array1<D>,
    t: D,
) -> (D, D, D, D, D, D) {
    let d = diameter_bh(p, t);
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

fn coefficients_bh<D: DualNum<f64> + Copy>(rep: D, att: D, d: D) -> [D; 3] {
    let c11 = d.powd(-rep + 6.0) * ((D::one() * 2.0f64).powd(-rep + 3.0) - d.powd(rep - 3.0))
        / (-rep + 3.0)
        + (-d.powi(3) * 8.0 + 1.0) / 24.0;
    let c12 = (d.powd(-rep + 6.0) * ((D::one() * 2.0f64).powd(-rep + 4.0) - d.powd(rep - 4.0))
        / (-rep + 4.0)
        + (-d.powi(2) * 4.0 + 1.0) / 8.0)
        * -0.75;
    let c13 = (((d * 2.0).powd(-rep + 6.0) - 1.0) / (-rep + 6.0)
        - (d * 2.0).ln() * d.powd(-att + 6.0))
        / 16.0;
    let rep_inv = rep.recip();
    let c1 = (c11 + c12 + c13) * FRAC_PI_3 * 4.0;
    let c2 = rep_inv * C_BH[0][1] + C_BH[0][0] - (rep_inv * C_BH[0][3] + C_BH[0][2]) * (-d + 1.0);
    let c3 = rep_inv * C_BH[1][1] + C_BH[1][0] - (rep_inv * C_BH[1][3] + C_BH[1][2]) * (-d + 1.0);
    [c1, c2, c3]
}

fn delta_b2<D: DualNum<f64> + Copy>(reduced_temperature: D, rep: f64, att: f64) -> D {
    let rc = 5.0;
    let alpha = mean_field_constant(rep, att, rc);
    let yeff = y_eff(reduced_temperature, rep, att);
    -(yeff * (rc.powi(3) - 1.0) / 3.0 + reduced_temperature.recip() * alpha) * 2.0 * PI
}

fn y_eff<D: DualNum<f64> + Copy>(reduced_temperature: D, rep: f64, att: f64) -> D {
    // optimize: move this part to parameter initialization
    let rc = 5.0;
    let rs = 1.0;
    let c0 = 1.0
        - 3.0 * (mean_field_constant(rep, att, rs) - mean_field_constant(rep, att, rc))
            / (rc.powi(3) - rs.powi(3));
    let c1 = C2[0][0] + C2[0][1] / rep;
    let c2 = C2[1][0] + C2[1][1] / rep;
    let c3 = C2[2][0] + C2[2][1] / rep;

    let beta = reduced_temperature.recip();
    let beta_eff = beta * (-(beta * (beta * c2 + beta.powi(3) * c3 + c1) + 1.0).recip() * c0 + 1.0);
    beta_eff.exp() - 1.0
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::uvtheory::parameters::utils::methane_parameters;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_attractive_perturbation() {
        // m = 12, t = 4.0, rho = 1.0
        let moles = arr1(&[2.0]);
        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let reduced_volume = moles[0] / reduced_density;

        let p = methane_parameters(24.0, 6.0);
        let pt = AttractivePerturbationBH {
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

        let mean_field_constant_x = mean_field_constant(rep_x, att_x, 1.0);

        let i_bh = correlation_integral_bh(rho_x, mean_field_constant_x, rep_x, att_x, d_x);
        let delta_a1u = state.partial_density.sum() / t_x * i_bh * 2.0 * PI * weighted_sigma3_ij;
        dbg!(delta_a1u);
        //assert!(delta_a1u.re() == -1.1470186919354);
        assert_relative_eq!(delta_a1u.re(), -1.1470186919354, epsilon = 1e-12);

        let u_fraction_bh = u_fraction_bh(
            rep_x,
            state.partial_density.sum() * (x * &p.sigma.mapv(|s| s.powi(3))).sum(),
            t_x.recip(),
        );
        dbg!(u_fraction_bh);
        //assert!(u_fraction_bh.re() == 0.743451055308332);
        assert_relative_eq!(u_fraction_bh.re(), 0.743451055308332, epsilon = 1e-5);

        let b21u = delta_b12u(t_x, mean_field_constant_x, weighted_sigma3_ij);
        dbg!(b21u);
        assert!(b21u.re() / p.sigma[0].powi(3) == -0.949898568221715);

        let b2bar = residual_virial_coefficient(&p, x, state.temperature);
        dbg!(b2bar);
        assert_relative_eq!(
            b2bar.re() / p.sigma[0].powi(3),
            -1.00533412744652,
            epsilon = 1e-12
        );
        //assert!(b2bar.re() ==-1.00533412744652);

        //let a_test = state.moles.sum()
        //  * (delta_a1u + (-u_fraction_bh + 1.0) * (b2bar - b21u) * state.partial_density.sum());
        let a = pt.helmholtz_energy(&state) / moles[0];
        dbg!(a.re());
        //assert!(-1.16124062615291 == a.re())
        assert_relative_eq!(-1.16124062615291, a.re(), epsilon = 1e-5);
    }
}
