use crate::uvtheory::parameters::UVParameters;
use feos_core::{HelmholtzEnergyDual, StateHD};
use lazy_static::lazy_static;
use ndarray::prelude::*;
use num_dual::DualNum;
use std::fmt;
use std::rc::Rc;

lazy_static! {
    static ref BH_CONSTANTS_ETA_B: Array2<f64> = arr2(&[
        [-0.960919783, -0.921097447],
        [-0.547468020, -3.508014069],
        [-2.253750186, 3.581161364],
    ]);
    static ref BH_CONSTANTS_ETA_A: Array2<f64> = arr2(&[
        [-1.217417282, 6.754987582, -0.5919326153, -28.99719604],
        [1.579548775, -26.93879416, 0.3998915410, 106.9446266],
        [-1.993990512, 44.11863355, -40.10916106, -29.6130848],
        [0.0, 0.0, 0.0, 0.0],
    ]);
}

#[derive(Debug, Clone)]
pub struct HardSphere {
    pub parameters: Rc<UVParameters>,
}

impl<D: DualNum<f64>> HelmholtzEnergyDual<D> for HardSphere {
    /// Helmholtz energy for hard spheres, eq. 19 (check Volume)
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let d = diameter_bh(&self.parameters, state.temperature);
        let zeta = zeta(&state.partial_density, &d);
        let frac_1mz3 = -(zeta[3] - 1.0).recip();
        let zeta_23 = zeta_23(&state.molefracs, &d);
        state.volume * 6.0 / std::f64::consts::PI
            * (zeta[1] * zeta[2] * frac_1mz3 * 3.0
                + zeta[2].powi(2) * frac_1mz3.powi(2) * zeta_23
                + (zeta[2] * zeta_23.powi(2) - zeta[0]) * (zeta[3] * (-1.0)).ln_1p())
    }
}

impl fmt::Display for HardSphere {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hard Sphere")
    }
}

/// Dimensionless Hard-sphere diameter according to Barker-Henderson division.
/// Eq. S23 and S24.
///
pub fn diameter_bh<D: DualNum<f64>>(parameters: &UVParameters, temperature: D) -> Array1<D> {
    parameters
        .cd_bh_pure
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let t = temperature / parameters.epsilon_k[i];
            let d = t.powf(0.25) * c[1] + t.powf(0.75) * c[2] + t.powf(1.25) * c[3];
            (t * c[0] + d * (t + 1.0).ln() + t.powi(2) * c[4] + 1.0)
                .powf(-0.5 / parameters.rep[i] as f64)
                * parameters.sigma[i]
        })
        .collect()
}

pub fn zeta<D: DualNum<f64>>(partial_density: &Array1<D>, diameter: &Array1<D>) -> [D; 4] {
    let mut zeta: [D; 4] = [D::zero(), D::zero(), D::zero(), D::zero()];
    for i in 0..partial_density.len() {
        for k in 0..4 {
            zeta[k] +=
                partial_density[i] * diameter[i].powi(k as i32) * (std::f64::consts::PI / 6.0);
        }
    }
    zeta
}

pub fn packing_fraction<D: DualNum<f64>>(partial_density: &Array1<D>, diameter: &Array1<D>) -> D {
    (0..partial_density.len()).fold(D::zero(), |acc, i| {
        acc + partial_density[i] * diameter[i].powi(3) * (std::f64::consts::PI / 6.0)
    })
}

pub fn zeta_23<D: DualNum<f64>>(molefracs: &Array1<D>, diameter: &Array1<D>) -> D {
    let mut zeta: [D; 2] = [D::zero(), D::zero()];
    for i in 0..molefracs.len() {
        for k in 0..2 {
            zeta[k] += molefracs[i] * diameter[i].powi((k + 2) as i32);
        }
    }
    zeta[0] / zeta[1]
}

//#[inline]
//pub fn dimensionless_length_scale<D: DualNum<f64>>(
//  parameters: &UVParameters,
//temperature: D,
//rep: f64,
//) -> Array1<D> {
//  -diameter_bh(parameters, temperature) + 1.0
//}

pub fn packing_fraction_b<D: DualNum<f64>>(
    parameters: &UVParameters,
    diameter: &Array1<D>,
    eta: D,
) -> Array2<D> {
    let n = parameters.att.len();
    Array2::from_shape_fn((n, n), |(i, j)| {
        let tau =
            -(diameter[i] / parameters.sigma[i] + diameter[j] / parameters.sigma[j]) * 0.5 + 1.0; //dimensionless
        let tau2 = tau * tau;
        let c = arr1(&[
            tau * BH_CONSTANTS_ETA_B[[0, 0]] + tau2 * BH_CONSTANTS_ETA_B[[0, 1]],
            tau * BH_CONSTANTS_ETA_B[[1, 0]] + tau2 * BH_CONSTANTS_ETA_B[[1, 1]],
            tau * BH_CONSTANTS_ETA_B[[2, 0]] + tau2 * BH_CONSTANTS_ETA_B[[2, 1]],
        ]);
        eta + eta * c[0] + eta * eta * c[1] + eta.powi(3) * c[2]
    })
}

pub fn packing_fraction_a<D: DualNum<f64>>(
    parameters: &UVParameters,
    diameter: &Array1<D>,
    eta: D,
) -> Array2<D> {
    let n = parameters.att.len();
    Array2::from_shape_fn((n, n), |(i, j)| {
        let tau =
            -(diameter[i] / parameters.sigma[i] + diameter[j] / parameters.sigma[j]) * 0.5 + 1.0; //dimensionless//-(diameter[i] + diameter[j]) * 0.5 + 1.0;
        let tau2 = tau * tau;
        let rep_inv = 1.0 / parameters.rep_ij[[i, j]];
        let c = arr1(&[
            tau * (BH_CONSTANTS_ETA_A[[0, 0]] + BH_CONSTANTS_ETA_A[[0, 1]] * rep_inv)
                + tau2 * (BH_CONSTANTS_ETA_A[[0, 2]] + BH_CONSTANTS_ETA_A[[0, 3]] * rep_inv),
            tau * (BH_CONSTANTS_ETA_A[[1, 0]] + BH_CONSTANTS_ETA_A[[1, 1]] * rep_inv)
                + tau2 * (BH_CONSTANTS_ETA_A[[1, 2]] + BH_CONSTANTS_ETA_A[[1, 3]] * rep_inv),
            tau * (BH_CONSTANTS_ETA_A[[2, 0]] + BH_CONSTANTS_ETA_A[[2, 1]] * rep_inv)
                + tau2 * (BH_CONSTANTS_ETA_A[[2, 2]] + BH_CONSTANTS_ETA_A[[2, 3]] * rep_inv),
            tau * (BH_CONSTANTS_ETA_A[[3, 0]] + BH_CONSTANTS_ETA_A[[3, 1]] * rep_inv)
                + tau2 * (BH_CONSTANTS_ETA_A[[3, 2]] + BH_CONSTANTS_ETA_A[[3, 3]] * rep_inv),
        ]);
        eta + eta * c[0] + eta * eta * c[1] + eta.powi(3) * c[2] + eta.powi(4) * c[3]
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::uvtheory::parameters::utils::{methane_parameters, test_parameters};

    #[test]
    fn test_bh_diameter() {
        let p = test_parameters(12.0, 6.0, 1.0, 1.0);
        assert_eq!(diameter_bh(&p, 2.0)[0], 0.95777257352360246);
        let p = test_parameters(24.0, 6.0, 1.0, 1.0);
        assert_eq!(diameter_bh(&p, 5.0)[0], 0.95583586434435486);

        // Methane
        let p = methane_parameters(12.0, 6.0);
        assert_eq!(
            diameter_bh(&p, 2.0 * p.epsilon_k[0])[0] / p.sigma[0],
            0.95777257352360246
        );
        let p = methane_parameters(24.0, 6.0);
        assert_eq!(
            diameter_bh(&p, 5.0 * p.epsilon_k[0])[0] / p.sigma[0],
            0.95583586434435486
        );
    }

    // #[test]
    // fn helmholtz_energy() {
    //     let p = methane_parameters(12);
    //     let p = test_parameters(12, 1.0, 1.0);
    //     let reduced_density = 1.0;
    //     let reduced_temperature = 4.0;

    //     let hs = HardSphere {
    //         parameters: Rc::new(p.clone()),
    //     };
    //     let particles = arr1(&[1000.0]);
    //     let n = &particles / 6.02214076e23;
    //     let volume = particles[0] / reduced_density * p.sigma[0].powi(3);
    //     let temperature = reduced_temperature * p.epsilon_k[0];
    //     let s = StateHD::new(temperature, volume, n.clone());
    //     dbg!(particles[0] / volume * p.sigma[0].powi(3));
    //     dbg!(&s.temperature);
    //     dbg!(&s.volume);
    //     dbg!(&s.moles);
    //     let a = hs.helmholtz_energy(&s);
    //     assert_relative_eq!(a / particles[0], 1.9859860794723039, epsilon = 1e-10)
    // }
}
