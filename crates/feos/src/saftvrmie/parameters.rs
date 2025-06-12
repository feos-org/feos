use crate::hard_sphere::{HardSphereProperties, MonomerShape};
use feos_core::parameter::{Parameters, PureRecord};
use ndarray::{Array, Array1, Array2};
use num_dual::DualNum;
use num_traits::Zero;
use serde::{Deserialize, Serialize};

/// 10-point Gauss-Legendre quadrature [position, weight]
const GLQ10: [[f64; 2]; 10] = [
    [-0.1488743389816312, 0.2955242247147529],
    [0.1488743389816312, 0.2955242247147529],
    [-0.4333953941292472, 0.2692667193099963],
    [0.4333953941292472, 0.2692667193099963],
    [-0.6794095682990244, 0.219086362515982],
    [0.6794095682990244, 0.219086362515982],
    [-0.8650633666889845, 0.1494513491505806],
    [0.8650633666889845, 0.1494513491505806],
    [-0.9739065285171717, 0.0666713443086881],
    [0.9739065285171717, 0.0666713443086881],
];

/// SAFT-VR Mie pure-component parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct SaftVRMieRecord {
    /// Segment number
    pub m: f64,
    /// Segment diameter in units of Angstrom
    pub sigma: f64,
    /// Energetic parameter in units of Kelvin
    pub epsilon_k: f64,
    /// Repulsive Mie exponent
    pub lr: f64,
    /// Attractive Mie exponent
    pub la: f64,
}

impl SaftVRMieRecord {
    pub fn new(m: f64, sigma: f64, epsilon_k: f64, lr: f64, la: f64) -> Self {
        Self {
            m,
            sigma,
            epsilon_k,
            lr,
            la,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct SaftVRMieAssociationRecord {
    /// Association radius parameter
    pub rc_ab: f64,
    /// Association energy parameter in units of Kelvin
    pub epsilon_k_ab: f64,
}

impl SaftVRMieAssociationRecord {
    pub fn new(rc_ab: f64, epsilon_k_ab: f64) -> Self {
        Self {
            rc_ab,
            epsilon_k_ab,
        }
    }
}

/// SAFT-VR Mie binary interaction parameters.
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct SaftVRMieBinaryRecord {
    /// Binary dispersion energy interaction parameter
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub k_ij: f64,
    /// Binary interaction parameter for repulsive exponent
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub gamma_ij: f64,
}

impl SaftVRMieBinaryRecord {
    pub fn new(k_ij: Option<f64>, gamma_ij: Option<f64>) -> Self {
        let k_ij = k_ij.unwrap_or_default();
        let gamma_ij = gamma_ij.unwrap_or_default();
        Self { k_ij, gamma_ij }
    }
}

/// Parameter set required for the SAFT-VR Mie equation of state.
pub type SaftVRMieParameters =
    Parameters<SaftVRMieRecord, SaftVRMieBinaryRecord, SaftVRMieAssociationRecord>;

/// The SAFT-VR Mie parameters in an easier accessible format.
pub struct SaftVRMiePars {
    pub m: Array1<f64>,
    pub sigma: Array1<f64>,
    pub epsilon_k: Array1<f64>,
    pub lr: Array1<f64>,
    pub la: Array1<f64>,
    pub sigma_ij: Array2<f64>,
    pub epsilon_k_ij: Array2<f64>,
    pub lr_ij: Array2<f64>,
    pub la_ij: Array2<f64>,
    pub c_ij: Array2<f64>,
    pub alpha_ij: Array2<f64>,
}

impl SaftVRMiePars {
    pub fn new(parameters: &SaftVRMieParameters) -> Self {
        let n = parameters.pure_records.len();

        let [m, sigma, epsilon_k] = parameters.collate(|pr| [pr.m, pr.sigma, pr.epsilon_k]);
        let [lr, la] = parameters.collate(|pr| [pr.lr, pr.la]);
        let [k_ij, gamma_ij] = parameters.collate_binary(|br| {
            let br = br.unwrap_or_default();
            [br.k_ij, br.gamma_ij]
        });

        let mut sigma_ij = Array::zeros((n, n));
        let mut e_k_ij = Array::zeros((n, n));
        let mut epsilon_k_ij = Array::zeros((n, n));
        let mut lr_ij = Array::zeros((n, n));
        let mut la_ij = Array::zeros((n, n));
        let mut c_ij = Array::zeros((n, n));
        let mut alpha_ij = Array::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                sigma_ij[[i, j]] = 0.5 * (sigma[i] + sigma[j]);
                e_k_ij[[i, j]] = (sigma[i].powi(3) * sigma[j].powi(3)).sqrt()
                    / sigma_ij[[i, j]].powi(3)
                    * (epsilon_k[i] * epsilon_k[j]).sqrt();
                epsilon_k_ij[[i, j]] = (1.0 - k_ij[[i, j]]) * e_k_ij[[i, j]];
                lr_ij[[i, j]] =
                    (1.0 - gamma_ij[[i, j]]) * ((lr[i] - 3.0) * (lr[j] - 3.0)).sqrt() + 3.0;
                la_ij[[i, j]] = ((la[i] - 3.0) * (la[j] - 3.0)).sqrt() + 3.0;
                c_ij[[i, j]] = lr_ij[[i, j]] / (lr_ij[[i, j]] - la_ij[[i, j]])
                    * (lr_ij[[i, j]] / la_ij[[i, j]])
                        .powf(la_ij[[i, j]] / (lr_ij[[i, j]] - la_ij[[i, j]]));
                alpha_ij[[i, j]] =
                    c_ij[[i, j]] * ((la_ij[[i, j]] - 3.0).recip() - (lr_ij[[i, j]] - 3.0).recip())
            }
        }

        Self {
            m,
            sigma,
            epsilon_k,
            lr,
            la,
            sigma_ij,
            epsilon_k_ij,
            lr_ij,
            la_ij,
            c_ij,
            alpha_ij,
        }
    }
}

impl SaftVRMiePars {
    #[inline]
    pub fn hs_diameter_ij<D: DualNum<f64> + Copy>(
        &self,
        i: usize,
        j: usize,
        inverse_temperature: D,
    ) -> D {
        let lr = self.lr_ij[[i, j]];
        let la = self.la_ij[[i, j]];
        let c_eps_t = inverse_temperature * self.c_ij[[i, j]] * self.epsilon_k_ij[[i, j]];

        // perform integration in reduced distances, then multiply sigma
        // r0 is dimensionless
        let r0 = lower_integratal_limit(la, lr, c_eps_t);
        let width = (-r0 + 1.0) * 0.5;
        GLQ10.iter().fold(r0, |d, &[x, w]| {
            let r = width * x + width + r0;
            let u = beta_u_mie(r, la, lr, 1.0, c_eps_t);
            let f_u = -(-u).exp_m1();
            d + width * f_u * w
        }) * self.sigma_ij[[i, j]]
    }
}

/// Find lower limit for integration of the temperature dependent diameter
///
/// Method of Aasen et al.
/// Using starting value proposed in Clapeyron.jl (Andrés Riedemann)
fn lower_integratal_limit<D: DualNum<f64> + Copy>(la: f64, lr: f64, c_eps_t: D) -> D {
    // initial value from repulsive contribution
    let k = (-c_eps_t.recip() * f64::EPSILON.ln()).ln();
    let mut r = (-k / lr).exp();
    // Halley's method
    for _ in 1..5 {
        let [u, u_du, du_d2u] = mie_potential_halley(r, la, lr, c_eps_t);
        let dr = u_du / (-u_du / du_d2u * 0.5 + 1.0);
        // if dr.re() < f64::EPSILON {
        if u.re() < 0.0 {
            return r;
        }
        r -= dr;
    }
    r // error instead?
}

/// Calculate the fractions f / df and df / d2f used for Halley's method,
///
/// f is the function to find the root of.
/// Here, f = -beta u_mie(r) - ln(EPS)
#[inline]
fn mie_potential_halley<D: DualNum<f64> + Copy>(r: D, la: f64, lr: f64, c_eps_t: D) -> [D; 3] {
    let ri = r.recip();
    let plr = ri.powf(lr);
    let pla = ri.powf(la);
    let u = plr - pla;
    let dplr = plr * (-lr) * ri;
    let dpla = pla * (-la) * ri;
    let du_dr = dplr - dpla;
    let d2u_dr2 = (dplr * (-lr - 1.0) - dpla * (-la - 1.0)) * ri;

    let f = -c_eps_t * u - f64::EPSILON.ln();
    let df = -c_eps_t * du_dr;
    let d2f = -c_eps_t * d2u_dr2;
    [f, f / df, df / d2f]
}

/// Dimensionless Mie potential (divided by kT)
#[inline]
fn beta_u_mie<D: DualNum<f64> + Copy>(r: D, la: f64, lr: f64, sigma: f64, c_eps_t: D) -> D {
    let ri = r.recip() * sigma;
    (ri.powf(lr) - ri.powf(la)) * c_eps_t
}

impl HardSphereProperties for SaftVRMiePars {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<N> {
        MonomerShape::NonSpherical(self.m.mapv(N::from))
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D> {
        let t_inv = temperature.recip();
        Array1::from_shape_fn(self.m.len(), |i| -> D { self.hs_diameter_ij(i, i, t_inv) })
    }
}

/// Utilities for running tests
#[doc(hidden)]
pub mod test_utils {
    use super::*;
    use feos_core::parameter::{AssociationRecord, Identifier};
    use std::collections::HashMap;

    /// Parameters from Lafitte et al. (2013)
    pub fn test_parameters() -> HashMap<&'static str, SaftVRMieParameters> {
        let mut parameters = HashMap::new();
        let id = Identifier::default();

        parameters.insert(
            "methane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                16.031,
                SaftVRMieRecord::new(1.0, 3.7412, 153.36, 12.65, 6.0),
            )),
        );

        parameters.insert(
            "ethane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                30.047,
                SaftVRMieRecord::new(1.4373, 3.7257, 206.12, 12.4, 6.0),
            )),
        );

        parameters.insert(
            "propane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                44.063,
                SaftVRMieRecord::new(1.6845, 3.9056, 239.89, 13.006, 6.0),
            )),
        );

        parameters.insert(
            "n-butane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                58.078,
                SaftVRMieRecord::new(1.8514, 4.0887, 273.64, 13.65, 6.0),
            )),
        );

        parameters.insert(
            "pentane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                72.094,
                SaftVRMieRecord::new(1.9606, 4.2928, 321.94, 15.847, 6.0),
            )),
        );

        parameters.insert(
            "hexane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                86.11,
                SaftVRMieRecord::new(2.1097, 4.423, 354.38, 17.203, 6.0),
            )),
        );

        parameters.insert(
            "heptane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                100.125,
                SaftVRMieRecord::new(2.3949, 4.4282, 358.51, 17.092, 6.0),
            )),
        );

        parameters.insert(
            "octane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                114.141,
                SaftVRMieRecord::new(2.6253, 4.4696, 369.18, 17.378, 6.0),
            )),
        );

        parameters.insert(
            "nonane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                128.157,
                SaftVRMieRecord::new(2.8099, 4.5334, 387.55, 18.324, 6.0),
            )),
        );

        parameters.insert(
            "decane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                142.172,
                SaftVRMieRecord::new(2.9976, 4.589, 400.79, 18.885, 6.0),
            )),
        );

        parameters.insert(
            "dodecane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                170.203,
                SaftVRMieRecord::new(3.2519, 4.7484, 437.72, 20.862, 6.0),
            )),
        );

        parameters.insert(
            "pentadecane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                212.25,
                SaftVRMieRecord::new(3.9325, 4.7738, 444.51, 20.822, 6.0),
            )),
        );

        parameters.insert(
            "eicosane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                282.329,
                SaftVRMieRecord::new(4.8794, 4.8788, 475.76, 22.926, 6.0),
            )),
        );

        parameters.insert(
            "methanol",
            SaftVRMieParameters::new_pure(PureRecord::with_association(
                id.clone(),
                32.026,
                SaftVRMieRecord::new(1.5283, 3.3063, 167.72, 8.6556, 6.0),
                vec![AssociationRecord::new(
                    Some(SaftVRMieAssociationRecord::new(0.41314, 2904.7)),
                    1.0,
                    1.0,
                    0.0,
                )],
            )),
        );

        parameters.insert(
            "ethanol",
            SaftVRMieParameters::new_pure(PureRecord::with_association(
                id.clone(),
                46.042,
                SaftVRMieRecord::new(1.96, 3.4914, 168.15, 7.6134, 6.0),
                vec![AssociationRecord::new(
                    Some(SaftVRMieAssociationRecord::new(0.34558, 2833.7)),
                    1.0,
                    1.0,
                    0.0,
                )],
            )),
        );

        parameters.insert(
            "1-propanol",
            SaftVRMieParameters::new_pure(PureRecord::with_association(
                id.clone(),
                60.058,
                SaftVRMieRecord::new(2.3356, 3.5612, 227.66, 10.179, 6.0),
                vec![AssociationRecord::new(
                    Some(SaftVRMieAssociationRecord::new(0.35377, 2746.2)),
                    1.0,
                    1.0,
                    0.0,
                )],
            )),
        );

        parameters.insert(
            "1-butanol",
            SaftVRMieParameters::new_pure(PureRecord::with_association(
                id.clone(),
                74.073,
                SaftVRMieRecord::new(2.4377, 3.7856, 278.92, 11.66, 6.0),
                vec![AssociationRecord::new(
                    Some(SaftVRMieAssociationRecord::new(0.32449, 2728.1)),
                    1.0,
                    1.0,
                    0.0,
                )],
            )),
        );

        parameters.insert(
            "tetrafluoromethane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                87.994,
                SaftVRMieRecord::new(1.0, 4.3372, 232.62, 42.553, 5.1906),
            )),
        );

        parameters.insert(
            "hexafluoroethane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                137.99,
                SaftVRMieRecord::new(1.8529, 3.9336, 211.46, 19.192, 5.7506),
            )),
        );

        parameters.insert(
            "perfluoropropane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                187.987,
                SaftVRMieRecord::new(1.9401, 4.2983, 263.26, 22.627, 5.7506),
            )),
        );

        parameters.insert(
            "perfluorobutane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                237.984,
                SaftVRMieRecord::new(2.1983, 4.4495, 290.49, 24.761, 5.7506),
            )),
        );

        parameters.insert(
            "perfluoropentane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                287.981,
                SaftVRMieRecord::new(2.3783, 4.6132, 328.56, 29.75, 5.7506),
            )),
        );

        parameters.insert(
            "perfluorohexane",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                337.978,
                SaftVRMieRecord::new(2.5202, 4.7885, 349.3, 30.741, 5.7506),
            )),
        );

        parameters.insert(
            "fluorine",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                37.997,
                SaftVRMieRecord::new(1.3211, 2.9554, 96.268, 11.606, 6.0),
            )),
        );

        parameters.insert(
            "carbon dioxide",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                43.99,
                SaftVRMieRecord::new(1.5, 3.1916, 231.88, 27.557, 5.1646),
            )),
        );

        parameters.insert(
            "benzene",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                78.047,
                SaftVRMieRecord::new(1.9163, 4.0549, 372.59, 14.798, 6.0),
            )),
        );

        parameters.insert(
            "toluene",
            SaftVRMieParameters::new_pure(PureRecord::new(
                id.clone(),
                92.063,
                SaftVRMieRecord::new(1.9977, 4.2777, 409.73, 16.334, 6.0),
            )),
        );

        parameters
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use num_dual::Dual2;
    use test::test_utils::test_parameters;

    use super::*;

    #[test]
    fn test_mie_potential() {
        let la = 6.0;
        let lr = 12.0;
        let c_eps_t = Dual2::from_re(4.0);
        let r = 0.9;
        let rd = Dual2::from_re(r).derivative();
        let u = beta_u_mie(rd, la, lr, 1.0, c_eps_t);
        let [_, u_du, du_d2u] = mie_potential_halley(r, la, lr, 4.0);
        assert_relative_eq!((-u.re - f64::EPSILON.ln()) / -u.v1, u_du);
        assert_relative_eq!(u.v1 / u.v2, du_d2u);
    }

    #[test]
    fn hs_diameter_ethane() {
        let temperature = 50.0;
        let ethane = SaftVRMiePars::new(&test_parameters()["ethane"]);
        let d_hs = ethane.hs_diameter(temperature);
        dbg!(&d_hs);
        assert_relative_eq!(
            3.694019351651498,
            d_hs[0],
            max_relative = 1e-9,
            epsilon = 1e-9
        )
    }

    #[test]
    fn test_zero_integrant() {
        let temperature = 50.0;
        let ethane = SaftVRMiePars::new(&test_parameters()["ethane"]);
        let la = ethane.la[0];
        let lr = ethane.lr[0];
        let c_eps_t = ethane.c_ij[[0, 0]] * ethane.epsilon_k[0] / temperature;
        let r0 = lower_integratal_limit(la, lr, c_eps_t);
        assert_relative_eq!(
            (-beta_u_mie(r0, la, lr, 1.0, c_eps_t)).exp(),
            f64::EPSILON,
            max_relative = 1e-15,
            epsilon = 1e-15
        )
    }
}
