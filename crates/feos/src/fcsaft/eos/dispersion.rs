use feos_core::StateHD;
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_6, PI};

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

/// Model constants of the dispersion contribution: `[a1, a2, b1, b2]`.
pub type DispersionConstants = [[f64; 7]; 4];

/// The dispersion contribution of FC-SAFT.
///
/// The contribution is evaluated for "segments", which are either the individual
/// segments of the heterosegmented model, or the components of the homosegmented
/// model (with the reduced surface $m$ and volume $m^*$ of the whole molecule).
pub(crate) struct Dispersion {
    component_index: DVector<usize>,
    sigma3_ij: DMatrix<f64>,
    epsilon_k_ij: DMatrix<f64>,
    a1: [f64; 7],
    a2: [f64; 7],
    b1: [f64; 7],
    b2: [f64; 7],
}

impl Dispersion {
    pub(crate) fn new(
        component_index: &DVector<usize>,
        sigma_ij: &DMatrix<f64>,
        epsilon_k_ij: &DMatrix<f64>,
        model_constants: Option<DispersionConstants>,
    ) -> Self {
        let [a1, a2, b1, b2] = model_constants.unwrap_or([A1, A2, B1, B2]);
        Self {
            component_index: component_index.clone(),
            sigma3_ij: sigma_ij.map(|s| s.powi(3)),
            epsilon_k_ij: epsilon_k_ij.clone(),
            a1,
            a2,
            b1,
            b2,
        }
    }

    /// The Helmholtz energy density for given temperature dependent diameters,
    /// surface and volume coefficients $a$ and $v$, and compressibility term $C_1$.
    pub(crate) fn helmholtz_energy_density<D: DualNum<Primitive = f64> + Copy>(
        &self,
        state: &StateHD<D>,
        diameter: &DVector<D>,
        [a, v]: &[DVector<D>; 2],
        c1: D,
    ) -> D {
        // auxiliary variables
        let n = diameter.len();
        let c = &self.component_index;
        let rho = &state.partial_density;
        let x = &state.molefracs;

        // packing fraction
        let eta: D = (0..n)
            .map(|i| rho[c[i]] * v[i] * diameter[i].powi(3) * FRAC_PI_6)
            .sum();

        // mean segment number
        let m: D = (0..n).map(|i| x[c[i]] * a[i]).sum();

        // mixture densities, crosswise interactions of all segments on all chains
        let mut rho1mix = D::zero();
        let mut rho2mix = D::zero();
        for i in 0..n {
            for j in 0..n {
                let eps_ij = state.temperature.recip() * self.epsilon_k_ij[(i, j)];
                let rho1 = rho[c[i]] * rho[c[j]] * a[i] * a[j] * eps_ij * self.sigma3_ij[(i, j)];
                rho1mix += rho1;
                rho2mix += rho1 * eps_ij;
            }
        }

        // I1 and I2
        let mut i1 = D::zero();
        let mut i2 = D::zero();
        let mut eta_i = D::one();
        for i in 0..=6 {
            i1 += ((m - 1.0) / m * ((m - 2.0) / m * self.a2[i] + self.a1[i]) + A0[i]) * eta_i;
            i2 += ((m - 1.0) / m * ((m - 2.0) / m * self.b2[i] + self.b1[i]) + B0[i]) * eta_i;
            eta_i *= eta;
        }

        // Helmholtz energy density
        (-rho1mix * i1 * 2.0 - rho2mix * m * c1 * i2) * PI
    }
}
