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
    -0.7967554406765953,
    2.6790885480460225,
    -3.487146197861997,
    12.880375286361222,
    -63.678387954754264,
    135.2293725600522,
    -92.36174418380334,
];
pub const A2: [f64; 7] = [
    0.07899144809821383,
    -0.46265604871501964,
    -0.16852522804500122,
    -0.5690540299027762,
    -0.10891595731602507,
    6.978853948715148,
    -5.413574400329618,
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
    0.2384607176760574,
    6.430183782977854,
    -16.03207359098296,
    6.551695787529511,
    31.444257990183306,
    -500.9301850495875,
    762.2725790356507,
];
pub const B2: [f64; 7] = [
    -0.3521344090059574,
    -4.861304748454132,
    49.256434792398586,
    -153.85773541338776,
    82.30476426689317,
    1467.9798510614366,
    -2009.0462529305892,
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
