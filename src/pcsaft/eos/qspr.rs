use super::PcSaftParameters;
use feos_core::IdealGasContributionDual;
use ndarray::Array1;
use num_dual::*;
use std::fmt;
use std::sync::Arc;

const RGAS: f64 = 6.022140857 * 1.38064852;
const KB: f64 = 1.38064852e-23;
const T300: f64 = 300.0;
const T400: f64 = 400.0;
const T0: f64 = 298.15;
const P0: f64 = 1.0e5;
const A3: f64 = 1e-30;

// Heat capacity parameters @ T = 300 K (col 1) and T = 400 K (col 2)
const NA_NP_300: [f64; 6] = [
    -5763.04893,
    1232.30607,
    -239.3513996,
    0.0,
    0.0,
    -15174.28321,
];
const NA_NP_400: [f64; 6] = [
    -8171.26676935062,
    1498.01217504596,
    -315.515836223387,
    0.0,
    0.0,
    -19389.5468655708,
];
const NA_P_300: [f64; 6] = [
    5177.19095226181,
    919.565206504576,
    -108.829105648889,
    0.0,
    -3.93917830677682,
    -13504.5671858292,
];
const NA_P_400: [f64; 6] = [
    10656.1018362315,
    1146.10782703748,
    -131.023645998081,
    0.0,
    -9.93789225413177,
    -24430.12952497,
];
const AP_300: [f64; 6] = [
    3600.32322462175,
    1006.20461224949,
    -151.688378113974,
    7.81876773647109e-07,
    8.01001754473385,
    -8959.37140957179,
];
const AP_400: [f64; 6] = [
    7248.0697641199,
    1267.44346171358,
    -208.738557800023,
    0.000170238690157906,
    -6.7841792685616,
    -12669.4196622924,
];

#[allow(clippy::upper_case_acronyms)]
pub struct QSPR {
    pub parameters: Arc<PcSaftParameters>,
}

impl<D: DualNum<f64>> IdealGasContributionDual<D> for QSPR {
    fn de_broglie_wavelength(&self, temperature: D, components: usize) -> Array1<D> {
        let (c_300, c_400) = if self.parameters.association.is_empty() {
            match self.parameters.ndipole + self.parameters.nquadpole {
                0 => (NA_NP_300, NA_NP_400),
                _ => (NA_P_300, NA_P_400),
            }
        } else {
            (AP_300, AP_400)
        };

        Array1::from_shape_fn(components, |i| {
            let epsilon_kt = temperature.recip() * self.parameters.epsilon_k[i];
            let sigma3 = self.parameters.sigma[i].powi(3);

            let p1 = epsilon_kt * self.parameters.m[i];
            let p2 = sigma3 * self.parameters.m[i];
            let p3 = epsilon_kt * p2;
            let p4 = self.parameters.pure_records[i]
                .model_record
                .association_record
                .as_ref()
                .map_or(D::zero(), |a| {
                    (temperature.recip() * a.epsilon_k_ab).exp_m1() * p2 * sigma3 * a.kappa_ab
                });
            let p5 = p2 * self.parameters.q[i];
            let p6 = 1.0;

            let icpc300 = (p1 * c_300[0] / T300
                + p2 * c_300[1]
                + p3 * c_300[2] / T300
                + p4 * c_300[3] / T300
                + p5 * c_300[4]
                + p6 * c_300[5])
                * 0.001;
            let icpc400 = (p1 * c_400[0] / T400
                + p2 * c_400[1]
                + p3 * c_400[2] / T400
                + p4 * c_400[3] / T400
                + p5 * c_400[4]
                + p6 * c_400[5])
                * 0.001;

            // linear approximation
            let b = (icpc400 - icpc300) / (T400 - T300);
            let a = icpc300 - b * T300;

            // integration
            let k = a * (temperature - T0 - temperature * (temperature / T0).ln())
                - b * (temperature - T0).powi(2) * 0.5;

            // de Broglie wavelength
            k / (temperature * RGAS) + (temperature * KB / (P0 * A3)).ln()
        })
    }
}

impl fmt::Display for QSPR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ideal gas (QSPR)")
    }
}
