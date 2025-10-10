use feos_core::{IdealGas, Molarweight, ResidualDyn, StateHD, Subset};
use nalgebra::{DVector, dvector};
use num_dual::{Dual2, DualNum, partial, second_derivative};
use quantity::{GRAM, JOULE, KELVIN, MOL, MolarWeight, RGAS};
use std::f64::consts::E;

const TC: f64 = 647.096;
const RHOC: f64 = 322.;
const R: f64 = 0.46151805;

// used to convert from reduced density to delta
const DELTA_OVER_RHO: f64 = 1.380649e4 / (RHOC * R);

const P1_7: [(i32, f64, f64); 7] = [
    (1, -0.5, 0.125335479355233e-1),
    (1, 0.875, 0.789576347228283e1),
    (1, 1.0, -0.878032033035613e1),
    (2, 0.5, 0.31802509345418),
    (2, 0.75, -0.26145533859358),
    (3, 0.375, -0.781997516879813e-2),
    (4, 1.0, 0.880894931021343e-2),
];
const P8_51: [(i32, i32, i32, f64); 44] = [
    (1, 1, 4, -0.66856572307965),
    (1, 1, 6, 0.20433810950965),
    (1, 1, 12, -0.662126050396873e-4),
    (1, 2, 1, -0.19232721156002),
    (1, 2, 5, -0.25709043003438),
    (1, 3, 4, 0.16074868486251),
    (1, 4, 2, -0.400928289258073e-1),
    (1, 4, 13, 0.393434226032543e-6),
    (1, 5, 9, -0.759413770881443e-5),
    (1, 7, 3, 0.562509793518883e-3),
    (1, 9, 4, -0.156086522571353e-4),
    (1, 10, 11, 0.115379964229513e-8),
    (1, 11, 4, 0.365821651442043e-6),
    (1, 13, 13, -0.132511800746683e-11),
    (1, 15, 1, -0.626395869124543e-9),
    (2, 1, 7, -0.10793600908932),
    (2, 2, 1, 0.176114910087523e-1),
    (2, 2, 9, 0.22132295167546),
    (2, 2, 10, -0.40247669763528),
    (2, 3, 10, 0.58083399985759),
    (2, 4, 3, 0.499691469908063e-2),
    (2, 4, 7, -0.313587007125493e-1),
    (2, 4, 10, -0.74315929710341),
    (2, 5, 10, 0.47807329915480),
    (2, 6, 6, 0.205279408959483e-1),
    (2, 6, 10, -0.13636435110343),
    (2, 7, 10, 0.141806344006173e-1),
    (2, 9, 1, 0.833265048807133e-2),
    (2, 9, 2, -0.290523360095853e-1),
    (2, 9, 3, 0.386150855742063e-1),
    (2, 9, 4, -0.203934865137043e-1),
    (2, 9, 8, -0.165540500637343e-2),
    (2, 10, 6, 0.199555719795413e-2),
    (2, 10, 9, 0.158703083241573e-3),
    (2, 12, 8, -0.163885683425303e-4),
    (3, 3, 16, 0.436136157238113e-1),
    (3, 4, 22, 0.349940054637653e-1),
    (3, 4, 23, -0.767881978446213e-1),
    (3, 5, 23, 0.224462773320063e-1),
    (4, 14, 10, -0.626897104146853e-4),
    (6, 3, 50, -0.557111185656453e-9),
    (6, 6, 44, -0.19905718354408),
    (6, 6, 46, 0.31777497330738),
    (6, 6, 50, -0.11841182425981),
];

const P52_54: [(i32, i32, f64, f64, f64, f64); 3] = [
    (3, 0, -0.313062603234353e2, 20., 150., 1.21),
    (3, 1, 0.315461402377813e2, 20., 150., 1.21),
    (3, 4, -0.252131543416953e4, 20., 250., 1.25),
];

const P55_56: [[f64; 8]; 2] = [
    [3.5, 0.85, 0.2, -0.14874640856724, 28., 700., 0.32, 0.3],
    [3.5, 0.95, 0.2, 0.31806110878444, 32., 800., 0.32, 0.3],
];

const I1_3: [f64; 3] = [-8.32044648201, 6.6832105268, 3.00632];

const I4_8: [[f64; 2]; 5] = [
    [0.012436, 1.28728967],
    [0.97315, 3.53734222],
    [1.27950, 7.74073708],
    [0.96956, 9.24437796],
    [0.24873, 27.5075105],
];

const SAT_LIQ: [f64; 6] = [
    1.99274064,
    1.09965342,
    -0.510839303,
    -1.75493479,
    -45.5170352,
    -6.74694450e5,
];

const SAT_VAP: [f64; 6] = [
    -2.03105240,
    -2.68302940,
    -5.38626492,
    -17.2991605,
    -44.7586581,
    -63.9201063,
];

fn delta_sat<D: DualNum<f64> + Copy>(temperature: D) -> [D; 2] {
    let tau = -temperature / TC + 1.0;
    let tau_6 = tau.powf(1.0 / 6.0);
    let rho_liquid = tau_6.powi(2) * SAT_LIQ[0]
        + tau_6.powi(4) * SAT_LIQ[1]
        + tau_6.powi(10) * SAT_LIQ[2]
        + tau_6.powi(32) * SAT_LIQ[3]
        + tau_6.powi(86) * SAT_LIQ[4]
        + tau_6.powi(220) * SAT_LIQ[5]
        + 1.0;
    let rho_vapor = (tau_6.powi(2) * SAT_VAP[0]
        + tau_6.powi(4) * SAT_VAP[1]
        + tau_6.powi(8) * SAT_VAP[2]
        + tau_6.powi(18) * SAT_VAP[3]
        + tau_6.powi(37) * SAT_VAP[4]
        + tau_6.powi(71) * SAT_VAP[5])
        .exp();

    [rho_liquid, rho_vapor]
}

fn phi_r<D: DualNum<f64> + Copy>(delta: D, tau: D) -> D {
    let delta_2 = (delta - 1.0).powi(2);
    let mut phi = D::zero();
    for (d, t, n) in P1_7 {
        phi += delta.powi(d) * tau.powf(t) * n;
    }
    for (c, d, t, n) in P8_51 {
        phi += delta.powi(d) * tau.powi(t) * n * (-delta.powi(c)).exp();
    }
    for (d, t, n, alpha, beta, gamma) in P52_54 {
        phi += (delta.powi(d) * tau.powi(t) * n)
            * (-delta_2 * alpha - (tau - gamma).powi(2) * beta).exp();
    }
    for [a, b, bb, n, cc, dd, aa, beta] in P55_56 {
        let psi = (-delta_2 * cc - (tau - 1.0).powi(2) * dd).exp();
        let theta = -(tau - 1.0) + delta_2.powf(0.5 / beta) * aa;
        let ddelta = theta.powi(2) + delta_2.powf(a) * bb;
        phi += ddelta.powf(b) * delta * psi * n;
    }
    phi
}

fn phi_o<D: DualNum<f64> + Copy>(delta: D, tau: D) -> D {
    let mut phi = delta.ln() + I1_3[0] + tau * I1_3[1] + tau.ln() * I1_3[2];
    for [n, gamma] in I4_8 {
        phi += (-(-tau * gamma).exp()).ln_1p() * n
    }
    phi
}

fn phi_extrapolated<D: DualNum<f64> + Copy>(delta: D, t: D, t_stb: f64) -> D {
    let (a, at, att) = second_derivative(
        partial(
            |t: Dual2<D, f64>, &delta| {
                let tau = t.recip() * TC;
                phi_r(delta, tau) * t
            },
            &delta,
        ),
        D::from(t_stb * TC),
    );
    let dt = t - t_stb * TC;

    (a + at * dt + att * dt * dt * 0.5) / t
}

#[derive(Clone, Copy)]
pub enum IAPWS {
    Base,
    Smooth(f64, f64),
    Extrapolated(f64),
}

impl ResidualDyn for IAPWS {
    fn components(&self) -> usize {
        1
    }

    fn compute_max_density<D: DualNum<f64> + Copy>(&self, _: &DVector<D>) -> D {
        // Not sure what value works well here. This one is based on a mass density of 1000 kg/m³.
        D::from(0.033456)
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        let rho = state.partial_density.sum();
        let delta = rho * DELTA_OVER_RHO;
        match *self {
            Self::Extrapolated(t_stb) => {
                vec![(
                    "IAPWS95 (Extrapolated)",
                    phi_extrapolated(delta, state.temperature, t_stb) * rho,
                )]
            }
            Self::Smooth(t_stb, d) => {
                let [delta_liq, delta_vap] = delta_sat(state.temperature);
                fn f<D: DualNum<f64> + Copy>(x: D) -> D {
                    if x.re() > 0.0 {
                        (-x.recip()).exp()
                    } else {
                        D::zero()
                    }
                }
                fn g<D: DualNum<f64> + Copy>(x: D) -> D {
                    f(x) / (f(x) + f(-x + 1.0))
                }
                let bump = g((delta - delta_vap) / d) * g((delta_liq - delta) / d);
                let ex = phi_extrapolated(delta, state.temperature, t_stb);
                let tau = state.temperature.recip() * TC;
                let phi = phi_r(delta, tau);
                vec![(
                    "IAPWS95 (Smoothed)",
                    (ex * bump - phi * (bump - 1.0) * 0.0) * rho,
                )]
            }
            Self::Base => {
                let tau = state.temperature.recip() * TC;
                vec![("IAPWS95", phi_r(delta, tau) * rho)]
            }
        }
    }
}

impl IdealGas for IAPWS {
    fn ln_lambda3<D: DualNum<f64, Inner = f64> + Copy>(&self, temperature: D) -> D {
        let tau = temperature.recip() * TC;
        // bit of a hack to convert from phi^0 into ln Lambda^3
        phi_o(D::from(E * DELTA_OVER_RHO), tau)
    }

    fn ideal_gas_model(&self) -> &'static str {
        "IAPWS95"
    }
}

impl Subset for IAPWS {
    fn subset(&self, _: &[usize]) -> Self {
        *self
    }
}

impl Molarweight for IAPWS {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        dvector![RGAS.convert_into(JOULE / (KELVIN * MOL)) / R] * GRAM / MOL
    }
}

#[cfg(test)]
mod test {
    use feos_core::{EquationOfState, State, Total};
    use nalgebra::{SVector, dvector};
    use num_dual::{hessian, partial, third_derivative};
    use quantity::{JOULE, KELVIN, KILO, KILOGRAM, METER, MOL, RGAS};
    use typenum::P3;

    use super::*;

    #[test]
    fn test_phi_r_1() {
        let t = 500.;
        let rho = 838.025;
        let tau = TC / t;
        let delta = rho / RHOC;
        let (phi, dphi, d2phi) = hessian(
            |x| {
                let [delta, tau] = x.data.0[0];
                phi_r(delta, tau)
            },
            &SVector::from([delta, tau]),
        );
        println!("{}\n{}\n{}", phi, dphi, d2phi);
        assert_eq!(format!("{phi:.8}"), "-3.42693206");
        assert_eq!(format!("{:.8}", dphi[0]), "-0.36436665");
        assert_eq!(format!("{:.8}", dphi[1]), "-5.81403435");
        assert_eq!(format!("{:.8}", d2phi[(0, 0)]), "0.85606370");
        assert_eq!(format!("{:.8}", d2phi[(0, 1)]), "-1.12176915");
        assert_eq!(format!("{:.8}", d2phi[(1, 1)]), "-2.23440737");
    }

    #[test]
    fn test_phi_r_2() {
        let t = 647.;
        let rho = 358.;
        let tau = TC / t;
        let delta = rho / RHOC;
        let (phi, dphi, d2phi) = hessian(
            |x| {
                let [delta, tau] = x.data.0[0];
                phi_r(delta, tau)
            },
            &SVector::from([delta, tau]),
        );
        println!("{}\n{}\n{}", phi, dphi, d2phi);
        assert_eq!(format!("{phi:.8}"), "-1.21202657");
        assert_eq!(format!("{:.8}", dphi[0]), "-0.71401202");
        assert_eq!(format!("{:.8}", dphi[1]), "-3.21722501");
        assert_eq!(format!("{:.8}", d2phi[(0, 0)]), "0.47573070");
        assert_eq!(format!("{:.8}", d2phi[(0, 1)]), "-1.33214720");
        assert_eq!(format!("{:.8}", d2phi[(1, 1)]), "-9.96029507");
    }

    #[test]
    fn test_phi_o_1() {
        let t = 500.;
        let rho = 838.025;
        let tau = TC / t;
        let delta = rho / RHOC;
        let (phi, dphi, d2phi) = hessian(
            |x| {
                let [delta, tau] = x.data.0[0];
                phi_o(delta, tau)
            },
            &SVector::from([delta, tau]),
        );
        println!("{}\n{}\n{}", phi, dphi, d2phi);
        assert_eq!(format!("{phi:.8}"), "2.04797734");
        assert_eq!(format!("{:.8}", dphi[0]), "0.38423675");
        assert_eq!(format!("{:.8}", dphi[1]), "9.04611106");
        assert_eq!(format!("{:.8}", d2phi[(0, 0)]), "-0.14763788");
        assert_eq!(format!("{:.8}", d2phi[(0, 1)]), "-0.00000000");
        assert_eq!(format!("{:.8}", d2phi[(1, 1)]), "-1.93249185");
    }

    #[test]
    fn test_phi_o_2() {
        let t = 647.;
        let rho = 358.;
        let tau = TC / t;
        let delta = rho / RHOC;
        let (phi, dphi, d2phi) = hessian(
            |x| {
                let [delta, tau] = x.data.0[0];
                phi_o(delta, tau)
            },
            &SVector::from([delta, tau]),
        );
        println!("{}\n{}\n{}", phi, dphi, d2phi);
        assert_eq!(format!("{phi:.8}"), "-1.56319605");
        assert_eq!(format!("{:.8}", dphi[0]), "0.89944134");
        assert_eq!(format!("{:.8}", dphi[1]), "9.80343918");
        assert_eq!(format!("{:.8}", d2phi[(0, 0)]), "-0.80899473");
        assert_eq!(format!("{:.8}", d2phi[(0, 1)]), "-0.00000000");
        assert_eq!(format!("{:.8}", d2phi[(1, 1)]), "-3.43316334");
    }

    #[test]
    fn test_ideal_gas_hack() {
        let t = 647. * KELVIN;
        let rho = 358. * KILOGRAM / METER.powi::<P3>();
        let mw = RGAS / (R * KILO * JOULE / (KILOGRAM * KELVIN));
        let moles = dvector![1.8] * MOL;
        let eos = &EquationOfState::ideal_gas(vec![IAPWS::Base]);
        let a_feos = eos.ideal_gas_helmholtz_energy(t, moles.sum() * mw / rho, &moles);
        let phi_feos = (a_feos / RGAS / moles.sum() / t).into_value();
        println!("A:          {a_feos}");
        println!("phi(feos):  {phi_feos}");
        let delta = (rho / (RHOC * KILOGRAM / METER.powi::<P3>())).into_value();
        let tau = (TC * KELVIN / t).into_value();
        let phi = phi_o(delta, tau);
        println!("phi(IAPWS): {phi}");
        assert_eq!(phi_feos, phi)
    }

    #[test]
    fn test_critical_point() {
        let (phi, dphi, d2phi, d3phi) =
            third_derivative(partial(|delta, &tau| phi_o(delta, tau), &1.), 1.);
        println!("{}\n{}\n{}\n{}", phi, dphi, d2phi, d3phi);
        // assert_eq!(format!("{phi:.8}"), "-1.56319605");
        // assert_eq!(format!("{:.8}", dphi[0]), "0.89944134");
        // assert_eq!(format!("{:.8}", dphi[1]), "9.80343918");
        // assert_eq!(format!("{:.8}", d2phi[(0, 0)]), "-0.80899473");
        // assert_eq!(format!("{:.8}", d2phi[(0, 1)]), "-0.00000000");
        // assert_eq!(format!("{:.8}", d2phi[(1, 1)]), "-3.43316334");
        let iapws = &IAPWS::Base;
        let cp: State<_, _, f64> =
            State::critical_point(&iapws, None, Some(TC * KELVIN), Default::default()).unwrap();
        println!("{cp}");
        let cp: State<_, _, f64> =
            State::critical_point(&iapws, None, None, Default::default()).unwrap();
        println!("{cp}");
        let cp: State<_, _, f64> =
            State::critical_point(&iapws, None, Some(700.0 * KELVIN), Default::default()).unwrap();
        println!("{cp}");
    }
}
