use super::dispersion::{A0, A1, A2, B0, B1, B2};
use super::polar::{AD, BD, CD};
use feos_core::{ParametersAD, ResidualConst, StateHD};
use nalgebra::{SVector, U2};
use num_dual::{DualNum, DualSVec, DualVec, jacobian};
use std::f64::consts::{FRAC_PI_6, PI};

const PI_SQ_43: f64 = 4.0 / 3.0 * PI * PI;

const MAX_ETA: f64 = 0.5;

/// Optimized implementation of PC-SAFT for a binary mixture.
#[derive(Clone, Copy)]
pub struct PcSaftBinary<D, const N: usize>(([[D; N]; 2], D));

impl<D, const N: usize> PcSaftBinary<D, N> {
    pub fn new(parameters: [[D; N]; 2], kij: D) -> Self {
        Self((parameters, kij))
    }
}

impl<D: DualNum<f64> + Copy, const N: usize> From<&[f64]> for PcSaftBinary<D, N> {
    fn from(parameters: &[f64]) -> Self {
        if parameters.len() != 2 * N + 1 {
            panic!(
                "This version of PC-SAFT requires exactly {} parameters!",
                2 * N + 1
            )
        }
        let (Ok(p1), Ok(p2)): (Result<[f64; N], _>, Result<[f64; N], _>) =
            (parameters[..N].try_into(), parameters[N..2 * N].try_into())
        else {
            unreachable!()
        };
        let kij = D::from(parameters[2 * N]);
        Self::new([p1.map(D::from), p2.map(D::from)], kij)
    }
}

impl<const N: usize, const P: usize> ParametersAD<P> for PcSaftBinary<DualSVec<f64, f64, P>, N> {
    fn index_parameters_mut<'a>(&'a mut self, index: &str) -> &'a mut DualSVec<f64, f64, P> {
        match index {
            "k_ij" => &mut self.0.1,
            _ => panic!("{index} is not a valid binary PC-SAFT parameter!"),
        }
    }
}

fn hard_sphere<D: DualNum<f64> + Copy>(
    [m1, m2]: [D; 2],
    [x1, x2]: [D; 2],
    [d1, d2]: [D; 2],
    density: D,
) -> (D, [D; 7], D, D, D) {
    // Packing fractions
    let zeta = [0, 1, 2, 3].map(|k| (m1 * x1 * d1.powi(k) + m2 * x2 * d2.powi(k)) * FRAC_PI_6);
    let zeta_23 = zeta[2] / zeta[3];
    let [zeta0, zeta1, zeta2, zeta3] = zeta.map(|z| z * density);
    let frac_1mz3 = (-zeta3 + 1.0).recip();

    let eta = zeta3;
    let eta2 = eta * eta;
    let eta3 = eta2 * eta;
    let etas = [
        D::one(),
        eta,
        eta2,
        eta3,
        eta2 * eta2,
        eta2 * eta3,
        eta3 * eta3,
    ];

    // hard sphere
    let hs = (zeta1 * zeta2 * frac_1mz3 * 3.0
        + zeta2.powi(2) * frac_1mz3.powi(2) * zeta_23
        + (zeta2 * zeta_23.powi(2) - zeta0) * (-zeta3).ln_1p())
        / std::f64::consts::FRAC_PI_6;

    (hs, etas, zeta2, zeta3, frac_1mz3)
}

fn hard_chain<D: DualNum<f64> + Copy>(
    [m1, m2]: [D; 2],
    [d1, d2]: [D; 2],
    [rho1, rho2]: [D; 2],
    zeta2: D,
    zeta3: D,
    frac_1mz3: D,
) -> D {
    let c = zeta2 * frac_1mz3 * frac_1mz3;
    let g_hs = [d1, d2].map(|d| frac_1mz3 + d * c * 1.5 - (d * c).powi(2) * (zeta3 - 1.0) * 0.5);
    -(rho1 * (m1 - 1.0) * g_hs[0].ln() + rho2 * (m2 - 1.0) * g_hs[1].ln())
}

#[expect(clippy::too_many_arguments)]
fn dispersion<D: DualNum<f64> + Copy>(
    [m1, m2]: [D; 2],
    [sigma1, sigma2]: [D; 2],
    [epsilon_k1, epsilon_k2]: [D; 2],
    kij: D,
    [x1, x2]: [D; 2],
    t_inv: D,
    [rho1, rho2]: [D; 2],
    etas: [D; 7],
    frac_1mz3: D,
) -> (D, [D; 3], [D; 3]) {
    // binary interactions
    let m = x1 * m1 + x2 * m2;
    let epsilon_k11 = epsilon_k1 * t_inv;
    let epsilon_k12 = (epsilon_k1 * epsilon_k2).sqrt() * t_inv;
    let epsilon_k22 = epsilon_k2 * t_inv;
    let sigma11_3 = sigma1.powi(3);
    let sigma12_3 = ((sigma1 + sigma2) * 0.5).powi(3);
    let sigma22_3 = sigma2.powi(3);
    let d11 = rho1 * rho1 * m1 * m1 * epsilon_k11 * sigma11_3;
    let d12 = rho1 * rho2 * m1 * m2 * epsilon_k12 * sigma12_3 * (-kij + 1.0);
    let d22 = rho2 * rho2 * m2 * m2 * epsilon_k22 * sigma22_3;
    let rho1mix = d11 + d12 * 2.0 + d22;
    let rho2mix = d11 * epsilon_k11 + d12 * epsilon_k12 * (-kij + 1.0) * 2.0 + d22 * epsilon_k22;

    // I1, I2 and C1
    let mm1 = (m - 1.0) / m;
    let mm2 = (m - 2.0) / m;
    let mut i1 = D::zero();
    let mut i2 = D::zero();
    for i in 0..7 {
        i1 += (mm1 * (mm2 * A2[i] + A1[i]) + A0[i]) * etas[i];
        i2 += (mm1 * (mm2 * B2[i] + B1[i]) + B0[i]) * etas[i];
    }
    let eta_m2 = frac_1mz3 * frac_1mz3;
    let c1 = (m * (etas[1] * 8.0 - etas[2] * 2.0) * eta_m2 * eta_m2 + 1.0
        - (m - 1.0) * (etas[1] * 20.0 - etas[2] * 27.0 + etas[3] * 12.0 - etas[4] * 2.0)
            / ((etas[1] - 1.0) * (etas[1] - 2.0)).powi(2))
    .recip();

    // dispersion
    let disp = (-rho1mix * i1 * 2.0 - rho2mix * m * c1 * i2) * PI;

    (
        disp,
        [sigma11_3, sigma12_3, sigma22_3],
        [epsilon_k11, epsilon_k12, epsilon_k22],
    )
}

#[expect(clippy::too_many_arguments)]
fn dipoles<D: DualNum<f64> + Copy>(
    [m1, m2]: [D; 2],
    [sigma1, sigma2]: [D; 2],
    [sigma11_3, sigma12_3, sigma22_3]: [D; 3],
    [epsilon_k11, epsilon_k12, epsilon_k22]: [D; 3],
    [mu1, mu2]: [D; 2],
    temperature: D,
    [rho1, rho2]: [D; 2],
    etas: [D; 7],
) -> D {
    let mu_term1 = mu1 * mu1 / (m1 * temperature * 1.380649e-4) * rho1;
    let mu_term2 = mu2 * mu2 / (m2 * temperature * 1.380649e-4) * rho2;
    let sigma111 = sigma11_3;
    let sigma112 = sigma1 * ((sigma1 + sigma2) * 0.5).powi(2);
    let sigma122 = sigma2 * ((sigma1 + sigma2) * 0.5).powi(2);
    let sigma222 = sigma22_3;

    let m11_dipole = if m1.re() > 2.0 { D::from(2.0) } else { m1 };
    let m22_dipole = if m2.re() > 2.0 { D::from(2.0) } else { m2 };
    let m12_dipole = (m11_dipole * m22_dipole).sqrt();
    let [j2_11, j2_12, j2_22] = [
        (m11_dipole, epsilon_k11),
        (m12_dipole, epsilon_k12),
        (m22_dipole, epsilon_k22),
    ]
    .map(|(m, e)| {
        let m1 = (m - 1.0) / m;
        let m2 = m1 * (m - 2.0) / m;
        let mut j2 = D::zero();
        for i in 0..5 {
            let a = m2 * AD[i][2] + m1 * AD[i][1] + AD[i][0];
            let b = m2 * BD[i][2] + m1 * BD[i][1] + BD[i][0];
            j2 += (a + b * e) * etas[i];
        }
        j2
    });
    let m112_dipole = (m11_dipole * m11_dipole * m22_dipole).cbrt();
    let m122_dipole = (m11_dipole * m22_dipole * m22_dipole).cbrt();
    let [j3_111, j3_112, j3_122, j3_222] =
        [m11_dipole, m112_dipole, m122_dipole, m22_dipole].map(|m| {
            let m1 = (m - 1.0) / m;
            let m2 = m1 * (m - 2.0) / m;
            let mut j3 = D::zero();
            for i in 0..4 {
                j3 += (m2 * CD[i][2] + m1 * CD[i][1] + CD[i][0]) * etas[i];
            }
            j3
        });

    let phi2 = (mu_term1 * mu_term1 / sigma11_3 * j2_11
        + mu_term1 * mu_term2 / sigma12_3 * j2_12 * 2.0
        + mu_term2 * mu_term2 / sigma22_3 * j2_22)
        * (-PI);
    let phi3 = (mu_term1.powi(3) / sigma111 * j3_111
        + mu_term1.powi(2) * mu_term2 / sigma112 * j3_112 * 3.0
        + mu_term1 * mu_term2.powi(2) / sigma122 * j3_122 * 3.0
        + mu_term2.powi(3) / sigma222 * j3_222)
        * (-PI_SQ_43);

    let mut polar = phi2 * phi2 / (phi2 - phi3);
    if polar.re().is_nan() {
        polar = phi2
    }

    polar
}

fn association<D: DualNum<f64> + Copy>(
    assoc_params: [[D; 4]; 2],
    [sigma11_3, _, sigma22_3]: [D; 3],
    t_inv: D,
    [rho1, rho2]: [D; 2],
    [d1, d2]: [D; 2],
    zeta2: D,
    frac_1mz3: D,
) -> D {
    let [
        [kappa_ab1, epsilon_k_ab1, na1, nb1],
        [kappa_ab2, epsilon_k_ab2, na2, nb2],
    ] = assoc_params;

    let d11 = d1 * 0.5;
    let d12 = d1 * d2 / (d1 + d2);
    let d22 = d2 * 0.5;
    let [k11, k12, k22] = [d11, d12, d22].map(|d| d * zeta2 * frac_1mz3);
    let s11 = sigma11_3 * kappa_ab1;
    let mut s12 = (sigma11_3 * sigma22_3 * kappa_ab1 * kappa_ab2).sqrt();
    if s12.re() == 0.0 {
        s12 = D::zero();
    }
    let s22 = sigma22_3 * kappa_ab2;
    let e11 = (epsilon_k_ab1 * t_inv).exp() - 1.0;
    let e12 = ((epsilon_k_ab1 + epsilon_k_ab2) * 0.5 * t_inv).exp() - 1.0;
    let e22 = (epsilon_k_ab2 * t_inv).exp() - 1.0;
    let d11 = frac_1mz3 * (k11 * (k11 * 2.0 + 3.0) + 1.0) * s11 * e11;
    let d12 = frac_1mz3 * (k12 * (k12 * 2.0 + 3.0) + 1.0) * s12 * e12;
    let d22 = frac_1mz3 * (k22 * (k22 * 2.0 + 3.0) + 1.0) * s22 * e22;
    let rhoa1 = rho1 * na1;
    let rhob1 = rho1 * nb1;
    let rhoa2 = rho2 * na2;
    let rhob2 = rho2 * nb2;

    let [mut xa1, mut xa2] = [D::from(0.2); 2];
    for _ in 0..50 {
        let (g, j) = jacobian(
            |x| {
                let [xa1, xa2] = x.data.0[0];
                let xb1_i =
                    xa1 * DualVec::from_re(rhoa1 * d11) + xa2 * DualVec::from_re(rhoa2 * d12) + 1.0;
                let xb2_i =
                    xa1 * DualVec::from_re(rhoa1 * d12) + xa2 * DualVec::from_re(rhoa2 * d22) + 1.0;

                let f1 = xa1 - 1.0
                    + xa1 / xb1_i * DualVec::from_re(rhob1 * d11)
                    + xa1 / xb2_i * DualVec::from_re(rhob2 * d12);
                let f2 = xa2 - 1.0
                    + xa2 / xb1_i * DualVec::from_re(rhob1 * d12)
                    + xa2 / xb2_i * DualVec::from_re(rhob2 * d22);

                SVector::from([f1, f2])
            },
            SVector::from([xa1, xa2]),
        );

        let [g1, g2] = g.data.0[0];
        let [[j11, j12], [j21, j22]] = j.data.0;
        let det = j11 * j22 - j12 * j21;

        let delta_xa1 = (j22 * g1 - j12 * g2) / det;
        let delta_xa2 = (-j21 * g1 + j11 * g2) / det;
        if delta_xa1.re() < xa1.re() * 0.8 {
            xa1 -= delta_xa1;
        } else {
            xa1 *= 0.2;
        }
        if delta_xa2.re() < xa2.re() * 0.8 {
            xa2 -= delta_xa2;
        } else {
            xa2 *= 0.2;
        }

        if g1.re().abs() < 1e-15 && g2.re().abs() < 1e-15 {
            break;
        }
    }

    let xb1 = (xa1 * rhoa1 * d11 + xa2 * rhoa2 * d12 + 1.0).recip();
    let xb2 = (xa1 * rhoa1 * d12 + xa2 * rhoa2 * d22 + 1.0).recip();
    let f = |x: D| x.ln() - x * 0.5 + 0.5;

    rhoa1 * f(xa1) + rhoa2 * f(xa2) + rhob1 * f(xb1) + rhob2 * f(xb2)
}

#[expect(clippy::too_many_arguments)]
fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
    temperature: D,
    rho: [D; 2],
    m: [D; 2],
    sigma: [D; 2],
    epsilon_k: [D; 2],
    mu: [D; 2],
    kij: D,
    assoc_params: Option<[[D; 4]; 2]>,
) -> D {
    // temperature dependent segment diameter
    let t_inv = temperature.recip();
    let [sigma1, sigma2] = sigma;
    let [epsilon_k1, epsilon_k2] = epsilon_k;
    let d1 = sigma1 * (-(epsilon_k1 * -3. * t_inv).exp() * 0.12 + 1.0);
    let d2 = sigma2 * (-(epsilon_k2 * -3. * t_inv).exp() * 0.12 + 1.0);
    let d = [d1, d2];

    // density and composition
    let [rho1, rho2] = rho;
    let density = rho1 + rho2;
    let x = [rho1 / density, rho2 / density];

    // hard sphere
    let (hs, etas, zeta2, zeta3, frac_1mz3) = hard_sphere(m, x, d, density);

    // hard chain
    let hc = hard_chain(m, d, rho, zeta2, zeta3, frac_1mz3);

    // dispersion
    let (disp, sigma_3, epsilon_k_mix) =
        dispersion(m, sigma, epsilon_k, kij, x, t_inv, rho, etas, frac_1mz3);

    // dipoles
    let polar = dipoles(m, sigma, sigma_3, epsilon_k_mix, mu, temperature, rho, etas);

    // association
    if let Some(p) = assoc_params {
        let assoc = association(p, sigma_3, t_inv, rho, d, zeta2, frac_1mz3);
        hs + hc + disp + polar + assoc
    } else {
        hs + hc + disp + polar
    }
}

impl<D: DualNum<f64> + Copy> ResidualConst<2, D> for PcSaftBinary<D, 4> {
    const NAME: &str = "PC-SAFT (binary, non-assoc)";
    type Real = PcSaftBinary<f64, 4>;
    type Lifted<D2: DualNum<f64, Inner = D> + Copy> = PcSaftBinary<D2, 4>;
    fn re(&self) -> Self::Real {
        PcSaftBinary((
            self.0.0.each_ref().map(|x| x.each_ref().map(D::re)),
            self.0.1.re(),
        ))
    }
    fn lift<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        PcSaftBinary((
            self.0
                .0
                .each_ref()
                .map(|x| x.each_ref().map(D2::from_inner)),
            D2::from_inner(&self.0.1),
        ))
    }

    fn compute_max_density(&self, molefracs: &SVector<D, 2>) -> D {
        let &([p1, p2], _) = &self.0;
        let [x1, x2] = molefracs.data.0[0];
        let [m1, sigma1, ..] = p1;
        let [m2, sigma2, ..] = p2;
        ((m1 * sigma1.powi(3) * x1 + m2 * sigma2.powi(3) * x2) * FRAC_PI_6).recip() * MAX_ETA
    }

    fn reduced_residual_helmholtz_energy_density(&self, state: &StateHD<D, U2>) -> D {
        let ([p1, p2], kij) = self.0;
        let [m1, sigma1, epsilon_k1, mu1] = p1;
        let [m2, sigma2, epsilon_k2, mu2] = p2;
        let m = [m1, m2];
        let sigma = [sigma1, sigma2];
        let epsilon_k = [epsilon_k1, epsilon_k2];
        let mu = [mu1, mu2];

        let [rho1, rho2] = state.partial_density.data.0[0];
        let rho = [rho1, rho2];

        helmholtz_energy_density(state.temperature, rho, m, sigma, epsilon_k, mu, kij, None)
    }
}

impl<D: DualNum<f64> + Copy> ResidualConst<2, D> for PcSaftBinary<D, 8> {
    const NAME: &str = "PC-SAFT (binary)";
    type Real = PcSaftBinary<f64, 8>;
    type Lifted<D2: DualNum<f64, Inner = D> + Copy> = PcSaftBinary<D2, 8>;
    fn re(&self) -> Self::Real {
        PcSaftBinary((
            self.0.0.each_ref().map(|x| x.each_ref().map(D::re)),
            self.0.1.re(),
        ))
    }
    fn lift<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        PcSaftBinary((
            self.0
                .0
                .each_ref()
                .map(|x| x.each_ref().map(D2::from_inner)),
            D2::from_inner(&self.0.1),
        ))
    }

    fn compute_max_density(&self, molefracs: &SVector<D, 2>) -> D {
        let &([p1, p2], _) = &self.0;
        let [x1, x2] = molefracs.data.0[0];
        let [m1, sigma1, ..] = p1;
        let [m2, sigma2, ..] = p2;
        ((m1 * sigma1.powi(3) * x1 + m2 * sigma2.powi(3) * x2) * FRAC_PI_6).recip() * MAX_ETA
    }

    fn reduced_residual_helmholtz_energy_density(&self, state: &StateHD<D, U2>) -> D {
        let ([p1, p2], kij) = self.0;
        let [
            m1,
            sigma1,
            epsilon_k1,
            mu1,
            kappa_ab1,
            epsilon_k_ab1,
            na1,
            nb1,
        ] = p1;
        let [
            m2,
            sigma2,
            epsilon_k2,
            mu2,
            kappa_ab2,
            epsilon_k_ab2,
            na2,
            nb2,
        ] = p2;
        let m = [m1, m2];
        let sigma = [sigma1, sigma2];
        let epsilon_k = [epsilon_k1, epsilon_k2];
        let mu = [mu1, mu2];
        let assoc_params = Some([
            [kappa_ab1, epsilon_k_ab1, na1, nb1],
            [kappa_ab2, epsilon_k_ab2, na2, nb2],
        ]);

        let [rho1, rho2] = state.partial_density.data.0[0];
        let rho = [rho1, rho2];

        helmholtz_energy_density(
            state.temperature,
            rho,
            m,
            sigma,
            epsilon_k,
            mu,
            kij,
            assoc_params,
        )
    }
}

#[cfg(test)]
pub mod test {
    use super::PcSaftBinary;
    use crate::pcsaft::{
        PcSaft, PcSaftAssociationRecord, PcSaftBinaryRecord, PcSaftParameters, PcSaftRecord,
    };
    use approx::assert_relative_eq;
    use feos_core::parameter::{AssociationRecord, PureRecord};
    use feos_core::{Contributions::Total, FeosResult, State};
    use nalgebra::{dvector, vector};
    use quantity::{KELVIN, KILO, METER, MOL};

    pub fn pcsaft_binary() -> FeosResult<(PcSaftBinary<f64, 8>, PcSaft)> {
        let params = [
            [1.5, 3.4, 180.0, 2.2, 0.03, 2500., 2.0, 1.0],
            [2.5, 3.6, 250.0, 1.2, 0.015, 1500., 1.0, 2.0],
        ];
        let kij = 0.15;
        let records = params.map(|p| {
            PureRecord::with_association(
                Default::default(),
                0.0,
                PcSaftRecord::new(p[0], p[1], p[2], p[3], 0.0, None, None, None),
                vec![AssociationRecord::new(
                    Some(PcSaftAssociationRecord::new(p[4], p[5])),
                    p[6],
                    p[7],
                    0.0,
                )],
            )
        });
        let params_feos =
            PcSaftParameters::new_binary(records, Some(PcSaftBinaryRecord::new(kij)), vec![])?;
        let eos = PcSaft::new(params_feos);
        Ok((PcSaftBinary::new(params, kij), eos))
    }

    #[test]
    fn test_pcsaft_binary() -> FeosResult<()> {
        let (pcsaft, eos) = pcsaft_binary()?;

        let temperature = 300.0 * KELVIN;
        let volume = 2.3 * METER * METER * METER;
        let moles = dvector![1.3, 2.5] * KILO * MOL;

        let state = State::new_nvt(&&eos, temperature, volume, &moles)?;
        let a_feos = state.residual_molar_helmholtz_energy();
        let mu_feos = state.residual_chemical_potential();
        let p_feos = state.pressure(Total);
        let s_feos = state.residual_molar_entropy();
        let h_feos = state.residual_molar_enthalpy();

        let moles = vector![1.3, 2.5] * KILO * MOL;
        let state = State::new_nvt(&pcsaft, temperature, volume, &moles)?;
        let a_ad = state.residual_molar_helmholtz_energy();
        let mu_ad = state.residual_chemical_potential();
        let p_ad = state.pressure(Total);
        let s_ad = state.residual_molar_entropy();
        let h_ad = state.residual_molar_enthalpy();

        for (s, c) in state.residual_molar_helmholtz_energy_contributions() {
            println!("{s:20} {c}");
        }

        println!("\nMolar Helmholtz energy:\n{a_feos}");
        println!("{a_ad}");
        assert_relative_eq!(a_feos, a_ad, max_relative = 1e-14);

        println!("\nChemical potential:\n{}", mu_feos.get(0));
        println!("{}", mu_ad.get(0));
        assert_relative_eq!(mu_feos.get(0), mu_ad.get(0), max_relative = 1e-14);

        println!("\nPressure:\n{p_feos}");
        println!("{p_ad}");
        assert_relative_eq!(p_feos, p_ad, max_relative = 1e-14);

        println!("\nMolar entropy:\n{s_feos}");
        println!("{s_ad}");
        assert_relative_eq!(s_feos, s_ad, max_relative = 1e-14);

        println!("\nMolar enthalpy:\n{h_feos}");
        println!("{h_ad}");
        assert_relative_eq!(h_feos, h_ad, max_relative = 1e-14);

        Ok(())
    }
}
