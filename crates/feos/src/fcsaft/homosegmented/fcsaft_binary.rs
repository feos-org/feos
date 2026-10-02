use super::fcsaft_pure::reduced_volume;
use super::polar::{AD, BD, CD};
use crate::fcsaft::eos::dispersion::{A0, A1, A2, B0, B1, B2};
use crate::hard_sphere::HardSphere;
use feos_core::{Residual, StateHD, ad::ParametersAD};
use nalgebra::{SVector, U2};
use num_dual::{Dual2, DualNum, DualVec, jacobian, second_derivative};
use std::f64::consts::{FRAC_PI_6, PI};

const PI_SQ_43: f64 = 4.0 / 3.0 * PI * PI;

const MAX_ETA: f64 = 0.5;

/// Optimized implementation of FC-SAFT for a binary mixture of (homosegmented) components.
#[derive(Clone, Copy)]
pub struct FcSaftBinary<D, const N: usize>(pub ([[D; N]; 2], D));

impl<D, const N: usize> FcSaftBinary<D, N> {
    pub fn new(parameters: [[D; N]; 2], kij: D) -> Self {
        Self((parameters, kij))
    }
}

impl ParametersAD<U2> for FcSaftBinary<f64, 5> {
    fn build<D: DualNum<Primitive = f64, Inner = f64> + Copy>(
        mut f: impl FnMut(&'static str, bool) -> D,
    ) -> FcSaftBinary<D, 5> {
        FcSaftBinary::new(
            [
                [
                    f("s1", false),
                    f("l1", true),
                    f("sigma1", true),
                    f("epsilon_k1", true),
                    f("mu1", true),
                ],
                [
                    f("s2", false),
                    f("l2", true),
                    f("sigma2", true),
                    f("epsilon_k2", true),
                    f("mu2", true),
                ],
            ],
            f("k_ij", true),
        )
    }
}

impl ParametersAD<U2> for FcSaftBinary<f64, 9> {
    fn build<D: DualNum<Primitive = f64, Inner = f64> + Copy>(
        mut f: impl FnMut(&'static str, bool) -> D,
    ) -> FcSaftBinary<D, 9> {
        FcSaftBinary::new(
            [
                [
                    f("s1", false),
                    f("l1", true),
                    f("sigma1", true),
                    f("epsilon_k1", true),
                    f("mu1", true),
                    f("kappa_ab1", true),
                    f("epsilon_k_ab1", true),
                    f("na1", false),
                    f("nb1", false),
                ],
                [
                    f("s2", false),
                    f("l2", true),
                    f("sigma2", true),
                    f("epsilon_k2", true),
                    f("mu2", true),
                    f("kappa_ab2", true),
                    f("epsilon_k_ab2", true),
                    f("na2", false),
                    f("nb2", false),
                ],
            ],
            f("k_ij", true),
        )
    }
}

/// Hard spheres and fused chains, and the compressibility term C1 from the
/// second derivative of the reference fluid w.r.t. the density.
fn reference<D: DualNum<Primitive = f64> + Copy>(
    [s1, s2]: [D; 2],
    [l1, l2]: [D; 2],
    [m1, m2]: [D; 2],
    [m_star1, m_star2]: [D; 2],
    [x1, x2]: [D; 2],
    [d1, d2]: [D; 2],
    density: D,
) -> (D, [D; 7], D, D, D) {
    // Packing fractions (divided by the density)
    let zeta = [
        s1 * x1 + s2 * x2,
        m1 * x1 * d1 + m2 * x2 * d2,
        m1 * x1 * d1.powi(2) + m2 * x2 * d2.powi(2),
        m_star1 * x1 * d1.powi(3) + m_star2 * x2 * d2.powi(3),
    ]
    .map(|z| z * FRAC_PI_6);
    let zeta_23 = Dual2::from_re(zeta[2] / zeta[3]);

    let (reference, _, d2_reference) = second_derivative(
        |rho: Dual2<D>| {
            let zeta = zeta.map(|z| rho.scale(&z));
            let hs = HardSphere::bmcsl_helmholtz_energy_density(zeta, zeta_23);
            let frac_1mz3 = -(zeta[3] - 1.0).recip();
            let [ln_y1, ln_y2] = [l1, l2].map(|l| {
                let z2l = zeta[2].scale(&l);
                (z2l * frac_1mz3 * frac_1mz3 * (z2l * frac_1mz3 * 0.5 + 1.5) + frac_1mz3).ln()
            });
            let hc = -rho * (ln_y1.scale(&(x1 * (s1 - 1.0))) + ln_y2.scale(&(x2 * (s2 - 1.0))));
            hs + hc
        },
        density,
    );
    let c1 = (d2_reference * density + 1.0).recip();

    let zeta2 = zeta[2] * density;
    let eta = zeta[3] * density;
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
    let frac_1mz3 = (-eta + 1.0).recip();

    (reference, etas, zeta2, frac_1mz3, c1)
}

#[expect(clippy::too_many_arguments)]
fn dispersion<D: DualNum<Primitive = f64> + Copy>(
    [m1, m2]: [D; 2],
    [sigma1, sigma2]: [D; 2],
    [epsilon_k1, epsilon_k2]: [D; 2],
    kij: D,
    [x1, x2]: [D; 2],
    t_inv: D,
    [rho1, rho2]: [D; 2],
    etas: [D; 7],
    c1: D,
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

    // I1 and I2
    let mm1 = (m - 1.0) / m;
    let mm2 = (m - 2.0) / m;
    let mut i1 = D::zero();
    let mut i2 = D::zero();
    for i in 0..7 {
        i1 += (mm1 * (mm2 * A2[i] + A1[i]) + A0[i]) * etas[i];
        i2 += (mm1 * (mm2 * B2[i] + B1[i]) + B0[i]) * etas[i];
    }

    // dispersion
    let disp = (-rho1mix * i1 * 2.0 - rho2mix * m * c1 * i2) * PI;

    (
        disp,
        [sigma11_3, sigma12_3, sigma22_3],
        [epsilon_k11, epsilon_k12, epsilon_k22],
    )
}

#[expect(clippy::too_many_arguments)]
fn dipoles<D: DualNum<Primitive = f64> + Copy>(
    [m1, m2]: [D; 2],
    [m_star1, m_star2]: [D; 2],
    [sigma1, sigma2]: [D; 2],
    [sigma11_3, sigma12_3, sigma22_3]: [D; 3],
    [epsilon_k11, epsilon_k12, epsilon_k22]: [D; 3],
    [mu1, mu2]: [D; 2],
    temperature: D,
    [rho1, rho2]: [D; 2],
    etas: [D; 7],
) -> D {
    // Note: the reduced dipole moment is normalized with the reduced volume m*.
    // Using the reduced surface m (which replaces the segment number of PC-SAFT
    // everywhere else) would be an equally valid generalization to fused chains.
    let mu_term1 = mu1 * mu1 / (m_star1 * temperature * 1.380649e-4) * rho1;
    let mu_term2 = mu2 * mu2 / (m_star2 * temperature * 1.380649e-4) * rho2;
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

fn association<D: DualNum<Primitive = f64> + Copy>(
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
            &SVector::from([xa1, xa2]),
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
fn helmholtz_energy_density<D: DualNum<Primitive = f64> + Copy>(
    temperature: D,
    rho: [D; 2],
    s: [D; 2],
    l: [D; 2],
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

    // reduced surface and volume of the fused chains
    let m = [0, 1].map(|i| l[i] / d[i] * (s[i] - 1.0) + 1.0);
    let m_star = [0, 1].map(|i| reduced_volume(s[i], l[i], d[i]));

    // density and composition
    let [rho1, rho2] = rho;
    let density = rho1 + rho2;
    let x = [rho1 / density, rho2 / density];

    // reference fluid (hard spheres and fused chains)
    let (reference, etas, zeta2, frac_1mz3, c1) = reference(s, l, m, m_star, x, d, density);

    // dispersion
    let (disp, sigma_3, epsilon_k_mix) =
        dispersion(m, sigma, epsilon_k, kij, x, t_inv, rho, etas, c1);

    // dipoles
    let m_star_sigma = [0, 1].map(|i| reduced_volume(s[i], l[i], sigma[i]));
    let polar = dipoles(
        m,
        m_star_sigma,
        sigma,
        sigma_3,
        epsilon_k_mix,
        mu,
        temperature,
        rho,
        etas,
    );

    // association
    if let Some(p) = assoc_params {
        let assoc = association(p, sigma_3, t_inv, rho, d, zeta2, frac_1mz3);
        reference + disp + polar + assoc
    } else {
        reference + disp + polar
    }
}

fn max_density<D: DualNum<Primitive = f64> + Copy, const N: usize>(
    [p1, p2]: &[[D; N]; 2],
    molefracs: &SVector<D, 2>,
) -> D {
    let [x1, x2] = molefracs.data.0[0];
    let [v1, v2] = [p1, p2].map(|p| reduced_volume(p[0], p[1], p[2]) * p[2].powi(3));
    ((v1 * x1 + v2 * x2) * FRAC_PI_6).recip() * MAX_ETA
}

impl<D: DualNum<Primitive = f64> + Copy> Residual<U2, D> for FcSaftBinary<D, 5> {
    fn components(&self) -> usize {
        2
    }

    type Real = FcSaftBinary<f64, 5>;
    type Lifted<D2: DualNum<Primitive = f64, Inner = D> + Copy> = FcSaftBinary<D2, 5>;
    fn re(&self) -> Self::Real {
        FcSaftBinary((
            self.0.0.each_ref().map(|x| x.each_ref().map(D::re)),
            self.0.1.re(),
        ))
    }
    fn lift<D2: DualNum<Primitive = f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        FcSaftBinary((
            self.0
                .0
                .each_ref()
                .map(|x| x.each_ref().map(D2::from_inner)),
            D2::from_inner(&self.0.1),
        ))
    }

    fn compute_max_density(&self, molefracs: &SVector<D, 2>) -> D {
        max_density(&self.0.0, molefracs)
    }

    fn reduced_helmholtz_energy_density_contributions(
        &self,
        state: &StateHD<D, U2>,
    ) -> Vec<(&'static str, D)> {
        vec![(
            "FC-SAFT (binary, non-assoc)",
            self.reduced_residual_helmholtz_energy_density(state),
        )]
    }

    fn reduced_residual_helmholtz_energy_density(&self, state: &StateHD<D, U2>) -> D {
        let ([p1, p2], kij) = self.0;
        let [s1, l1, sigma1, epsilon_k1, mu1] = p1;
        let [s2, l2, sigma2, epsilon_k2, mu2] = p2;
        let s = [s1, s2];
        let l = [l1, l2];
        let sigma = [sigma1, sigma2];
        let epsilon_k = [epsilon_k1, epsilon_k2];
        let mu = [mu1, mu2];

        let [rho1, rho2] = state.partial_density.data.0[0];
        let rho = [rho1, rho2];

        helmholtz_energy_density(
            state.temperature,
            rho,
            s,
            l,
            sigma,
            epsilon_k,
            mu,
            kij,
            None,
        )
    }
}

impl<D: DualNum<Primitive = f64> + Copy> Residual<U2, D> for FcSaftBinary<D, 9> {
    fn components(&self) -> usize {
        2
    }

    type Real = FcSaftBinary<f64, 9>;
    type Lifted<D2: DualNum<Primitive = f64, Inner = D> + Copy> = FcSaftBinary<D2, 9>;
    fn re(&self) -> Self::Real {
        FcSaftBinary((
            self.0.0.each_ref().map(|x| x.each_ref().map(D::re)),
            self.0.1.re(),
        ))
    }
    fn lift<D2: DualNum<Primitive = f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        FcSaftBinary((
            self.0
                .0
                .each_ref()
                .map(|x| x.each_ref().map(D2::from_inner)),
            D2::from_inner(&self.0.1),
        ))
    }

    fn compute_max_density(&self, molefracs: &SVector<D, 2>) -> D {
        max_density(&self.0.0, molefracs)
    }

    fn reduced_helmholtz_energy_density_contributions(
        &self,
        state: &StateHD<D, U2>,
    ) -> Vec<(&'static str, D)> {
        vec![(
            "FC-SAFT (binary)",
            self.reduced_residual_helmholtz_energy_density(state),
        )]
    }

    fn reduced_residual_helmholtz_energy_density(&self, state: &StateHD<D, U2>) -> D {
        let ([p1, p2], kij) = self.0;
        let [
            s1,
            l1,
            sigma1,
            epsilon_k1,
            mu1,
            kappa_ab1,
            epsilon_k_ab1,
            na1,
            nb1,
        ] = p1;
        let [
            s2,
            l2,
            sigma2,
            epsilon_k2,
            mu2,
            kappa_ab2,
            epsilon_k_ab2,
            na2,
            nb2,
        ] = p2;
        let s = [s1, s2];
        let l = [l1, l2];
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
            s,
            l,
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
    use super::FcSaftBinary;
    use crate::fcsaft::FcSaftAssociationRecord;
    use crate::fcsaft::FcSaftBinaryRecord;
    use crate::fcsaft::homosegmented::{FcSaftHomo, FcSaftHomoParameters, FcSaftHomoRecord};
    use approx::assert_relative_eq;
    use feos_core::ad::{BubblePointPressure, ParametersAD, PropertyAD};
    use feos_core::parameter::{AssociationRecord, PureRecord};
    use feos_core::{Contributions::Total, FeosResult, PhaseEquilibrium, State};
    use nalgebra::{U1, U13, dvector, vector};
    use quantity::{KELVIN, KILO, METER, MOL, PASCAL};

    pub fn fcsaft_binary() -> FeosResult<(FcSaftBinary<f64, 9>, FcSaftHomo)> {
        let params = [
            [3.0, 2.5, 3.4, 180.0, 2.2, 0.03, 2500., 2.0, 1.0],
            [2.0, 3.0, 3.6, 250.0, 1.2, 0.015, 1500., 1.0, 2.0],
        ];
        let kij = 0.15;
        let records = params.map(|p| {
            PureRecord::with_association(
                Default::default(),
                0.0,
                FcSaftHomoRecord::new(p[0] as usize, p[1], p[2], p[3], p[4]),
                vec![AssociationRecord::new(
                    Some(FcSaftAssociationRecord::new(p[5], p[6])),
                    p[7],
                    p[8],
                    0.0,
                )],
            )
        });
        let params_feos =
            FcSaftHomoParameters::new_binary(records, Some(FcSaftBinaryRecord::new(kij)), vec![])?;
        let eos = FcSaftHomo::new(params_feos);
        Ok((FcSaftBinary::new(params, kij), eos))
    }

    #[test]
    fn test_fcsaft_binary() -> FeosResult<()> {
        let (fcsaft, eos) = fcsaft_binary()?;

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
        let state = State::new_nvt(&fcsaft, temperature, volume, moles)?;
        let a_ad = state.residual_molar_helmholtz_energy();
        let mu_ad = state.residual_chemical_potential();
        let p_ad = state.pressure(Total);
        let s_ad = state.residual_molar_entropy();
        let h_ad = state.residual_molar_enthalpy();

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

    #[test]
    fn test_bubble_point_pressure_derivatives() -> FeosResult<()> {
        let fcsaft_params = [
            "l1",
            "sigma1",
            "epsilon_k1",
            "mu1",
            "kappa_ab1",
            "epsilon_k_ab1",
            "l2",
            "sigma2",
            "epsilon_k2",
            "mu2",
            "kappa_ab2",
            "epsilon_k_ab2",
            "k_ij",
        ];
        let indices = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 18];
        let (fcsaft, _) = fcsaft_binary()?;
        let ([p1, p2], kij) = fcsaft.0;
        let mut flat_params = p1.to_vec();
        flat_params.extend_from_slice(&p2);
        flat_params.push(kij);

        let temperature = 450.0 * KELVIN;
        let x = 0.4;
        let fcsaft_ad = FcSaftBinary::<f64, 9>::seed_derivatives(&flat_params, fcsaft_params);
        let p = BubblePointPressure(temperature, x, None).evaluate(&fcsaft_ad)?;
        let p = p.convert_into(PASCAL);
        let (p, grad) = (p.re, p.eps.unwrap_generic(U13, U1));

        println!("{p:.5}");
        println!("{grad:.5?}");

        for ((i, par), j) in fcsaft_params.into_iter().enumerate().zip(indices) {
            let h = flat_params[j] * 1e-6;
            let p_h = |h: f64| {
                let mut params = flat_params.clone();
                params[j] += h;
                let eos: FcSaftBinary<f64, 9> = FcSaftBinary::new(
                    [0, 9].map(|k| std::array::from_fn(|l| params[k + l])),
                    params[18],
                );
                PhaseEquilibrium::bubble_point(
                    &eos,
                    temperature,
                    vector![x, 1.0 - x],
                    None,
                    None,
                    Default::default(),
                )
                .map(|vle| vle.vapor().pressure(Total).convert_into(PASCAL))
            };
            let dp_h = (p_h(h)? - p_h(-h)?) / (2.0 * h);
            let dp = grad[i];
            println!(
                "{par:14}: {:11.5} {:11.5} {:.3e}",
                dp_h,
                dp,
                ((dp_h - dp) / dp).abs()
            );
            assert_relative_eq!(dp, dp_h, max_relative = 1e-6);
        }
        Ok(())
    }
}
