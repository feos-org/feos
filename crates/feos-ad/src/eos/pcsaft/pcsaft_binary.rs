use super::{A0, A1, A2, AD, B0, B1, B2, BD, CD, MAX_ETA};
use crate::{NamedParameters, ParametersAD, ResidualHelmholtzEnergy};
use nalgebra::SVector;
use num_dual::{jacobian, DualNum, DualVec};
use std::f64::consts::{FRAC_PI_6, PI};

const PI_SQ_43: f64 = 4.0 / 3.0 * PI * PI;

/// Optimized implementation of PC-SAFT for a binary mixture.
pub struct PcSaftBinary<const N: usize> {
    parameters: [[f64; N]; 2],
    kij: f64,
}

impl<const N: usize> PcSaftBinary<N> {
    pub fn new(parameters: [[f64; N]; 2], kij: f64) -> Self {
        Self { parameters, kij }
    }
}

impl<const N: usize> ParametersAD for PcSaftBinary<N> {
    type Parameters<D: DualNum<f64> + Copy> = ([[D; N]; 2], D);

    fn params<D: DualNum<f64> + Copy>(&self) -> Self::Parameters<D> {
        (self.parameters.map(|p| p.map(D::from)), D::from(self.kij))
    }

    fn params_from_inner<D: DualNum<f64> + Copy, D2: DualNum<f64, Inner = D> + Copy>(
        &(parameters, kij): &Self::Parameters<D>,
    ) -> Self::Parameters<D2> {
        (
            parameters.map(|p| p.map(D2::from_inner)),
            D2::from_inner(kij),
        )
    }
}

impl<const N: usize> NamedParameters for PcSaftBinary<N> {
    fn index_parameters_mut<'a, D: DualNum<f64> + Copy>(
        (_, kij): &'a mut Self::Parameters<D>,
        index: &str,
    ) -> &'a mut D {
        match index {
            "k_ij" => kij,
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
    let [[kappa_ab1, epsilon_k_ab1, na1, nb1], [kappa_ab2, epsilon_k_ab2, na2, nb2]] = assoc_params;

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
    let phi = if let Some(p) = assoc_params {
        let assoc = association(p, sigma_3, t_inv, rho, d, zeta2, frac_1mz3);
        hs + hc + disp + polar + assoc
    } else {
        hs + hc + disp + polar
    };

    phi * temperature
}

impl ResidualHelmholtzEnergy<2> for PcSaftBinary<4> {
    const RESIDUAL: &str = "PC-SAFT (binary)";

    fn compute_max_density(&self, molefracs: &SVector<f64, 2>) -> f64 {
        let [p1, p2] = self.parameters;
        let [x1, x2] = molefracs.data.0[0];
        let [m1, sigma1, ..] = p1;
        let [m2, sigma2, ..] = p2;
        MAX_ETA / (FRAC_PI_6 * (m1 * sigma1.powi(3) * x1 + m2 * sigma2.powi(3) * x2))
    }

    fn residual_helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &([p1, p2], kij): &Self::Parameters<D>,
        temperature: D,
        partial_density: &SVector<D, 2>,
    ) -> D {
        let [m1, sigma1, epsilon_k1, mu1] = p1;
        let [m2, sigma2, epsilon_k2, mu2] = p2;
        let m = [m1, m2];
        let sigma = [sigma1, sigma2];
        let epsilon_k = [epsilon_k1, epsilon_k2];
        let mu = [mu1, mu2];

        let [rho1, rho2] = partial_density.data.0[0];
        let rho = [rho1, rho2];

        helmholtz_energy_density(temperature, rho, m, sigma, epsilon_k, mu, kij, None)
    }
}

impl ResidualHelmholtzEnergy<2> for PcSaftBinary<8> {
    const RESIDUAL: &str = "PC-SAFT (binary)";

    fn compute_max_density(&self, molefracs: &SVector<f64, 2>) -> f64 {
        let [p1, p2] = self.parameters;
        let [x1, x2] = molefracs.data.0[0];
        let [m1, sigma1, ..] = p1;
        let [m2, sigma2, ..] = p2;
        MAX_ETA / (FRAC_PI_6 * (m1 * sigma1.powi(3) * x1 + m2 * sigma2.powi(3) * x2))
    }

    fn residual_helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &([p1, p2], kij): &Self::Parameters<D>,
        temperature: D,
        partial_density: &SVector<D, 2>,
    ) -> D {
        let [m1, sigma1, epsilon_k1, mu1, kappa_ab1, epsilon_k_ab1, na1, nb1] = p1;
        let [m2, sigma2, epsilon_k2, mu2, kappa_ab2, epsilon_k_ab2, na2, nb2] = p2;
        let m = [m1, m2];
        let sigma = [sigma1, sigma2];
        let epsilon_k = [epsilon_k1, epsilon_k2];
        let mu = [mu1, mu2];
        let assoc_params = Some([
            [kappa_ab1, epsilon_k_ab1, na1, nb1],
            [kappa_ab2, epsilon_k_ab2, na2, nb2],
        ]);

        let [rho1, rho2] = partial_density.data.0[0];
        let rho = [rho1, rho2];

        helmholtz_energy_density(temperature, rho, m, sigma, epsilon_k, mu, kij, assoc_params)
    }
}

#[cfg(test)]
pub mod test {
    use super::{PcSaftBinary, ResidualHelmholtzEnergy};
    use crate::eos::pcsaft::test::pcsaft_binary;
    use approx::assert_relative_eq;
    use feos_core::{Contributions::Total, FeosResult, ReferenceSystem, State};
    use nalgebra::SVector;
    use ndarray::arr1;
    use quantity::{KELVIN, KILO, METER, MOL};

    #[test]
    fn test_pcsaft_binary() -> FeosResult<()> {
        let (pcsaft, eos) = pcsaft_binary()?;
        let pcsaft = (pcsaft.parameters, pcsaft.kij);

        let temperature = 300.0 * KELVIN;
        let volume = 2.3 * METER * METER * METER;
        let moles = arr1(&[1.3, 2.5]) * KILO * MOL;

        let state = State::new_nvt(&eos, temperature, volume, &moles)?;
        let a_feos = state.residual_molar_helmholtz_energy();
        let mu_feos = state.residual_chemical_potential();
        let p_feos = state.pressure(Total);
        let s_feos = state.residual_molar_entropy();
        let h_feos = state.residual_molar_enthalpy();

        let total_moles = moles.sum();
        let t = temperature.to_reduced();
        let v = (volume / total_moles).to_reduced();
        let x = SVector::from_fn(|i, _| moles.get(i).convert_into(total_moles));
        let a_ad = PcSaftBinary::residual_molar_helmholtz_energy(&pcsaft, t, v, &x);
        let mu_ad = PcSaftBinary::residual_chemical_potential(&pcsaft, t, v, &x);
        let p_ad = PcSaftBinary::pressure(&pcsaft, t, v, &x);
        let s_ad = PcSaftBinary::residual_molar_entropy(&pcsaft, t, v, &x);
        let h_ad = PcSaftBinary::residual_molar_enthalpy(&pcsaft, t, v, &x);

        for (s, c) in state.residual_helmholtz_energy_contributions() {
            println!("{s:20} {}", (c / state.volume).to_reduced());
        }

        println!("\nMolar Helmholtz energy:\n{}", a_feos.to_reduced(),);
        println!("{a_ad}");
        assert_relative_eq!(a_feos.to_reduced(), a_ad, max_relative = 1e-14);

        println!("\nChemical potential:\n{}", mu_feos.get(0).to_reduced());
        println!("{}", mu_ad[0]);
        assert_relative_eq!(mu_feos.get(0).to_reduced(), mu_ad[0], max_relative = 1e-14);

        println!("\nPressure:\n{}", p_feos.to_reduced());
        println!("{p_ad}");
        assert_relative_eq!(p_feos.to_reduced(), p_ad, max_relative = 1e-14);

        println!("\nMolar entropy:\n{}", s_feos.to_reduced());
        println!("{s_ad}");
        assert_relative_eq!(s_feos.to_reduced(), s_ad, max_relative = 1e-14);

        println!("\nMolar enthalpy:\n{}", h_feos.to_reduced());
        println!("{h_ad}");
        assert_relative_eq!(h_feos.to_reduced(), h_ad, max_relative = 1e-14);

        Ok(())
    }
}
