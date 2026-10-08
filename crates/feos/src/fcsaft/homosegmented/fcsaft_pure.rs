use super::polar::{AD, BD, CD};
use crate::fcsaft::eos::dispersion::{A0, A1, A2, B0, B1, B2};
use crate::hard_sphere::HardSphere;
use feos_core::{Residual, StateHD, ad::ParametersAD};
use nalgebra::{SVector, U1};
use num_dual::{Dual2, DualNum, second_derivative};
use std::f64::consts::{FRAC_PI_6, PI};

const PI_SQ_43: f64 = 4.0 / 3.0 * PI * PI;

const MAX_ETA: f64 = 0.5;

/// Optimized implementation of FC-SAFT for a single (homosegmented) component.
///
/// The parameters are `[s, l, sigma, epsilon_k, mu]` (`N = 5`), extended by
/// `[kappa_ab, epsilon_k_ab, na, nb]` for associating components (`N = 9`).
/// The variants `N = 33` and `N = 37` additionally contain the 28 model
/// constants `[a1, a2, b1, b2]` (7 coefficients each) of the dispersion
/// contribution, so that they can be adjusted together with the
/// pure-component parameters.
#[derive(Clone, Copy)]
pub struct FcSaftPure<D: DualNum<Primitive = f64> + Copy, const N: usize>(pub [D; N]);

/// Names of the model constants of the dispersion contribution.
#[rustfmt::skip]
const DISPERSION_CONSTANTS: [&str; 28] = [
    "a1_0", "a1_1", "a1_2", "a1_3", "a1_4", "a1_5", "a1_6",
    "a2_0", "a2_1", "a2_2", "a2_3", "a2_4", "a2_5", "a2_6",
    "b1_0", "b1_1", "b1_2", "b1_3", "b1_4", "b1_5", "b1_6",
    "b2_0", "b2_1", "b2_2", "b2_3", "b2_4", "b2_5", "b2_6",
];

/// The default model constants of the dispersion contribution.
fn default_constants<D: DualNum<Primitive = f64> + Copy>() -> [[D; 7]; 4] {
    [A1, A2, B1, B2].map(|c| c.map(D::from))
}

/// The model constants of the dispersion contribution from a flat slice.
fn constants<D: Copy>(c: &[D]) -> [[D; 7]; 4] {
    std::array::from_fn(|i| std::array::from_fn(|j| c[7 * i + j]))
}

/// Reduced volume $m^*$ of a chain of `s` fused spheres with bond length `l` and diameter `d`.
pub(super) fn reduced_volume<D: DualNum<Primitive = f64> + Copy>(s: D, l: D, d: D) -> D {
    let ld = l / d;
    ld * (-ld * ld + 3.0) * (s - 1.0) * 0.5 + 1.0
}

#[expect(clippy::too_many_arguments)]
fn helmholtz_energy_density_non_assoc<D: DualNum<Primitive = f64> + Copy>(
    s: D,
    l: D,
    sigma: D,
    epsilon_k: D,
    mu: D,
    [a1, a2, b1, b2]: &[[D; 7]; 4],
    temperature: D,
    density: D,
) -> (D, [D; 3]) {
    // temperature dependent segment diameter
    let diameter = sigma * (-(epsilon_k * (-3.) / temperature).exp() * 0.12 + 1.0);

    // reduced surface and volume of the fused chain
    let m = l / diameter * (s - 1.0) + 1.0;
    let m_star = reduced_volume(s, l, diameter);

    let eta = m_star * density * diameter.powi(3) * FRAC_PI_6;
    let eta2 = eta * eta;
    let eta3 = eta2 * eta;
    let eta_m1 = (-eta + 1.0).recip();
    let etas = [
        D::one(),
        eta,
        eta2,
        eta3,
        eta2 * eta2,
        eta2 * eta3,
        eta3 * eta3,
    ];

    // reference fluid (hard spheres and fused chain) and its second derivative
    // w.r.t. the density, which yields the compressibility term C1
    let (reference, _, d2_reference) = second_derivative(
        |rho: Dual2<D>| {
            let zeta = [
                s,
                m * diameter,
                m * diameter.powi(2),
                m_star * diameter.powi(3),
            ]
            .map(|c| rho.scale(&(c * FRAC_PI_6)));
            let zeta_23 = Dual2::from_re(m / (m_star * diameter));
            let hs = HardSphere::bmcsl_helmholtz_energy_density(zeta, zeta_23);
            let frac_1mz3 = -(zeta[3] - 1.0).recip();
            let z2l = zeta[2].scale(&l);
            let y = z2l * frac_1mz3 * frac_1mz3 * (z2l * frac_1mz3 * 0.5 + 1.5) + frac_1mz3;
            let hc = -rho * y.ln().scale(&(s - 1.0));
            hs + hc
        },
        density,
    );
    let c1 = (d2_reference * density + 1.0).recip();

    // dispersion
    let e = epsilon_k / temperature;
    let s3 = sigma.powi(3);
    let mut i1 = D::zero();
    let mut i2 = D::zero();
    let m1 = (m - 1.0) / m;
    let m2 = (m - 2.0) / m;
    for i in 0..7 {
        i1 += (m1 * (m2 * a2[i] + a1[i]) + A0[i]) * etas[i];
        i2 += (m1 * (m2 * b2[i] + b1[i]) + B0[i]) * etas[i];
    }
    let i = i1 * 2.0 + c1 * i2 * m * e;
    let disp = -density * density * m.powi(2) * e * s3 * i * PI;

    // dipoles
    // Note: the reduced dipole moment is normalized with the reduced volume m*.
    // Using the reduced surface m (which replaces the segment number of PC-SAFT
    // everywhere else) would be an equally valid generalization to fused chains.
    let mu2 = mu.powi(2) / (reduced_volume(s, l, sigma) * temperature * 1.380649e-4);
    let m_dipole = if m.re() > 2.0 { D::from(2.0) } else { m };
    let m1 = (m_dipole - 1.0) / m_dipole;
    let m2 = m1 * (m_dipole - 2.0) / m_dipole;
    let mut j1 = D::zero();
    let mut j2 = D::zero();
    for i in 0..5 {
        let a = m2 * AD[i][2] + m1 * AD[i][1] + AD[i][0];
        let b = m2 * BD[i][2] + m1 * BD[i][1] + BD[i][0];
        j1 += (a + b * e) * etas[i];
        if i < 4 {
            j2 += (m2 * CD[i][2] + m1 * CD[i][1] + CD[i][0]) * etas[i];
        }
    }

    // mu is factored out of these expressions to deal with the case where mu=0
    let phi2 = -density * density * j1 / s3 * PI;
    let phi3 = -density * density * density * j2 / s3 * PI_SQ_43;
    let dipole = phi2 * phi2 * mu2 * mu2 / (phi2 - phi3 * mu2);

    (reference + disp + dipole, [eta, eta_m1, m / m_star])
}

fn helmholtz_energy_density<D: DualNum<Primitive = f64> + Copy>(
    parameters: &[D; 9],
    constants: &[[D; 7]; 4],
    temperature: D,
    density: D,
) -> D {
    let [s, l, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab, na, nb] = *parameters;
    let (non_assoc, [eta, eta_m1, m_m_star]) = helmholtz_energy_density_non_assoc(
        s,
        l,
        sigma,
        epsilon_k,
        mu,
        constants,
        temperature,
        density,
    );

    // association
    let delta_assoc = ((epsilon_k_ab / temperature).exp() - 1.0) * sigma.powi(3) * kappa_ab;
    let k = eta * eta_m1 * m_m_star;
    let delta = (k * (k * 0.5 + 1.5) + 1.0) * eta_m1 * delta_assoc;
    let rhoa = na * density;
    let rhob = nb * density;
    let aux = (rhoa - rhob) * delta + 1.0;
    let sqrt = (aux * aux + rhob * delta * 4.0).sqrt();
    let xa = (sqrt + 1.0 + (rhob - rhoa) * delta).recip() * 2.0;
    let xb = (sqrt + 1.0 - (rhob - rhoa) * delta).recip() * 2.0;
    let assoc = rhoa * (xa.ln() - xa * 0.5 + 0.5) + rhob * (xb.ln() - xb * 0.5 + 0.5);

    non_assoc + assoc
}

impl<D: DualNum<Primitive = f64> + Copy> Residual<U1, D> for FcSaftPure<D, 9> {
    fn components(&self) -> usize {
        1
    }

    type Real = FcSaftPure<f64, 9>;
    type Lifted<D2: DualNum<Primitive = f64, Inner = D> + Copy> = FcSaftPure<D2, 9>;
    fn re(&self) -> Self::Real {
        FcSaftPure(self.0.each_ref().map(D::re))
    }
    fn lift<D2: DualNum<Primitive = f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        FcSaftPure(self.0.each_ref().map(D2::from_inner))
    }

    fn compute_max_density(&self, _: &SVector<D, 1>) -> D {
        let &[s, l, sigma, ..] = &self.0;
        (reduced_volume(s, l, sigma) * sigma.powi(3) * FRAC_PI_6).recip() * MAX_ETA
    }

    fn reduced_helmholtz_energy_density_contributions(
        &self,
        state: &StateHD<D, U1>,
    ) -> Vec<(&'static str, D)> {
        vec![(
            "FC-SAFT (pure)",
            self.reduced_residual_helmholtz_energy_density(state),
        )]
    }

    fn reduced_residual_helmholtz_energy_density(&self, state: &StateHD<D, U1>) -> D {
        let density = state.partial_density.data.0[0][0];
        helmholtz_energy_density(&self.0, &default_constants(), state.temperature, density)
    }
}

impl<D: DualNum<Primitive = f64> + Copy> Residual<U1, D> for FcSaftPure<D, 5> {
    fn components(&self) -> usize {
        1
    }

    type Real = FcSaftPure<f64, 5>;
    type Lifted<D2: DualNum<Primitive = f64, Inner = D> + Copy> = FcSaftPure<D2, 5>;
    fn re(&self) -> Self::Real {
        FcSaftPure(self.0.each_ref().map(D::re))
    }
    fn lift<D2: DualNum<Primitive = f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        FcSaftPure(self.0.each_ref().map(D2::from_inner))
    }

    fn compute_max_density(&self, _: &SVector<D, 1>) -> D {
        let &[s, l, sigma, ..] = &self.0;
        (reduced_volume(s, l, sigma) * sigma.powi(3) * FRAC_PI_6).recip() * MAX_ETA
    }

    fn reduced_helmholtz_energy_density_contributions(
        &self,
        state: &StateHD<D, U1>,
    ) -> Vec<(&'static str, D)> {
        vec![(
            "FC-SAFT (pure, non-assoc)",
            self.reduced_residual_helmholtz_energy_density(state),
        )]
    }

    fn reduced_residual_helmholtz_energy_density(&self, state: &StateHD<D, U1>) -> D {
        let density = state.partial_density.data.0[0][0];
        let [s, l, sigma, epsilon_k, mu] = self.0;
        let c = default_constants();
        helmholtz_energy_density_non_assoc(
            s,
            l,
            sigma,
            epsilon_k,
            mu,
            &c,
            state.temperature,
            density,
        )
        .0
    }
}

impl<D: DualNum<Primitive = f64> + Copy> Residual<U1, D> for FcSaftPure<D, 37> {
    fn components(&self) -> usize {
        1
    }

    type Real = FcSaftPure<f64, 37>;
    type Lifted<D2: DualNum<Primitive = f64, Inner = D> + Copy> = FcSaftPure<D2, 37>;
    fn re(&self) -> Self::Real {
        FcSaftPure(self.0.each_ref().map(D::re))
    }
    fn lift<D2: DualNum<Primitive = f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        FcSaftPure(self.0.each_ref().map(D2::from_inner))
    }

    fn compute_max_density(&self, _: &SVector<D, 1>) -> D {
        let &[s, l, sigma, ..] = &self.0;
        (reduced_volume(s, l, sigma) * sigma.powi(3) * FRAC_PI_6).recip() * MAX_ETA
    }

    fn reduced_helmholtz_energy_density_contributions(
        &self,
        state: &StateHD<D, U1>,
    ) -> Vec<(&'static str, D)> {
        vec![(
            "FC-SAFT (pure, model constants)",
            self.reduced_residual_helmholtz_energy_density(state),
        )]
    }

    fn reduced_residual_helmholtz_energy_density(&self, state: &StateHD<D, U1>) -> D {
        let density = state.partial_density.data.0[0][0];
        let (parameters, c) = self.0.split_at(9);
        let parameters = parameters.try_into().unwrap();
        helmholtz_energy_density(parameters, &constants(c), state.temperature, density)
    }
}

impl<D: DualNum<Primitive = f64> + Copy> Residual<U1, D> for FcSaftPure<D, 33> {
    fn components(&self) -> usize {
        1
    }

    type Real = FcSaftPure<f64, 33>;
    type Lifted<D2: DualNum<Primitive = f64, Inner = D> + Copy> = FcSaftPure<D2, 33>;
    fn re(&self) -> Self::Real {
        FcSaftPure(self.0.each_ref().map(D::re))
    }
    fn lift<D2: DualNum<Primitive = f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        FcSaftPure(self.0.each_ref().map(D2::from_inner))
    }

    fn compute_max_density(&self, _: &SVector<D, 1>) -> D {
        let &[s, l, sigma, ..] = &self.0;
        (reduced_volume(s, l, sigma) * sigma.powi(3) * FRAC_PI_6).recip() * MAX_ETA
    }

    fn reduced_helmholtz_energy_density_contributions(
        &self,
        state: &StateHD<D, U1>,
    ) -> Vec<(&'static str, D)> {
        vec![(
            "FC-SAFT (pure, non-assoc, model constants)",
            self.reduced_residual_helmholtz_energy_density(state),
        )]
    }

    fn reduced_residual_helmholtz_energy_density(&self, state: &StateHD<D, U1>) -> D {
        let density = state.partial_density.data.0[0][0];
        let [s, l, sigma, epsilon_k, mu, ..] = self.0;
        let c = constants(&self.0[5..]);
        helmholtz_energy_density_non_assoc(
            s,
            l,
            sigma,
            epsilon_k,
            mu,
            &c,
            state.temperature,
            density,
        )
        .0
    }
}

impl ParametersAD<U1> for FcSaftPure<f64, 5> {
    fn build<D: DualNum<Primitive = f64, Inner = f64> + Copy>(
        mut f: impl FnMut(&'static str, bool) -> D,
    ) -> FcSaftPure<D, 5> {
        FcSaftPure([
            f("s", false),
            f("l", true),
            f("sigma", true),
            f("epsilon_k", true),
            f("mu", true),
        ])
    }
}

impl ParametersAD<U1> for FcSaftPure<f64, 9> {
    fn build<D: DualNum<Primitive = f64, Inner = f64> + Copy>(
        mut f: impl FnMut(&'static str, bool) -> D,
    ) -> FcSaftPure<D, 9> {
        FcSaftPure([
            f("s", false),
            f("l", true),
            f("sigma", true),
            f("epsilon_k", true),
            f("mu", true),
            f("kappa_ab", true),
            f("epsilon_k_ab", true),
            f("na", false),
            f("nb", false),
        ])
    }
}

impl ParametersAD<U1> for FcSaftPure<f64, 33> {
    fn build<D: DualNum<Primitive = f64, Inner = f64> + Copy>(
        mut f: impl FnMut(&'static str, bool) -> D,
    ) -> FcSaftPure<D, 33> {
        let p = FcSaftPure::<f64, 5>::build(&mut f).0;
        let c = DISPERSION_CONSTANTS.map(|name| f(name, true));
        FcSaftPure(std::array::from_fn(|i| if i < 5 { p[i] } else { c[i - 5] }))
    }
}

impl ParametersAD<U1> for FcSaftPure<f64, 37> {
    fn build<D: DualNum<Primitive = f64, Inner = f64> + Copy>(
        mut f: impl FnMut(&'static str, bool) -> D,
    ) -> FcSaftPure<D, 37> {
        let p = FcSaftPure::<f64, 9>::build(&mut f).0;
        let c = DISPERSION_CONSTANTS.map(|name| f(name, true));
        FcSaftPure(std::array::from_fn(|i| if i < 9 { p[i] } else { c[i - 9] }))
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::fcsaft::homosegmented::{FcSaftHomo, FcSaftHomoParameters, FcSaftHomoRecord};
    use crate::fcsaft::{FcSaft, FcSaftAssociationRecord, new_pure_homosegmented};
    use approx::assert_relative_eq;
    use feos_core::parameter::{AssociationRecord, Identifier, PureRecord};
    use feos_core::{Contributions::Total, FeosResult, State};
    use nalgebra::{dvector, vector};
    use quantity::{KELVIN, KILO, METER, MOL};

    pub fn fcsaft() -> FeosResult<(FcSaftPure<f64, 9>, FcSaftHomo)> {
        let s = 3;
        let l = 2.5;
        let sigma = 3.4;
        let epsilon_k = 180.0;
        let mu = 2.2;
        let kappa_ab = 0.03;
        let epsilon_k_ab = 2500.;
        let na = 2.0;
        let nb = 1.0;
        let params = FcSaftHomoParameters::new_pure(PureRecord::with_association(
            Default::default(),
            0.0,
            FcSaftHomoRecord::new(s, l, sigma, epsilon_k, mu),
            vec![AssociationRecord::new(
                Some(FcSaftAssociationRecord::new(kappa_ab, epsilon_k_ab)),
                na,
                nb,
                0.0,
            )],
        ))?;
        let eos = FcSaftHomo::new(params);
        let params = [
            s as f64,
            l,
            sigma,
            epsilon_k,
            mu,
            kappa_ab,
            epsilon_k_ab,
            na,
            nb,
        ];
        Ok((FcSaftPure(params), eos))
    }

    #[test]
    fn test_fcsaft_pure() -> FeosResult<()> {
        let (fcsaft, eos) = fcsaft()?;

        let temperature = 350.0 * KELVIN;
        let volume = 2.3 * METER * METER * METER;
        let moles = dvector![1.3] * KILO * MOL;

        let state = State::new_nvt(&&eos, temperature, volume, &moles)?;
        let a_feos = state.residual_molar_helmholtz_energy();
        let mu_feos = state.residual_chemical_potential();
        let p_feos = state.pressure(Total);
        let s_feos = state.residual_molar_entropy();
        let h_feos = state.residual_molar_enthalpy();

        let moles = vector![1.3] * KILO * MOL;
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
    fn test_fcsaft_pure_heterosegmented() -> FeosResult<()> {
        let (s, l, sigma, epsilon_k) = (4, 2.1, 3.6, 220.0);
        let fcsaft = FcSaftPure([s as f64, l, sigma, epsilon_k, 0.0]);
        let params =
            new_pure_homosegmented(Identifier::default(), s, sigma, epsilon_k, l, 0.0, None)?;
        let eos = FcSaft::new(params);

        let temperature = 300.0 * KELVIN;
        let volume = 1.7 * METER * METER * METER;

        let state = State::new_nvt(&&eos, temperature, volume, &(dvector![2.1] * KILO * MOL))?;
        let a_feos = state.residual_molar_helmholtz_energy();
        let p_feos = state.pressure(Total);

        let state = State::new_nvt(&fcsaft, temperature, volume, vector![2.1] * KILO * MOL)?;
        let a_ad = state.residual_molar_helmholtz_energy();
        let p_ad = state.pressure(Total);

        assert_relative_eq!(a_feos, a_ad, max_relative = 1e-13);
        assert_relative_eq!(p_feos, p_ad, max_relative = 1e-13);
        Ok(())
    }

    #[test]
    fn test_fcsaft_pure_model_constants() -> FeosResult<()> {
        let temperature = 300.0 * KELVIN;
        let volume = 1.7 * METER * METER * METER;
        let moles = vector![2.1] * KILO * MOL;

        // default model constants: identical to the variants without constants
        let (fcsaft, _) = fcsaft()?;
        let default = [A1, A2, B1, B2].concat();
        let p = fcsaft.0;
        let fcsaft_c = FcSaftPure::<f64, 37>(std::array::from_fn(|i| {
            if i < 9 { p[i] } else { default[i - 9] }
        }));
        let a = State::new_nvt(&fcsaft, temperature, volume, moles)?;
        let a_c = State::new_nvt(&fcsaft_c, temperature, volume, moles)?;
        assert_relative_eq!(
            a.residual_molar_helmholtz_energy(),
            a_c.residual_molar_helmholtz_energy(),
            max_relative = 1e-14
        );

        // modified model constants: identical to the heterosegmented model
        let (s, l, sigma, epsilon_k) = (4, 2.1, 3.6, 220.0);
        let mut model_constants = [A1, A2, B1, B2];
        model_constants
            .iter_mut()
            .flatten()
            .enumerate()
            .for_each(|(i, c)| {
                *c *= 1.0 + 0.01 * i as f64;
            });
        let c = model_constants.concat();
        let p = [s as f64, l, sigma, epsilon_k, 0.0];
        let fcsaft =
            FcSaftPure::<f64, 33>(std::array::from_fn(|i| if i < 5 { p[i] } else { c[i - 5] }));
        let params =
            new_pure_homosegmented(Identifier::default(), s, sigma, epsilon_k, l, 0.0, None)?;
        let eos = FcSaft::with_options(params, Default::default(), Some(model_constants));
        let state = State::new_nvt(&&eos, temperature, volume, &(dvector![2.1] * KILO * MOL))?;
        let state_ad = State::new_nvt(&fcsaft, temperature, volume, moles)?;
        assert_relative_eq!(
            state.residual_molar_helmholtz_energy(),
            state_ad.residual_molar_helmholtz_energy(),
            max_relative = 1e-13
        );
        assert_relative_eq!(
            state.pressure(Total),
            state_ad.pressure(Total),
            max_relative = 1e-13
        );
        Ok(())
    }
}
