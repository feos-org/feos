use super::dispersion::{A0, A1, A2, B0, B1, B2};
use super::polar::{AD, BD, CD};
use feos_core::{ParametersAD, Residual, StateHD};
use nalgebra::{SVector, U1};
use num_dual::{DualNum, DualSVec64};
use std::f64::consts::{FRAC_PI_6, PI};

const PI_SQ_43: f64 = 4.0 / 3.0 * PI * PI;

const MAX_ETA: f64 = 0.5;

/// Optimized implementation of PC-SAFT for a single component.
#[derive(Clone, Copy)]
pub struct PcSaftPure<D: DualNum<f64> + Copy, const N: usize>(pub [D; N]);

fn helmholtz_energy_density_non_assoc<D: DualNum<f64> + Copy>(
    m: D,
    sigma: D,
    epsilon_k: D,
    mu: D,
    temperature: D,
    density: D,
) -> (D, [D; 2]) {
    // temperature dependent segment diameter
    let diameter = sigma * (-(epsilon_k * (-3.) / temperature).exp() * 0.12 + 1.0);

    let eta = m * density * diameter.powi(3) * FRAC_PI_6;
    let eta2 = eta * eta;
    let eta3 = eta2 * eta;
    let eta_m1 = (-eta + 1.0).recip();
    let eta_m2 = eta_m1 * eta_m1;
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
    let hs = m * density * (eta * 4.0 - eta2 * 3.0) * eta_m2;

    // hard chain
    let g = (-eta * 0.5 + 1.0) * eta_m1 * eta_m2;
    let hc = -density * (m - 1.0) * g.ln();

    // dispersion
    let e = epsilon_k / temperature;
    let s3 = sigma.powi(3);
    let mut i1 = D::zero();
    let mut i2 = D::zero();
    let m1 = (m - 1.0) / m;
    let m2 = (m - 2.0) / m;
    for i in 0..7 {
        i1 += (m1 * (m2 * A2[i] + A1[i]) + A0[i]) * etas[i];
        i2 += (m1 * (m2 * B2[i] + B1[i]) + B0[i]) * etas[i];
    }
    let c1 = (m * (eta * 8.0 - eta2 * 2.0) * eta_m2 * eta_m2 + 1.0
        - (m - 1.0) * (eta * 20.0 - eta2 * 27.0 + eta2 * eta * 12.0 - eta2 * eta2 * 2.0)
            / ((eta - 1.0) * (eta - 2.0)).powi(2))
    .recip();
    let i = i1 * 2.0 + c1 * i2 * m * e;
    let disp = -density * density * m.powi(2) * e * s3 * i * PI;

    // dipoles
    let mu2 = mu.powi(2) / (m * temperature * 1.380649e-4);
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

    (hs + hc + disp + dipole, [eta, eta_m1])
}

fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
    parameters: &[D; 8],
    temperature: D,
    density: D,
) -> D {
    let [m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab, na, nb] = *parameters;
    let (non_assoc, [eta, eta_m1]) =
        helmholtz_energy_density_non_assoc(m, sigma, epsilon_k, mu, temperature, density);

    // association
    let delta_assoc = ((epsilon_k_ab / temperature).exp() - 1.0) * sigma.powi(3) * kappa_ab;
    let k = eta * eta_m1;
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

impl<D: DualNum<f64> + Copy> Residual<U1, D> for PcSaftPure<D, 8> {
    fn components(&self) -> usize {
        1
    }

    type Real = PcSaftPure<f64, 8>;
    type Lifted<D2: DualNum<f64, Inner = D> + Copy> = PcSaftPure<D2, 8>;
    fn re(&self) -> Self::Real {
        PcSaftPure(self.0.each_ref().map(D::re))
    }
    fn lift<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        PcSaftPure(self.0.each_ref().map(D2::from_inner))
    }

    fn compute_max_density(&self, _: &SVector<D, 1>) -> D {
        let &[m, sigma, ..] = &self.0;
        (m * sigma.powi(3) * FRAC_PI_6).recip() * MAX_ETA
    }

    fn reduced_helmholtz_energy_density_contributions(
        &self,
        state: &StateHD<D, U1>,
    ) -> Vec<(&'static str, D)> {
        vec![(
            "PC-SAFT (pure)",
            self.reduced_residual_helmholtz_energy_density(state),
        )]
    }

    fn reduced_residual_helmholtz_energy_density(&self, state: &StateHD<D, U1>) -> D {
        let density = state.partial_density.data.0[0][0];
        helmholtz_energy_density(&self.0, state.temperature, density)
    }
}

impl<D: DualNum<f64> + Copy> Residual<U1, D> for PcSaftPure<D, 4> {
    fn components(&self) -> usize {
        1
    }

    type Real = PcSaftPure<f64, 4>;
    type Lifted<D2: DualNum<f64, Inner = D> + Copy> = PcSaftPure<D2, 4>;
    fn re(&self) -> Self::Real {
        PcSaftPure(self.0.each_ref().map(D::re))
    }
    fn lift<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        PcSaftPure(self.0.each_ref().map(D2::from_inner))
    }

    fn compute_max_density(&self, _: &SVector<D, 1>) -> D {
        let &[m, sigma, ..] = &self.0;
        (m * sigma.powi(3) * FRAC_PI_6).recip() * MAX_ETA
    }

    fn reduced_helmholtz_energy_density_contributions(
        &self,
        state: &StateHD<D, U1>,
    ) -> Vec<(&'static str, D)> {
        vec![(
            "PC-SAFT (pure, non-assoc)",
            self.reduced_residual_helmholtz_energy_density(state),
        )]
    }

    fn reduced_residual_helmholtz_energy_density(&self, state: &StateHD<D, U1>) -> D {
        let density = state.partial_density.data.0[0][0];
        let [m, sigma, epsilon_k, mu] = self.0;
        helmholtz_energy_density_non_assoc(m, sigma, epsilon_k, mu, state.temperature, density).0
    }
}

impl<D: DualNum<f64> + Copy, const N: usize> From<&[f64]> for PcSaftPure<D, N> {
    fn from(parameters: &[f64]) -> Self {
        let Ok(parameters): Result<[f64; N], _> = parameters.try_into() else {
            panic!("This version of PC-SAFT requires exactly {N} parameters!")
        };
        Self(parameters.map(D::from))
    }
}

impl ParametersAD<1> for PcSaftPure<f64, 4> {
    fn index_parameters_mut<'a, const P: usize>(
        eos: &'a mut Self::Lifted<DualSVec64<P>>,
        index: &str,
    ) -> &'a mut DualSVec64<P> {
        match index {
            "m" => &mut eos.0[0],
            "sigma" => &mut eos.0[1],
            "epsilon_k" => &mut eos.0[2],
            "mu" => &mut eos.0[3],
            _ => panic!("{index} is not a valid PC-SAFT parameter!"),
        }
    }
}

impl ParametersAD<1> for PcSaftPure<f64, 8> {
    fn index_parameters_mut<'a, const P: usize>(
        eos: &'a mut Self::Lifted<DualSVec64<P>>,
        index: &str,
    ) -> &'a mut DualSVec64<P> {
        match index {
            "m" => &mut eos.0[0],
            "sigma" => &mut eos.0[1],
            "epsilon_k" => &mut eos.0[2],
            "mu" => &mut eos.0[3],
            "kappa_ab" => &mut eos.0[4],
            "epsilon_k_ab" => &mut eos.0[5],
            "na" => &mut eos.0[6],
            "nb" => &mut eos.0[7],
            _ => panic!("{index} is not a valid PC-SAFT parameter!"),
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::super::{PcSaft, PcSaftAssociationRecord, PcSaftParameters};
    use super::*;
    use crate::pcsaft::PcSaftRecord;
    use approx::assert_relative_eq;
    use feos_core::parameter::{AssociationRecord, PureRecord};
    use feos_core::{Contributions::Total, FeosResult, State};
    use nalgebra::{dvector, vector};
    use quantity::{KELVIN, KILO, METER, MOL};

    pub fn pcsaft() -> FeosResult<(PcSaftPure<f64, 8>, PcSaft)> {
        let m = 1.5;
        let sigma = 3.4;
        let epsilon_k = 180.0;
        let mu = 2.2;
        let kappa_ab = 0.03;
        let epsilon_k_ab = 2500.;
        let na = 2.0;
        let nb = 1.0;
        let params = PcSaftParameters::new_pure(PureRecord::with_association(
            Default::default(),
            0.0,
            PcSaftRecord::new(m, sigma, epsilon_k, mu, 0.0, None, None, None),
            vec![AssociationRecord::new(
                Some(PcSaftAssociationRecord::new(kappa_ab, epsilon_k_ab)),
                na,
                nb,
                0.0,
            )],
        ))?;
        let eos = PcSaft::new(params);
        let params = [m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab, na, nb];
        Ok((PcSaftPure(params), eos))
    }

    #[test]
    fn test_pcsaft_pure() -> FeosResult<()> {
        let (pcsaft, eos) = pcsaft()?;

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
        let state = State::new_nvt(&pcsaft, temperature, volume, moles)?;
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
}
