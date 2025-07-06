use super::{A0, A1, A2, AD, B0, B1, B2, BD, CD, MAX_ETA};
use crate::{NamedParameters, ParametersAD, ResidualHelmholtzEnergy};
use nalgebra::SVector;
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_6, PI};

const PI_SQ_43: f64 = 4.0 / 3.0 * PI * PI;

/// Optimized implementation of PC-SAFT for a single component.
#[derive(Clone, Copy)]
pub struct PcSaftPure<const N: usize>(pub [f64; N]);

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

    ((hs + hc + disp + dipole) * temperature, [eta, eta_m1])
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

    non_assoc + assoc * temperature
}

impl<const N: usize> ParametersAD for PcSaftPure<N> {
    type Parameters<D: DualNum<f64> + Copy> = [D; N];

    fn params<D: DualNum<f64> + Copy>(&self) -> Self::Parameters<D> {
        self.0.map(D::from)
    }

    fn params_from_inner<D: DualNum<f64> + Copy, D2: DualNum<f64, Inner = D> + Copy>(
        parameters: &Self::Parameters<D>,
    ) -> Self::Parameters<D2> {
        parameters.map(D2::from_inner)
    }
}

impl ResidualHelmholtzEnergy<1> for PcSaftPure<8> {
    const RESIDUAL: &str = "PC-SAFT (pure)";

    fn compute_max_density(&self, _: &SVector<f64, 1>) -> f64 {
        let m = self.0[0];
        let sigma = self.0[1];
        MAX_ETA / (FRAC_PI_6 * m * sigma.powi(3))
    }

    fn residual_helmholtz_energy_density<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        partial_density: &SVector<D, 1>,
    ) -> D {
        let density = partial_density.data.0[0][0];
        helmholtz_energy_density(parameters, temperature, density)
    }
}

impl ResidualHelmholtzEnergy<1> for PcSaftPure<4> {
    const RESIDUAL: &str = "PC-SAFT (pure)";

    fn compute_max_density(&self, _: &SVector<f64, 1>) -> f64 {
        let m = self.0[0];
        let sigma = self.0[1];
        MAX_ETA / (FRAC_PI_6 * m * sigma.powi(3))
    }

    fn residual_helmholtz_energy_density<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        partial_density: &SVector<D, 1>,
    ) -> D {
        let density = partial_density.data.0[0][0];
        let [m, sigma, epsilon_k, mu] = *parameters;
        helmholtz_energy_density_non_assoc(m, sigma, epsilon_k, mu, temperature, density).0
    }
}

impl<const N: usize> NamedParameters for PcSaftPure<N> {
    fn index_parameters_mut<'a, D: DualNum<f64> + Copy>(
        parameters: &'a mut [D; N],
        index: &str,
    ) -> &'a mut D {
        match index {
            "m" => &mut parameters[0],
            "sigma" => &mut parameters[1],
            "epsilon_k" => &mut parameters[2],
            "mu" => &mut parameters[3],
            "kappa_ab" => &mut parameters[4],
            "epsilon_k_ab" => &mut parameters[5],
            "na" => &mut parameters[6],
            "nb" => &mut parameters[7],
            _ => panic!("{index} is not a valid PC-SAFT parameter!"),
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::{PcSaftPure, ResidualHelmholtzEnergy};
    use crate::eos::pcsaft::test::pcsaft;
    use approx::assert_relative_eq;
    use feos_core::{Contributions::Total, FeosResult, ReferenceSystem, State};
    use nalgebra::SVector;
    use ndarray::arr1;
    use quantity::{KELVIN, KILO, METER, MOL};

    #[test]
    fn test_pcsaft_pure() -> FeosResult<()> {
        let (pcsaft, eos) = pcsaft()?;
        let pcsaft = pcsaft.0;

        let temperature = 300.0 * KELVIN;
        let volume = 2.3 * METER * METER * METER;
        let moles = arr1(&[1.3]) * KILO * MOL;

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
        let a_ad = PcSaftPure::residual_molar_helmholtz_energy(&pcsaft, t, v, &x);
        let mu_ad = PcSaftPure::residual_chemical_potential(&pcsaft, t, v, &x);
        let p_ad = PcSaftPure::pressure(&pcsaft, t, v, &x);
        let s_ad = PcSaftPure::residual_molar_entropy(&pcsaft, t, v, &x);
        let h_ad = PcSaftPure::residual_molar_enthalpy(&pcsaft, t, v, &x);

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
