use crate::{IdealGasAD, ParametersAD};
use num_dual::DualNum;
use std::collections::HashMap;

const RGAS: f64 = 6.022140857 * 1.38064852;
const T0: f64 = 298.15;
const T0_2: f64 = 298.15 * 298.15;
const T0_3: f64 = T0 * T0_2;
const T0_4: f64 = T0_2 * T0_2;
const T0_5: f64 = T0 * T0_4;
const P0: f64 = 1.0e5;
const A3: f64 = 1e-30;
const KB: f64 = 1.38064852e-23;

const A: [f64; 22] = [
    19.5, -0.909, -23.0, -66.2, 23.6, -8.0, -28.0, 32.37, -6.03, -20.5, -6.03, -20.5, -2.14, -8.25,
    30.9, 6.45, 45.0, 24.591, 24.1, 24.5, 25.7, 26.9,
];
const B: [f64; 22] = [
    -0.00808, 0.095, 0.204, 0.427, -0.0381, 0.105, 0.208, -0.007, 0.0854, 0.162, 0.0854, 0.162,
    0.0574, 0.101, -0.0336, 0.067, -0.07128, 0.0318, 0.0427, 0.0402, -0.0691, -0.0412,
];
const C: [f64; 22] = [
    0.000153, -5.44e-05, -0.000265, -0.000641, 0.000172, -9.63e-05, -0.000306, 0.00010267, -8e-06,
    -0.00016, -8e-06, -0.00016, -1.64e-06, -0.000142, 0.00016, -3.57e-05, 0.000264, 5.66e-05,
    8.04e-05, 4.02e-05, 0.000177, 0.000164,
];
const D: [f64; 22] = [
    -9.67e-08, 1.19e-08, 1.2e-07, 3.01e-07, -1.03e-07, 3.56e-08, 1.46e-07, -6.641e-08, -1.8e-08,
    6.24e-08, -1.8e-08, 6.24e-08, -1.59e-08, 6.78e-08, -9.88e-08, 2.86e-09, -1.515e-07, -4.29e-08,
    -6.87e-08, -4.52e-08, -9.88e-08, -9.76e-08,
];
const GROUPS: [&str; 22] = [
    "CH3", "CH2", ">CH", ">C<", "=CH2", "=CH", "=C<", "C≡CH", "CH2_hex", "CH_hex", "CH2_pent",
    "CH_pent", "CH_arom", "C_arom", "CH=O", ">C=O", "OCH3", "OCH2", "HCOO", "COO", "OH", "NH2",
];

/// The GC method for the ideal gas heat capacity by Joback & Reid.
#[derive(Clone, Copy)]
pub struct Joback(pub [f64; 5]);

impl Joback {
    pub fn from_groups<D: DualNum<f64> + Copy>(group_counts: [D; 22]) -> [D; 5] {
        let a: D = A.into_iter().zip(group_counts).map(|(a, g)| g * a).sum();
        let b: D = B.into_iter().zip(group_counts).map(|(b, g)| g * b).sum();
        let c: D = C.into_iter().zip(group_counts).map(|(c, g)| g * c).sum();
        let d: D = D.into_iter().zip(group_counts).map(|(d, g)| g * d).sum();

        [a - 37.93, b + 0.21, c - 3.91e-4, d + 2.06e-7, D::zero()]
    }

    pub fn from_group_counts<D: DualNum<f64> + Copy>(group_counts: &HashMap<&str, D>) -> [D; 5] {
        Self::from_groups(GROUPS.map(|g| *group_counts.get(g).unwrap_or(&D::zero())))
    }
}

impl ParametersAD for Joback {
    type Parameters<D: DualNum<f64> + Copy> = [D; 5];

    fn params<D: DualNum<f64> + Copy>(&self) -> [D; 5] {
        self.0.map(D::from)
    }

    fn params_from_inner<D: DualNum<f64> + Copy, D2: DualNum<f64, Inner = D> + Copy>(
        parameters: &[D; 5],
    ) -> [D2; 5] {
        parameters.map(D2::from_inner)
    }
}

impl IdealGasAD for Joback {
    const IDEAL_GAS: &str = "Joback";

    fn ln_lambda3_dual<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
    ) -> D {
        let [a, b, c, d, e] = *parameters;
        let t = temperature;
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t2 * t2;
        let f = (temperature * KB / (P0 * A3)).ln();
        let h = (t2 - T0_2) * 0.5 * b
            + (t3 - T0_3) * c / 3.0
            + (t4 - T0_4) * 0.25 * d
            + (t4 * t - T0_5) * 0.2 * e
            + (t - T0) * a;
        let s = (t - T0) * b
            + (t2 - T0_2) * 0.5 * c
            + (t3 - T0_3) * d / 3.0
            + (t4 - T0_4) * 0.25 * e
            + (t / T0).ln() * a;
        (h - t * s) / (t * RGAS) + f
    }
}

#[cfg(test)]
pub mod test {
    use super::Joback as JobackAD;
    use crate::{EquationOfStateAD, ParametersAD, ResidualHelmholtzEnergy, TotalHelmholtzEnergy};
    use approx::assert_relative_eq;
    use feos::ideal_gas::{Joback, JobackParameters, JobackRecord};
    use feos_core::{Contributions::IdealGas, EquationOfState, FeosResult, ReferenceSystem, State};
    use nalgebra::SVector;
    use ndarray::arr1;
    use num_dual::DualNum;
    use quantity::{KELVIN, KILO, METER, MOL};
    use std::sync::Arc;

    pub fn joback() -> FeosResult<(JobackAD, Arc<Joback>)> {
        let a = 1.5;
        let b = 3.4e-2;
        let c = 180.0e-4;
        let d = 2.2e-6;
        let e = 0.03e-8;
        let eos = Arc::new(Joback::new(JobackParameters::from_model_records(vec![
            JobackRecord::new(a, b, c, d, e),
        ])?));
        let params = [a, b, c, d, e];
        let eos_ad = JobackAD(params);
        Ok((eos_ad, eos))
    }

    struct NoResidual;
    impl ParametersAD for NoResidual {
        type Parameters<D: DualNum<f64> + Copy> = [D; 0];

        fn params<D: DualNum<f64> + Copy>(&self) -> Self::Parameters<D> {
            []
        }

        fn params_from_inner<D: DualNum<f64> + Copy, D2: DualNum<f64, Inner = D> + Copy>(
            _: &Self::Parameters<D>,
        ) -> Self::Parameters<D2> {
            []
        }
    }
    impl<const N: usize> ResidualHelmholtzEnergy<N> for NoResidual {
        const RESIDUAL: &str = "No residual";

        fn compute_max_density(&self, _: &SVector<f64, N>) -> f64 {
            1.0
        }

        fn residual_helmholtz_energy_density<D: DualNum<f64> + Copy>(
            _: &Self::Parameters<D>,
            _: D,
            _: &SVector<D, N>,
        ) -> D {
            D::zero()
        }
    }

    type JobackEos = EquationOfStateAD<JobackAD, NoResidual, 1>;

    #[test]
    fn test_joback() -> FeosResult<()> {
        let (joback_ad, joback) = joback()?;
        let eos = Arc::new(EquationOfState::ideal_gas(joback));
        let eos_ad = EquationOfStateAD::new([joback_ad], NoResidual);

        let temperature = 300.0 * KELVIN;
        let volume = 2.3 * METER * METER * METER;
        let moles = arr1(&[1.3]) * KILO * MOL;

        let state = State::new_nvt(&eos, temperature, volume, &moles)?;
        let a_feos = state.molar_helmholtz_energy(IdealGas);
        let mu_feos = state.chemical_potential(IdealGas);
        let p_feos = state.pressure(IdealGas);
        let s_feos = state.molar_entropy(IdealGas);
        let h_feos = state.molar_enthalpy(IdealGas);

        let total_moles = moles.sum();
        let t = temperature.to_reduced();
        let v = (volume / total_moles).to_reduced();
        let x = SVector::from_fn(|i, _| moles.get(i).convert_into(total_moles));
        let a_ad = JobackEos::molar_helmholtz_energy(&eos_ad.params(), t, v, &x);
        let mu_ad = JobackEos::chemical_potential(&eos_ad.params(), t, v, &x);
        let p_ad = JobackEos::pressure(&eos_ad.params(), t, v, &x);
        let s_ad = JobackEos::molar_entropy(&eos_ad.params(), t, v, &x);
        let h_ad = JobackEos::molar_enthalpy(&eos_ad.params(), t, v, &x);

        println!("\nMolar Helmholtz energy:\n{}", a_feos.to_reduced());
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
