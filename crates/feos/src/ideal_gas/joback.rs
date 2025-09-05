//! Implementation of the ideal gas heat capacity (de Broglie wavelength)
//! of [Joback and Reid, 1987](https://doi.org/10.1080/00986448708960487).
use feos_core::parameter::{FromSegments, Parameters};
use feos_core::{FeosResult, IdealGas, ReferenceSystem};
use nalgebra::DVector;
use num_dual::*;
use quantity::{MolarEntropy, Temperature};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Coefficients used in the Joback model.
///
/// Contains an additional fourth order polynomial coefficient `e`
/// which is not used in the original publication but is used in
/// parametrization for additional molecules in other publications.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct JobackRecord {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
}

impl JobackRecord {
    /// Creates a new `JobackRecord`
    pub fn new(a: f64, b: f64, c: f64, d: f64, e: f64) -> Self {
        Self { a, b, c, d, e }
    }
}

/// Implementation of the combining rules as described in
/// [Joback and Reid, 1987](https://doi.org/10.1080/00986448708960487).
impl FromSegments for JobackRecord {
    fn from_segments(segments: &[(Self, f64)]) -> FeosResult<Self> {
        let mut a = -37.93;
        let mut b = 0.21;
        let mut c = -3.91e-4;
        let mut d = 2.06e-7;
        let mut e = 0.0;
        segments.iter().for_each(|(s, n)| {
            a += s.a * n;
            b += s.b * n;
            c += s.c * n;
            d += s.d * n;
            e += s.e * n;
        });
        Ok(Self { a, b, c, d, e })
    }
}

pub type JobackParameters = Parameters<JobackRecord, (), ()>;

/// The ideal gas contribution according to
/// [Joback and Reid, 1987](https://doi.org/10.1080/00986448708960487).
///
/// The thermal de Broglie wavelength is calculated by integrating
/// the heat capacity with the following reference values:
///
/// - T = 289.15 K
/// - p = 1e5 Pa
/// - V = 1e-30 A³
#[derive(Clone)]
pub struct Joback<D = f64>(pub [D; 5]);

impl Joback<f64> {
    pub fn new(parameters: JobackParameters) -> Vec<Self> {
        parameters
            .pure
            .into_iter()
            .map(|r| {
                let m = &r.model_record;
                Self([m.a, m.b, m.c, m.d, m.e])
            })
            .collect()
    }
}

impl Joback {
    /// Directly calculates the molar ideal gas heat capacity from the Joback model.
    pub fn molar_isobaric_heat_capacity(
        joback: &[Joback],
        temperature: Temperature,
        molefracs: &DVector<f64>,
    ) -> FeosResult<MolarEntropy> {
        let t = temperature.to_reduced();
        let c_p: f64 = molefracs
            .iter()
            .zip(joback)
            .map(|(x, p)| {
                let [a, b, c, d, e] = p.0;
                x * (a + b * t + c * t.powi(2) + d * t.powi(3) + e * t.powi(4))
            })
            .sum();
        Ok(c_p / RGAS * quantity::RGAS)
    }
}

impl<D: DualNum<f64> + Copy> IdealGas<D> for Joback<D> {
    fn ln_lambda3<D2: DualNum<f64, Inner = D> + Copy>(&self, temperature: D2) -> D2 {
        let [a, b, c, d, e] = self.0.each_ref().map(D2::from_inner);
        let t = temperature;
        let t2 = t * t;
        let t4 = t2 * t2;
        let f = (temperature * KB / (P0 * A3)).ln();
        let h = (t2 - T0_2) * 0.5 * b
            + (t * t2 - T0_3) * c / 3.0
            + (t4 - T0_4) * d / 4.0
            + (t4 * t - T0_5) * e / 5.0
            + (t - T0) * a;
        let s = (t - T0) * b
            + (t2 - T0_2) * 0.5 * c
            + (t2 * t - T0_3) * d / 3.0
            + (t4 - T0_4) * e / 4.0
            + (t / T0).ln() * a;
        (h - t * s) / (t * RGAS) + f
    }

    fn ideal_gas_model(&self) -> &'static str {
        "Ideal gas (Joback)"
    }
}

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

impl<D: DualNum<f64> + Copy> Joback<D> {
    pub fn from_groups(group_counts: [D; 22]) -> Self {
        let a: D = A.into_iter().zip(group_counts).map(|(a, g)| g * a).sum();
        let b: D = B.into_iter().zip(group_counts).map(|(b, g)| g * b).sum();
        let c: D = C.into_iter().zip(group_counts).map(|(c, g)| g * c).sum();
        let d: D = D.into_iter().zip(group_counts).map(|(d, g)| g * d).sum();

        Self([a - 37.93, b + 0.21, c - 3.91e-4, d + 2.06e-7, D::zero()])
    }

    pub fn from_group_counts(group_counts: &HashMap<&str, D>) -> Self {
        Self::from_groups(GROUPS.map(|g| *group_counts.get(g).unwrap_or(&D::zero())))
    }
}

const RGAS: f64 = 6.022140857 * 1.38064852;
const T0: f64 = 298.15;
const T0_2: f64 = T0 * T0;
const T0_3: f64 = T0 * T0_2;
const T0_4: f64 = T0_2 * T0_2;
const T0_5: f64 = T0 * T0_4;
const P0: f64 = 1.0e5;
const A3: f64 = 1e-30;
const KB: f64 = 1.38064852e-23;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use feos_core::{
        Contributions, EquationOfState, State, StateBuilder,
        parameter::{ChemicalRecord, GroupCount, Identifier, PureRecord, SegmentRecord},
    };
    use nalgebra::dvector;
    use quantity::*;
    use std::collections::HashMap;
    use typenum::P3;

    use super::*;

    #[test]
    fn paper_example() -> FeosResult<()> {
        let segments_json = r#"[
        {
          "identifier": "-Cl",
          "a": 33.3,
          "b": -0.0963,
          "c": 0.000187,
          "d": -9.96e-8,
          "e": 0.0,
          "molarweight": 35.453
        },
        {
          "identifier": "-CH=(ring)",
          "a": -2.14,
          "b": 5.74e-2,
          "c": -1.64e-6,
          "d": -1.59e-8,
          "e": 0.0,
          "molarweight": 13.01864
        },
        {
          "identifier": "=CH<(ring)",
          "a": -8.25,
          "b": 1.01e-1,
          "c": -1.42e-4,
          "d": 6.78e-8,
          "e": 0.0,
          "molarweight": 13.01864
        }
        ]"#;
        let segment_records: Vec<SegmentRecord<JobackRecord, ()>> =
            serde_json::from_str(segments_json).expect("Unable to parse json.");
        let segment_map: HashMap<_, _> =
            segment_records.iter().map(|s| (&s.identifier, s)).collect();
        let (_, segments, _) = GroupCount::into_groups(ChemicalRecord::new(
            Identifier::default(),
            vec![
                String::from("-Cl"),
                String::from("-Cl"),
                String::from("-CH=(ring)"),
                String::from("-CH=(ring)"),
                String::from("-CH=(ring)"),
                String::from("-CH=(ring)"),
                String::from("=CH<(ring)"),
                String::from("=CH<(ring)"),
            ],
            None,
        ));
        // .segment_map(&segment_records)?;
        let joback_segments: Vec<_> = segments
            .into_iter()
            .map(|(s, n)| (segment_map[&s].model_record.clone(), n))
            .collect();
        let jr = JobackRecord::from_segments(&joback_segments)?;
        assert_relative_eq!(
            jr.a,
            33.3 * 2.0 - 2.14 * 4.0 - 8.25 * 2.0 - 37.93,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            jr.b,
            -0.0963 * 2.0 + 5.74e-2 * 4.0 + 1.01e-1 * 2.0 + 0.21,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            jr.c,
            0.000187 * 2.0 - 1.64e-6 * 4.0 - 1.42e-4 * 2.0 - 3.91e-4,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            jr.d,
            -9.96e-8 * 2.0 - 1.59e-8 * 4.0 + 6.78e-8 * 2.0 + 2.06e-7,
            epsilon = 1e-10
        );
        assert_relative_eq!(jr.e, 0.0);

        let pr = PureRecord::new(Identifier::default(), 1.0, jr);
        let joback = Joback::new(JobackParameters::new_pure(pr)?);
        let eos = EquationOfState::ideal_gas(joback);
        let state = State::new_pure(&&eos, 1000.0 * KELVIN, 1.0 * MOL / METER.powi::<P3>())?;
        assert!(
            ((state.molar_isobaric_heat_capacity(Contributions::IdealGas)
                / (JOULE / MOL / KELVIN))
                .into_value()
                - 224.6)
                .abs()
                < 1.0
        );
        Ok(())
    }

    #[test]
    fn c_p_comparison() -> FeosResult<()> {
        let record1 = PureRecord::new(
            Identifier::default(),
            1.0,
            JobackRecord::new(1.0, 0.2, 0.03, 0.004, 0.005),
        );
        let record2 = PureRecord::new(
            Identifier::default(),
            1.0,
            JobackRecord::new(-5.0, 0.4, 0.03, 0.002, 0.001),
        );
        let joback = Joback::new(JobackParameters::new_binary(
            [record1, record2],
            None,
            vec![],
        )?);
        let eos = EquationOfState::ideal_gas(joback.clone());
        let temperature = 300.0 * KELVIN;
        let volume = METER.powi::<P3>();
        let moles = &dvector![1.0, 3.0] * MOL;
        let state = StateBuilder::new(&&eos)
            .temperature(temperature)
            .volume(volume)
            .moles(&moles)
            .build()?;
        println!(
            "{} {}",
            Joback::molar_isobaric_heat_capacity(&joback, temperature, &state.molefracs)?,
            state.molar_isobaric_heat_capacity(Contributions::IdealGas)
        );
        assert_relative_eq!(
            Joback::molar_isobaric_heat_capacity(&joback, temperature, &state.molefracs)?,
            state.molar_isobaric_heat_capacity(Contributions::IdealGas),
            max_relative = 1e-10
        );
        Ok(())
    }
}

#[cfg(test)]
pub mod test_ad {
    use super::{Joback, JobackParameters, JobackRecord};
    use approx::assert_relative_eq;
    use feos_core::{Contributions::IdealGas, EquationOfState, FeosResult, State};
    use feos_core::{ResidualConst, StateHD};
    use nalgebra::{SVector, U1};
    use num_dual::DualNum;
    use quantity::{KELVIN, KILO, METER, MOL};

    pub fn joback() -> FeosResult<(Joback<f64>, Vec<Joback>)> {
        let a = 1.5;
        let b = 3.4e-2;
        let c = 180.0e-4;
        let d = 2.2e-6;
        let e = 0.03e-8;
        let eos = Joback::new(JobackParameters::from_model_records(vec![
            JobackRecord::new(a, b, c, d, e),
        ])?);
        let params = [a, b, c, d, e];
        let eos_ad = Joback(params);
        Ok((eos_ad, eos))
    }

    #[derive(Clone, Copy)]
    struct NoResidual;

    impl<D: DualNum<f64> + Copy> ResidualConst<1, D> for NoResidual {
        const NAME: &str = "";

        type Real = Self;

        type Lifted<D2: DualNum<f64, Inner = D> + Copy> = Self;

        fn re(&self) -> Self::Real {
            *self
        }

        fn lift<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
            *self
        }

        fn compute_max_density(&self, _: &SVector<D, 1>) -> D {
            D::from(1.0)
        }

        fn reduced_residual_helmholtz_energy_density(&self, _: &StateHD<D, U1>) -> D {
            D::from(0.0)
        }
    }

    #[test]
    fn test_joback() -> FeosResult<()> {
        let (joback_ad, joback) = joback()?;
        let eos = EquationOfState::ideal_gas(joback);
        let eos_ad = EquationOfState::new([joback_ad], NoResidual);

        let temperature = 300.0 * KELVIN;
        let density = 2.3 * KILO * MOL / (METER * METER * METER);

        let state = State::new_pure(&&eos, temperature, density)?;
        let a_feos = state.molar_helmholtz_energy(IdealGas);
        let mu_feos = state.chemical_potential(IdealGas);
        let p_feos = state.pressure(IdealGas);
        let s_feos = state.molar_entropy(IdealGas);
        let h_feos = state.molar_enthalpy(IdealGas);

        let state_ad = State::new_pure(&eos_ad, temperature, density)?;
        let a_ad = state_ad.molar_helmholtz_energy(IdealGas);
        let mu_ad = state_ad.chemical_potential(IdealGas);
        let p_ad = state_ad.pressure(IdealGas);
        let s_ad = state_ad.molar_entropy(IdealGas);
        let h_ad = state_ad.molar_enthalpy(IdealGas);

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
