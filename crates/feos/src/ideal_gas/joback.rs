//! Implementation of the ideal gas heat capacity (de Broglie wavelength)
//! of [Joback and Reid, 1987](https://doi.org/10.1080/00986448708960487).
use feos_core::parameter::{FromSegments, Parameters, PureRecord};
use feos_core::{Components, FeosResult, IdealGas, ReferenceSystem};
use ndarray::Array1;
use num_dual::*;
use quantity::{MolarEntropy, Temperature};
use serde::{Deserialize, Serialize};

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
pub struct Joback(Vec<PureRecord<JobackRecord, ()>>);

impl Joback {
    pub fn new(parameters: JobackParameters) -> Self {
        Self(parameters.pure_records)
    }
}

impl Joback {
    /// Directly calculates the molar ideal gas heat capacity from the Joback model.
    pub fn molar_isobaric_heat_capacity(
        &self,
        temperature: Temperature,
        molefracs: &Array1<f64>,
    ) -> FeosResult<MolarEntropy> {
        let t = temperature.to_reduced();
        let c_p: f64 = molefracs
            .iter()
            .zip(&self.0)
            .map(|(x, p)| {
                let m = &p.model_record;
                x * (m.a + m.b * t + m.c * t.powi(2) + m.d * t.powi(3) + m.e * t.powi(4))
            })
            .sum();
        Ok(c_p / RGAS * quantity::RGAS)
    }
}

impl Components for Joback {
    fn components(&self) -> usize {
        self.0.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        let mut records = Vec::with_capacity(component_list.len());
        component_list
            .iter()
            .for_each(|&i| records.push(self.0[i].clone()));
        Self(records)
    }
}

impl IdealGas for Joback {
    fn ln_lambda3<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D> {
        let t = temperature;
        let t2 = t * t;
        let t4 = t2 * t2;
        let f = (temperature * KB / (P0 * A3)).ln();
        Array1::from_shape_fn(self.0.len(), |i| {
            let j = &self.0[i].model_record;
            let h = (t2 - T0_2) * 0.5 * j.b
                + (t * t2 - T0_3) * j.c / 3.0
                + (t4 - T0_4) * j.d / 4.0
                + (t4 * t - T0_5) * j.e / 5.0
                + (t - T0) * j.a;
            let s = (t - T0) * j.b
                + (t2 - T0_2) * 0.5 * j.c
                + (t2 * t - T0_3) * j.d / 3.0
                + (t4 - T0_4) * j.e / 4.0
                + (t / T0).ln() * j.a;
            (h - t * s) / (t * RGAS) + f
        })
    }

    fn ideal_gas_model(&self) -> String {
        "Ideal gas (Joback)".into()
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
        parameter::{ChemicalRecord, Identifier, SegmentRecord},
    };
    use ndarray::arr1;
    use quantity::*;
    use std::{collections::HashMap, sync::Arc};
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
        let (_, segments, _) = ChemicalRecord::new(
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
        )
        .into_groups();
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
        let joback = Arc::new(Joback::new(JobackParameters::new_pure(pr)));
        let eos = Arc::new(EquationOfState::ideal_gas(joback));
        let state = State::new_nvt(
            &eos,
            1000.0 * KELVIN,
            1.0 * ANGSTROM.powi::<P3>(),
            &(&arr1(&[1.0]) * MOL),
        )?;
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
        let joback = Arc::new(Joback::new(JobackParameters::new_binary(
            [record1, record2],
            None,
            vec![],
        )));
        let eos = Arc::new(EquationOfState::ideal_gas(joback.clone()));
        let temperature = 300.0 * KELVIN;
        let volume = METER.powi::<P3>();
        let moles = &arr1(&[1.0, 3.0]) * MOL;
        let state = StateBuilder::new(&eos)
            .temperature(temperature)
            .volume(volume)
            .moles(&moles)
            .build()?;
        println!(
            "{} {}",
            joback.molar_isobaric_heat_capacity(temperature, &state.molefracs)?,
            state.molar_isobaric_heat_capacity(Contributions::IdealGas)
        );
        assert_relative_eq!(
            joback.molar_isobaric_heat_capacity(temperature, &state.molefracs)?,
            state.molar_isobaric_heat_capacity(Contributions::IdealGas),
            max_relative = 1e-10
        );
        Ok(())
    }
}
