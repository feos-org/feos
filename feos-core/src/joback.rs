//! Implementation of the ideal gas heat capacity (de Broglie wavelength)
//! of [Joback and Reid, 1987](https://doi.org/10.1080/00986448708960487).

use crate::equation_of_state::{Components, DeBroglieWavelength, DeBroglieWavelengthDual};
use crate::parameter::*;
use crate::si::{MolarEntropy, Temperature};
use crate::{EosResult, IdealGas};
use conv::ValueInto;
use ndarray::{Array, Array1, Array2};
use num_dual::*;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// Coefficients used in the Joback model.
///
/// Contains an additional fourth order polynomial coefficient `e`
/// which is not used in the original publication but is used in
/// parametrization for additional molecules in other publications.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct JobackRecord {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
}

impl JobackRecord {
    /// Creates a new `JobackRecord`
    pub fn new(a: f64, b: f64, c: f64, d: f64, e: f64) -> Self {
        Self { a, b, c, d, e }
    }
}

impl fmt::Display for JobackRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "JobackRecord(a={}, b={}, c={}, d={}, e={})",
            self.a, self.b, self.c, self.d, self.e
        )
    }
}

/// Implementation of the combining rules as described in
/// [Joback and Reid, 1987](https://doi.org/10.1080/00986448708960487).
impl<T: Copy + ValueInto<f64>> FromSegments<T> for JobackRecord {
    fn from_segments(segments: &[(Self, T)]) -> Result<Self, ParameterError> {
        let mut a = -37.93;
        let mut b = 0.21;
        let mut c = -3.91e-4;
        let mut d = 2.06e-7;
        let mut e = 0.0;
        segments.iter().for_each(|(s, n)| {
            let n = (*n).value_into().unwrap();
            a += s.a * n;
            b += s.b * n;
            c += s.c * n;
            d += s.d * n;
            e += s.e * n;
        });
        Ok(Self { a, b, c, d, e })
    }
}

/// Parameters for one or more components for the Joback and Reid model.
pub struct JobackParameters {
    a: Array1<f64>,
    b: Array1<f64>,
    c: Array1<f64>,
    d: Array1<f64>,
    e: Array1<f64>,
    pure_records: Vec<PureRecord<JobackRecord>>,
}

impl Parameter for JobackParameters {
    type Pure = JobackRecord;
    type Binary = JobackBinaryRecord;

    fn from_records(
        pure_records: Vec<PureRecord<Self::Pure>>,
        _binary_records: Option<Array2<Self::Binary>>,
    ) -> Result<Self, ParameterError> {
        let n = pure_records.len();

        let mut a = Array::zeros(n);
        let mut b = Array::zeros(n);
        let mut c = Array::zeros(n);
        let mut d = Array::zeros(n);
        let mut e = Array::zeros(n);

        for (i, record) in pure_records.iter().enumerate() {
            let r = &record.model_record;
            a[i] = r.a;
            b[i] = r.b;
            c[i] = r.c;
            d[i] = r.d;
            e[i] = r.e;
        }

        Ok(Self {
            a,
            b,
            c,
            d,
            e,
            pure_records,
        })
    }

    fn records(&self) -> (&[PureRecord<Self::Pure>], Option<&Array2<Self::Binary>>) {
        (&self.pure_records, None)
    }
}

/// Dummy implementation to satisfy traits for parameter handling.
/// Not intended to be used.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct JobackBinaryRecord;

impl From<f64> for JobackBinaryRecord {
    fn from(_: f64) -> Self {
        Self
    }
}

impl From<JobackBinaryRecord> for f64 {
    fn from(_: JobackBinaryRecord) -> Self {
        0.0 // nasty hack - panic crashes Ipython kernel, actual value is never used
    }
}

impl<T: Copy + ValueInto<f64>> FromSegmentsBinary<T> for JobackBinaryRecord {
    fn from_segments_binary(_segments: &[(Self, T, T)]) -> Result<Self, ParameterError> {
        Ok(Self)
    }
}

impl std::fmt::Display for JobackBinaryRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

/// The ideal gas contribution according to
/// [Joback and Reid, 1987](https://doi.org/10.1080/00986448708960487).
///
/// The (cubic) de Broglie wavelength is calculated by integrating
/// the heat capacity with the following reference values:
///
/// - T = 289.15 K
/// - p = 1e5 Pa
/// - V = 1e-30 A³
pub struct Joback {
    pub parameters: Arc<JobackParameters>,
}

impl Joback {
    /// Creates a new Joback contribution.
    pub fn new(parameters: Arc<JobackParameters>) -> Self {
        Self { parameters }
    }

    /// Directly calculates the molar ideal gas heat capacity from the Joback model.
    pub fn molar_isobaric_heat_capacity(
        &self,
        temperature: Temperature,
        molefracs: &Array1<f64>,
    ) -> EosResult<MolarEntropy> {
        let t = temperature.to_reduced();
        let p = &self.parameters;
        let c_p = (molefracs
            * &(&p.a + &p.b * t + &p.c * t.powi(2) + &p.d * t.powi(3) + &p.e * t.powi(4)))
            .sum();
        Ok(c_p / RGAS * crate::si::RGAS)
    }
}

impl Components for Joback {
    fn components(&self) -> usize {
        self.parameters.pure_records.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        let mut records = Vec::with_capacity(component_list.len());
        component_list
            .iter()
            .for_each(|&i| records.push(self.parameters.pure_records[i].clone()));
        Self::new(Arc::new(
            JobackParameters::from_records(records, None).unwrap(),
        ))
    }
}

impl IdealGas for Joback {
    fn ideal_gas_model(&self) -> &dyn DeBroglieWavelength {
        self
    }
}

impl<D: DualNum<f64> + Copy> DeBroglieWavelengthDual<D> for Joback {
    fn ln_lambda3(&self, temperature: D) -> Array1<D> {
        let t = temperature;
        let t2 = t * t;
        let t4 = t2 * t2;
        let f = (temperature * KB / (P0 * A3)).ln();
        Array1::from_shape_fn(self.parameters.pure_records.len(), |i| {
            let j = &self.parameters.pure_records[i].model_record;
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
}

impl fmt::Display for Joback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ideal gas (Joback)")
    }
}

const RGAS: f64 = 6.022140857 * 1.38064852;
const T0: f64 = 298.15;
const T0_2: f64 = 298.15 * 298.15;
const T0_3: f64 = T0 * T0_2;
const T0_4: f64 = T0_2 * T0_2;
const T0_5: f64 = T0 * T0_4;
const P0: f64 = 1.0e5;
const A3: f64 = 1e-30;
const KB: f64 = 1.38064852e-23;

#[cfg(test)]
mod tests {
    use crate::si::*;
    use crate::{Contributions, Residual, State, StateBuilder};
    use approx::assert_relative_eq;
    use ndarray::arr1;
    use std::sync::Arc;
    use typenum::P3;

    use super::*;

    // implement Residual to test Joback as equation of state
    impl Residual for Joback {
        fn compute_max_density(&self, _moles: &Array1<f64>) -> f64 {
            1.0
        }

        fn contributions(&self) -> &[Box<dyn crate::HelmholtzEnergy>] {
            &[]
        }

        fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
            MolarWeight::from_shape_fn(self.components(), |i| {
                self.parameters.pure_records[i].molarweight * GRAM / MOL
            })
        }
    }

    #[test]
    fn paper_example() -> EosResult<()> {
        let segments_json = r#"[
        {
          "identifier": "-Cl",
          "model_record": {
            "a": 33.3,
            "b": -0.0963,
            "c": 0.000187,
            "d": -9.96e-8,
            "e": 0.0
          },
          "molarweight": 35.453
        },
        {
          "identifier": "-CH=(ring)",
          "model_record": {
            "a": -2.14,
            "b": 5.74e-2,
            "c": -1.64e-6,
            "d": -1.59e-8,
            "e": 0.0
          },
          "molarweight": 13.01864
        },
        {
          "identifier": "=CH<(ring)",
          "model_record": {
            "a": -8.25,
            "b": 1.01e-1,
            "c": -1.42e-4,
            "d": 6.78e-8,
            "e": 0.0
          },
          "molarweight": 13.01864
        }
        ]"#;
        let segment_records: Vec<SegmentRecord<JobackRecord>> =
            serde_json::from_str(segments_json).expect("Unable to parse json.");
        let segments = ChemicalRecord::new(
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
        .segment_map(&segment_records)?;
        assert_eq!(segments.get(&segment_records[0]), Some(&2));
        assert_eq!(segments.get(&segment_records[1]), Some(&4));
        assert_eq!(segments.get(&segment_records[2]), Some(&2));
        let joback_segments: Vec<_> = segments
            .iter()
            .map(|(s, &n)| (s.model_record.clone(), n))
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
        let eos = Arc::new(Joback::new(Arc::new(JobackParameters::new_pure(pr)?)));
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
    fn c_p_comparison() -> EosResult<()> {
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
        let parameters = Arc::new(JobackParameters::new_binary(vec![record1, record2], None)?);
        let joback = Arc::new(Joback::new(parameters));
        let temperature = 300.0 * KELVIN;
        let volume = METER.powi::<P3>();
        let moles = &arr1(&[1.0, 3.0]) * MOL;
        let state = StateBuilder::new(&joback)
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
