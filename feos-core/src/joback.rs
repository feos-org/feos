//! Implementation of the ideal gas heat capacity (de Broglie wavelength)
//! of [Joback and Reid, 1987](https://doi.org/10.1080/00986448708960487).

use crate::parameter::*;
use crate::{
    EosResult, EosUnit, EquationOfState, HelmholtzEnergy, IdealGasContribution,
    IdealGasContributionDual,
};
use conv::ValueInto;
use ndarray::Array1;
use num_dual::*;
use quantity::QuantityScalar;
use serde::{Deserialize, Serialize};
use std::fmt;

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

/// The ideal gas contribution according to
/// [Joback and Reid, 1987](https://doi.org/10.1080/00986448708960487).
#[derive(Debug, Clone)]
pub struct Joback {
    pub records: Vec<JobackRecord>,
}

impl Joback {
    /// Creates a new Joback contribution.
    pub fn new(records: Vec<JobackRecord>) -> Self {
        Self { records }
    }

    /// Creates a default ($c_p^\mathrm{ig}=0$) ideal gas contribution for the
    /// given number of components.
    pub fn default(components: usize) -> Self {
        Self::new(vec![JobackRecord::default(); components])
    }

    /// Directly calculates the ideal gas heat capacity from the Joback model.
    pub fn c_p<U: EosUnit>(
        &self,
        temperature: QuantityScalar<U>,
        molefracs: &Array1<f64>,
    ) -> EosResult<QuantityScalar<U>> {
        let t = temperature.to_reduced(U::reference_temperature())?;
        let mut c_p = 0.0;
        for (j, &x) in self.records.iter().zip(molefracs.iter()) {
            c_p += x * (j.a + j.b * t + j.c * t.powi(2) + j.d * t.powi(3) + j.e * t.powi(4));
        }
        Ok(c_p / RGAS * U::gas_constant())
    }
}

impl fmt::Display for Joback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ideal gas (Joback)")
    }
}

const RGAS: f64 = 6.022140857 * 1.38064852;
const T0: f64 = 298.15;
const P0: f64 = 1.0e5;
const A3: f64 = 1e-30;
const KB: f64 = 1.38064852e-23;

impl<D: DualNum<f64>> IdealGasContributionDual<D> for Joback {
    fn de_broglie_wavelength(&self, temperature: D, components: usize) -> Array1<D> {
        let t = temperature;
        let t2 = t * t;
        let f = (temperature * KB / (P0 * A3)).ln();
        Array1::from_shape_fn(components, |i| {
            let j = &self.records[i];
            let h = (t2 - T0 * T0) * 0.5 * j.b
                + (t * t2 - T0.powi(3)) * j.c / 3.0
                + (t2 * t2 - T0.powi(4)) * j.d / 4.0
                + (t2 * t2 * t - T0.powi(5)) * j.e / 5.0
                + (t - T0) * j.a;
            let s = (t - T0) * j.b
                + (t2 - T0.powi(2)) * 0.5 * j.c
                + (t2 * t - T0.powi(3)) * j.d / 3.0
                + (t2 * t2 - T0.powi(4)) * j.e / 4.0
                + (t / T0).ln() * j.a;
            (h - t * s) / (t * RGAS) + f
        })
    }
}

impl EquationOfState for Joback {
    fn components(&self) -> usize {
        self.records.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        let records = component_list
            .iter()
            .map(|&i| self.records[i].clone())
            .collect();
        Self::new(records)
    }

    fn compute_max_density(&self, _moles: &Array1<f64>) -> f64 {
        1.0
    }

    fn residual(&self) -> &[Box<dyn HelmholtzEnergy>] {
        &[]
    }

    fn ideal_gas(&self) -> &dyn IdealGasContribution {
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::{Contributions, State, StateBuilder};
    use approx::assert_relative_eq;
    use ndarray::arr1;
    use quantity::si::*;
    use std::rc::Rc;

    use super::*;

    #[derive(Deserialize, Clone, Debug)]
    struct ModelRecord;

    #[test]
    fn paper_example() -> EosResult<()> {
        let segments_json = r#"[
        {
          "identifier": "-Cl",
          "model_record": null,
          "ideal_gas_record": {
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
          "model_record": null,
          "ideal_gas_record": {
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
          "model_record": null,
          "ideal_gas_record": {
            "a": -8.25,
            "b": 1.01e-1,
            "c": -1.42e-4,
            "d": 6.78e-8,
            "e": 0.0
          },
          "molarweight": 13.01864
        }
        ]"#;
        let segment_records: Vec<SegmentRecord<ModelRecord, JobackRecord>> =
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
            .map(|(s, &n)| (s.ideal_gas_record.clone().unwrap(), n))
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

        let eos = Rc::new(Joback::new(vec![jr]));
        let state = State::new_nvt(
            &eos,
            1000.0 * KELVIN,
            1.0 * ANGSTROM.powi(3),
            &(arr1(&[1.0]) * MOL),
        )?;
        assert!(
            (state
                .c_p(Contributions::IdealGas)
                .to_reduced(JOULE / MOL / KELVIN)?
                - 224.6)
                .abs()
                < 1.0
        );
        Ok(())
    }

    #[test]
    fn c_p_comparison() -> EosResult<()> {
        let record1 = JobackRecord::new(1.0, 0.2, 0.03, 0.004, 0.005);
        let record2 = JobackRecord::new(-5.0, 0.4, 0.03, 0.002, 0.001);
        let joback = Rc::new(Joback::new(vec![record1, record2]));
        let temperature = 300.0 * KELVIN;
        let volume = METER.powi(3);
        let moles = arr1(&[1.0, 3.0]) * MOL;
        let state = StateBuilder::new(&joback)
            .temperature(temperature)
            .volume(volume)
            .moles(&moles)
            .build()?;
        println!(
            "{} {}",
            joback.c_p(temperature, &state.molefracs)?,
            state.c_p(Contributions::IdealGas)
        );
        assert_relative_eq!(
            joback.c_p(temperature, &state.molefracs)?,
            state.c_p(Contributions::IdealGas),
            max_relative = 1e-10
        );
        Ok(())
    }
}
