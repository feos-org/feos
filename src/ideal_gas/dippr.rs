use feos_core::parameter::{NoBinaryModelRecord, Parameter, ParameterError, PureRecord};
use feos_core::si::{MolarEntropy, Temperature, JOULE, KELVIN, KILO, MOL};
use feos_core::{Components, DeBroglieWavelength, DeBroglieWavelengthDual, EosResult, IdealGas};
use ndarray::{Array1, Array2};
use num_dual::DualNum;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DipprRecord {
    DIPPR100([f64; 7]),
    DIPPR107([f64; 5]),
    DIPPR127([f64; 7]),
}

impl DipprRecord {
    pub fn eq100(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64) -> Self {
        Self::DIPPR100([a, b, c, d, e, f, g])
    }

    pub fn eq107(a: f64, b: f64, c: f64, d: f64, e: f64) -> Self {
        Self::DIPPR107([a, b, c, d, e])
    }

    pub fn eq127(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64) -> Self {
        Self::DIPPR127([a, b, c, d, e, f, g])
    }

    fn c_p(&self, t: f64) -> f64 {
        match self {
            Self::DIPPR100(coefs) => coefs.iter().rev().fold(0.0, |acc, c| t * acc + c),
            Self::DIPPR107([a, b, c, d, e]) => {
                let ct = c / t;
                let et = e / t;
                a + b * (ct / ct.sinh()).powi(2) + d * (et / et.cosh()).powi(2)
            }
            Self::DIPPR127([a, b, c, d, e, f, g]) => {
                let ct = c / t;
                let et = e / t;
                let gt = g / t;
                let fun = |x: f64| x * x * x.exp() / (x.exp() - 1.0).powi(2);
                a + b * fun(ct) + d * fun(et) + f * fun(gt)
            }
        }
    }

    fn c_p_integral<D: DualNum<f64> + Copy>(&self, t: D) -> D {
        match self {
            Self::DIPPR100(coefs) => coefs
                .iter()
                .enumerate()
                .rev()
                .fold(D::zero(), |acc, (i, &c)| t * (acc + c / (i + 1) as f64)),
            Self::DIPPR107([a, b, c, d, e]) => {
                let t_inv = t.recip();
                let ct = t_inv * *c;
                let et = t_inv * *e;
                t * *a + ct.tanh().recip() * (b * c) - et.tanh() * (d * e)
            }
            Self::DIPPR127([a, b, c, d, e, f, g]) => {
                let t_inv = t.recip();
                let ct = t_inv * *c;
                let et = t_inv * *e;
                let gt = t_inv * *g;
                let fun = |p: f64, x: D| ((x.exp() - 1.0) * p).recip() * (p * p);
                fun(*c, ct) * *b + fun(*e, et) * *d + fun(*g, gt) * *f + t * *a
            }
        }
    }

    fn c_p_t_integral<D: DualNum<f64> + Copy>(&self, t: D) -> D {
        match self {
            Self::DIPPR100([a, coefs @ ..]) => {
                coefs
                    .iter()
                    .enumerate()
                    .rev()
                    .fold(D::zero(), |acc, (i, &c)| t * (acc + c / (i + 1) as f64))
                    + t.ln() * *a
            }
            Self::DIPPR107([a, b, c, d, e]) => {
                let t_inv = t.recip();
                let ct = t_inv * *c;
                let et = t_inv * *e;
                t.ln() * *a + (t * ct.tanh()).recip() * (b * c)
                    - ct.sinh().ln() * *b
                    - et.tanh() * t_inv * (d * e)
                    + et.cosh().ln() * *d
            }
            Self::DIPPR127([a, b, c, d, e, f, g]) => {
                let t_inv = t.recip();
                let ct = t_inv * *c;
                let et = t_inv * *e;
                let gt = t_inv * *g;
                let fun = |p: f64, x: D| {
                    (((x.exp() - 1.0) * t).recip() + t_inv) * p - (x.exp() - 1.0).ln()
                };
                fun(*c, ct) * *b + fun(*e, et) * *d + fun(*g, gt) * *f + t.ln() * *a
            }
        }
    }
}

impl fmt::Display for DipprRecord {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DIPPR100([a, b, c, d, e, f, g]) => write!(
                fmt,
                "DipprRecord(EQ100, a={a}, b={b}, c={c}, d={d}, e={e}, f={f}, g={g})"
            ),
            Self::DIPPR107([a, b, c, d, e]) => {
                write!(fmt, "DipprRecord(EQ107, a={a}, b={b}, c={c}, d={d}, e={e})")
            }
            Self::DIPPR127([a, b, c, d, e, f, g]) => write!(
                fmt,
                "DipprRecord(EQ127, a={a}, b={b}, c={c}, d={d}, e={e}, f={f}, g={g})"
            ),
        }
    }
}

pub struct Dippr(Vec<PureRecord<DipprRecord>>);

impl Parameter for Dippr {
    type Pure = DipprRecord;
    type Binary = NoBinaryModelRecord;

    fn from_records(
        pure_records: Vec<PureRecord<Self::Pure>>,
        _binary_records: Option<Array2<Self::Binary>>,
    ) -> Result<Self, ParameterError> {
        Ok(Self(pure_records))
    }

    fn records(&self) -> (&[PureRecord<Self::Pure>], Option<&Array2<Self::Binary>>) {
        (&self.0, None)
    }
}

impl Dippr {
    pub fn molar_isobaric_heat_capacity(
        &self,
        temperature: Temperature,
        molefracs: &Array1<f64>,
    ) -> EosResult<MolarEntropy> {
        let t = temperature.convert_into(KELVIN);
        let c_p: f64 = molefracs
            .iter()
            .zip(&self.0)
            .map(|(x, r)| x * r.model_record.c_p(t))
            .sum();
        Ok(c_p * (JOULE / (KILO * MOL * KELVIN)))
    }
}

impl Components for Dippr {
    fn components(&self) -> usize {
        self.0.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        let mut records = Vec::with_capacity(component_list.len());
        component_list
            .iter()
            .for_each(|&i| records.push(self.0[i].clone()));
        Self::from_records(records, None).unwrap()
    }
}

impl IdealGas for Dippr {
    fn ideal_gas_model(&self) -> &dyn DeBroglieWavelength {
        self
    }
}

const RGAS: f64 = 8.31446261815324 * 1000.0;
const T0: f64 = 298.15;
const P0: f64 = 1.0e5;
const A3: f64 = 1e-30;
const KB: f64 = 1.38064852e-23;

impl<D: DualNum<f64> + Copy> DeBroglieWavelengthDual<D> for Dippr {
    fn ln_lambda3(&self, temperature: D) -> Array1<D> {
        let t = temperature;
        let f = (temperature * KB / (P0 * A3)).ln();
        self.0
            .iter()
            .map(|r| {
                let m = &r.model_record;
                let h = m.c_p_integral(t) - m.c_p_integral(T0);
                let s = m.c_p_t_integral(t) - m.c_p_t_integral(T0);
                (h - t * s) / (t * RGAS) + f
            })
            .collect()
    }
}

impl fmt::Display for Dippr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ideal gas (DIPPR)")
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use feos_core::parameter::Identifier;
    use feos_core::si::*;
    use feos_core::{Contributions, EquationOfState, StateBuilder};
    use num_dual::first_derivative;
    use std::sync::Arc;
    use typenum::P3;

    use super::*;

    #[test]
    fn eq100() -> EosResult<()> {
        let record = PureRecord::new(
            Identifier::default(),
            0.0,
            DipprRecord::eq100(276370., -2090.1, 8.125, -0.014116, 0.0000093701, 0.0, 0.0),
        );
        let dippr = Arc::new(Dippr::new_pure(record.clone())?);
        let eos = Arc::new(EquationOfState::ideal_gas(dippr.clone()));
        let temperature = 300.0 * KELVIN;
        let volume = METER.powi::<P3>();
        let state = StateBuilder::new(&eos)
            .temperature(temperature)
            .volume(volume)
            .total_moles(MOL)
            .build()?;

        let t = temperature.convert_into(KELVIN);
        let c_p_direct = record.model_record.c_p(t);
        let (_, c_p) = first_derivative(|t| record.model_record.c_p_integral(t), t);
        let (_, c_p_t) = first_derivative(|t| record.model_record.c_p_t_integral(t), t);
        println!("{c_p_direct} {c_p} {}", c_p_t * t);
        assert_relative_eq!(c_p_direct, c_p, max_relative = 1e-10);
        assert_relative_eq!(c_p_direct, c_p_t * t, max_relative = 1e-10);

        println!(
            "{} {}",
            dippr.molar_isobaric_heat_capacity(temperature, &state.molefracs)?,
            state.molar_isobaric_heat_capacity(Contributions::IdealGas)
        );
        let reference = 75355.81000000003 * JOULE / (KILO * MOL * KELVIN);
        assert_relative_eq!(
            reference,
            dippr.molar_isobaric_heat_capacity(temperature, &state.molefracs)?,
            max_relative = 1e-10
        );
        assert_relative_eq!(
            reference,
            state.molar_isobaric_heat_capacity(Contributions::IdealGas),
            max_relative = 1e-10
        );
        Ok(())
    }

    #[test]
    fn eq107() -> EosResult<()> {
        let record = PureRecord::new(
            Identifier::default(),
            0.0,
            DipprRecord::eq107(33363., 26790., 2610.5, 8896., 1169.),
        );
        let dippr = Arc::new(Dippr::new_pure(record.clone())?);
        let eos = Arc::new(EquationOfState::ideal_gas(dippr.clone()));
        let temperature = 300.0 * KELVIN;
        let volume = METER.powi::<P3>();
        let state = StateBuilder::new(&eos)
            .temperature(temperature)
            .volume(volume)
            .total_moles(MOL)
            .build()?;

        let t = temperature.convert_into(KELVIN);
        let c_p_direct = record.model_record.c_p(t);
        let (_, c_p) = first_derivative(|t| record.model_record.c_p_integral(t), t);
        let (_, c_p_t) = first_derivative(|t| record.model_record.c_p_t_integral(t), t);
        println!("{c_p_direct} {c_p} {}", c_p_t * t);
        assert_relative_eq!(c_p_direct, c_p, max_relative = 1e-10);
        assert_relative_eq!(c_p_direct, c_p_t * t, max_relative = 1e-10);

        println!(
            "{} {}",
            dippr.molar_isobaric_heat_capacity(temperature, &state.molefracs)?,
            state.molar_isobaric_heat_capacity(Contributions::IdealGas)
        );
        let reference = 33585.90452768923 * JOULE / (KILO * MOL * KELVIN);
        assert_relative_eq!(
            reference,
            dippr.molar_isobaric_heat_capacity(temperature, &state.molefracs)?,
            max_relative = 1e-10
        );
        assert_relative_eq!(
            reference,
            state.molar_isobaric_heat_capacity(Contributions::IdealGas),
            max_relative = 1e-10
        );
        Ok(())
    }

    #[test]
    fn eq127() -> EosResult<()> {
        let record = PureRecord::new(
            Identifier::default(),
            0.0,
            DipprRecord::eq127(
                3.3258E4, 3.6199E4, 1.2057E3, 1.5373E7, 3.2122E3, -1.5318E7, 3.2122E3,
            ),
        );
        let dippr = Arc::new(Dippr::new_pure(record.clone())?);
        let eos = Arc::new(EquationOfState::ideal_gas(dippr.clone()));
        let temperature = 20.0 * KELVIN;
        let volume = METER.powi::<P3>();
        let state = StateBuilder::new(&eos)
            .temperature(temperature)
            .volume(volume)
            .total_moles(MOL)
            .build()?;

        let t = temperature.convert_into(KELVIN);
        let c_p_direct = record.model_record.c_p(t);
        let (_, c_p) = first_derivative(|t| record.model_record.c_p_integral(t), t);
        let (_, c_p_t) = first_derivative(|t| record.model_record.c_p_t_integral(t), t);
        println!("{c_p_direct} {c_p} {}", c_p_t * t);
        assert_relative_eq!(c_p_direct, c_p, max_relative = 1e-10);
        assert_relative_eq!(c_p_direct, c_p_t * t, max_relative = 1e-10);

        println!(
            "{} {}",
            dippr.molar_isobaric_heat_capacity(temperature, &state.molefracs)?,
            state.molar_isobaric_heat_capacity(Contributions::IdealGas)
        );
        let reference = 33258.0 * JOULE / (KILO * MOL * KELVIN);
        assert_relative_eq!(
            reference,
            dippr.molar_isobaric_heat_capacity(temperature, &state.molefracs)?,
            max_relative = 1e-10
        );
        assert_relative_eq!(
            reference,
            state.molar_isobaric_heat_capacity(Contributions::IdealGas),
            max_relative = 1e-10
        );
        Ok(())
    }
}
