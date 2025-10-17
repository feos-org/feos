use num_dual::DualNum;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone, Deserialize)]
pub struct IdealGasFunctionJson {
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(flatten)]
    parameters: HashMap<String, Value>,
}

pub struct IdealGasFunctionIterator {
    inner: IdealGasFunctionJson,
    count: usize,
    index: usize,
}

impl Iterator for IdealGasFunctionIterator {
    type Item = IdealGasFunction;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.count {
            return None;
        }
        let mut parameters: HashMap<_, _> = self
            .inner
            .parameters
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    if self.inner.ty == "IdealGasHelmholtzCP0AlyLee" {
                        // AlyLee parameters are stored as list instead of named parameters
                        v.clone()
                    } else {
                        v.as_array().map_or(v, |v| &v[self.index]).clone()
                    },
                )
            })
            .collect();
        if self.inner.ty == "IdealGasHelmholtzCP0AlyLee" {
            self.index += 5
        } else {
            self.index += 1;
        }
        parameters.insert("type".into(), serde_json::to_value(&self.inner.ty).unwrap());
        Some(serde_json::from_value(serde_json::to_value(parameters).unwrap()).unwrap())
    }
}

impl IntoIterator for IdealGasFunctionJson {
    type Item = IdealGasFunction;
    type IntoIter = IdealGasFunctionIterator;

    fn into_iter(self) -> Self::IntoIter {
        let count = self
            .parameters
            .values()
            .map(|e| e.as_array().map_or(1, Vec::len))
            .max()
            .unwrap();
        IdealGasFunctionIterator {
            count,
            inner: self,
            index: 0,
        }
    }
}

#[derive(Clone, Copy, Deserialize)]
#[serde(tag = "type")]
#[expect(non_snake_case)]
pub enum IdealGasFunction {
    IdealGasHelmholtzLead { a1: f64, a2: f64 },
    IdealGasHelmholtzLogTau { a: f64 },
    IdealGasHelmholtzPower { n: f64, t: f64 },
    IdealGasHelmholtzPlanckEinstein { n: f64, t: f64 },
    IdealGasHelmholtzPlanckEinsteinFunctionT { Tcrit: f64, n: f64, v: f64 },
    IdealGasHelmholtzPlanckEinsteinGeneralized { c: f64, d: f64, n: f64, t: f64 },
    IdealGasHelmholtzCP0AlyLee { T0: f64, Tc: f64, c: [f64; 5] },
    IdealGasHelmholtzCP0Constant { T0: f64, Tc: f64, cp_over_R: f64 },
    IdealGasHelmholtzCP0PolyT { T0: f64, Tc: f64, c: f64, t: f64 },
    IdealGasHelmholtzEnthalpyEntropyOffset { a1: f64, a2: f64 },
}

impl IdealGasFunction {
    pub fn evaluate<D: DualNum<f64> + Copy>(&self, delta: D, tau: D) -> D {
        match *self {
            IdealGasFunction::IdealGasHelmholtzLead { a1, a2 } => delta.ln() + a1 + tau * a2,
            IdealGasFunction::IdealGasHelmholtzLogTau { a } => tau.ln() * a,
            IdealGasFunction::IdealGasHelmholtzPower { n, t } => tau.powf(t) * n,
            IdealGasFunction::IdealGasHelmholtzPlanckEinstein { n, t } => {
                (-(-tau * t).exp()).ln_1p() * n
            }
            IdealGasFunction::IdealGasHelmholtzPlanckEinsteinFunctionT { Tcrit, n, v } => {
                (-(-tau * v / Tcrit).exp()).ln_1p() * n
            }
            IdealGasFunction::IdealGasHelmholtzPlanckEinsteinGeneralized { c, d, n, t } => {
                ((tau * t).exp() * d + c).ln() * n
            }
            IdealGasFunction::IdealGasHelmholtzCP0AlyLee { T0, Tc, c } => {
                let [a, b, c, d, e] = c;
                let mut res = D::zero();
                if a > 0.0 {
                    let tau0 = Tc / T0;
                    res += (-tau / tau0 + 1.0 + (tau / tau0).ln()) * a;
                }
                if b > 0.0 {
                    res += (-(-tau * 2.0 * c / Tc).exp()).ln_1p() * b;
                }
                if d > 0.0 {
                    res -= ((-tau * 2.0 * e / Tc).exp()).ln_1p() * d;
                }
                res
            }
            IdealGasFunction::IdealGasHelmholtzCP0Constant { T0, Tc, cp_over_R } => {
                let tau0 = Tc / T0;
                (-tau / tau0 + 1.0 + (tau / tau0).ln()) * cp_over_R
            }
            IdealGasFunction::IdealGasHelmholtzCP0PolyT { T0, Tc, c, t } => {
                // unfortunately some models use floats and other models use 0 or -1 which needs to be treated separately...
                if t.abs() < 10.0 * f64::EPSILON {
                    let tau0 = Tc / T0;
                    (-tau / tau0 + 1.0 + (tau / tau0).ln()) * c
                } else if (t + 1.0).abs() < 10.0 * f64::EPSILON {
                    let tau0 = Tc / T0;
                    (-tau / Tc * (tau / tau0).ln() + (tau - tau0) / Tc) * c
                } else {
                    (-tau.powf(-t) * Tc.powf(t) / (t * (t + 1.0))
                        - tau * T0.powf(t + 1.0) / (Tc * (t + 1.0))
                        + T0.powf(t) / t)
                        * c
                }
            }
            IdealGasFunction::IdealGasHelmholtzEnthalpyEntropyOffset { a1, a2 } => tau * a2 + a1,
        }
    }
}
