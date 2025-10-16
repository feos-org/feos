use num_dual::DualNum;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone, Deserialize)]
pub struct ResidualFunctionJson {
    #[serde(rename = "type")]
    ty: String,
    #[serde(flatten)]
    parameters: HashMap<String, Vec<Value>>,
}

pub struct ResidualFunctionIterator {
    inner: ResidualFunctionJson,
    count: usize,
    index: usize,
}

impl Iterator for ResidualFunctionIterator {
    type Item = ResidualFunction;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.count {
            return None;
        }
        let mut parameters: HashMap<_, _> = self
            .inner
            .parameters
            .iter()
            .map(|(k, v)| (k.clone(), v[self.index].clone()))
            .collect();
        self.index += 1;
        parameters.insert("type".into(), serde_json::to_value(&self.inner.ty).unwrap());
        Some(serde_json::from_value(serde_json::to_value(parameters).unwrap()).unwrap())
    }
}

impl IntoIterator for ResidualFunctionJson {
    type Item = ResidualFunction;
    type IntoIter = ResidualFunctionIterator;

    fn into_iter(self) -> Self::IntoIter {
        let count = self.parameters.values().next().unwrap().len();
        ResidualFunctionIterator {
            count,
            inner: self,
            index: 0,
        }
    }
}

#[derive(Clone, Copy, Deserialize)]
#[serde(tag = "type")]
pub enum ResidualFunction {
    ResidualHelmholtzPower {
        d: i32,
        l: i32,
        n: f64,
        t: f64,
    },
    ResidualHelmholtzGaussian {
        d: i32,
        n: f64,
        t: f64,
        beta: f64,
        epsilon: f64,
        eta: f64,
        gamma: f64,
    },
    ResidualHelmholtzNonAnalytic {
        #[serde(rename = "A")]
        aa: f64,
        #[serde(rename = "B")]
        bb: f64,
        #[serde(rename = "C")]
        cc: f64,
        #[serde(rename = "D")]
        dd: f64,
        a: f64,
        b: f64,
        beta: f64,
        n: f64,
    },
    ResidualHelmholtzExponential {
        d: i32,
        g: f64,
        l: i32,
        n: f64,
        t: i32,
    },
    ResidualHelmholtzDoubleExponential {
        d: i32,
        gd: f64,
        gt: f64,
        ld: i32,
        lt: i32,
        n: f64,
        t: i32,
    },
    ResidualHelmholtzGaoB {
        b: f64,
        beta: f64,
        d: i32,
        epsilon: f64,
        eta: f64,
        gamma: f64,
        n: f64,
        t: f64,
    },
    ResidualHelmholtzLemmon2005 {
        d: i32,
        l: i32,
        m: f64,
        n: f64,
        t: f64,
    },
}

impl ResidualFunction {
    pub fn evaluate<D: DualNum<f64> + Copy>(&self, delta: D, tau: D) -> D {
        match *self {
            ResidualFunction::ResidualHelmholtzPower { d, l, n, t } => {
                let mut pre = delta.powi(d) * tau.powf(t) * n;
                if l != 0 {
                    pre *= (-delta.powi(l)).exp()
                };
                pre
            }
            ResidualFunction::ResidualHelmholtzGaussian {
                d,
                n,
                t,
                beta,
                epsilon,
                eta,
                gamma,
            } => {
                (delta.powi(d) * tau.powf(t) * n)
                    * (-(delta - epsilon).powi(2) * eta - (tau - gamma).powi(2) * beta).exp()
            }
            ResidualFunction::ResidualHelmholtzNonAnalytic {
                aa,
                bb,
                cc,
                dd,
                a,
                b,
                beta,
                n,
            } => {
                let delta_m1 = (delta - 1.0).powi(2);
                let psi = (-delta_m1 * cc - (tau - 1.0).powi(2) * dd).exp();
                let theta = -tau + 1.0 + delta_m1.powf(0.5 / beta) * aa;
                let ddelta = theta * theta + delta_m1.powf(a) * bb;
                ddelta.powf(b) * delta * psi * n
            }
            ResidualFunction::ResidualHelmholtzExponential { d, g, l, n, t } => {
                delta.powi(d) * tau.powi(t) * n * (-delta.powi(l) * g).exp()
            }
            ResidualFunction::ResidualHelmholtzDoubleExponential {
                d,
                gd,
                gt,
                ld,
                lt,
                n,
                t,
            } => delta.powi(d) * tau.powi(t) * n * (-delta.powi(ld) * gd - tau.powi(lt) * gt).exp(),
            ResidualFunction::ResidualHelmholtzGaoB {
                b,
                beta,
                d,
                epsilon,
                eta,
                gamma,
                n,
                t,
            } => {
                let f_delta = delta.powi(d) * ((delta - epsilon).powi(2) * eta).exp();
                let f_tau = tau.powf(t) * ((tau - gamma).powi(2) * beta + b).recip().exp();
                f_tau * f_delta * n
            }
            ResidualFunction::ResidualHelmholtzLemmon2005 { d, l, m, n, t } => {
                let mut pre = delta.powi(d) * tau.powf(t) * n;
                if l != 0 {
                    pre *= (-delta.powi(l)).exp()
                }
                if m > 0.0 {
                    pre *= (-tau.powf(m)).exp()
                }
                pre
            }
        }
    }
}
