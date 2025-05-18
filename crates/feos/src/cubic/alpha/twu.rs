use super::AlphaFunction;
use crate::cubic::parameters::CubicParameters;
use feos_core::{FeosError, FeosResult};
use itertools::izip;
use ndarray::{Array1, Zip};
use num_dual::DualNum;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Generalized version of the Twu alpha function (1995).
///
/// Different parameters are used for sub- and supercritical conditions.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GeneralizedTwu([[[f64; 3]; 2]; 2]);

impl GeneralizedTwu {
    pub fn redlich_kwong() -> Self {
        GeneralizedTwu([
            [
                [2.496441 * (0.919422 - 1.0), 0.141599, 2.496441 * 0.919422],
                [3.291790 * (0.799457 - 1.0), 0.500315, 3.291790 * 0.799457],
            ],
            [
                [-0.2 * (6.500018 - 1.0), 0.441411, -0.2 * 6.500018],
                [-8.0 * (1.289098 - 1.0), 0.032580, -8.0 * 1.289098],
            ],
        ])
    }

    pub fn peng_robinson() -> Self {
        GeneralizedTwu([
            [
                [1.948150 * (0.911807 - 1.0), 0.125283, 1.948150 * 0.911807],
                [2.812520 * (0.784054 - 1.0), 0.511614, 2.812520 * 0.784054],
            ],
            [
                [-0.2 * (4.963070 - 1.0), 0.401219, -0.2 * 4.963070],
                [-0.8 * (1.248089 - 1.0), 0.024955, -0.8 * 1.248089],
            ],
        ])
    }
}

impl AlphaFunction for GeneralizedTwu {
    #[inline]
    fn alpha<D: DualNum<f64> + Copy>(
        &self,
        acentric_factor: &Array1<f64>,
        reduced_temperature: &Array1<D>,
    ) -> Array1<D> {
        Zip::from(acentric_factor)
            .and(reduced_temperature)
            .map_collect(|&w, &tr| {
                if tr.re() <= 1.0 {
                    let [nm_m1_0, l0, nm0] = self.0[0][0];
                    let [nm_m1_1, l1, nm1] = self.0[0][1];
                    let a0 = tr.powf(nm_m1_0) * ((-tr.powf(nm0) + 1.0) * l0).exp();
                    let a1 = tr.powf(nm_m1_1) * ((-tr.powf(nm1) + 1.0) * l1).exp();
                    a0 + (a1 - a0) * w
                } else {
                    let [nm_m1_0, l0, nm0] = self.0[1][0];
                    let [nm_m1_1, l1, nm1] = self.0[1][1];
                    let a0 = tr.powf(nm_m1_0) * ((-tr.powf(nm0) + 1.0) * l0).exp();
                    let a1 = tr.powf(nm_m1_1) * ((-tr.powf(nm1) + 1.0) * l1).exp();
                    a0 + (a1 - a0) * w
                }
            })
    }

    fn validate(&self, _: &Arc<CubicParameters>) -> FeosResult<()> {
        Ok(())
    }

    fn subset(&self, _: &[usize]) -> Self {
        self.clone()
    }
}

/// Generalized version of the Twu alpha function (1995).
///
/// Different parameters are used for sub- and supercritical conditions.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Twu(Vec<[f64; 3]>);

impl Twu {
    pub fn new(l: Vec<f64>, m: Vec<f64>, n: Option<Vec<f64>>) -> Self {
        let _n = n.unwrap_or(vec![2.0; l.len()]);
        let input = izip!(l, m, _n)
            .map(|(l, m, n)| [n * (m - 1.0), l, n * m])
            .collect();
        Self(input)
    }
}

impl AlphaFunction for Twu {
    #[inline]
    fn alpha<D: DualNum<f64> + Copy>(
        &self,
        _: &Array1<f64>,
        reduced_temperature: &Array1<D>,
    ) -> Array1<D> {
        izip!(reduced_temperature, &self.0)
            .map(|(tr, &[nmm1, l, nm])| tr.powf(nmm1) * ((-tr.powf(nm) + 1.0) * l).exp())
            .collect()
    }

    fn validate(&self, parameters: &Arc<CubicParameters>) -> FeosResult<()> {
        if self.0.len() == parameters.tc.len() {
            Ok(())
        } else {
            Err(FeosError::IncompatibleParameters(
                format!(
                    "Twu alpha function was initialized for {} components, but the equation of state contains {}.",
                    self.0.len(), parameters.tc.len()
                )
            ).into())
        }
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        let c = component_list.iter().map(|&i| self.0[i]).collect();
        Self(c)
    }
}
