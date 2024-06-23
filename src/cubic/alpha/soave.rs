use std::sync::Arc;

use feos_core::EosResult;
use ndarray::{Array1, Zip};
use num_dual::DualNum;
use serde::{Deserialize, Serialize};

use crate::cubic::parameters::CubicParameters;

use super::AlphaFunction;

#[derive(Serialize, Deserialize, Clone, Debug)]
/// Generic version of Soave's function using 3rd order polynomial.
pub struct Soave {
    /// coefficients for m-polynomial
    mi: Array1<f64>,
}

impl Soave {
    pub fn new(mi: Array1<f64>) -> Self {
        Soave { mi }
    }
}

impl AlphaFunction for Soave {
    #[inline]
    fn alpha<D: DualNum<f64>>(
        &self,
        acentric_factor: &Array1<f64>,
        reduced_temperature: &Array1<D>,
    ) -> Array1<D> {
        let m = self
            .mi
            .iter()
            .enumerate()
            .fold(Array1::zeros(acentric_factor.len()), |m, (i, &mi)| {
                &m + acentric_factor.mapv(|w| w.powi(i as i32)) * mi
            });
        ((-reduced_temperature.mapv(|t| t.sqrt()) + 1.0) * m + 1.0).mapv(|a| a.powi(2))
    }

    fn validate(&self, _: &Arc<CubicParameters>) -> EosResult<()> {
        Ok(())
    }

    fn subset(&self, _: &[usize]) -> Self {
        self.clone()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RedlichKwong1972;

impl AlphaFunction for RedlichKwong1972 {
    #[inline]
    fn alpha<D: DualNum<f64>>(
        &self,
        acentric_factor: &Array1<f64>,
        reduced_temperature: &Array1<D>,
    ) -> Array1<D> {
        let m = acentric_factor.mapv(|w| 0.48 + w * (1.574 - w * 0.176));
        ((-reduced_temperature.mapv(|t| t.sqrt()) + 1.0) * m + 1.0).mapv(|a| a.powi(2))
    }

    fn validate(&self, _: &Arc<CubicParameters>) -> EosResult<()> {
        Ok(())
    }

    fn subset(&self, _: &[usize]) -> Self {
        Self
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PengRobinson1976;

impl AlphaFunction for PengRobinson1976 {
    #[inline]
    fn alpha<D: DualNum<f64>>(
        &self,
        acentric_factor: &Array1<f64>,
        reduced_temperature: &Array1<D>,
    ) -> Array1<D> {
        let m = acentric_factor.mapv(|w| 0.37464 + w * (1.54226 - w * 0.26992));
        ((-reduced_temperature.mapv(|t| t.sqrt()) + 1.0) * m + 1.0).mapv(|a| a.powi(2))
    }

    fn validate(&self, _: &Arc<CubicParameters>) -> EosResult<()> {
        Ok(())
    }

    fn subset(&self, _: &[usize]) -> Self {
        Self
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PengRobinson1978;

impl AlphaFunction for PengRobinson1978 {
    #[inline]
    fn alpha<D: DualNum<f64> + Copy>(
        &self,
        acentric_factor: &Array1<f64>,
        reduced_temperature: &Array1<D>,
    ) -> Array1<D> {
        Zip::from(acentric_factor)
            .and(reduced_temperature)
            .map_collect(|&w, &tr| {
                let m = if w <= 0.491 {
                    0.37464 + w * (1.54226 - w * 0.26992)
                } else {
                    // use higher-order polynomial if w > w(n-decane)
                    0.379642 + w * (1.48503 + w * (-0.164423 + w * 0.016666))
                };
                ((-tr.sqrt() + 1.0) * m + 1.0).powi(2)
            })
    }

    fn validate(&self, _: &Arc<CubicParameters>) -> EosResult<()> {
        Ok(())
    }

    fn subset(&self, _: &[usize]) -> Self {
        Self
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RedlichKwong2019;

impl AlphaFunction for RedlichKwong2019 {
    #[inline]
    fn alpha<D: DualNum<f64>>(
        &self,
        acentric_factor: &Array1<f64>,
        reduced_temperature: &Array1<D>,
    ) -> Array1<D> {
        let m = acentric_factor.mapv(|w| 0.481 + w * (1.5963 + w * (-0.2963 + w * 0.1223)));
        ((-reduced_temperature.mapv(|t| t.sqrt()) + 1.0) * m + 1.0).mapv(|a| a.powi(2))
    }

    fn validate(&self, _: &Arc<CubicParameters>) -> EosResult<()> {
        Ok(())
    }

    fn subset(&self, _: &[usize]) -> Self {
        Self
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PengRobinson2019;

impl AlphaFunction for PengRobinson2019 {
    #[inline]
    fn alpha<D: DualNum<f64>>(
        &self,
        acentric_factor: &Array1<f64>,
        reduced_temperature: &Array1<D>,
    ) -> Array1<D> {
        let m = acentric_factor.mapv(|w| 0.3919 + w * (1.4996 + w * (-0.2721 + w * 0.1063)));
        ((-reduced_temperature.mapv(|t| t.sqrt()) + 1.0) * m + 1.0).mapv(|a| a.powi(2))
    }

    fn validate(&self, _: &Arc<CubicParameters>) -> EosResult<()> {
        Ok(())
    }

    fn subset(&self, _: &[usize]) -> Self {
        Self
    }
}
