use super::{BarkerHenderson, Perturbation, WeeksChandlerAndersen};
use crate::hard_sphere::{HardSphereProperties, MonomerShape};
use feos_core::parameter::Parameters;
use ndarray::Array2;
use ndarray::concatenate;
use ndarray::prelude::*;
use num_dual::DualNum;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// uv-theory parameters for a pure substance
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UVTheoryRecord {
    rep: f64,
    att: f64,
    sigma: f64,
    epsilon_k: f64,
}

impl UVTheoryRecord {
    /// Single substance record for uv-theory
    pub fn new(rep: f64, att: f64, sigma: f64, epsilon_k: f64) -> Self {
        Self {
            rep,
            att,
            sigma,
            epsilon_k,
        }
    }
}

/// Constants for BH temperature dependent HS diameter.
static CD_BH: LazyLock<Array2<f64>> = LazyLock::new(|| {
    arr2(&[
        [0.0, 1.09360455168912E-02, 0.0],
        [-2.00897880971934E-01, -1.27074910870683E-02, 0.0],
        [
            1.40422470174053E-02,
            7.35946850956932E-02,
            1.28463973950737E-02,
        ],
        [
            3.71527116894441E-03,
            5.05384813757953E-03,
            4.91003312452622E-02,
        ],
    ])
});

#[inline]
pub fn mie_prefactor<D: DualNum<f64> + Copy>(rep: D, att: D) -> D {
    rep / (rep - att) * (rep / att).powd(att / (rep - att))
}

#[inline]
pub fn mean_field_constant<D: DualNum<f64> + Copy>(rep: D, att: D, x: D) -> D {
    mie_prefactor(rep, att) * (x.powd(-att + 3.0) / (att - 3.0) - x.powd(-rep + 3.0) / (rep - 3.0))
}

/// Parameters for all substances for uv-theory equation of state and Helmholtz energy functional
pub type UVTheoryParameters = Parameters<UVTheoryRecord, f64, ()>;

/// Parameters for all substances for uv-theory equation of state and Helmholtz energy functional
#[derive(Debug, Clone)]
pub struct UVTheoryPars {
    pub perturbation: Perturbation,
    pub rep: Array1<f64>,
    pub att: Array1<f64>,
    pub sigma: Array1<f64>,
    pub epsilon_k: Array1<f64>,
    pub rep_ij: Array2<f64>,
    pub att_ij: Array2<f64>,
    pub sigma_ij: Array2<f64>,
    pub eps_k_ij: Array2<f64>,
    pub cd_bh_pure: Vec<Array1<f64>>,
    pub cd_bh_binary: Array2<Array1<f64>>,
}

impl UVTheoryPars {
    pub fn new(parameters: &UVTheoryParameters, perturbation: Perturbation) -> Self {
        let n = parameters.pure.len();

        let [rep, att, sigma, epsilon_k] =
            parameters.collate(|pr| [pr.rep, pr.att, pr.sigma, pr.epsilon_k]);

        let mut rep_ij = Array2::zeros((n, n));
        let mut att_ij = Array2::zeros((n, n));
        let mut sigma_ij = Array2::zeros((n, n));
        let mut eps_k_ij = Array2::zeros((n, n));
        let [k_ij] = parameters.collate_binary(|&br| [br]);

        for i in 0..n {
            rep_ij[[i, i]] = rep[i];
            att_ij[[i, i]] = att[i];
            sigma_ij[[i, i]] = sigma[i];
            eps_k_ij[[i, i]] = epsilon_k[i];
            for j in i + 1..n {
                rep_ij[[i, j]] = (rep[i] * rep[j]).sqrt();
                rep_ij[[j, i]] = rep_ij[[i, j]];
                att_ij[[i, j]] = (att[i] * att[j]).sqrt();
                att_ij[[j, i]] = att_ij[[i, j]];
                sigma_ij[[i, j]] = 0.5 * (sigma[i] + sigma[j]);
                sigma_ij[[j, i]] = sigma_ij[[i, j]];
                eps_k_ij[[i, j]] = (1.0 - k_ij[[i, j]]) * (epsilon_k[i] * epsilon_k[j]).sqrt();
                eps_k_ij[[j, i]] = eps_k_ij[[i, j]];
            }
        }

        // BH temperature dependent HS diameter, eq. 21
        let cd_bh_pure: Vec<Array1<f64>> = rep.iter().map(|&mi| bh_coefficients(mi, 6.0)).collect();
        let cd_bh_binary =
            Array2::from_shape_fn((n, n), |(i, j)| bh_coefficients(rep_ij[[i, j]], 6.0));

        Self {
            perturbation,
            rep,
            att,
            sigma,
            epsilon_k,
            rep_ij,
            att_ij,
            sigma_ij,
            eps_k_ij,
            cd_bh_pure,
            cd_bh_binary,
        }
    }
}

fn bh_coefficients(rep: f64, att: f64) -> Array1<f64> {
    let inv_a76 = 1.0 / mean_field_constant(7.0, att, 1.0);
    let am6 = mean_field_constant(rep, att, 1.0);
    let alpha = 1.0 / am6 - inv_a76;
    let c0 = arr1(&[-2.0 * rep / ((att - rep) * mie_prefactor(rep, att))]);
    concatenate![Axis(0), c0, CD_BH.dot(&arr1(&[1.0, alpha, alpha * alpha]))]
}

impl HardSphereProperties for UVTheoryPars {
    fn monomer_shape<D: DualNum<f64> + Copy>(&self, _: D) -> MonomerShape<D> {
        MonomerShape::Spherical(self.sigma.len())
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D> {
        match self.perturbation {
            Perturbation::BarkerHenderson => BarkerHenderson::diameter_bh(self, temperature),
            Perturbation::WeeksChandlerAndersen => {
                WeeksChandlerAndersen::diameter_wca(self, temperature)
            }
            Perturbation::WeeksChandlerAndersenB3 => {
                WeeksChandlerAndersen::diameter_wca(self, temperature)
            }
        }
    }
}

#[cfg(test)]
pub mod utils {
    use super::*;
    use feos_core::parameter::{Identifier, PureRecord};
    use std::f64;

    pub fn new_simple(rep: f64, att: f64, sigma: f64, epsilon_k: f64) -> UVTheoryParameters {
        UVTheoryParameters::new_pure(PureRecord::new(
            Default::default(),
            0.0,
            UVTheoryRecord::new(rep, att, sigma, epsilon_k),
        ))
        .unwrap()
    }

    pub fn test_parameters(
        rep: f64,
        att: f64,
        sigma: f64,
        epsilon: f64,
        p: Perturbation,
    ) -> UVTheoryPars {
        let identifier = Identifier::new(Some("1"), None, None, None, None, None);
        let model_record = UVTheoryRecord::new(rep, att, sigma, epsilon);
        let pr = PureRecord::new(identifier, 1.0, model_record);
        UVTheoryPars::new(&UVTheoryParameters::new_pure(pr).unwrap(), p)
    }

    pub fn test_parameters_mixture(
        rep: Array1<f64>,
        att: Array1<f64>,
        sigma: Array1<f64>,
        epsilon: Array1<f64>,
    ) -> UVTheoryParameters {
        let identifier = Identifier::new(Some("1"), None, None, None, None, None);
        let model_record = UVTheoryRecord::new(rep[0], att[0], sigma[0], epsilon[0]);
        let pr1 = PureRecord::new(identifier, 1.0, model_record);
        //
        let identifier2 = Identifier::new(Some("1"), None, None, None, None, None);
        let model_record2 = UVTheoryRecord::new(rep[1], att[1], sigma[1], epsilon[1]);
        let pr2 = PureRecord::new(identifier2, 1.0, model_record2);
        UVTheoryParameters::new_binary([pr1, pr2], None, vec![]).unwrap()
    }

    pub fn methane_parameters(rep: f64, att: f64) -> UVTheoryParameters {
        let identifier = Identifier::new(Some("1"), None, None, None, None, None);
        let model_record = UVTheoryRecord::new(rep, att, 3.7039, 150.03);
        let pr = PureRecord::new(identifier, 1.0, model_record);
        UVTheoryParameters::new_pure(pr).unwrap()
    }
}
