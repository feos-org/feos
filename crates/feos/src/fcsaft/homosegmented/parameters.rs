use crate::association::AssociationStrength;
use crate::fcsaft::record::{FcSaftAssociationRecord, FcSaftBinaryRecord};
use crate::hard_sphere::{HardSphereProperties, MonomerShape};
use feos_core::parameter::{CombiningRule, Parameters};
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use num_traits::Zero;
use quantity::{JOULE, KB, KELVIN};
use serde::{Deserialize, Serialize};

/// Pure-component parameters of the homosegmented FC-SAFT equation of state.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct FcSaftHomoRecord {
    /// Segment number
    pub s: usize,
    /// Bond length in units of Angstrom
    pub l: f64,
    /// Segment diameter in units of Angstrom
    pub sigma: f64,
    /// Energetic parameter in units of Kelvin
    pub epsilon_k: f64,
    /// Dipole moment in units of Debye
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub mu: f64,
}

impl FcSaftHomoRecord {
    pub fn new(s: usize, l: f64, sigma: f64, epsilon_k: f64, mu: f64) -> Self {
        Self {
            s,
            l,
            sigma,
            epsilon_k,
            mu,
        }
    }
}

impl CombiningRule<FcSaftHomoRecord> for FcSaftAssociationRecord {
    fn combining_rule(
        _: &FcSaftHomoRecord,
        _: &FcSaftHomoRecord,
        parameters_i: &Self,
        parameters_j: &Self,
    ) -> Self {
        Self::combine(parameters_i, parameters_j)
    }
}

/// Parameter set required for the homosegmented FC-SAFT equation of state.
pub type FcSaftHomoParameters =
    Parameters<FcSaftHomoRecord, FcSaftBinaryRecord, FcSaftAssociationRecord>;

/// The homosegmented FC-SAFT parameters in an easier accessible format.
pub struct FcSaftHomoPars {
    pub component_index: DVector<usize>,
    pub s: DVector<f64>,
    pub l: DVector<f64>,
    pub sigma: DVector<f64>,
    pub epsilon_k: DVector<f64>,
    pub mu2: DVector<f64>,
    pub sigma_ij: DMatrix<f64>,
    pub epsilon_k_ij: DMatrix<f64>,
    pub e_k_ij: DMatrix<f64>,
    pub dipole_comp: Vec<usize>,
}

impl FcSaftHomoPars {
    pub fn new(parameters: &FcSaftHomoParameters) -> Self {
        let n = parameters.pure.len();

        let [s, l, sigma, epsilon_k, mu] =
            parameters.collate(|pr| [pr.s as f64, pr.l, pr.sigma, pr.epsilon_k, pr.mu]);
        let [k_ij] = parameters.collate_binary(|br| [br.k_ij]);

        let ld = l.component_div(&sigma);
        let m_star = DVector::from_fn(n, |i, _| {
            1.0 + 0.5 * ld[i] * (3.0 - ld[i] * ld[i]) * (s[i] - 1.0)
        });
        // Note: the reduced dipole moment is normalized with the reduced volume m*.
        // Using the reduced surface m (which replaces the segment number of PC-SAFT
        // everywhere else) would be an equally valid generalization to fused chains.
        let mu2 = DVector::from_fn(n, |i, _| {
            mu[i] * mu[i] / (m_star[i] * sigma[i].powi(3) * epsilon_k[i])
                * 1e-19
                * (JOULE / KELVIN / KB).into_value()
        });
        let dipole_comp = mu2
            .iter()
            .enumerate()
            .filter_map(|(i, &mu2)| (mu2.abs() > 0.0).then_some(i))
            .collect();

        let sigma_ij = DMatrix::from_fn(n, n, |i, j| 0.5 * (sigma[i] + sigma[j]));
        let e_k_ij = DMatrix::from_fn(n, n, |i, j| (epsilon_k[i] * epsilon_k[j]).sqrt());
        let epsilon_k_ij = (-k_ij).add_scalar(1.0).component_mul(&e_k_ij);

        Self {
            component_index: DVector::from_fn(n, |i, _| i),
            s,
            l,
            sigma,
            epsilon_k,
            mu2,
            sigma_ij,
            epsilon_k_ij,
            e_k_ij,
            dipole_comp,
        }
    }

    /// The mean segment number $m$ and the reduced volume $m^*$ of every component.
    pub fn m_values<D: DualNum<Primitive = f64> + Copy>(
        &self,
        diameter: &DVector<D>,
    ) -> [DVector<D>; 2] {
        let ld = DVector::from_fn(diameter.len(), |i, _| diameter[i].recip() * self.l[i]);
        let m = DVector::from_fn(ld.len(), |i, _| ld[i] * (self.s[i] - 1.0) + 1.0);
        let m_star = DVector::from_fn(ld.len(), |i, _| {
            (-(ld[i] * ld[i]) + 3.0) * ld[i] * 0.5 * (self.s[i] - 1.0) + 1.0
        });
        [m, m_star]
    }
}

impl HardSphereProperties for FcSaftHomoPars {
    fn monomer_shape<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
    ) -> MonomerShape<'_, N> {
        let [m, m_star] = self.m_values(&self.hs_diameter(temperature));
        MonomerShape::Heterosegmented(
            [self.s.map(N::from), m.clone(), m, m_star],
            &self.component_index,
        )
    }

    fn hs_diameter<D: DualNum<Primitive = f64> + Copy>(&self, temperature: D) -> DVector<D> {
        let ti = temperature.recip() * -3.0;
        DVector::from_fn(self.sigma.len(), |i, _| {
            -((ti * self.epsilon_k[i]).exp() * 0.12 - 1.0) * self.sigma[i]
        })
    }
}

impl AssociationStrength for FcSaftHomoPars {
    type Record = FcSaftAssociationRecord;

    fn association_strength_ij<D: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: D,
        comp_i: usize,
        comp_j: usize,
        assoc_ij: &Self::Record,
    ) -> D {
        let si = self.sigma[comp_i];
        let sj = self.sigma[comp_j];
        (temperature.recip() * assoc_ij.epsilon_k_ab).exp_m1()
            * assoc_ij.kappa_ab
            * (si * sj).powf(1.5)
    }
}
