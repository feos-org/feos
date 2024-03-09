use std::collections::HashMap;

use super::eos::association::{AssociationParameters, AssociationRecord, BinaryAssociationRecord};
use crate::hard_sphere::{HardSphereProperties, MonomerShape};
use feos_core::{
    parameter::{Parameter, ParameterError, PureRecord},
    EosResult,
};
use ndarray::{Array, Array1, Array2};
use num_dual::DualNum;
use num_traits::Zero;
use serde::{Deserialize, Serialize};

/// 10-point Gauss-Legendre quadrature [position, weight]
const GLQ10: [[f64; 2]; 10] = [
    [-0.1488743389816312, 0.2955242247147529],
    [0.1488743389816312, 0.2955242247147529],
    [-0.4333953941292472, 0.2692667193099963],
    [0.4333953941292472, 0.2692667193099963],
    [-0.6794095682990244, 0.219086362515982],
    [0.6794095682990244, 0.219086362515982],
    [-0.8650633666889845, 0.1494513491505806],
    [0.8650633666889845, 0.1494513491505806],
    [-0.9739065285171717, 0.0666713443086881],
    [0.9739065285171717, 0.0666713443086881],
];

/// SAFT-VR Mie pure-component parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct SaftVRMieRecord {
    /// Segment number
    pub m: f64,
    /// Segment diameter in units of Angstrom
    pub sigma: f64,
    /// Energetic parameter in units of Kelvin
    pub epsilon_k: f64,
    /// Repulsive Mie exponent
    pub lr: f64,
    /// Attractive Mie exponent
    pub la: f64,
    /// Association
    #[serde(flatten)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub association_record: Option<AssociationRecord>,
    /// Entropy scaling coefficients for the viscosity
    #[serde(skip_serializing_if = "Option::is_none")]
    pub viscosity: Option<[f64; 4]>,
    /// Entropy scaling coefficients for the diffusion coefficient
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diffusion: Option<[f64; 5]>,
    /// Entropy scaling coefficients for the thermal conductivity
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thermal_conductivity: Option<[f64; 4]>,
}

impl SaftVRMieRecord {
    pub fn new(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        lr: f64,
        la: f64,
        kappa_ab: Option<f64>,
        epsilon_k_ab: Option<f64>,
        na: Option<f64>,
        nb: Option<f64>,
        nc: Option<f64>,
        viscosity: Option<[f64; 4]>,
        diffusion: Option<[f64; 5]>,
        thermal_conductivity: Option<[f64; 4]>,
    ) -> Self {
        let association_record = if kappa_ab.is_none()
            && epsilon_k_ab.is_none()
            && na.is_none()
            && nb.is_none()
            && nc.is_none()
        {
            None
        } else {
            Some(AssociationRecord::new(
                kappa_ab.unwrap_or_default(),
                epsilon_k_ab.unwrap_or_default(),
                na.unwrap_or_default(),
                nb.unwrap_or_default(),
                nc.unwrap_or_default(),
            ))
        };
        Self {
            m,
            sigma,
            epsilon_k,
            lr,
            la,
            association_record,
            viscosity,
            diffusion,
            thermal_conductivity,
        }
    }
}

impl std::fmt::Display for SaftVRMieRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SaftVRMieRecord(m={}", self.m)?;
        write!(f, ", sigma={}", self.sigma)?;
        write!(f, ", epsilon_k={}", self.epsilon_k)?;
        write!(f, ", lr={}", self.lr)?;
        write!(f, ", la={}", self.la)?;
        if let Some(n) = &self.association_record {
            write!(f, ", association_record={}", n)?;
        }
        if let Some(n) = &self.viscosity {
            write!(f, ", viscosity={:?}", n)?;
        }
        if let Some(n) = &self.diffusion {
            write!(f, ", diffusion={:?}", n)?;
        }
        if let Some(n) = &self.thermal_conductivity {
            write!(f, ", thermal_conductivity={:?}", n)?;
        }
        write!(f, ")")
    }
}

/// PC-SAFT binary interaction parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct SaftVRMieBinaryRecord {
    /// Binary dispersion energy interaction parameter
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub k_ij: f64,
    /// Binary interaction parameter for repulsive exponent
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub gamma_ij: f64,
    /// Binary association parameters
    #[serde(flatten)]
    association: Option<BinaryAssociationRecord>,
}

impl SaftVRMieBinaryRecord {
    pub fn new(
        k_ij: Option<f64>,
        gamma_ij: Option<f64>,
        kappa_ab: Option<f64>,
        epsilon_k_ab: Option<f64>,
    ) -> Self {
        let k_ij = k_ij.unwrap_or_default();
        let gamma_ij = gamma_ij.unwrap_or_default();
        let association = if kappa_ab.is_none() && epsilon_k_ab.is_none() {
            None
        } else {
            Some(BinaryAssociationRecord::new(kappa_ab, epsilon_k_ab, None))
        };
        Self {
            k_ij,
            gamma_ij,
            association,
        }
    }
}

impl From<f64> for SaftVRMieBinaryRecord {
    fn from(k_ij: f64) -> Self {
        Self {
            k_ij,
            gamma_ij: f64::default(),
            association: None,
        }
    }
}

impl From<SaftVRMieBinaryRecord> for f64 {
    fn from(binary_record: SaftVRMieBinaryRecord) -> Self {
        binary_record.k_ij
    }
}

impl std::fmt::Display for SaftVRMieBinaryRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut tokens = vec![];
        if !self.k_ij.is_zero() {
            tokens.push(format!("k_ij={}", self.k_ij));
        }
        if !self.gamma_ij.is_zero() {
            tokens.push(format!("gamma_ij={}", self.gamma_ij));
        }
        if let Some(association) = self.association {
            if let Some(kappa_ab) = association.kappa_ab {
                tokens.push(format!("kappa_ab={}", kappa_ab));
            }
            if let Some(epsilon_k_ab) = association.epsilon_k_ab {
                tokens.push(format!("epsilon_k_ab={}", epsilon_k_ab));
            }
        }
        write!(f, "SaftVRMieBinaryRecord({})", tokens.join(", "))
    }
}

pub struct SaftVRMieParameters {
    pub molarweight: Array1<f64>,
    pub m: Array1<f64>,
    pub sigma: Array1<f64>,
    pub epsilon_k: Array1<f64>,
    pub lr: Array1<f64>,
    pub la: Array1<f64>,
    pub association: AssociationParameters,
    pub sigma_ij: Array2<f64>,
    pub epsilon_k_ij: Array2<f64>,
    pub e_k_ij: Array2<f64>,
    pub lr_ij: Array2<f64>,
    pub la_ij: Array2<f64>,
    pub c_ij: Array2<f64>,
    pub alpha_ij: Array2<f64>,
    pub viscosity: Option<Array2<f64>>,
    pub diffusion: Option<Array2<f64>>,
    pub thermal_conductivity: Option<Array2<f64>>,
    pub pure_records: Vec<PureRecord<SaftVRMieRecord>>,
    pub binary_records: Option<Array2<SaftVRMieBinaryRecord>>,
}

impl Parameter for SaftVRMieParameters {
    type Pure = SaftVRMieRecord;
    type Binary = SaftVRMieBinaryRecord;

    fn from_records(
        pure_records: Vec<PureRecord<Self::Pure>>,
        binary_records: Option<Array2<SaftVRMieBinaryRecord>>,
    ) -> Result<Self, ParameterError> {
        let n = pure_records.len();

        let mut molarweight = Array::zeros(n);
        let mut m = Array::zeros(n);
        let mut sigma = Array::zeros(n);
        let mut epsilon_k = Array::zeros(n);
        let mut lr = Array::zeros(n);
        let mut la = Array::zeros(n);
        let mut association_records = Vec::with_capacity(n);
        let mut viscosity = Vec::with_capacity(n);
        let mut diffusion = Vec::with_capacity(n);
        let mut thermal_conductivity = Vec::with_capacity(n);

        let mut component_index = HashMap::with_capacity(n);

        for (i, record) in pure_records.iter().enumerate() {
            component_index.insert(record.identifier.clone(), i);
            let r = &record.model_record;
            m[i] = r.m;
            sigma[i] = r.sigma;
            epsilon_k[i] = r.epsilon_k;
            lr[i] = r.lr;
            la[i] = r.la;
            association_records.push(r.association_record.into_iter().collect());
            viscosity.push(r.viscosity);
            diffusion.push(r.diffusion);
            thermal_conductivity.push(r.thermal_conductivity);
            molarweight[i] = record.molarweight;
        }

        let binary_association: Vec<_> = binary_records
            .iter()
            .flat_map(|r| {
                r.indexed_iter()
                    .filter_map(|(i, record)| record.association.map(|r| (i, r)))
            })
            .collect();
        let association =
            AssociationParameters::new(&association_records, &binary_association, None);

        let br = binary_records.as_ref();
        let k_ij = br.map_or_else(|| Array2::zeros([n; 2]), |br| br.mapv(|br| br.k_ij));
        let gamma_ij = br.map_or_else(|| Array2::zeros([n; 2]), |br| br.mapv(|br| br.gamma_ij));
        let mut sigma_ij = Array::zeros((n, n));
        let mut e_k_ij = Array::zeros((n, n));
        let mut epsilon_k_ij = Array::zeros((n, n));
        let mut lr_ij = Array::zeros((n, n));
        let mut la_ij = Array::zeros((n, n));
        let mut c_ij = Array::zeros((n, n));
        let mut alpha_ij = Array::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                sigma_ij[[i, j]] = 0.5 * (sigma[i] + sigma[j]);
                e_k_ij[[i, j]] = (sigma[i].powi(3) * sigma[j].powi(3)).sqrt()
                    / sigma_ij[[i, j]].powi(3)
                    * (epsilon_k[i] * epsilon_k[j]).sqrt();
                epsilon_k_ij[[i, j]] = (1.0 - k_ij[[i, j]]) * e_k_ij[[i, j]];
                lr_ij[[i, j]] =
                    (1.0 - gamma_ij[[i, j]]) * ((lr[i] - 3.0) * (lr[j] - 3.0)).sqrt() + 3.0;
                la_ij[[i, j]] = ((la[i] - 3.0) * (la[j] - 3.0)).sqrt() + 3.0;
                c_ij[[i, j]] = lr_ij[[i, j]] / (lr_ij[[i, j]] - la_ij[[i, j]])
                    * (lr_ij[[i, j]] / la_ij[[i, j]])
                        .powf(la_ij[[i, j]] / (lr_ij[[i, j]] - la_ij[[i, j]]));
                alpha_ij[[i, j]] =
                    c_ij[[i, j]] * ((la_ij[[i, j]] - 3.0).recip() - (lr_ij[[i, j]] - 3.0).recip())
            }
        }

        let viscosity_coefficients = if viscosity.iter().any(|v| v.is_none()) {
            None
        } else {
            let mut v = Array2::zeros((4, viscosity.len()));
            for (i, vi) in viscosity.iter().enumerate() {
                v.column_mut(i).assign(&Array1::from(vi.unwrap().to_vec()));
            }
            Some(v)
        };

        let diffusion_coefficients = if diffusion.iter().any(|v| v.is_none()) {
            None
        } else {
            let mut v = Array2::zeros((5, diffusion.len()));
            for (i, vi) in diffusion.iter().enumerate() {
                v.column_mut(i).assign(&Array1::from(vi.unwrap().to_vec()));
            }
            Some(v)
        };

        let thermal_conductivity_coefficients = if thermal_conductivity.iter().any(|v| v.is_none())
        {
            None
        } else {
            let mut v = Array2::zeros((4, thermal_conductivity.len()));
            for (i, vi) in thermal_conductivity.iter().enumerate() {
                v.column_mut(i).assign(&Array1::from(vi.unwrap().to_vec()));
            }
            Some(v)
        };

        Ok(Self {
            molarweight,
            m,
            sigma,
            epsilon_k,
            lr,
            la,
            association,
            sigma_ij,
            epsilon_k_ij,
            e_k_ij,
            lr_ij,
            la_ij,
            c_ij,
            alpha_ij,
            viscosity: viscosity_coefficients,
            diffusion: diffusion_coefficients,
            thermal_conductivity: thermal_conductivity_coefficients,
            pure_records,
            binary_records,
        })
    }

    fn records(
        &self,
    ) -> (
        &[PureRecord<SaftVRMieRecord>],
        Option<&Array2<SaftVRMieBinaryRecord>>,
    ) {
        (&self.pure_records, self.binary_records.as_ref())
    }
}

impl SaftVRMieParameters {
    #[inline]
    pub fn hs_diameter_ij<D: DualNum<f64> + Copy>(
        &self,
        i: usize,
        j: usize,
        inverse_temperature: D,
    ) -> D {
        let lr = self.lr_ij[[i, j]];
        let la = self.la_ij[[i, j]];
        let c_eps_t = inverse_temperature * self.c_ij[[i, j]] * self.epsilon_k_ij[[i, j]];

        // perform integration in reduced distances, then multiply sigma
        // r0 is dimensionless
        let r0 = lower_integratal_limit(la, lr, c_eps_t);
        let width = (-r0 + 1.0) * 0.5;
        GLQ10.iter().fold(r0, |d, &[x, w]| {
            let r = width * x + width + r0;
            let u = beta_u_mie(r, la, lr, 1.0, c_eps_t);
            let f_u = -(-u).exp_m1();
            d + width * f_u * w
        }) * self.sigma_ij[[i, j]]
    }
}

// Find lower limit for integration of diameter
// Method of Aasen et al.
// Using starting value proposed in Clapeyron.jl (Andrés Riedemann)
fn lower_integratal_limit<D: DualNum<f64> + Copy>(la: f64, lr: f64, c_eps_t: D) -> D {
    // initial value from repulsive contribution
    let k = (-c_eps_t.recip() * f64::EPSILON.ln()).ln();
    let mut r = (-k / lr).exp();
    // Halley's method
    for _ in 1..5 {
        let [u, u_du, du_d2u] = mie_potential_halley(r, la, lr, c_eps_t);
        let dr = u_du / (-u_du / du_d2u * 0.5 + 1.0);
        // if dr.re() < f64::EPSILON {
        if u.re() < 0.0 {
            return r;
        }
        r -= dr;
    }
    r // error instead?
}

// Calculate the fractions f / df and df / d2f used for Halley's method,
// where f is the function to find the root of.
// Here, f = -beta u_mie(r) - ln(EPS)
#[inline]
fn mie_potential_halley<D: DualNum<f64> + Copy>(r: D, la: f64, lr: f64, c_eps_t: D) -> [D; 3] {
    let ri = r.recip();
    let plr = ri.powf(lr);
    let pla = ri.powf(la);
    let u = plr - pla;
    let dplr = plr * (-lr) * ri;
    let dpla = pla * (-la) * ri;
    let du_dr = dplr - dpla;
    let d2u_dr2 = (dplr * (-lr - 1.0) - dpla * (-la - 1.0)) * ri;

    // exp(-beta u) < EPS
    // -beta u < ln(EPS)
    let f = -c_eps_t * u - f64::EPSILON.ln();
    let df = -c_eps_t * du_dr;
    let d2f = -c_eps_t * d2u_dr2;
    [f, f / df, df / d2f]
}

/// Dimensionless Mie potential (divided by kT)
#[inline]
fn beta_u_mie<D: DualNum<f64> + Copy>(r: D, la: f64, lr: f64, sigma: f64, c_eps_t: D) -> D {
    let ri = r.recip() * sigma;
    (ri.powf(lr) - ri.powf(la)) * c_eps_t
}

impl HardSphereProperties for SaftVRMieParameters {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<N> {
        MonomerShape::NonSpherical(self.m.mapv(N::from))
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D> {
        let t_inv = temperature.recip();
        Array1::from_shape_fn(self.m.len(), |i| -> D { self.hs_diameter_ij(i, i, t_inv) })
    }
}

pub(super) mod utils {
    use super::*;
    use feos_core::{
        parameter::Identifier,
        si::{ANGSTROM, METER},
    };
    use typenum::P3;

    /// Methane TABLE III or Lafitte et al. (2013)
    pub fn methane() -> SaftVRMieParameters {
        let m = 1.0;
        let epsilon_k = 152.58;
        let sigma = 3.737;
        let lr = 12.504;
        let la = 6.0;
        let molarweight = 16.04;

        let model_record = SaftVRMieRecord::new(
            m, sigma, epsilon_k, lr, la, None, None, None, None, None, None, None, None,
        );
        SaftVRMieParameters::new_pure(PureRecord {
            identifier: Identifier::default(),
            molarweight,
            model_record,
        })
        .unwrap()
    }

    pub fn methane2() -> SaftVRMieParameters {
        let m = 1.0;
        let epsilon_k = 152.58;
        let sigma = 3.737;
        let lr = 12.0;
        let la = 6.0;
        let molarweight = 16.04;

        let model_record = SaftVRMieRecord::new(
            m, sigma, epsilon_k, lr, la, None, None, None, None, None, None, None, None,
        );
        SaftVRMieParameters::new_pure(PureRecord {
            identifier: Identifier::default(),
            molarweight,
            model_record,
        })
        .unwrap()
    }

    /// Ethane TABLE III or Lafitte et al. (2013)
    pub fn ethane() -> SaftVRMieParameters {
        let m = 1.4373;
        let epsilon_k = 206.12;
        let sigma = 3.7257;
        let lr = 12.4;
        let la = 6.0;
        let molarweight = 30.07;

        let model_record = SaftVRMieRecord::new(
            m, sigma, epsilon_k, lr, la, None, None, None, None, None, None, None, None,
        );
        SaftVRMieParameters::new_pure(PureRecord {
            identifier: Identifier::default(),
            molarweight,
            model_record,
        })
        .unwrap()
    }

    /// Ethane TABLE III or Lafitte et al. (2013)
    pub fn methane_ethane() -> SaftVRMieParameters {
        let m = 1.0;
        let epsilon_k = 152.58;
        let sigma = 3.737;
        let lr = 12.504;
        let la = 6.0;
        let molarweight = 16.04;

        let model_record = SaftVRMieRecord::new(
            m, sigma, epsilon_k, lr, la, None, None, None, None, None, None, None, None,
        );
        let methane = PureRecord {
            identifier: Identifier {
                name: Some("methane".to_string()),
                ..Default::default()
            },
            molarweight,
            model_record,
        };

        let m = 1.4373;
        let epsilon_k = 206.12;
        let sigma = 3.7257;
        let lr = 12.4;
        let la = 6.0;
        let molarweight = 30.07;

        let model_record = SaftVRMieRecord::new(
            m, sigma, epsilon_k, lr, la, None, None, None, None, None, None, None, None,
        );
        let ethane = PureRecord {
            identifier: Identifier {
                name: Some("ethane".to_string()),
                ..Default::default()
            },
            molarweight,
            model_record,
        };
        SaftVRMieParameters::new_binary(vec![methane, ethane], None).unwrap()
    }

    pub fn mixture_thijs() -> SaftVRMieParameters {
        let m = 1.0;
        let epsilon_k = 1.0;
        let sigma = 1.0;
        let lr = 12.0;
        let la = 6.0;
        let molarweight = 1.0;

        let model_record = SaftVRMieRecord::new(
            m, sigma, epsilon_k, lr, la, None, None, None, None, None, None, None, None,
        );
        let methane = PureRecord {
            identifier: Identifier {
                name: Some("methane".to_string()),
                ..Default::default()
            },
            molarweight,
            model_record,
        };

        let m = 1.0;
        let epsilon_k = 1.0;
        let sigma = 1.5;
        let lr = 12.0;
        let la = 6.0;
        let molarweight = 1.0;

        let model_record = SaftVRMieRecord::new(
            m, sigma, epsilon_k, lr, la, None, None, None, None, None, None, None, None,
        );
        let ethane = PureRecord {
            identifier: Identifier {
                name: Some("ethane".to_string()),
                ..Default::default()
            },
            molarweight,
            model_record,
        };
        SaftVRMieParameters::new_binary(vec![methane, ethane], None).unwrap()
    }

    /// Ethane TABLE III or Lafitte et al. (2013)
    pub fn methanol() -> SaftVRMieParameters {
        let m = 1.67034;
        let epsilon_k = 307.69;
        let sigma = 3.2462;
        let lr = 19.235;
        let la = 6.0;
        let molarweight = 32.042;
        let kappa_ab = 1.0657e-28 * METER.powi::<P3>().convert_into(ANGSTROM.powi::<P3>());
        let epsilon_k_ab = 2062.1;
        let na = 2.0;
        let nb = 1.0;
        let nc = 0.0;

        let model_record = SaftVRMieRecord::new(
            m,
            sigma,
            epsilon_k,
            lr,
            la,
            Some(kappa_ab),
            Some(epsilon_k_ab),
            // None, None,
            Some(na),
            Some(nb),
            Some(nc),
            None,
            None,
            None,
        );
        SaftVRMieParameters::new_pure(PureRecord {
            identifier: Identifier::default(),
            molarweight,
            model_record,
        })
        .unwrap()
    }

    /// Ethane TABLE III or Lafitte et al. (2013)
    pub fn methanol_propanol() -> SaftVRMieParameters {
        let m = 1.67034;
        let epsilon_k = 307.69;
        let sigma = 3.2462;
        let lr = 19.235;
        let la = 6.0;
        let molarweight = 32.042;
        let kappa_ab = 1.0657e-28 * METER.powi::<P3>().convert_into(ANGSTROM.powi::<P3>());
        let epsilon_k_ab = 2062.1;
        let na = 2.0;
        let nb = 1.0;
        let nc = 0.0;

        let model_record = SaftVRMieRecord::new(
            m,
            sigma,
            epsilon_k,
            lr,
            la,
            Some(kappa_ab),
            Some(epsilon_k_ab),
            Some(na),
            Some(nb),
            None,
            None,
            None,
            None,
        );
        let methanol = PureRecord {
            identifier: Identifier {
                name: Some("methanol".to_string()),
                ..Default::default()
            },
            molarweight,
            model_record,
        };
        let m = 2.3356;
        let epsilon_k = 227.66;
        let sigma = 3.5612;
        let lr = 10.179;
        let la = 6.0;
        let molarweight = 60.096;
        let kappa_ab = 6.2309e-29 * METER.powi::<P3>().convert_into(ANGSTROM.powi::<P3>());
        let epsilon_k_ab = 2097.9;
        let na = 1.0;
        let nb = 1.0;

        let model_record = SaftVRMieRecord::new(
            m,
            sigma,
            epsilon_k,
            lr,
            la,
            Some(kappa_ab),
            Some(epsilon_k_ab),
            Some(na),
            Some(nb),
            None,
            None,
            None,
            None,
        );
        let propanol = PureRecord {
            identifier: Identifier {
                name: Some("propanol".to_string()),
                ..Default::default()
            },
            molarweight,
            model_record,
        };
        SaftVRMieParameters::new_binary(vec![methanol, propanol], None).unwrap()
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use num_dual::Dual2;
    use test::utils::ethane;
    use utils::methane;

    use super::*;

    #[test]
    fn test_mie_potential() {
        let la = 6.0;
        let lr = 12.0;
        let c_eps_t = Dual2::from_re(4.0);
        let r = 0.9;
        let rd = Dual2::from_re(r).derivative();
        let u = beta_u_mie(rd, la, lr, 1.0, c_eps_t);
        let [_, u_du, du_d2u] = mie_potential_halley(r, la, lr, 4.0);
        assert_relative_eq!(u.re / u.v1, u_du);
        assert_relative_eq!(u.v1 / u.v2, du_d2u);
    }

    #[test]
    fn hs_diameter() {
        let temperature = 50.0;
        let ethane = ethane();
        let d_hs = ethane.hs_diameter(temperature);
        dbg!(&d_hs);
        assert_relative_eq!(
            3.694019351651498,
            d_hs[0],
            max_relative = 1e-9,
            epsilon = 1e-9
        )
    }

    #[test]
    fn test_zero_integrant() {
        let la = 6.0;
        let lr = 12.4;
        let eps_k = 206.12;
        let temperature = 50.0;
        let c = lr / (lr - la) * (lr / la).powf(la / (lr - la));
        let c_eps_t = c * eps_k / temperature;
        dbg!(c_eps_t);
        let r0 = lower_integratal_limit(la, lr, c_eps_t);
        dbg!(&r0);
        assert_relative_eq!(3.694019351651498, r0, max_relative = 1e-9, epsilon = 1e-9)
    }
}
