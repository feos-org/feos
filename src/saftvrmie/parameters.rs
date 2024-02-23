use std::collections::HashMap;

use crate::association::{AssociationParameters, AssociationRecord, BinaryAssociationRecord};
use crate::hard_sphere::{HardSphereProperties, MonomerShape};
use feos_core::parameter::{Parameter, ParameterError, PureRecord};
use ndarray::{Array, Array1, Array2};
use num_dual::DualNum;
use num_traits::Zero;
use serde::{Deserialize, Serialize};

const X_K21: [f64; 21] = [
    -0.995657163025808080735527280689003,
    -0.973906528517171720077964012084452,
    -0.930157491355708226001207180059508,
    -0.865063366688984510732096688423493,
    -0.780817726586416897063717578345042,
    -0.679409568299024406234327365114874,
    -0.562757134668604683339000099272694,
    -0.433395394129247190799265943165784,
    -0.294392862701460198131126603103866,
    -0.148874338981631210884826001129720,
    0.000000000000000000000000000000000,
    0.148874338981631210884826001129720,
    0.294392862701460198131126603103866,
    0.433395394129247190799265943165784,
    0.562757134668604683339000099272694,
    0.679409568299024406234327365114874,
    0.780817726586416897063717578345042,
    0.865063366688984510732096688423493,
    0.930157491355708226001207180059508,
    0.973906528517171720077964012084452,
    0.995657163025808080735527280689003,
];

const W_K21: [f64; 21] = [
    0.011694638867371874278064396062192,
    0.032558162307964727478818972459390,
    0.054755896574351996031381300244580,
    0.075039674810919952767043140916190,
    0.093125454583697605535065465083366,
    0.109387158802297641899210590325805,
    0.123491976262065851077958109831074,
    0.134709217311473325928054001771707,
    0.142775938577060080797094273138717,
    0.147739104901338491374841515972068,
    0.149445554002916905664936468389821,
    0.147739104901338491374841515972068,
    0.142775938577060080797094273138717,
    0.134709217311473325928054001771707,
    0.123491976262065851077958109831074,
    0.109387158802297641899210590325805,
    0.093125454583697605535065465083366,
    0.075039674810919952767043140916190,
    0.054755896574351996031381300244580,
    0.032558162307964727478818972459390,
    0.011694638867371874278064396062192,
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
            AssociationParameters::new(&association_records, &sigma, &binary_association, None);

        let br = binary_records.as_ref();
        let k_ij = br.map_or_else(|| Array2::zeros([n; 2]), |br| br.mapv(|br| br.k_ij));
        let gamma_ij = br.map_or_else(|| Array2::zeros([n; 2]), |br| br.mapv(|br| br.gamma_ij));
        let mut sigma_ij = Array::zeros((n, n));
        let mut e_k_ij = Array::zeros((n, n));
        let mut epsilon_k_ij = Array::zeros((n, n));
        let mut lr_ij = Array::zeros((n, n));
        let mut la_ij = Array::zeros((n, n));
        let mut c_ij = Array::zeros((n, n));
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
    /// Energy of the Mie potential, the first and second derivative.
    pub fn mie_potential_ij<D: DualNum<f64> + Copy>(&self, i: usize, j: usize, r: D) -> [D; 3] {
        let lr = self.lr_ij[[i, j]];
        let la = self.la_ij[[i, j]];
        let s = self.sigma_ij[[i, j]];
        let c_eps = self.c_ij[[i, j]] * self.epsilon_k_ij[[i, j]];

        let u = (r.powf(lr).recip() * s.powf(lr) - r.powf(la).recip() * s.powf(la)) * c_eps;
        let u_r = (-r.powf(lr + 1.0).recip() * lr * s.powf(lr)
            + r.powf(la + 1.0).recip() * la * s.powf(la))
            * c_eps;
        let u_rr = (r.powf(lr + 2.0).recip() * lr * (lr + 1.0) * s.powf(lr)
            - r.powf(la + 2.0).recip() * la * (la + 1.0) * s.powf(la))
            * c_eps;
        [u, u_r, u_rr]
    }

    /// Find the lower limit for integration of HS diameter
    pub fn zero_integrand<D: DualNum<f64> + Copy>(
        &self,
        i: usize,
        j: usize,
        inverse_temperature: D,
    ) -> D {
        let mut r = D::one() * self.sigma_ij[[i, j]] * 0.7;
        let mut f = D::zero();
        for _k in 1..20 {
            let u_vec = self.mie_potential_ij(i, j, r);
            f = inverse_temperature * u_vec[0] + f64::EPSILON.ln();
            if f.re().abs() < 1.0e-12 {
                break;
            }
            let dfdr = inverse_temperature * u_vec[1];
            let mut dr = -(f / dfdr);
            if dr.re().abs() > 0.5 {
                dr *= 0.5 / dr.re().abs();
            }
            r += dr;
        }
        if f.re().abs() > 1.0e-12 {
            println!("zero_integrand calculation failed {}", f.re().abs());
        }
        r
    }

    #[inline]
    pub fn hs_diameter_ij<D: DualNum<f64> + Copy>(
        &self,
        i: usize,
        j: usize,
        inverse_temperature: D,
    ) -> D {
        let r0 = self.zero_integrand(i, j, inverse_temperature);
        let sigma = self.sigma_ij[[i, j]];
        let mut d_hs = r0;
        for k in 0..21 {
            let width = (-r0 + sigma) * 0.5;
            let r = width * X_K21[k] + width + r0;
            let u = self.mie_potential_ij(i, j, r);
            let f_u = -(-u[0] * inverse_temperature).exp() + 1.0;
            d_hs += width * f_u * W_K21[k];
        }
        d_hs
    }
}

impl HardSphereProperties for SaftVRMieParameters {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<N> {
        MonomerShape::NonSpherical(self.m.mapv(N::from))
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D> {
        Array1::from_shape_fn(self.m.len(), |i| -> D {
            self.hs_diameter_ij(i, i, temperature.recip())
        })
    }
}

pub(super) mod utils {
    use super::*;
    use feos_core::parameter::Identifier;

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
}

#[cfg(test)]
mod test {
    use test::utils::ethane;

    use super::*;

    #[test]
    fn hs_diameter() {
        let temperature = 50.0;
        let ethane = ethane();
        let d_hs = ethane.hs_diameter(temperature);
        assert!((3.694019351651498 - d_hs[0]) / 3.694019351651498 < 1e-15)
    }
}
