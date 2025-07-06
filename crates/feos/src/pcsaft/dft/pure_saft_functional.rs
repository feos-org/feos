use super::polar::{pair_integral_ij, triplet_integral_ijk};
use crate::association::AssociationFunctional;
use crate::hard_sphere::{FMTVersion, HardSphereProperties};
use crate::pcsaft::eos::dispersion::{A0, A1, A2, B0, B1, B2};
use crate::pcsaft::eos::polar::{AD, AQ, BD, BQ, CD, CQ, PI_SQ_43};
use crate::pcsaft::parameters::PcSaftPars;
use feos_core::{FeosError, FeosResult};
use feos_dft::{FunctionalContribution, WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use ndarray::*;
use num_dual::*;
use std::f64::consts::{FRAC_PI_6, PI};

const PI36M1: f64 = 1.0 / (36.0 * PI);
const N3_CUTOFF: f64 = 1e-5;
const N0_CUTOFF: f64 = 1e-9;

pub(crate) struct PureFMTAssocFunctional<'a> {
    parameters: &'a PcSaftPars,
    association: Option<AssociationFunctional<'a, PcSaftPars>>,
    version: FMTVersion,
}

impl<'a> PureFMTAssocFunctional<'a> {
    pub(crate) fn new(
        params: &'a PcSaftPars,
        association: Option<AssociationFunctional<'a, PcSaftPars>>,
        version: FMTVersion,
    ) -> Self {
        Self {
            parameters: params,
            association,
            version,
        }
    }
}

impl<'a> FunctionalContribution for PureFMTAssocFunctional<'a> {
    fn name(&self) -> &'static str {
        "Pure FMT+association"
    }

    fn weight_functions<N: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        temperature: N,
    ) -> WeightFunctionInfo<N> {
        let r = self.parameters.hs_diameter(temperature) * 0.5;
        WeightFunctionInfo::new(arr1(&[0]), false).extend(
            vec![
                WeightFunctionShape::Delta,
                WeightFunctionShape::Theta,
                WeightFunctionShape::DeltaVec,
            ]
            .into_iter()
            .map(|s| WeightFunction {
                prefactor: self.parameters.m.mapv(|m| m.into()),
                kernel_radius: r.clone(),
                shape: s,
            })
            .collect(),
            false,
        )
    }

    fn helmholtz_energy_density<N: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> FeosResult<Array1<N>> {
        let p = &self.parameters;

        // weighted densities
        let n2 = weighted_densities.index_axis(Axis(0), 0);
        let n3 = weighted_densities.index_axis(Axis(0), 1);
        let n2v = weighted_densities.slice_axis(Axis(0), Slice::new(2, None, 1));

        // temperature dependent segment radius
        let r = self.parameters.hs_diameter(temperature)[0] * 0.5;

        // auxiliary variables
        if n3.iter().any(|n3| n3.re() > 1.0) {
            return Err(FeosError::IterationFailed(String::from(
                "PureFMTAssocFunctional",
            )));
        }
        let ln31 = n3.mapv(|n3| (-n3).ln_1p());
        let n3rec = n3.mapv(|n3| n3.recip());
        let n3m1 = n3.mapv(|n3| -n3 + 1.0);
        let n3m1rec = n3m1.mapv(|n3m1| n3m1.recip());
        let n1 = n2.mapv(|n2| n2 / (r * 4.0 * PI));
        let n0 = n2.mapv(|n2| n2 / (r * r * 4.0 * PI));
        let n1v = n2v.mapv(|n2v| n2v / (r * 4.0 * PI));

        let (n1n2, n2n2) = match self.version {
            FMTVersion::WhiteBear => (
                &n1 * &n2 - (&n1v * &n2v).sum_axis(Axis(0)),
                &n2 * &n2 - (&n2v * &n2v).sum_axis(Axis(0)) * 3.0,
            ),
            FMTVersion::AntiSymWhiteBear => {
                let mut xi2 = (&n2v * &n2v).sum_axis(Axis(0)) / n2.map(|n| n.powi(2));
                xi2.iter_mut().for_each(|x| {
                    if x.re() > 1.0 {
                        *x = N::one()
                    }
                });
                (
                    &n1 * &n2 - (&n1v * &n2v).sum_axis(Axis(0)),
                    &n2 * &n2 * xi2.mapv(|x| (-x + 1.0).powi(3)),
                )
            }
            _ => unreachable!(),
        };

        // The f3 term contains a 0/0, therefore a taylor expansion is used for small values of n3
        let mut f3 = (&n3m1 * &n3m1 * &ln31 + n3) * &n3rec * n3rec * &n3m1rec * &n3m1rec;
        f3.iter_mut().zip(n3).for_each(|(f3, &n3)| {
            if n3.re() < N3_CUTOFF {
                *f3 = (((n3 * 35.0 / 6.0 + 4.8) * n3 + 3.75) * n3 + 8.0 / 3.0) * n3 + 1.5;
            }
        });
        let mut phi = -(&n0 * &ln31) + n1n2 * &n3m1rec + n2n2 * n2 * PI36M1 * f3;

        // association
        if let Some(a) = self.association.as_ref() {
            let mut xi = -(&n2v * &n2v).sum_axis(Axis(0)) / (&n2 * &n2) + 1.0;
            xi.iter_mut().zip(&n2).for_each(|(xi, &n2)| {
                if n2.re() < N0_CUTOFF * 4.0 * PI * p.m[0] * r.re().powi(2) {
                    *xi = N::one();
                }
            });
            let rho0 = (&n0 / p.m[0] * &xi).insert_axis(Axis(0));

            phi += &(a._helmholtz_energy_density(temperature, &rho0, &n2, &n3m1rec, &xi))?;
        }

        Ok(phi)
    }
}

#[derive(Clone)]
pub(crate) struct PureChainFunctional<'a> {
    parameters: &'a PcSaftPars,
}

impl<'a> PureChainFunctional<'a> {
    pub(crate) fn new(parameters: &'a PcSaftPars) -> Self {
        Self { parameters }
    }
}

impl<'a> FunctionalContribution for PureChainFunctional<'a> {
    fn name(&self) -> &'static str {
        "Pure chain"
    }

    fn weight_functions<N: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        temperature: N,
    ) -> WeightFunctionInfo<N> {
        let d = self.parameters.hs_diameter(temperature);
        WeightFunctionInfo::new(arr1(&[0]), true)
            .add(
                WeightFunction::new_scaled(d.clone(), WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction {
                    prefactor: (&self.parameters.m / 8.0).mapv(|x| x.into()),
                    kernel_radius: d,
                    shape: WeightFunctionShape::Theta,
                },
                false,
            )
    }

    fn helmholtz_energy_density<N: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        _: N,
        weighted_densities: ArrayView2<N>,
    ) -> FeosResult<Array1<N>> {
        let rho = weighted_densities.index_axis(Axis(0), 0);
        // negative lambdas lead to nan, therefore the absolute value is used
        let lambda = weighted_densities
            .index_axis(Axis(0), 1)
            .map(|&l| if l.re() < 0.0 { -l } else { l } + N::from(f64::EPSILON));
        let eta = weighted_densities.index_axis(Axis(0), 2);

        let y = eta.mapv(|eta| (eta * 0.5 - 1.0) / (eta - 1.0).powi(3));
        Ok(-(y * lambda).mapv(|x| (x.ln() - 1.0) * (self.parameters.m[0] - 1.0)) * rho)
    }
}

#[derive(Clone)]
pub(crate) struct PureAttFunctional<'a> {
    parameters: &'a PcSaftPars,
}

impl<'a> PureAttFunctional<'a> {
    pub(crate) fn new(parameters: &'a PcSaftPars) -> Self {
        Self { parameters }
    }
}

impl<'a> FunctionalContribution for PureAttFunctional<'a> {
    fn name(&self) -> &'static str {
        "Pure attractive"
    }

    fn weight_functions<N: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        temperature: N,
    ) -> WeightFunctionInfo<N> {
        let d = self.parameters.hs_diameter(temperature);
        const PSI: f64 = 1.3862; // Homosegmented DFT (Sauer2017)
        WeightFunctionInfo::new(arr1(&[0]), false).add(
            WeightFunction::new_scaled(d * PSI, WeightFunctionShape::Theta),
            false,
        )
    }

    fn weight_functions_pdgt<N: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        temperature: N,
    ) -> WeightFunctionInfo<N> {
        let d = self.parameters.hs_diameter(temperature);
        const PSI: f64 = 1.3286; // pDGT (Rehner2018)
        WeightFunctionInfo::new(arr1(&[0]), false).add(
            WeightFunction::new_scaled(d * PSI, WeightFunctionShape::Theta),
            false,
        )
    }

    fn helmholtz_energy_density<N: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> FeosResult<Array1<N>> {
        let p = &self.parameters;
        let rho = weighted_densities.index_axis(Axis(0), 0);

        // temperature dependent segment radius
        let d = p.hs_diameter(temperature)[0];

        let eta = rho.mapv(|rho| rho * FRAC_PI_6 * p.m[0] * d.powi(3));
        let m1 = (p.m[0] - 1.0) / p.m[0];
        let m2 = m1 * (p.m[0] - 2.0) / p.m[0];
        let e = temperature.recip() * p.epsilon_k[0];
        let s3 = p.sigma[0].powi(3);

        // I1, I2 and C1
        let mut i1: Array1<N> = Array::zeros(eta.raw_dim());
        let mut i2: Array1<N> = Array::zeros(eta.raw_dim());
        for i in 0..=6 {
            i1 = i1 + eta.mapv(|eta| eta.powi(i as i32) * (A0[i] + m1 * A1[i] + m2 * A2[i]));
            i2 = i2 + eta.mapv(|eta| eta.powi(i as i32) * (B0[i] + m1 * B1[i] + m2 * B2[i]));
        }
        let c1 = eta.mapv(|eta| {
            ((eta * 8.0 - eta.powi(2) * 2.0) / (eta - 1.0).powi(4) * p.m[0]
                + (eta * 20.0 - eta.powi(2) * 27.0 + eta.powi(3) * 12.0 - eta.powi(4) * 2.0)
                    / ((eta - 1.0) * (eta - 2.0)).powi(2)
                    * (1.0 - p.m[0])
                + 1.0)
                .recip()
        });
        let mut phi = rho.mapv(|rho| -(rho * p.m[0]).powi(2) * e * s3 * PI)
            * (i1 * 2.0 + c1 * i2.mapv(|i2| i2 * p.m[0] * e));

        // dipoles
        if p.ndipole > 0 {
            let mu2_term = e * s3 * p.mu2[0];
            let m = p.m[0].min(2.0);
            let m1 = (m - 1.0) / m;
            let m2 = m1 * (m - 2.0) / m;

            let phi2 = -(&rho * &rho)
                * pair_integral_ij(m1, m2, &eta, &AD, &BD, e)
                * (mu2_term * mu2_term / s3 * PI);
            let phi3 = -(&rho * &rho * rho)
                * triplet_integral_ijk(m1, m2, &eta, &CD)
                * (mu2_term * mu2_term * mu2_term / s3 * PI_SQ_43);

            let mut phi_d = &phi2 * &phi2 / (&phi2 - &phi3);
            phi_d.iter_mut().zip(phi2.iter()).for_each(|(p, &p2)| {
                if p.re().is_nan() {
                    *p = p2;
                }
            });
            phi += &phi_d;
        }

        // quadrupoles
        if p.nquadpole > 0 {
            let q2_term = e * p.sigma[0].powi(5) * p.q2[0];
            let m = p.m[0].min(2.0);
            let m1 = (m - 1.0) / m;
            let m2 = m1 * (m - 2.0) / m;

            let phi2 = -(&rho * &rho)
                * pair_integral_ij(m1, m2, &eta, &AQ, &BQ, e)
                * (q2_term * q2_term / p.sigma[0].powi(7) * PI * 0.5625);
            let phi3 = (&rho * &rho * rho)
                * triplet_integral_ijk(m1, m2, &eta, &CQ)
                * (q2_term * q2_term * q2_term / s3.powi(3) * PI * PI * 0.5625);

            let mut phi_q = &phi2 * &phi2 / (&phi2 - &phi3);
            phi_q.iter_mut().zip(phi2.iter()).for_each(|(p, &p2)| {
                if p.re().is_nan() {
                    *p = p2;
                }
            });
            phi += &phi_q;
        }

        Ok(phi)
    }
}
