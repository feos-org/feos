use crate::hard_sphere::{FMTVersion, HardSphereProperties};
use crate::pets::eos::dispersion::{A, B};
use crate::pets::parameters::PetsParameters;
use feos_core::{EosError, EosResult};
use feos_dft::{
    FunctionalContributionDual, WeightFunction, WeightFunctionInfo, WeightFunctionShape,
};
use ndarray::*;
use num_dual::*;
use std::f64::consts::{FRAC_PI_6, PI};
use std::fmt;
use std::sync::Arc;

const PI36M1: f64 = 1.0 / (36.0 * PI);
const N3_CUTOFF: f64 = 1e-5;

#[derive(Clone)]
pub struct PureFMTFunctional {
    parameters: Arc<PetsParameters>,
    version: FMTVersion,
}

impl PureFMTFunctional {
    pub fn new(parameters: Arc<PetsParameters>, version: FMTVersion) -> Self {
        Self {
            parameters,
            version,
        }
    }
}

impl<N: DualNum<f64> + Copy + ScalarOperand> FunctionalContributionDual<N> for PureFMTFunctional {
    fn weight_functions(&self, temperature: N) -> WeightFunctionInfo<N> {
        let r = self.parameters.hs_diameter(temperature) * 0.5;
        WeightFunctionInfo::new(arr1(&[0]), false).extend(
            vec![
                WeightFunctionShape::Delta,
                WeightFunctionShape::Theta,
                WeightFunctionShape::DeltaVec,
            ]
            .into_iter()
            .map(|s| WeightFunction {
                prefactor: Array1::<N>::ones(self.parameters.sigma.len()),
                kernel_radius: r.clone(),
                shape: s,
            })
            .collect(),
            false,
        )
    }

    fn calculate_helmholtz_energy_density(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> EosResult<Array1<N>> {
        // Weighted densities
        let n2 = weighted_densities.index_axis(Axis(0), 0);
        let n3 = weighted_densities.index_axis(Axis(0), 1);
        let n2v = weighted_densities.slice_axis(Axis(0), Slice::new(2, None, 1));

        // Temperature dependent segment radius
        let r = self.parameters.hs_diameter(temperature)[0] * 0.5;

        // Auxiliary variables
        if n3.iter().any(|n3| n3.re() > 1.0) {
            return Err(EosError::IterationFailed(String::from("PureFMTFunctional")));
        }
        let ln31 = n3.mapv(|n3| (-n3).ln_1p());
        let n3rec = n3.mapv(|n3| n3.recip());
        let n3m1 = n3.mapv(|n3| -n3 + 1.0);
        let n3m1rec = n3m1.mapv(|n3m1| n3m1.recip());
        let n1 = n2.mapv(|n2| n2 / (r * 4.0 * PI));
        let n0 = n2.mapv(|n2| n2 / (r * r * 4.0 * PI));
        let n1v = n2v.mapv(|n2v| n2v / (r * 4.0 * PI));

        // Different FMT versions
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
        let phi = -(&n0 * &ln31) + n1n2 * &n3m1rec + n2n2 * n2 * PI36M1 * f3;

        Ok(phi)
    }
}

impl fmt::Display for PureFMTFunctional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pure FMT")
    }
}

#[derive(Clone)]
pub struct PureAttFunctional {
    parameters: Arc<PetsParameters>,
}

impl PureAttFunctional {
    pub fn new(parameters: Arc<PetsParameters>) -> Self {
        Self { parameters }
    }
}

impl<N: DualNum<f64> + Copy + ScalarOperand> FunctionalContributionDual<N> for PureAttFunctional {
    fn weight_functions(&self, temperature: N) -> WeightFunctionInfo<N> {
        let d = self.parameters.hs_diameter(temperature);
        const PSI: f64 = 1.21; // Homosegmented DFT (Heier2018)
        WeightFunctionInfo::new(arr1(&[0]), false).add(
            WeightFunction::new_scaled(d * PSI, WeightFunctionShape::Theta),
            false,
        )
    }

    fn weight_functions_pdgt(&self, temperature: N) -> WeightFunctionInfo<N> {
        let d = self.parameters.hs_diameter(temperature);
        const PSI: f64 = 1.21; // pDGT (not yet determined)
        WeightFunctionInfo::new(arr1(&[0]), false).add(
            WeightFunction::new_scaled(d * PSI, WeightFunctionShape::Theta),
            false,
        )
    }

    fn calculate_helmholtz_energy_density(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> Result<Array1<N>, EosError> {
        let p = &self.parameters;
        let rho = weighted_densities.index_axis(Axis(0), 0);

        // temperature dependent segment radius
        let d = p.hs_diameter(temperature)[0];

        let eta = rho.mapv(|rho| rho * FRAC_PI_6 * d.powi(3));
        let e = temperature.recip() * p.epsilon_k[0];
        let s3 = p.sigma[0].powi(3);

        // I1, I2 and C1
        let mut i1: Array1<N> = Array::zeros(eta.raw_dim());
        let mut i2: Array1<N> = Array::zeros(eta.raw_dim());
        for i in 0..=6 {
            i1 = i1 + eta.mapv(|eta| eta.powi(i as i32) * A[i]);
            i2 = i2 + eta.mapv(|eta| eta.powi(i as i32) * B[i]);
        }
        let c1 =
            eta.mapv(|eta| ((eta * 8.0 - eta.powi(2) * 2.0) / (eta - 1.0).powi(4) + 1.0).recip());
        let phi =
            rho.mapv(|rho| -(rho).powi(2) * e * s3 * PI) * (i1 * 2.0 + c1 * i2.mapv(|i2| i2 * e));

        Ok(phi)
    }
}

impl fmt::Display for PureAttFunctional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pure attractive")
    }
}
