use crate::hard_sphere::FMTVersion;
use crate::saftvrqmie::eos::dispersion::{dispersion_energy_density, Alpha};
use crate::saftvrqmie::parameters::SaftVRQMieParameters;
use feos_core::{EosError, EosResult};
use feos_dft::{
    FunctionalContributionDual, WeightFunction, WeightFunctionInfo, WeightFunctionShape,
};
use ndarray::*;
use num_dual::*;
use std::f64::consts::PI;
use std::fmt;
use std::sync::Arc;

const PI36M1: f64 = 1.0 / (36.0 * PI);
const N3_CUTOFF: f64 = 1e-5;

#[derive(Clone)]
pub struct PureFMTAssocFunctional {
    parameters: Arc<SaftVRQMieParameters>,
    version: FMTVersion,
}

impl PureFMTAssocFunctional {
    pub fn new(parameters: Arc<SaftVRQMieParameters>, version: FMTVersion) -> Self {
        Self {
            parameters,
            version,
        }
    }
}

impl<N: DualNum<f64> + ScalarOperand> FunctionalContributionDual<N> for PureFMTAssocFunctional {
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
                prefactor: self.parameters.m.mapv(|m| m.into()),
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
        //let p = &self.parameters;

        // weighted densities
        let n2 = weighted_densities.index_axis(Axis(0), 0);
        let n3 = weighted_densities.index_axis(Axis(0), 1);
        let n2v = weighted_densities.slice_axis(Axis(0), Slice::new(2, None, 1));

        // temperature dependent segment radius
        let r = self.parameters.hs_diameter(temperature)[0] * 0.5;

        // auxiliary variables
        if n3.iter().any(|n3| n3.re() > 1.0) {
            return Err(EosError::IterationFailed(String::from(
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
        let phi = -(&n0 * &ln31) + n1n2 * &n3m1rec + n2n2 * n2 * PI36M1 * f3;

        // association
        // if p.nassoc == 1 {
        //     let mut xi = -(&n2v * &n2v).sum_axis(Axis(0)) / (&n2 * &n2) + 1.0;
        //     xi.iter_mut().zip(&n2).for_each(|(xi, &n2)| {
        //         if n2.re() < N0_CUTOFF * 4.0 * PI * p.m[0] * r.re().powi(2) {
        //             *xi = N::one();
        //         }
        //     });

        //     let k = &n2 * &n3m1rec * r;
        //     let deltarho = (((&k / 18.0 + 0.5) * &k * &xi + 1.0) * n3m1rec)
        //         * ((temperature.recip() * p.epsilon_k_aibj[(0, 0)]).exp_m1()
        //             * (p.sigma[0].powi(3) * p.kappa_aibj[(0, 0)]))
        //         * (&n0 / p.m[0] * &xi);

        //     let f = |x: N| x.ln() - x * 0.5 + 0.5;
        //     phi = phi
        //         + if p.nb[0] > 0.0 {
        //             let xa = deltarho.mapv(|d| assoc_site_frac_ab(d, p.na[0], p.nb[0]));
        //             let xb = (xa.clone() - 1.0) * p.na[0] / p.nb[0] + 1.0;
        //             (n0 / p.m[0] * xi) * (xa.mapv(f) * p.na[0] + xb.mapv(f) * p.nb[0])
        //         } else {
        //             let xa = deltarho.mapv(|d| assoc_site_frac_a(d, p.na[0]));
        //             n0 / p.m[0] * xi * (xa.mapv(f) * p.na[0])
        //         };
        // }

        Ok(phi)
    }
}

impl fmt::Display for PureFMTAssocFunctional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pure FMT+association")
    }
}

// #[derive(Clone)]
// pub struct PureChainFunctional {
//     parameters: Arc<SaftVRQMieParameters>,
// }

// impl PureChainFunctional {
//     pub fn new(parameters: Arc<SaftVRQMieParameters>) -> Self {
//         Self { parameters }
//     }
// }

// impl<N: DualNum<f64> + ScalarOperand> FunctionalContributionDual<N> for PureChainFunctional {
//     fn weight_functions(&self, temperature: N) -> WeightFunctionInfo<N> {
//         let d = self.parameters.hs_diameter(temperature);
//         WeightFunctionInfo::new(arr1(&[0]), true)
//             .add(
//                 WeightFunction::new_scaled(d.clone(), WeightFunctionShape::Delta),
//                 false,
//             )
//             .add(
//                 WeightFunction {
//                     prefactor: (&self.parameters.m / 8.0).mapv(|x| x.into()),
//                     kernel_radius: d,
//                     shape: WeightFunctionShape::Theta,
//                 },
//                 false,
//             )
//     }

//     fn calculate_helmholtz_energy_density(
//         &self,
//         _: N,
//         weighted_densities: ArrayView2<N>,
//     ) -> EosResult<Array1<N>> {
//         let rho = weighted_densities.index_axis(Axis(0), 0);
//         // negative lambdas lead to nan, therefore the absolute value is used
//         let lambda = weighted_densities
//             .index_axis(Axis(0), 1)
//             .map(|&l| if l.re() < 0.0 { -l } else { l } + N::from(f64::EPSILON));
//         let eta = weighted_densities.index_axis(Axis(0), 2);

//         let y = eta.mapv(|eta| (eta * 0.5 - 1.0) / (eta - 1.0).powi(3));
//         Ok(-(y * lambda).mapv(|x| (x.ln() - 1.0) * (self.parameters.m[0] - 1.0)) * rho)
//     }
// }

// impl fmt::Display for PureChainFunctional {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "Pure chain")
//     }
// }

#[derive(Clone)]
pub struct PureAttFunctional {
    parameters: Arc<SaftVRQMieParameters>,
}

impl PureAttFunctional {
    pub fn new(parameters: Arc<SaftVRQMieParameters>) -> Self {
        Self { parameters }
    }
}

impl<N: DualNum<f64> + ScalarOperand> FunctionalContributionDual<N> for PureAttFunctional {
    fn weight_functions(&self, temperature: N) -> WeightFunctionInfo<N> {
        let d = self.parameters.hs_diameter(temperature);
        const PSI: f64 = 1.3862; // Homosegmented DFT (Sauer2017)
        WeightFunctionInfo::new(arr1(&[0]), false).add(
            WeightFunction::new_scaled(d * PSI, WeightFunctionShape::Theta),
            false,
        )
    }

    fn weight_functions_pdgt(&self, temperature: N) -> WeightFunctionInfo<N> {
        let d = self.parameters.hs_diameter(temperature);
        const PSI: f64 = 1.3286; // pDGT (Rehner2018)
        WeightFunctionInfo::new(arr1(&[0]), false).add(
            WeightFunction::new_scaled(d * PSI, WeightFunctionShape::Theta),
            false,
        )
    }

    fn calculate_helmholtz_energy_density(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> EosResult<Array1<N>> {
        let p = &self.parameters;
        let rho = weighted_densities.index_axis(Axis(0), 0);
        let n = p.m.len();

        // temperature dependent segment radius
        let s_eff_ij =
            Array2::from_shape_fn((n, n), |(i, j)| p.calc_sigma_eff_ij(i, j, temperature));

        // temperature dependent segment radius
        let d_hs_ij = Array2::from_shape_fn((n, n), |(i, j)| {
            p.hs_diameter_ij(i, j, temperature, s_eff_ij[[i, j]])
        });

        // temperature dependent well depth
        let epsilon_k_eff_ij =
            Array2::from_shape_fn((n, n), |(i, j)| p.calc_epsilon_k_eff_ij(i, j, temperature));

        // temperature dependent well depth
        let dq_ij = Array2::from_shape_fn((n, n), |(i, j)| p.quantum_d_ij(i, j, temperature));

        // alphas ....
        let alpha = Alpha::new(p, &s_eff_ij, &epsilon_k_eff_ij, temperature);

        let phi = rho.mapv(|rho_cell| {
            dispersion_energy_density(
                p,
                &d_hs_ij,
                &s_eff_ij,
                &epsilon_k_eff_ij,
                &dq_ij,
                &alpha,
                &arr1(&[rho_cell]),
                temperature,
            )
        });
        Ok(phi)
    }
}

impl fmt::Display for PureAttFunctional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pure attractive")
    }
}
