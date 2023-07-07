use crate::saftvrqmie::parameters::SaftVRQMieParameters;
use feos_core::EosResult;
use feos_dft::{
    FunctionalContributionDual, WeightFunction, WeightFunctionInfo, WeightFunctionShape,
};
use ndarray::*;
use num_dual::DualNum;
use std::f64::consts::PI;
use std::fmt;
use std::sync::Arc;

pub const N0_CUTOFF: f64 = 1e-9;

#[derive(Clone)]
pub struct NonAddHardSphereFunctional {
    parameters: Arc<SaftVRQMieParameters>,
}

impl NonAddHardSphereFunctional {
    pub fn new(parameters: Arc<SaftVRQMieParameters>) -> Self {
        Self { parameters }
    }
}

impl<N> FunctionalContributionDual<N> for NonAddHardSphereFunctional
where
    N: DualNum<f64> + Copy + ScalarOperand,
{
    fn weight_functions(&self, temperature: N) -> WeightFunctionInfo<N> {
        let p = &self.parameters;
        let r = p.hs_diameter(temperature) * 0.5;
        WeightFunctionInfo::new(Array1::from_shape_fn(r.len(), |i| i), false)
            .add(
                WeightFunction::new_scaled(r.clone(), WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction {
                    prefactor: p.m.mapv(N::from),
                    kernel_radius: r.clone(),
                    shape: WeightFunctionShape::DeltaVec,
                },
                false,
            )
            .add(
                WeightFunction {
                    prefactor: p.m.mapv(N::from),
                    kernel_radius: r,
                    shape: WeightFunctionShape::Theta,
                },
                true,
            )
    }

    fn calculate_helmholtz_energy_density(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> EosResult<Array1<N>> {
        let p = &self.parameters;
        // number of components
        let n = p.m.len();
        // number of dimensions
        let dim = (weighted_densities.shape()[0] - 1) / n - 1;

        // weighted densities
        let n0i = weighted_densities.slice_axis(Axis(0), Slice::new(0, Some(n as isize), 1));
        let n2vi: Vec<_> = (0..dim)
            .map(|i| {
                weighted_densities.slice_axis(
                    Axis(0),
                    Slice::new((n * (i + 1)) as isize, Some((n * (i + 2)) as isize), 1),
                )
            })
            .collect();
        let n3 = weighted_densities.index_axis(Axis(0), n * (dim + 1));

        // calculate rho0
        let r = p.hs_diameter(temperature) * 0.5;
        let mut n2i = Array::zeros(n0i.raw_dim());
        for i in 0..n {
            n2i.index_axis_mut(Axis(0), i)
                .assign(&(&n0i.index_axis(Axis(0), i) * (r[i].powi(2) * (p.m[i] * 4.0 * PI))));
        }
        let mut rho0: Array2<N> = (n2vi
            .iter()
            .map(|n2vi| n2vi * n2vi)
            .fold(Array::zeros(n0i.raw_dim()), |acc, x| acc + x)
            / -(&n2i * &n2i)
            + 1.0)
            * n0i;
        rho0.iter_mut().zip(&n0i).for_each(|(rho0, &n0i)| {
            if n0i.re() < N0_CUTOFF {
                *rho0 = n0i;
            }
        });

        // calculate xi
        let n2v: Vec<_> = n2vi.iter().map(|n2vi| n2vi.sum_axis(Axis(0))).collect();
        let n2 = n2i.sum_axis(Axis(0));
        let mut xi = n2v
            .iter()
            .map(|n2v| n2v * n2v)
            .fold(Array::zeros(n3.raw_dim()), |acc, x| acc + x)
            / -(&n2 * &n2)
            + 1.0;
        xi.iter_mut()
            .zip(&n0i.sum_axis(Axis(0)))
            .for_each(|(xi, &n0i)| {
                if n0i.re() < N0_CUTOFF {
                    *xi = N::one();
                }
            });

        // auxiliary variables
        let n3i = n3.mapv(|n3| (-n3 + 1.0).recip());

        // temperature dependent segment radius // calc & store this in struct
        let s_eff_ij =
            Array2::from_shape_fn((n, n), |(i, j)| p.calc_sigma_eff_ij(i, j, temperature));

        // temperature dependent segment radius // calc & store this in struct
        let d_hs_ij = Array2::from_shape_fn((n, n), |(i, j)| {
            p.hs_diameter_ij(i, j, temperature, s_eff_ij[[i, j]])
        });

        // Additive hard-sphere diameter
        let d_hs_add_ij =
            Array2::from_shape_fn((n, n), |(i, j)| (d_hs_ij[[i, i]] + d_hs_ij[[j, j]]) * 0.5);

        Ok(rho0
            .view()
            .into_shape([n, rho0.len() / n])
            .unwrap()
            .axis_iter(Axis(1))
            .zip(n2.iter())
            .zip(n3i.iter())
            .zip(xi.iter())
            .map(|(((rho0, &n2), &n3i), &xi)| {
                non_additive_hs_energy_density(p, &d_hs_ij, &d_hs_add_ij, &rho0, n2, n3i, xi)
            })
            .collect::<Array1<N>>()
            .into_shape(n2.raw_dim())
            .unwrap())
    }
}

pub fn non_additive_hs_energy_density<S, N: DualNum<f64> + Copy + ScalarOperand>(
    parameters: &SaftVRQMieParameters,
    d_hs_ij: &Array2<N>,
    d_hs_add_ij: &Array2<N>,
    rho0: &ArrayBase<S, Ix1>,
    n2: N,
    n3i: N,
    xi: N,
) -> N
where
    S: Data<Elem = N>,
{
    // auxiliary variables
    let n = rho0.len();
    let p = parameters;
    let d = Array1::from_shape_fn(n, |i| d_hs_ij[[i, i]]);
    let g_hs_ij = Array2::from_shape_fn((n, n), |(i, j)| {
        let mu = d[i] * d[j] / (d[i] + d[j]);
        n3i + mu * n2 * xi * n3i.powi(2) / 2.0 + (mu * n2 * xi).powi(2) * n3i.powi(3) / 18.0
    });

    // segment densities
    let rho0_s = Array1::from_shape_fn(n, |i| -> N { rho0[i] * p.m[i] });

    Array2::from_shape_fn((n, n), |(i, j)| {
        -rho0_s[i]
            * rho0_s[j]
            * d_hs_add_ij[[i, j]].powi(2)
            * g_hs_ij[[i, j]]
            * (d_hs_add_ij[[i, j]] - d_hs_ij[[i, j]])
            * 2.0
            * PI
    })
    .sum()
}

impl fmt::Display for NonAddHardSphereFunctional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Non-additive hard-sphere functional")
    }
}
