use super::PcSaftParameters;
use crate::pcsaft::eos::association::{
    assoc_site_frac_a, assoc_site_frac_ab, helmholtz_energy_density_cross_association,
};
use feos_core::EosError;
use feos_dft::{
    FunctionalContributionDual, WeightFunction, WeightFunctionInfo, WeightFunctionShape,
};
use feos_saft::HardSphereProperties;
use ndarray::*;
use num_dual::DualNum;
use std::f64::consts::PI;
use std::fmt;
use std::rc::Rc;

pub const N0_CUTOFF: f64 = 1e-9;

#[derive(Clone)]
pub struct AssociationFunctional {
    parameters: Rc<PcSaftParameters>,
    max_iter: usize,
    tol: f64,
}

impl AssociationFunctional {
    pub fn new(parameters: Rc<PcSaftParameters>, max_iter: usize, tol: f64) -> Self {
        Self {
            parameters,
            max_iter,
            tol,
        }
    }
}

impl<N> FunctionalContributionDual<N> for AssociationFunctional
where
    N: DualNum<f64> + ScalarOperand,
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
    ) -> Result<Array1<N>, EosError> {
        let p = &self.parameters;
        // number of components
        let n = p.m.len();
        // number of dimensions
        let dim = (weighted_densities.shape()[0] - 1) / n - 1;

        // weighted densities
        let n0i = weighted_densities.slice_axis(Axis(0), Slice::new(0, Some(n as isize), 1));
        let n2vi: Vec<_> = (0..dim)
            .into_iter()
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

        // only one associating component
        if p.nassoc == 1 {
            // association strength
            let a = p.assoc_comp[0];
            let k = &n2 * &n3i * r[a];
            let deltarho = (((&k / 18.0 + 0.5) * &k * xi + 1.0) * n3i)
                * ((temperature.recip() * p.epsilon_k_aibj[(a, a)]).exp_m1()
                    * (p.sigma[a].powi(3) * p.kappa_aibj[(a, a)]))
                * rho0.index_axis(Axis(0), a);

            let na = p.na[a];
            let nb = p.nb[a];
            let f = |x: N| x.ln() - x * 0.5 + 0.5;
            if nb > 0.0 {
                // no cross association, two association sites
                let xa = deltarho.mapv(|d| assoc_site_frac_ab(d, na, nb));
                let xb = (&xa - 1.0) * (na / nb) + 1.0;
                Ok((xa.mapv(f) * na + xb.mapv(f) * nb) * rho0.index_axis(Axis(0), a))
            } else {
                // no cross association, one association site
                let xa = deltarho.mapv(|d| assoc_site_frac_a(d, na));

                Ok(xa.mapv(f) * na * rho0.index_axis(Axis(0), a))
            }
        } else {
            let mut x: Array1<f64> = Array::from_elem(2 * p.nassoc, 0.2);
            Ok(rho0
                .view()
                .into_shape([n, rho0.len() / n])
                .unwrap()
                .axis_iter(Axis(1))
                .zip(n2.iter())
                .zip(n3i.iter())
                .zip(xi.iter())
                .map(|(((rho0, &n2), &n3i), &xi)| {
                    helmholtz_energy_density_cross_association(
                        p,
                        temperature,
                        &rho0,
                        &r,
                        n2,
                        n3i,
                        xi,
                        self.max_iter,
                        self.tol,
                        Some(&mut x),
                    )
                })
                .collect::<Result<Array1<N>, _>>()?
                .into_shape(n2.raw_dim())
                .unwrap())
        }
    }
}

impl fmt::Display for AssociationFunctional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Association functional")
    }
}
