use super::*;
use crate::HardSphereProperties;
use feos_core::EosResult;
use feos_dft::{
    FunctionalContributionDual, WeightFunction, WeightFunctionInfo, WeightFunctionShape,
};
use num_dual::DualNum;
use std::f64::consts::PI;
use std::ops::MulAssign;

pub const N0_CUTOFF: f64 = 1e-9;

impl<N, P> FunctionalContributionDual<N> for Association<P>
where
    N: DualNum<f64> + ScalarOperand,
    P: HardSphereProperties,
{
    fn weight_functions(&self, temperature: N) -> WeightFunctionInfo<N> {
        let p = &self.parameters;
        let r = p.hs_diameter(temperature) * 0.5;
        let [_, _, _, c3] = p.geometry_coefficients(temperature);
        WeightFunctionInfo::new(p.component_index().into_owned(), false)
            .add(
                WeightFunction::new_scaled(r.clone(), WeightFunctionShape::Delta),
                false,
            )
            .add(
                WeightFunction {
                    prefactor: c3.clone(),
                    kernel_radius: r.clone(),
                    shape: WeightFunctionShape::DeltaVec,
                },
                false,
            )
            .add(
                WeightFunction {
                    prefactor: c3,
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

        // number of segments
        let n = self.association_parameters.component_index.len();

        // number of associating segments
        let nassoc = self.association_parameters.assoc_comp.len();

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

        // calculate rho0 (only associating segments)
        let [_, _, c2, _] = p.geometry_coefficients(temperature);
        let diameter = p.hs_diameter(temperature);
        let mut n2i = n0i.to_owned();
        for (i, mut n2i) in n2i.outer_iter_mut().enumerate() {
            n2i.mul_assign(diameter[i].powi(2) * c2[i] * PI);
        }
        let mut rho0: Array2<N> = (n2vi
            .iter()
            .fold(Array::zeros(n0i.raw_dim()), |acc, n2vi| acc + n2vi * n2vi)
            / -(&n2i * &n2i)
            + 1.0)
            * n0i;
        rho0.iter_mut().zip(&n0i).for_each(|(rho0, &n0i)| {
            if n0i.re() < N0_CUTOFF {
                *rho0 = n0i;
            }
        });
        let rho0 = Array2::from_shape_fn((nassoc, n3.len()), |(i, j)| {
            rho0[(self.association_parameters.assoc_comp[i], j)]
        });

        // calculate xi
        let n2v: Vec<_> = n2vi.iter().map(|n2vi| n2vi.sum_axis(Axis(0))).collect();
        let n2 = n2i.sum_axis(Axis(0));
        let mut xi = n2v
            .iter()
            .fold(Array::zeros(n2.raw_dim()), |acc, n2v| acc + n2v * n2v)
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
        if nassoc == 1 {
            // association strength
            let k = &n2 * &n3i * diameter[self.association_parameters.assoc_comp[0]] * 0.5;
            let deltarho = (((&k / 18.0 + 0.5) * &k * xi + 1.0) * n3i)
                * ((temperature.recip() * self.association_parameters.epsilon_k_aibj[(0, 0)])
                    .exp_m1()
                    * self.association_parameters.sigma3_kappa_aibj[(0, 0)])
                * rho0.index_axis(Axis(0), 0);

            let na = self.association_parameters.na[0];
            let nb = self.association_parameters.nb[0];
            let f = |x: N| x.ln() - x * 0.5 + 0.5;
            if nb > 0.0 {
                // no cross association, two association sites
                let xa = deltarho.mapv(|d| Self::assoc_site_frac_ab(d, na, nb));
                let xb = (&xa - 1.0) * (na / nb) + 1.0;
                Ok((xa.mapv(f) * na + xb.mapv(f) * nb) * rho0.index_axis(Axis(0), 0))
            } else {
                // no cross association, one association site
                let xa = deltarho.mapv(|d| Self::assoc_site_frac_a(d, na));

                Ok(xa.mapv(f) * na * rho0.index_axis(Axis(0), 0))
            }
        } else {
            let mut x: Array1<f64> = Array::from_elem(2 * nassoc, 0.2);
            Ok(rho0
                .view()
                .into_shape([nassoc, rho0.len() / nassoc])
                .unwrap()
                .axis_iter(Axis(1))
                .zip(n2.iter())
                .zip(n3i.iter())
                .zip(xi.iter())
                .map(|(((rho0, &n2), &n3i), &xi)| {
                    self.helmholtz_energy_density_cross_association(
                        temperature,
                        &rho0,
                        &diameter,
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
