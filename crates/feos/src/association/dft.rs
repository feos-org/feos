use super::*;
use feos_core::FeosResult;
use feos_dft::{FunctionalContribution, WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1, Slice};
use num_dual::DualNum;
use std::f64::consts::PI;
use std::ops::MulAssign;

pub const N0_CUTOFF: f64 = 1e-9;

/// Implementation of the SAFT association Helmholtz energy functional.
pub struct AssociationFunctional<'a, A: AssociationStrength> {
    model: &'a A,
    association_parameters: &'a AssociationParameters<A::Record>,
    association: &'a Association<A>,
}

impl<'a, A: AssociationStrength> AssociationFunctional<'a, A> {
    pub fn new<P, B, Bo, C, Data>(
        model: &'a A,
        parameters: &'a GenericParameters<P, B, A::Record, Bo, C, Data>,
        association: &'a Option<Association<A>>,
    ) -> Option<Self> {
        association.as_ref().map(|a| Self {
            model,
            association_parameters: &parameters.association,
            association: a,
        })
    }
}

impl<'a, A: AssociationStrength + Sync + Send> FunctionalContribution
    for AssociationFunctional<'a, A>
where
    A::Record: Sync + Send,
{
    fn name(&self) -> &'static str {
        "Association"
    }

    fn weight_functions<N: DualNum<f64> + Copy>(&self, temperature: N) -> WeightFunctionInfo<N> {
        let p = self.model;
        let r = p.hs_diameter(temperature) * N::from(0.5);
        let [_, _, _, c3] = p.geometry_coefficients(temperature);
        WeightFunctionInfo::new(p.component_index().into_owned().into(), false)
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

    fn helmholtz_energy_density<N: DualNum<f64> + Copy>(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> FeosResult<Array1<N>> {
        // number of segments
        let n = self.association_parameters.component_index.len();

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
        let [_, _, c2, _] = self.model.geometry_coefficients(temperature);
        let diameter = self.model.hs_diameter(temperature);
        let mut n2i = n0i.to_owned();
        for (i, mut n2i) in n2i.outer_iter_mut().enumerate() {
            n2i.mul_assign(diameter[i].powi(2) * c2[i] * PI);
        }
        let mut rho0: Array2<N> = (n2vi
            .iter()
            .fold(Array2::zeros(n0i.raw_dim()), |acc, n2vi| acc + n2vi * n2vi)
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
            .fold(Array1::zeros(n2.raw_dim()), |acc, n2v| acc + n2v * n2v)
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

        self._helmholtz_energy_density(temperature, &rho0, &n2, &n3i, &xi)
    }
}

impl<'a, A: AssociationStrength> AssociationFunctional<'a, A> {
    pub fn _helmholtz_energy_density<N: DualNum<f64> + Copy, S: Data<Elem = N>>(
        &self,
        temperature: N,
        rho0: &Array2<N>,
        n2: &ArrayBase<S, Ix1>,
        n3i: &Array1<N>,
        xi: &Array1<N>,
    ) -> FeosResult<Array1<N>> {
        let a = &self.association_parameters;

        let d = self.model.hs_diameter(temperature);

        match (
            a.sites_a.len() * a.sites_b.len(),
            a.sites_c.len(),
            self.association.force_cross_association,
        ) {
            (0, 0, _) => Ok(Array1::zeros(n3i.len())),
            (1, 0, false) => {
                Ok(self.helmholtz_energy_density_ab_analytic(temperature, rho0, &d, n2, n3i, xi))
            }
            (0, 1, false) => {
                Ok(self.helmholtz_energy_density_cc_analytic(temperature, rho0, &d, n2, n3i, xi))
            }
            (1, 1, false) => {
                Ok(
                    self.helmholtz_energy_density_ab_analytic(temperature, rho0, &d, n2, n3i, xi)
                        + self.helmholtz_energy_density_cc_analytic(
                            temperature,
                            rho0,
                            &d,
                            n2,
                            n3i,
                            xi,
                        ),
                )
            }
            _ => {
                let mut x = DVector::from_element(a.sites_a.len() + a.sites_b.len(), 0.2);
                let (assoc_comp_ab, n_ab): (Vec<_>, Vec<_>) = a
                    .sites_a
                    .iter()
                    .chain(a.sites_b.iter())
                    .map(|s| (s.assoc_comp, s.n))
                    .unzip();
                let rhoab = Array2::from_shape_fn((x.len(), n3i.len()), |(i, j)| {
                    rho0[(assoc_comp_ab[i], j)] * n_ab[i]
                });
                rhoab
                    .axis_iter(Axis(1))
                    .zip(n2.iter())
                    .zip(n3i.iter())
                    .zip(xi.iter())
                    .map(|(((rho, &n2), &n3i), &xi)| {
                        let [delta_ab, delta_cc] = self.association.association_strength(
                            self.association_parameters,
                            self.model,
                            temperature,
                            &d,
                            n2,
                            n3i,
                            xi,
                        );
                        let rho = DVector::from(rho.to_vec());
                        self.association.helmholtz_energy_density_cross_association(
                            &rho,
                            &delta_ab,
                            &delta_cc,
                            Some(&mut x),
                        )
                    })
                    .collect()
            }
        }
    }

    fn helmholtz_energy_density_ab_analytic<N: DualNum<f64> + Copy, S: Data<Elem = N>>(
        &self,
        temperature: N,
        rho0: &Array2<N>,
        diameter: &DVector<N>,
        n2: &ArrayBase<S, Ix1>,
        n3i: &Array1<N>,
        xi: &Array1<N>,
    ) -> Array1<N> {
        let a = &self.association_parameters;
        let Some(par) = &self.association.parameters_ab[(0, 0)] else {
            return Array1::zeros(xi.len());
        };

        // site densities
        let i = a.sites_a[0].assoc_comp;
        let j = a.sites_b[0].assoc_comp;
        let rhoa = &rho0.index_axis(Axis(0), i) * a.sites_a[0].n;
        let rhob = &rho0.index_axis(Axis(0), j) * a.sites_b[0].n;

        // association strength
        let di = diameter[i];
        let dj = diameter[j];
        let k = n2 * n3i * (di * dj / (di + dj));
        let delta = (((&k / 18.0 + 0.5) * &k * xi + 1.0) * n3i)
            * self.model.association_strength(temperature, 0, 0, par);

        // no cross association, two association sites
        let aux = &delta * (&rhob - &rhoa) + 1.0;
        let xa = ((&aux * &aux + &delta * &rhoa * 4.0).map(N::sqrt) + &aux).map(N::recip) * 2.0;
        let aux = -aux + 2.0;
        let xb = ((&aux * &aux + delta * &rhob * 4.0).map(N::sqrt) + aux).map(N::recip) * 2.0;

        let f = |x: N| x.ln() - x * 0.5 + 0.5;
        rhoa * xa.mapv(f) + rhob * xb.mapv(f)
    }

    fn helmholtz_energy_density_cc_analytic<N: DualNum<f64> + Copy, S: Data<Elem = N>>(
        &self,
        temperature: N,
        rho0: &Array2<N>,
        diameter: &DVector<N>,
        n2: &ArrayBase<S, Ix1>,
        n3i: &Array1<N>,
        xi: &Array1<N>,
    ) -> Array1<N> {
        let a = &self.association_parameters;
        let Some(par) = &self.association.parameters_cc[(0, 0)] else {
            return Array1::zeros(xi.len());
        };

        // site densities
        let i = a.sites_c[0].assoc_comp;
        let rhoc = &rho0.index_axis(Axis(0), i) * a.sites_c[0].n;

        // association strength
        let di = diameter[i];
        let k = n2 * n3i * (di * 0.5);
        let delta = (((&k / 18.0 + 0.5) * &k * xi + 1.0) * n3i)
            * self.model.association_strength(temperature, 0, 0, par);

        // no cross association, two association sites
        let xc = ((delta * 4.0 * &rhoc + 1.0).map(N::sqrt) + 1.0).map(N::recip) * 2.0;

        let f = |x: N| x.ln() - x * 0.5 + 0.5;
        rhoc * xc.mapv(f)
    }
}
