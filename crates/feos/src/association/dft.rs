use super::*;
use feos_core::FeosResult;
use feos_core::parameter::GenericParameters;
use feos_dft::{FunctionalContribution, WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix1, Slice};
use num_dual::DualNum;
use std::f64::consts::PI;
use std::ops::MulAssign;

pub const N0_CUTOFF: f64 = 1e-9;

impl Association {
    /// Association strength for functional of Yu and Wu with xi parameter for inhomogeneity.
    ///
    /// Uses the contact value of hard-sphere pair correlation function and model-specific
    /// implementations for the bonding volume.
    #[expect(clippy::too_many_arguments)]
    fn yu_wu_association_strength<A: AssociationStrength, D: DualNum<f64> + Copy>(
        &self,
        parameters: &AssociationParameters<A::Record>,
        model: &A,
        temperature: D,
        diameter: &DVector<D>,
        n2: D,
        n3i: D,
        xi: D,
    ) -> [DMatrix<D>; 2] {
        let p = parameters;

        let mut delta_ab = DMatrix::zeros(p.sites_a.len(), p.sites_b.len());
        for b in &p.binary_ab {
            let [i, j] = [b.id1, b.id2];
            let di = diameter[p.sites_a[i].assoc_comp];
            let dj = diameter[p.sites_b[j].assoc_comp];
            let k = di * dj / (di + dj) * (n2 * n3i);
            delta_ab[(i, j)] = n3i
                * (k * xi * (k / 18.0 + 0.5) + 1.0)
                * model.association_strength_ij(
                    temperature,
                    p.sites_a[i].assoc_comp,
                    p.sites_b[j].assoc_comp,
                    &b.model_record,
                )
        }
        let mut delta_cc = DMatrix::zeros(p.sites_c.len(), p.sites_c.len());
        for b in &p.binary_cc {
            let [i, j] = [b.id1, b.id2];
            let di = diameter[p.sites_c[i].assoc_comp];
            let dj = diameter[p.sites_c[j].assoc_comp];
            let k = di * dj / (di + dj) * (n2 * n3i);
            delta_cc[(i, j)] = n3i
                * (k * xi * (k / 18.0 + 0.5) + 1.0)
                * model.association_strength_ij(
                    temperature,
                    p.sites_c[i].assoc_comp,
                    p.sites_c[j].assoc_comp,
                    &b.model_record,
                )
        }
        [delta_ab, delta_cc]
    }
}

/// Implementation of the association Helmholtz energy functional of Yu and Wu.
///
/// [Yang-Xin Yu and Jianzhong Wu (2002)](https://aip.scitation.org/doi/abs/10.1063/1.1463435)
///
/// # Note
///  
/// Uses the contact value of the hard-sphere pair correlation function and model-specific
/// implementations for the bonding volume.
pub struct YuWuAssociationFunctional<'a, A: AssociationStrength> {
    model: &'a A,
    association_parameters: &'a AssociationParameters<A::Record>,
    association: Association,
}

impl<'a, A: AssociationStrength> YuWuAssociationFunctional<'a, A> {
    pub fn new<P, B, Bo, C, Data>(
        model: &'a A,
        parameters: &'a GenericParameters<P, B, A::Record, Bo, C, Data>,
        association: Option<Association>,
    ) -> Option<Self> {
        association.map(|a| Self {
            model,
            association_parameters: &parameters.association,
            association: a,
        })
    }
}

impl<'a, A: AssociationStrength + Sync + Send> FunctionalContribution
    for YuWuAssociationFunctional<'a, A>
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

impl<'a, A: AssociationStrength> YuWuAssociationFunctional<'a, A> {
    pub fn _helmholtz_energy_density<N: DualNum<f64> + Copy, S: Data<Elem = N>>(
        &self,
        temperature: N,
        rho0: &Array2<N>,
        n2: &ArrayBase<S, Ix1>,
        n3i: &Array1<N>,
        xi: &Array1<N>,
    ) -> FeosResult<Array1<N>> {
        let a = &self.association_parameters;
        let t = temperature;

        let d = self.model.hs_diameter(t);

        match (
            a.sites_a.len() * a.sites_b.len(),
            a.sites_c.len(),
            self.association.force_cross_association,
        ) {
            (0, 0, _) => Ok(Array1::zeros(n3i.len())),
            (1, 0, false) => {
                let params = a.binary_ab.first().map(|r| &r.model_record);
                Ok(self.helmholtz_energy_density_ab_analytic(params, t, rho0, &d, n2, n3i, xi))
            }
            (0, 1, false) => {
                let params = a.binary_cc.first().map(|r| &r.model_record);
                Ok(self.helmholtz_energy_density_cc_analytic(params, t, rho0, &d, n2, n3i, xi))
            }
            (1, 1, false) => {
                let params_ab = a.binary_ab.first().map(|r| &r.model_record);
                let params_cc = a.binary_cc.first().map(|r| &r.model_record);
                Ok(
                    self.helmholtz_energy_density_ab_analytic(params_ab, t, rho0, &d, n2, n3i, xi)
                        + self.helmholtz_energy_density_cc_analytic(
                            params_cc, t, rho0, &d, n2, n3i, xi,
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
                        let [delta_ab, delta_cc] = self.association.yu_wu_association_strength(
                            self.association_parameters,
                            self.model,
                            t,
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

    #[expect(clippy::too_many_arguments)]
    fn helmholtz_energy_density_ab_analytic<N: DualNum<f64> + Copy, S: Data<Elem = N>>(
        &self,
        parameters: Option<&A::Record>,
        temperature: N,
        rho0: &Array2<N>,
        diameter: &DVector<N>,
        n2: &ArrayBase<S, Ix1>,
        n3i: &Array1<N>,
        xi: &Array1<N>,
    ) -> Array1<N> {
        let a = &self.association_parameters;
        let Some(par) = parameters else {
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
            * self.model.association_strength_ij(temperature, 0, 0, par);

        // no cross association, two association sites
        let aux = &delta * (&rhob - &rhoa) + 1.0;
        let xa = ((&aux * &aux + &delta * &rhoa * 4.0).map(N::sqrt) + &aux).map(N::recip) * 2.0;
        let aux = -aux + 2.0;
        let xb = ((&aux * &aux + delta * &rhob * 4.0).map(N::sqrt) + aux).map(N::recip) * 2.0;

        let f = |x: N| x.ln() - x * 0.5 + 0.5;
        rhoa * xa.mapv(f) + rhob * xb.mapv(f)
    }

    #[expect(clippy::too_many_arguments)]
    fn helmholtz_energy_density_cc_analytic<N: DualNum<f64> + Copy, S: Data<Elem = N>>(
        &self,
        parameters: Option<&A::Record>,
        temperature: N,
        rho0: &Array2<N>,
        diameter: &DVector<N>,
        n2: &ArrayBase<S, Ix1>,
        n3i: &Array1<N>,
        xi: &Array1<N>,
    ) -> Array1<N> {
        let a = &self.association_parameters;
        let Some(par) = parameters else {
            return Array1::zeros(xi.len());
        };

        // site densities
        let i = a.sites_c[0].assoc_comp;
        let rhoc = &rho0.index_axis(Axis(0), i) * a.sites_c[0].n;

        // association strength
        let di = diameter[i];
        let k = n2 * n3i * (di * 0.5);
        let delta = (((&k / 18.0 + 0.5) * &k * xi + 1.0) * n3i)
            * self.model.association_strength_ij(temperature, 0, 0, par);

        // no cross association, two association sites
        let xc = ((delta * 4.0 * &rhoc + 1.0).map(N::sqrt) + 1.0).map(N::recip) * 2.0;

        let f = |x: N| x.ln() - x * 0.5 + 0.5;
        rhoc * xc.mapv(f)
    }
}
