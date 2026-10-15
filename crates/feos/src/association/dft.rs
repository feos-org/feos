use super::*;
use feos_dft::{AssociationSites, WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use ndarray::{Array1, Array2, ArrayView2, Axis, Slice, Zip};
use num_dual::DualNum;
use std::f64::consts::PI;

pub const N0_CUTOFF: f64 = 1e-9;

/// Versions of the association functional (see [SaftAssociationFunctional]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AssociationVersion {
    /// Inhomogeneous TPT1 with local site densities and a contact shell as association kernel
    /// ([Bymaster and Chapman 2010](https://doi.org/10.1021/jp102964e)).
    ISaft,
    /// Site densities and contact values from FMT weighted densities
    /// ([Yu and Wu 2002](https://aip.scitation.org/doi/abs/10.1063/1.1463435)).
    YuWu,
    /// Bulk association at weighted densities averaged over a sphere with the segment diameter
    /// ([Camacho Vergara et al. 2020](https://doi.org/10.1080/00268976.2020.1725668)).
    Wda,
}

/// Association functional (Michelsen's Q function) for all models that implement
/// [AssociationStrength].
pub struct SaftAssociationFunctional<'a, A: AssociationStrength> {
    model: &'a A,
    parameters: &'a AssociationParameters<A::Record>,
    version: AssociationVersion,
}

impl<'a, A: AssociationStrength> SaftAssociationFunctional<'a, A> {
    pub fn new(
        model: &'a A,
        parameters: &'a AssociationParameters<A::Record>,
        version: AssociationVersion,
    ) -> Self {
        Self {
            model,
            parameters,
            version,
        }
    }

    /// Segment indices (i, j) of every binary association record (AB, then CC).
    fn records(&self) -> impl Iterator<Item = (usize, usize, &A::Record)> {
        let p = self.parameters;
        let ab = p.binary_ab.iter().map(|b| {
            let [i, j] = [p.sites_a[b.id1].assoc_comp, p.sites_b[b.id2].assoc_comp];
            (i, j, &b.model_record)
        });
        let cc = p.binary_cc.iter().map(|b| {
            let [i, j] = [p.sites_c[b.id1].assoc_comp, p.sites_c[b.id2].assoc_comp];
            (i, j, &b.model_record)
        });
        ab.chain(cc)
    }

    /// Contact values of the hard-sphere pair correlation function for every record
    /// (records x grid) from $n_2$, $n_3$, and the inhomogeneity factor $\xi$ (Yu and Wu).
    fn bmcsl_contact_values<N: DualNum<Primitive = f64> + Copy>(
        &self,
        diameter: &DVector<N>,
        n2: &Array1<N>,
        n3: &Array1<N>,
        xi: &Array1<N>,
    ) -> Array2<N> {
        let records: Vec<_> = self.records().map(|(i, j, _)| (i, j)).collect();
        let mut y = Array2::zeros((records.len(), n2.len()));
        for (mut y, &(i, j)) in y.outer_iter_mut().zip(&records) {
            let (di, dj) = (diameter[i], diameter[j]);
            let dij = di * dj / (di + dj);
            Zip::from(&mut y)
                .and(n2)
                .and(n3)
                .and(xi)
                .for_each(|y, &n2, &n3, &xi| {
                    let n3i = (-n3 + 1.0).recip();
                    let k = dij * n2 * n3i;
                    *y = n3i * (k * xi * (k / 18.0 + 0.5) + 1.0);
                });
        }
        y
    }
}

impl<A: AssociationStrength + Sync + Send> feos_dft::AssociationFunctional
    for SaftAssociationFunctional<'_, A>
where
    A::Record: Sync + Send,
{
    fn sites(&self) -> AssociationSites<'_> {
        AssociationSites::new(self.parameters)
    }

    fn weight_functions<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
    ) -> WeightFunctionInfo<N> {
        let p = self.model;
        let d = p.hs_diameter(temperature);
        let segments = p.component_index().len();
        match self.version {
            AssociationVersion::YuWu => {
                let r = d * N::from(0.5);
                let [_, _, _, c3] = p.geometry_coefficients(temperature);
                WeightFunctionInfo::new(segments, false)
                    // n0 per segment
                    .add(
                        WeightFunction::new_scaled(r.clone(), WeightFunctionShape::Delta),
                        false,
                    )
                    // n2v per segment
                    .add(
                        WeightFunction {
                            prefactor: c3.clone(),
                            kernel_radius: r.clone(),
                            shape: WeightFunctionShape::DeltaVec,
                        },
                        false,
                    )
                    // n3
                    .add(
                        WeightFunction {
                            prefactor: c3,
                            kernel_radius: r,
                            shape: WeightFunctionShape::Theta,
                        },
                        true,
                    )
            }
            AssociationVersion::Wda => WeightFunctionInfo::new(segments, false)
                // rhobar per segment
                .add(
                    WeightFunction::new_scaled(d, WeightFunctionShape::Theta),
                    false,
                ),
            AssociationVersion::ISaft => {
                let [_, _, c2, c3] = p.geometry_coefficients(temperature);
                // local_density
                WeightFunctionInfo::new(segments, true)
                    // n2
                    .add(
                        WeightFunction {
                            prefactor: c2.zip_map(&d, |c, d| c * 0.75 / d),
                            kernel_radius: d.clone(),
                            shape: WeightFunctionShape::Theta,
                        },
                        true,
                    )
                    // n3
                    .add(
                        WeightFunction {
                            prefactor: c3.map(|c| c * 0.125),
                            kernel_radius: d,
                            shape: WeightFunctionShape::Theta,
                        },
                        true,
                    )
            }
        }
    }

    fn contact_distances<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
    ) -> Option<Array1<N>> {
        (self.version == AssociationVersion::ISaft).then(|| {
            let d = self.model.hs_diameter(temperature);
            self.records()
                .map(|(i, j, _)| (d[i] + d[j]) * 0.5)
                .collect()
        })
    }

    fn contact_values<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> (Array2<N>, Array2<N>) {
        let p = self.model;
        let wd = weighted_densities;
        let n = p.component_index().len();
        let d = p.hs_diameter(temperature);
        let [_, _, c2, c3] = p.geometry_coefficients(temperature);
        match self.version {
            AssociationVersion::YuWu => {
                let dim = (wd.shape()[0] - 1) / n - 1;
                let n0i = wd.slice_axis(Axis(0), Slice::from(0..n));
                let n2vi: Vec<_> = (0..dim)
                    .map(|k| wd.slice_axis(Axis(0), Slice::from(n * (k + 1)..n * (k + 2))))
                    .collect();
                let n3 = wd.index_axis(Axis(0), n * (dim + 1)).to_owned();

                // segment-wise inhomogeneity factors and site densities (|n2v| > n2 and n0 < 0
                // only from FFT ringing; xi is bound to [0, 1] and the site densities have to be
                // non-negative)
                let mut n2i = n0i.to_owned();
                for (mut n2i, i) in n2i.outer_iter_mut().zip(0..n) {
                    n2i.mapv_inplace(|n0| n0 * d[i].powi(2) * c2[i] * PI);
                }
                let n2vi_sq: Array2<N> = n2vi
                    .iter()
                    .fold(Array2::zeros(n0i.raw_dim()), |acc, v| acc + &(v * v));
                let rho =
                    Zip::from(&n0i)
                        .and(&n2i)
                        .and(&n2vi_sq)
                        .map_collect(|&n0, &n2, &n2v_sq| {
                            if n0.re() < N0_CUTOFF {
                                clip0(n0)
                            } else {
                                n0 * clip01(-n2v_sq / (n2 * n2) + 1.0)
                            }
                        });

                // total inhomogeneity factor
                let n2 = n2i.sum_axis(Axis(0));
                let n2v_sq: Array1<N> = n2vi
                    .iter()
                    .map(|v| v.sum_axis(Axis(0)))
                    .fold(Array1::zeros(n2.raw_dim()), |acc, v| acc + &v * &v);
                let n0 = n0i.sum_axis(Axis(0));
                let xi = Zip::from(&n0)
                    .and(&n2)
                    .and(&n2v_sq)
                    .map_collect(|&n0, &n2, &n2v_sq| {
                        if n0.re() < N0_CUTOFF {
                            N::one()
                        } else {
                            clip01(-n2v_sq / (n2 * n2) + 1.0)
                        }
                    });
                let y = self.bmcsl_contact_values(&d, &n2, &n3, &xi);
                (rho, y)
            }
            AssociationVersion::Wda => {
                let mut n2 = Array1::zeros(wd.shape()[1]);
                let mut n3 = Array1::zeros(wd.shape()[1]);
                for (rhobar, i) in wd.outer_iter().zip(0..n) {
                    n2 += &rhobar.mapv(|r| r * d[i].powi(2) * c2[i] * PI);
                    n3 += &rhobar.mapv(|r| r * d[i].powi(3) * c3[i] * (PI / 6.0));
                }
                let xi = Array1::ones(n2.raw_dim());
                let y = self.bmcsl_contact_values(&d, &n2, &n3, &xi);
                // rhobar < 0 only from FFT ringing; the site densities have to be non-negative
                (wd.mapv(clip0), y)
            }
            AssociationVersion::ISaft => {
                let rho = wd.slice_axis(Axis(0), Slice::from(0..n)).to_owned();
                let n2 = wd.index_axis(Axis(0), n).to_owned();
                let n3 = wd.index_axis(Axis(0), n + 1).to_owned();
                let xi = Array1::ones(n2.raw_dim());
                let y = self.bmcsl_contact_values(&d, &n2, &n3, &xi);
                (rho, y)
            }
        }
    }

    fn association_strength<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
    ) -> Array1<N> {
        self.records()
            .map(|(i, j, record)| {
                self.model
                    .association_strength_ij(temperature, i, j, record)
            })
            .collect()
    }
}

/// Clip a value to non-negative values.
fn clip0<N: DualNum<Primitive = f64> + Copy>(x: N) -> N {
    if x.re() < 0.0 { N::zero() } else { x }
}

/// Clip a value to the interval [0, 1].
fn clip01<N: DualNum<Primitive = f64> + Copy>(x: N) -> N {
    if x.re() < 0.0 {
        N::zero()
    } else if x.re() > 1.0 {
        N::one()
    } else {
        x
    }
}
