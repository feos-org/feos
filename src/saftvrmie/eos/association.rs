//! Generic implementation of the SAFT association contribution
//! that can be used across models.
use crate::hard_sphere::HardSphereProperties;
use feos_core::{EosError, EosResult, StateHD};
use ndarray::*;
use num_dual::linalg::{norm, LU};
use num_dual::*;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
struct AssociationSite {
    assoc_comp: usize,
    site_index: usize,
    n: f64,
    rc_ab: f64,
    epsilon_k_ab: f64,
}

impl AssociationSite {
    fn new(assoc_comp: usize, site_index: usize, n: f64, rc_ab: f64, epsilon_k_ab: f64) -> Self {
        Self {
            assoc_comp,
            site_index,
            n,
            rc_ab,
            epsilon_k_ab,
        }
    }
}

/// Pure component association parameters.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct AssociationRecord {
    /// Dimensionless association range parameter
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub rc_ab: f64,
    /// Association energy parameter in units of Kelvin
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub epsilon_k_ab: f64,
    /// \# of association sites of type A
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub na: f64,
    /// \# of association sites of type B
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub nb: f64,
    /// \# of association sites of type C
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub nc: f64,
}

impl AssociationRecord {
    pub fn new(rc_ab: f64, epsilon_k_ab: f64, na: f64, nb: f64, nc: f64) -> Self {
        Self {
            rc_ab,
            epsilon_k_ab,
            na,
            nb,
            nc,
        }
    }
}

impl fmt::Display for AssociationRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AssociationRecord(rc_ab={}", self.rc_ab)?;
        write!(f, ", epsilon_k_ab={}", self.epsilon_k_ab)?;
        if self.na > 0.0 {
            write!(f, ", na={}", self.na)?;
        }
        if self.nb > 0.0 {
            write!(f, ", nb={}", self.nb)?;
        }
        if self.nc > 0.0 {
            write!(f, ", nc={}", self.nc)?;
        }
        write!(f, ")")
    }
}

/// Binary association parameters.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct BinaryAssociationRecord {
    /// Dimensionless cross-association association range parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rc_ab: Option<f64>,
    /// Cross-association energy parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epsilon_k_ab: Option<f64>,
    /// Indices of sites that the record refers to.
    #[serde(skip_serializing_if = "is_default_site_indices")]
    #[serde(default)]
    pub site_indices: [usize; 2],
}

fn is_default_site_indices([i, j]: &[usize; 2]) -> bool {
    *i == 0 && *j == 0
}

impl BinaryAssociationRecord {
    pub fn new(
        rc_ab: Option<f64>,
        epsilon_k_ab: Option<f64>,
        site_indices: Option<[usize; 2]>,
    ) -> Self {
        Self {
            rc_ab,
            epsilon_k_ab,
            site_indices: site_indices.unwrap_or_default(),
        }
    }
}

/// Parameter set required for the SAFT association Helmoltz energy
/// contribution and functional.
#[derive(Clone)]
pub struct AssociationParameters {
    component_index: Array1<usize>,
    sites_a: Array1<AssociationSite>,
    sites_b: Array1<AssociationSite>,
    sites_c: Array1<AssociationSite>,
    pub rc_ab: Array2<f64>,
    pub rc_cc: Array2<f64>,
    pub rd_ab: Array2<f64>,
    pub rd_cc: Array2<f64>,
    pub epsilon_k_ab: Array2<f64>,
    pub epsilon_k_cc: Array2<f64>,
}

impl AssociationParameters {
    pub fn new(
        records: &[Vec<AssociationRecord>],
        sigma: &Array1<f64>,
        binary_records: &[((usize, usize), BinaryAssociationRecord)],
        component_index: Option<&Array1<usize>>,
    ) -> Self {
        let mut sites_a = Vec::new();
        let mut sites_b = Vec::new();
        let mut sites_c = Vec::new();

        for (i, record) in records.iter().enumerate() {
            for (s, site) in record.iter().enumerate() {
                if site.na > 0.0 {
                    sites_a.push(AssociationSite::new(
                        i,
                        s,
                        site.na,
                        site.rc_ab,
                        site.epsilon_k_ab,
                    ));
                }
                if site.nb > 0.0 {
                    sites_b.push(AssociationSite::new(
                        i,
                        s,
                        site.nb,
                        site.rc_ab,
                        site.epsilon_k_ab,
                    ));
                }
                if site.nc > 0.0 {
                    sites_c.push(AssociationSite::new(
                        i,
                        s,
                        site.nc,
                        site.rc_ab,
                        site.epsilon_k_ab,
                    ));
                }
            }
        }

        let indices_a: HashMap<_, _> = sites_a
            .iter()
            .enumerate()
            .map(|(i, site)| ((site.assoc_comp, site.site_index), i))
            .collect();

        let indices_b: HashMap<_, _> = sites_b
            .iter()
            .enumerate()
            .map(|(i, site)| ((site.assoc_comp, site.site_index), i))
            .collect();

        let indices_c: HashMap<_, _> = sites_c
            .iter()
            .enumerate()
            .map(|(i, site)| ((site.assoc_comp, site.site_index), i))
            .collect();

        // rc_ab and rc_cc are *dimensioned* distance parameters (i.e. multiplied by sigma, in Angstrom)
        let mut rc_ab = Array2::from_shape_fn([sites_a.len(), sites_b.len()], |(i, j)| {
            (sites_a[i].rc_ab * sigma[sites_a[i].assoc_comp]
                + sites_b[j].rc_ab * sigma[sites_b[j].assoc_comp])
                * 0.5
        });
        let mut rc_cc = Array2::from_shape_fn([sites_c.len(); 2], |(i, j)| {
            (sites_c[i].rc_ab * sigma[sites_c[i].assoc_comp]
                + sites_c[j].rc_ab * sigma[sites_c[j].assoc_comp])
                * 0.5
        });

        // r_d_AB is the distance between an association site and the segment centre.
        // It is fixed at 0.4 sigma, leading to 0.4 * 0.5 = 0.2 in the combining rule.
        let rd_ab = Array2::from_shape_fn([sites_a.len(), sites_b.len()], |(i, j)| {
            (sigma[sites_a[i].assoc_comp] + sigma[sites_b[j].assoc_comp]) * 0.2
        });
        let rd_cc = Array2::from_shape_fn([sites_c.len(); 2], |(i, j)| {
            (sigma[sites_c[i].assoc_comp] + sigma[sites_c[j].assoc_comp]) * 0.2
        });

        let mut epsilon_k_ab = Array2::from_shape_fn([sites_a.len(), sites_b.len()], |(i, j)| {
            (sites_a[i].epsilon_k_ab * sites_b[j].epsilon_k_ab).sqrt()
        });
        let mut epsilon_k_cc = Array2::from_shape_fn([sites_c.len(); 2], |(i, j)| {
            (sites_c[i].epsilon_k_ab * sites_c[j].epsilon_k_ab).sqrt()
        });

        for &((i, j), record) in binary_records.iter() {
            let [a, b] = record.site_indices;
            if let (Some(x), Some(y)) = (indices_a.get(&(i, a)), indices_b.get(&(j, b))) {
                if let Some(epsilon_k_aibj) = record.epsilon_k_ab {
                    epsilon_k_ab[[*x, *y]] = epsilon_k_aibj;
                }
                if let Some(rc_aibj) = record.rc_ab {
                    rc_ab[[*x, *y]] = rc_aibj;
                }
            }
            if let (Some(y), Some(x)) = (indices_b.get(&(i, a)), indices_a.get(&(j, b))) {
                if let Some(epsilon_k_aibj) = record.epsilon_k_ab {
                    epsilon_k_ab[[*x, *y]] = epsilon_k_aibj;
                }
                if let Some(rc_aibj) = record.rc_ab {
                    rc_ab[[*x, *y]] = rc_aibj;
                }
            }
            if let (Some(x), Some(y)) = (indices_c.get(&(i, a)), indices_c.get(&(j, b))) {
                if let Some(epsilon_k_aibj) = record.epsilon_k_ab {
                    epsilon_k_cc[[*x, *y]] = epsilon_k_aibj;
                    epsilon_k_cc[[*y, *x]] = epsilon_k_aibj;
                }
                if let Some(rc_aibj) = record.rc_ab {
                    rc_cc[[*x, *y]] = rc_aibj;
                    rc_cc[[*y, *x]] = rc_aibj;
                }
            }
        }

        Self {
            component_index: component_index
                .cloned()
                .unwrap_or_else(|| Array1::from_shape_fn(records.len(), |i| i)),
            sites_a: Array1::from_vec(sites_a),
            sites_b: Array1::from_vec(sites_b),
            sites_c: Array1::from_vec(sites_c),
            rc_ab,
            rc_cc,
            rd_ab,
            rd_cc,
            epsilon_k_ab,
            epsilon_k_cc,
        }
    }

    pub fn is_empty(&self) -> bool {
        (self.sites_a.is_empty() | self.sites_b.is_empty()) & self.sites_c.is_empty()
    }
}

/// Implementation of the SAFT association Helmholtz energy
/// contribution and functional.
pub struct Association<P> {
    parameters: Arc<P>,
    association_parameters: Arc<AssociationParameters>,
    max_iter: usize,
    tol: f64,
    force_cross_association: bool,
}

impl<P: HardSphereProperties> Association<P> {
    pub fn new(
        parameters: &Arc<P>,
        association_parameters: &AssociationParameters,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        Self {
            parameters: parameters.clone(),
            association_parameters: Arc::new(association_parameters.clone()),
            max_iter,
            tol,
            force_cross_association: false,
        }
    }

    #[allow(dead_code)]
    pub fn new_cross_association(
        parameters: &Arc<P>,
        association_parameters: &AssociationParameters,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        let mut res = Self::new(parameters, association_parameters, max_iter, tol);
        res.force_cross_association = true;
        res
    }

    fn association_strength<D: DualNum<f64> + Copy>(
        &self,
        temperature: D,
        diameter: &Array1<D>,
        n2: D,
        n3i: D,
        xi: D,
    ) -> [Array2<D>; 2] {
        let p = &self.association_parameters;
        let delta_ab = Array2::from_shape_fn([p.sites_a.len(), p.sites_b.len()], |(i, j)| {
            let di = diameter[p.sites_a[i].assoc_comp];
            let dj = diameter[p.sites_b[j].assoc_comp];
            let d = (di + dj) * 0.5;
            let k = di * dj / (di + dj) * (n2 * n3i);
            // temperature dependent association volume
            // rc and rd are dimensioned in units of Angstrom
            let rc = p.rc_ab[(i, j)];
            let rd = p.rd_ab[(i, j)];
            let v = d * d * PI * 4.0 / (72.0 * rd.powi(2))
                * ((d.recip() * (rc + 2.0 * rd)).ln()
                    * (6.0 * rc.powi(3) + 18.0 * rc.powi(2) * rd - 24.0 * rd.powi(3))
                    + (-d + rc + 2.0 * rd)
                        * (d.powi(2) + d * rc + 22.0 * rd.powi(2)
                            - 5.0 * rc * rd
                            - d * 7.0 * rd
                            - 8.0 * rc.powi(2)));
            n3i * (k * xi * (k / 18.0 + 0.5) + 1.0)
                * v
                * (temperature.recip() * p.epsilon_k_ab[(i, j)]).exp_m1()
        });
        let delta_cc = Array2::from_shape_fn([p.sites_c.len(); 2], |(i, j)| {
            let di = diameter[p.sites_c[i].assoc_comp];
            let dj = diameter[p.sites_c[j].assoc_comp];
            let d = (di + dj) * 0.5;
            let k = di * dj / (di + dj) * (n2 * n3i);
            // temperature dependent association volume
            // rc and rd are dimensioned in units of Angstrom
            let rc = p.rc_cc[(i, j)];
            let rd = p.rd_cc[(i, j)];
            let v = d * d * PI * 4.0 / (72.0 * rd.powi(2))
                * ((d.recip() * (rc + 2.0 * rd)).ln()
                    * (6.0 * rc.powi(3) + 18.0 * rc.powi(2) * rd - 24.0 * rd.powi(3))
                    + (-d + rc + 2.0 * rd)
                        * (d.powi(2) + d * rc + 22.0 * rd.powi(2)
                            - 5.0 * rc * rd
                            - d * 7.0 * rd
                            - 8.0 * rc.powi(2)));
            n3i * (k * xi * (k / 18.0 + 0.5) + 1.0)
                * v
                * (temperature.recip() * p.epsilon_k_ab[(i, j)]).exp_m1()
        });
        [delta_ab, delta_cc]
    }
}

impl<P: HardSphereProperties> Association<P> {
    #[inline]
    pub fn helmholtz_energy<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
        diameter: &Array1<D>,
    ) -> D {
        let p: &P = &self.parameters;
        let a = &self.association_parameters;

        // auxiliary variables
        let [zeta2, n3] = p.zeta(state.temperature, &state.partial_density, [2, 3]);
        let n2 = zeta2 * 6.0;
        let n3i = (-n3 + 1.0).recip();

        // association strength
        let [delta_ab, delta_cc] =
            self.association_strength(state.temperature, diameter, n2, n3i, D::one());

        match (
            a.sites_a.len() * a.sites_b.len(),
            a.sites_c.len(),
            self.force_cross_association,
        ) {
            (0, 0, _) => D::zero(),
            (1, 0, false) => self.helmholtz_energy_ab_analytic(state, delta_ab[(0, 0)]),
            (0, 1, false) => self.helmholtz_energy_cc_analytic(state, delta_cc[(0, 0)]),
            (1, 1, false) => {
                self.helmholtz_energy_ab_analytic(state, delta_ab[(0, 0)])
                    + self.helmholtz_energy_cc_analytic(state, delta_cc[(0, 0)])
            }
            _ => {
                // extract site densities of associating segments
                let rho: Array1<_> = a
                    .sites_a
                    .iter()
                    .chain(a.sites_b.iter())
                    .chain(a.sites_c.iter())
                    .map(|s| state.partial_density[a.component_index[s.assoc_comp]] * s.n)
                    .collect();

                // Helmholtz energy
                Self::helmholtz_energy_density_cross_association(
                    &rho,
                    &delta_ab,
                    &delta_cc,
                    self.max_iter,
                    self.tol,
                    None,
                )
                .unwrap_or_else(|_| D::from(f64::NAN))
                    * state.volume
            }
        }
    }
}

impl<P> fmt::Display for Association<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Association")
    }
}

impl<P: HardSphereProperties> Association<P> {
    fn helmholtz_energy_ab_analytic<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
        delta: D,
    ) -> D {
        let a = &self.association_parameters;

        // site densities
        let rhoa =
            state.partial_density[a.component_index[a.sites_a[0].assoc_comp]] * a.sites_a[0].n;
        let rhob =
            state.partial_density[a.component_index[a.sites_b[0].assoc_comp]] * a.sites_b[0].n;

        // fraction of non-bonded association sites
        let sqrt = ((delta * (rhoa - rhob) + 1.0).powi(2) + delta * rhob * 4.0).sqrt();
        let xa = (sqrt + (delta * (rhob - rhoa) + 1.0)).recip() * 2.0;
        let xb = (sqrt + (delta * (rhoa - rhob) + 1.0)).recip() * 2.0;

        (rhoa * (xa.ln() - xa * 0.5 + 0.5) + rhob * (xb.ln() - xb * 0.5 + 0.5)) * state.volume
    }

    fn helmholtz_energy_cc_analytic<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
        delta: D,
    ) -> D {
        let a = &self.association_parameters;

        // site density
        let rhoc =
            state.partial_density[a.component_index[a.sites_c[0].assoc_comp]] * a.sites_c[0].n;

        // fraction of non-bonded association sites
        let xc = ((delta * 4.0 * rhoc + 1.0).sqrt() + 1.0).recip() * 2.0;

        rhoc * (xc.ln() - xc * 0.5 + 0.5) * state.volume
    }

    #[allow(clippy::too_many_arguments)]
    fn helmholtz_energy_density_cross_association<D: DualNum<f64> + Copy, S: Data<Elem = D>>(
        rho: &ArrayBase<S, Ix1>,
        delta_ab: &Array2<D>,
        delta_cc: &Array2<D>,
        max_iter: usize,
        tol: f64,
        x0: Option<&mut Array1<f64>>,
    ) -> EosResult<D> {
        // check if density is close to 0
        if rho.sum().re() < f64::EPSILON {
            if let Some(x0) = x0 {
                x0.fill(1.0);
            }
            return Ok(D::zero());
        }

        // cross-association according to Michelsen2006
        // initialize monomer fraction
        let mut x = match &x0 {
            Some(x0) => (*x0).clone(),
            None => Array::from_elem(rho.len(), 0.2),
        };

        let delta_ab_re = delta_ab.map(D::re);
        let delta_cc_re = delta_cc.map(D::re);
        let rho_re = rho.map(D::re);
        for k in 0..max_iter {
            if Self::newton_step_cross_association(
                &mut x,
                &delta_ab_re,
                &delta_cc_re,
                &rho_re,
                tol,
            )? {
                break;
            }
            if k == max_iter - 1 {
                return Err(EosError::NotConverged("Cross association".into()));
            }
        }

        // calculate derivatives
        let mut x_dual = x.mapv(D::from);
        for _ in 0..D::NDERIV {
            Self::newton_step_cross_association(&mut x_dual, delta_ab, delta_cc, rho, tol)?;
        }

        // save monomer fraction
        if let Some(x0) = x0 {
            *x0 = x;
        }

        // Helmholtz energy density
        let f = |x: D| x.ln() - x * 0.5 + 0.5;
        Ok((rho * x_dual.mapv(f)).sum())
    }

    fn newton_step_cross_association<D: DualNum<f64> + Copy, S: Data<Elem = D>>(
        x: &mut Array1<D>,
        delta_ab: &Array2<D>,
        delta_cc: &Array2<D>,
        rho: &ArrayBase<S, Ix1>,
        tol: f64,
    ) -> EosResult<bool> {
        let nassoc = x.len();
        // gradient
        let mut g = x.map(D::recip);
        // Hessian
        let mut h: Array2<D> = Array::zeros([nassoc; 2]);

        // split arrays
        let &[a, b] = delta_ab.shape() else {
            panic!("wrong shape!")
        };
        let c = delta_cc.shape()[0];
        let (xa, xc) = x.view().split_at(Axis(0), a + b);
        let (xa, xb) = xa.split_at(Axis(0), a);
        let (rhoa, rhoc) = rho.view().split_at(Axis(0), a + b);
        let (rhoa, rhob) = rhoa.split_at(Axis(0), a);

        for i in 0..nassoc {
            // calculate gradients
            let (d, dnx) = if i < a {
                let d = delta_ab.index_axis(Axis(0), i);
                (d, (&xb * &rhob * d).sum() + 1.0)
            } else if i < a + b {
                let d = delta_ab.index_axis(Axis(1), i - a);
                (d, (&xa * &rhoa * d).sum() + 1.0)
            } else {
                let d = delta_cc.index_axis(Axis(0), i - a - b);
                (d, (&xc * &rhoc * d).sum() + 1.0)
            };
            g[i] -= dnx;

            // approximate hessian
            h[(i, i)] = -dnx / x[i];
            if i < a {
                for j in 0..b {
                    h[(i, a + j)] = -d[j] * rhob[j];
                }
            } else if i < a + b {
                for j in 0..a {
                    h[(i, j)] = -d[j] * rhoa[j];
                }
            } else {
                for j in 0..c {
                    h[(i, a + b + j)] -= d[j] * rhoc[j];
                }
            }
        }

        // Newton step
        // avoid stepping to negative values for x (see Michelsen 2006)
        let delta_x = LU::new(h)?.solve(&g);
        Zip::from(x).and(&delta_x).for_each(|x, &delta_x| {
            if delta_x.re() < x.re() * 0.8 {
                *x -= delta_x
            } else {
                *x *= 0.2
            }
        });

        // check convergence
        Ok(norm(&g.map(D::re)) < tol)
    }
}
