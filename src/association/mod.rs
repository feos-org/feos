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
use std::fmt;
use std::sync::Arc;

#[cfg(feature = "dft")]
mod dft;

#[derive(Clone, Copy, Debug)]
struct AssociationSite<A> {
    assoc_comp: usize,
    site_index: usize,
    n: f64,
    parameters: A,
}

impl<A> AssociationSite<A> {
    fn new(assoc_comp: usize, site_index: usize, n: f64, parameters: A) -> Self {
        Self {
            assoc_comp,
            site_index,
            n,
            parameters,
        }
    }
}

/// Pure component association parameters.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct AssociationRecord<A> {
    #[serde(flatten)]
    pub parameters: A,
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

impl<A> AssociationRecord<A> {
    pub fn new(parameters: A, na: f64, nb: f64, nc: f64) -> Self {
        Self {
            parameters,
            na,
            nb,
            nc,
        }
    }
}

impl<A: fmt::Display> fmt::Display for AssociationRecord<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AssociationRecord(parameters={}", self.parameters)?;
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
pub struct BinaryAssociationRecord<B> {
    // Binary association parameters
    #[serde(flatten)]
    pub parameters: B,
    /// Indices of sites that the record refers to.
    #[serde(skip_serializing_if = "is_default_site_indices")]
    #[serde(default)]
    pub site_indices: [usize; 2],
}

fn is_default_site_indices([i, j]: &[usize; 2]) -> bool {
    *i == 0 && *j == 0
}

impl<B> BinaryAssociationRecord<B> {
    pub fn new(parameters: B, site_indices: Option<[usize; 2]>) -> Self {
        Self {
            parameters,
            site_indices: site_indices.unwrap_or_default(),
        }
    }
}

/// Parameter set required for the SAFT association Helmoltz energy
/// contribution and functional.
#[derive(Clone)]
pub struct AssociationParameters<P: AssociationStrength> {
    component_index: Array1<usize>,
    sites_a: Array1<AssociationSite<P::Record>>,
    sites_b: Array1<AssociationSite<P::Record>>,
    sites_c: Array1<AssociationSite<P::Record>>,
    parameters_ab: Array2<P::Record>,
    parameters_cc: Array2<P::Record>,
}

impl<P: AssociationStrength> AssociationParameters<P> {
    pub fn new(
        records: &[Vec<AssociationRecord<P::Record>>],
        binary_records: &[([usize; 2], BinaryAssociationRecord<P::BinaryRecord>)],
        component_index: Option<&Array1<usize>>,
    ) -> Self {
        let mut sites_a = Vec::new();
        let mut sites_b = Vec::new();
        let mut sites_c = Vec::new();

        for (i, record) in records.iter().enumerate() {
            for (s, site) in record.iter().enumerate() {
                if site.na > 0.0 {
                    sites_a.push(AssociationSite::new(i, s, site.na, site.parameters));
                }
                if site.nb > 0.0 {
                    sites_b.push(AssociationSite::new(i, s, site.nb, site.parameters));
                }
                if site.nc > 0.0 {
                    sites_c.push(AssociationSite::new(i, s, site.nc, site.parameters));
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

        let mut parameters_ab = Array2::from_shape_fn([sites_a.len(), sites_b.len()], |(i, j)| {
            P::combining_rule(sites_a[i].parameters, sites_b[j].parameters)
        });
        let mut parameters_cc = Array2::from_shape_fn([sites_c.len(); 2], |(i, j)| {
            P::combining_rule(sites_c[i].parameters, sites_c[j].parameters)
        });

        for &([i, j], record) in binary_records.iter() {
            let [a, b] = record.site_indices;
            if let (Some(x), Some(y)) = (indices_a.get(&(i, a)), indices_b.get(&(j, b))) {
                P::update_binary(&mut parameters_ab[[*x, *y]], record.parameters);
            }
            if let (Some(y), Some(x)) = (indices_b.get(&(i, a)), indices_a.get(&(j, b))) {
                P::update_binary(&mut parameters_ab[[*x, *y]], record.parameters);
            }
            if let (Some(x), Some(y)) = (indices_c.get(&(i, a)), indices_c.get(&(j, b))) {
                P::update_binary(&mut parameters_cc[[*x, *y]], record.parameters);
                P::update_binary(&mut parameters_cc[[*y, *x]], record.parameters);
            }
        }

        Self {
            component_index: component_index
                .cloned()
                .unwrap_or_else(|| Array1::from_shape_fn(records.len(), |i| i)),
            sites_a: Array1::from_vec(sites_a),
            sites_b: Array1::from_vec(sites_b),
            sites_c: Array1::from_vec(sites_c),
            parameters_ab,
            parameters_cc,
        }
    }

    pub fn is_empty(&self) -> bool {
        (self.sites_a.is_empty() | self.sites_b.is_empty()) & self.sites_c.is_empty()
    }
}

/// Implementation of the SAFT association Helmholtz energy
/// contribution and functional.
pub struct Association<P: AssociationStrength> {
    parameters: Arc<P>,
    association_parameters: Arc<AssociationParameters<P>>,
    max_iter: usize,
    tol: f64,
    force_cross_association: bool,
}

impl<P: AssociationStrength> Association<P> {
    pub fn new(
        parameters: &Arc<P>,
        association_parameters: &Arc<AssociationParameters<P>>,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        Self {
            parameters: parameters.clone(),
            association_parameters: association_parameters.clone(),
            max_iter,
            tol,
            force_cross_association: false,
        }
    }

    pub fn new_cross_association(
        parameters: &Arc<P>,
        association_parameters: &Arc<AssociationParameters<P>>,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        let mut res = Self::new(parameters, association_parameters, max_iter, tol);
        res.force_cross_association = true;
        res
    }
}

/// Implementation of the association strength in the SAFT association model.
pub trait AssociationStrength: HardSphereProperties {
    type Record: Copy;
    type BinaryRecord: Copy;

    fn association_strength<D: DualNum<f64> + Copy>(
        &self,
        temperature: D,
        comp_i: usize,
        comp_j: usize,
        assoc_ij: Self::Record,
    ) -> D;

    fn combining_rule(parameters_i: Self::Record, parameters_j: Self::Record) -> Self::Record;

    fn update_binary(_parameters_ij: &mut Self::Record, _binary_parameters: Self::BinaryRecord) {}
}

impl<P: AssociationStrength> Association<P> {
    #[inline]
    pub fn helmholtz_energy<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
        diameter: &Array1<D>,
    ) -> D {
        let a = &self.association_parameters;

        // auxiliary variables
        let [zeta2, n3] = self
            .parameters
            .zeta(state.temperature, &state.partial_density, [2, 3]);
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
                .unwrap_or_else(|_| D::from(std::f64::NAN))
                    * state.volume
            }
        }
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
            let k = di * dj / (di + dj) * (n2 * n3i);
            n3i * (k * xi * (k / 18.0 + 0.5) + 1.0)
                * self.parameters.association_strength(
                    temperature,
                    p.sites_a[i].assoc_comp,
                    p.sites_b[j].assoc_comp,
                    p.parameters_ab[(i, j)],
                )
        });
        let delta_cc = Array2::from_shape_fn([p.sites_c.len(); 2], |(i, j)| {
            let di = diameter[p.sites_c[i].assoc_comp];
            let dj = diameter[p.sites_c[j].assoc_comp];
            let k = di * dj / (di + dj) * (n2 * n3i);
            n3i * (k * xi * (k / 18.0 + 0.5) + 1.0)
                * self.parameters.association_strength(
                    temperature,
                    p.sites_c[i].assoc_comp,
                    p.sites_c[j].assoc_comp,
                    p.parameters_cc[(i, j)],
                )
        });
        [delta_ab, delta_cc]
    }
}

impl<P: AssociationStrength> fmt::Display for Association<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Association")
    }
}

impl<P: AssociationStrength> Association<P> {
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

#[cfg(test)]
#[cfg(feature = "pcsaft")]
mod tests_pcsaft {
    use super::*;
    use crate::hard_sphere::HardSphereProperties;
    use crate::pcsaft::parameters::utils::water_parameters;
    use crate::pcsaft::parameters::{PcSaftAssociationRecord, PcSaftBinaryAssociationRecord};
    use crate::pcsaft::PcSaftParameters;
    use approx::assert_relative_eq;
    use feos_core::parameter::{Parameter, ParameterError};

    fn record(
        kappa_ab: f64,
        epsilon_k_ab: f64,
        na: f64,
        nb: f64,
    ) -> AssociationRecord<PcSaftAssociationRecord> {
        let pcsaft = PcSaftAssociationRecord::new(kappa_ab, epsilon_k_ab);
        AssociationRecord::new(pcsaft, na, nb, 0.0)
    }

    fn binary_record(
        kappa_ab: f64,
        epsilon_k_ab: f64,
        indices: Option<[usize; 2]>,
    ) -> BinaryAssociationRecord<PcSaftBinaryAssociationRecord> {
        let pcsaft = PcSaftBinaryAssociationRecord::new(Some(kappa_ab), Some(epsilon_k_ab));
        BinaryAssociationRecord::new(pcsaft, indices)
    }

    #[test]
    fn test_binary_parameters() {
        let comp1 = vec![record(0.1, 2500., 1.0, 1.0)];
        let comp2 = vec![record(0.2, 1500., 1.0, 1.0)];
        let comp3 = vec![record(0.3, 500., 0.0, 1.0)];
        let comp4 = vec![record(0.3, 1000., 1.0, 0.0), record(0.3, 2000., 0.0, 1.0)];
        let records = [comp1, comp2, comp3, comp4];
        let binary = [
            ([0, 1], binary_record(3.5, 1234., Some([0, 0]))),
            ([0, 2], binary_record(3.5, 3140., Some([0, 0]))),
            ([1, 3], binary_record(3.5, 3333., Some([0, 1]))),
        ];
        let assoc = AssociationParameters::<PcSaftParameters>::new(&records, &binary, None);
        println!("{}", assoc.parameters_ab);
        let epsilon_k_ab = arr2(&[
            [2500., 1234., 3140., 2250.],
            [1234., 1500., 1000., 3333.],
            [1750., 1250., 750., 1500.],
        ]);
        assert_eq!(assoc.parameters_ab.mapv(|p| p.epsilon_k_ab), epsilon_k_ab);
    }

    #[test]
    fn test_induced_association() {
        let comp1 = vec![record(0.1, 2500., 1.0, 1.0)];
        let comp2 = vec![record(0.1, -500., 0.0, 1.0)];
        let comp3 = vec![record(0.0, 0.0, 0.0, 1.0)];
        let binary = [([0, 1], binary_record(0.1, 1000., None))];
        let assoc1 =
            AssociationParameters::<PcSaftParameters>::new(&[comp1.clone(), comp2], &[], None);
        let assoc2 = AssociationParameters::<PcSaftParameters>::new(&[comp1, comp3], &binary, None);
        println!("{}", assoc1.parameters_ab);
        println!("{}", assoc2.parameters_ab);
        assert_eq!(
            assoc1.parameters_ab.mapv(|p| p.epsilon_k_ab),
            assoc2.parameters_ab.mapv(|p| p.epsilon_k_ab)
        );
        assert_eq!(
            assoc1.parameters_ab.mapv(|p| p.kappa_ab),
            assoc2.parameters_ab.mapv(|p| p.kappa_ab)
        );
    }

    #[test]
    fn helmholtz_energy() {
        let params = Arc::new(water_parameters());
        let assoc = Association::new(&params, &params.association, 50, 1e-10);
        let t = 350.0;
        let v = 41.248289328513216;
        let n = 1.23;
        let s = StateHD::new(t, v, arr1(&[n]));
        let d = params.hs_diameter(t);
        let a_rust = assoc.helmholtz_energy(&s, &d) / n;
        assert_relative_eq!(a_rust, -4.229878997054543, epsilon = 1e-10);
    }

    #[test]
    fn helmholtz_energy_cross() {
        let params = Arc::new(water_parameters());
        let assoc = Association::new_cross_association(&params, &params.association, 50, 1e-10);
        let t = 350.0;
        let v = 41.248289328513216;
        let n = 1.23;
        let s = StateHD::new(t, v, arr1(&[n]));
        let d = params.hs_diameter(t);
        let a_rust = assoc.helmholtz_energy(&s, &d) / n;
        assert_relative_eq!(a_rust, -4.229878997054543, epsilon = 1e-10);
    }

    #[test]
    fn helmholtz_energy_cross_3b() -> Result<(), ParameterError> {
        let mut params = water_parameters();
        let mut record = params.pure_records.pop().unwrap();
        let mut association_record = record.model_record.association_record.unwrap();
        association_record.na = 2.0;
        record.model_record.association_record = Some(association_record);
        let params = Arc::new(PcSaftParameters::new_pure(record)?);
        let assoc = Association::new(&params, &params.association, 50, 1e-10);
        let cross_assoc =
            Association::new_cross_association(&params, &params.association, 50, 1e-10);
        let t = 350.0;
        let v = 41.248289328513216;
        let n = 1.23;
        let s = StateHD::new(t, v, arr1(&[n]));
        let d = params.hs_diameter(t);
        let a_assoc = assoc.helmholtz_energy(&s, &d) / n;
        let a_cross_assoc = cross_assoc.helmholtz_energy(&s, &d) / n;
        assert_relative_eq!(a_assoc, a_cross_assoc, epsilon = 1e-10);
        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "gc_pcsaft")]
mod tests_gc_pcsaft {
    use super::*;
    use crate::gc_pcsaft::eos::parameter::test::*;
    use approx::assert_relative_eq;
    use feos_core::si::{Pressure, METER, MOL, PASCAL};
    use ndarray::arr1;
    use num_dual::Dual64;
    use typenum::P3;

    #[test]
    fn test_assoc_propanol() {
        let params = Arc::new(propanol());
        let contrib = Association::new(&params, &params.association, 50, 1e-10);
        let temperature = 300.0;
        let volume = METER.powi::<P3>().to_reduced();
        let moles = (1.5 * MOL).to_reduced();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derivative(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let diameter = params.hs_diameter(state.temperature);
        let pressure =
            Pressure::from_reduced(-contrib.helmholtz_energy(&state, &diameter).eps * temperature);
        assert_relative_eq!(pressure, -3.6819598891967344 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn test_cross_assoc_propanol() {
        let params = Arc::new(propanol());
        let contrib = Association::new_cross_association(&params, &params.association, 50, 1e-10);
        let temperature = 300.0;
        let volume = METER.powi::<P3>().to_reduced();
        let moles = (1.5 * MOL).to_reduced();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derivative(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let diameter = params.hs_diameter(state.temperature);
        let pressure =
            Pressure::from_reduced(-contrib.helmholtz_energy(&state, &diameter).eps * temperature);
        assert_relative_eq!(pressure, -3.6819598891967344 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn test_cross_assoc_ethanol_propanol() {
        let params = Arc::new(ethanol_propanol(false));
        let contrib = Association::new(&params, &params.association, 50, 1e-10);
        let temperature = 300.0;
        let volume = METER.powi::<P3>().to_reduced();
        let moles = (arr1(&[1.5, 2.5]) * MOL).to_reduced();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derivative(),
            moles.mapv(Dual64::from_re),
        );
        let diameter = params.hs_diameter(state.temperature);
        let pressure =
            Pressure::from_reduced(-contrib.helmholtz_energy(&state, &diameter).eps * temperature);
        assert_relative_eq!(pressure, -26.105606376765632 * PASCAL, max_relative = 1e-10);
    }
}
