//! Generic implementation of the SAFT association contribution
//! that can be used across models.
use crate::hard_sphere::HardSphereProperties;
use feos_core::parameter::{AssociationParameters, GenericParameters};
use feos_core::{FeosError, FeosResult, StateHD};
use nalgebra::{DMatrix, DVector};
use num_dual::linalg::LU;
use num_dual::*;
use std::collections::HashMap;
use std::fmt::Debug;

#[cfg(feature = "dft")]
mod dft;
#[cfg(feature = "dft")]
pub use dft::AssociationFunctional;

/// Implementation of the association strength in the SAFT association model.
pub trait AssociationStrength: HardSphereProperties {
    type Pure;
    type Record: Clone + Debug + PartialEq + 'static;

    fn association_strength<D: DualNum<f64> + Copy>(
        &self,
        temperature: D,
        comp_i: usize,
        comp_j: usize,
        assoc_ij: &Self::Record,
    ) -> D;

    fn combining_rule(
        comp_i: &Self::Pure,
        comp_j: &Self::Pure,
        parameters_i: &Self::Record,
        parameters_j: &Self::Record,
    ) -> Self::Record;
}

/// Implementation of the SAFT association Helmholtz energy
/// contribution and functional.
pub struct Association<A: AssociationStrength> {
    max_iter: usize,
    tol: f64,
    force_cross_association: bool,
    parameters_ab: DMatrix<Option<A::Record>>,
    parameters_cc: DMatrix<Option<A::Record>>,
}

impl<A: AssociationStrength> Association<A> {
    pub fn new<B, Bo, C, Data>(
        parameters: &GenericParameters<A::Pure, B, A::Record, Bo, C, Data>,
        max_iter: usize,
        tol: f64,
    ) -> FeosResult<Option<Self>> {
        if parameters.association.is_empty() {
            return Ok(None);
        };

        let a = &parameters.association;

        let binary_ab: HashMap<_, _> = a
            .binary_ab
            .iter()
            .map(|br| ((br.id1, br.id2), &br.model_record))
            .collect();

        let binary_cc: HashMap<_, _> = a
            .binary_cc
            .iter()
            .map(|br| ((br.id1, br.id2), &br.model_record))
            .collect();

        let parameters_ab = DMatrix::from_fn(a.sites_a.len(), a.sites_b.len(), |i, j| {
            if let Some(&record) = binary_ab.get(&(i, j)) {
                Some(record.clone())
            } else if let (Some(p1), Some(p2)) =
                (&a.sites_a[i].parameters, &a.sites_b[j].parameters)
            {
                Some(A::combining_rule(
                    &parameters.pure[a.sites_a[i].assoc_comp].model_record,
                    &parameters.pure[a.sites_b[j].assoc_comp].model_record,
                    p1,
                    p2,
                ))
            } else {
                None
            }
        });
        let parameters_cc = DMatrix::from_fn(a.sites_c.len(), a.sites_c.len(), |i, j| {
            if let Some(&record) = binary_cc.get(&(i, j)) {
                Some(record.clone())
            } else if let (Some(p1), Some(p2)) =
                (&a.sites_c[i].parameters, &a.sites_c[j].parameters)
            {
                Some(A::combining_rule(
                    &parameters.pure[a.sites_c[i].assoc_comp].model_record,
                    &parameters.pure[a.sites_c[j].assoc_comp].model_record,
                    p1,
                    p2,
                ))
            } else {
                None
            }
        });

        Ok(Some(Self {
            max_iter,
            tol,
            force_cross_association: false,
            parameters_ab,
            parameters_cc,
        }))
    }

    pub fn new_cross_association<B, Bo, C, Data>(
        parameters: &GenericParameters<A::Pure, B, A::Record, Bo, C, Data>,
        max_iter: usize,
        tol: f64,
    ) -> FeosResult<Option<Self>> {
        let mut res = Self::new(parameters, max_iter, tol)?;
        if let Some(res) = &mut res {
            res.force_cross_association = true;
        }
        Ok(res)
    }

    #[inline]
    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        model: &A,
        parameters: &AssociationParameters<A::Record>,
        state: &StateHD<D>,
        diameter: &DVector<D>,
    ) -> D {
        let a = parameters;

        // auxiliary variables
        let [zeta2, n3] = model.zeta(state.temperature, &state.partial_density, [2, 3]);
        let n2 = zeta2 * 6.0;
        let n3i = (-n3 + 1.0).recip();

        // association strength
        let [delta_ab, delta_cc] = self.association_strength(
            parameters,
            model,
            state.temperature,
            diameter,
            n2,
            n3i,
            D::one(),
        );

        match (
            a.sites_a.len() * a.sites_b.len(),
            a.sites_c.len(),
            self.force_cross_association,
        ) {
            (0, 0, _) => D::zero(),
            (1, 0, false) => self.helmholtz_energy_density_ab_analytic(a, state, delta_ab[(0, 0)]),
            (0, 1, false) => self.helmholtz_energy_density_cc_analytic(a, state, delta_cc[(0, 0)]),
            (1, 1, false) => {
                self.helmholtz_energy_density_ab_analytic(a, state, delta_ab[(0, 0)])
                    + self.helmholtz_energy_density_cc_analytic(a, state, delta_cc[(0, 0)])
            }
            _ => {
                // extract site densities of associating segments
                let rho: Vec<_> = a
                    .sites_a
                    .iter()
                    .chain(a.sites_b.iter())
                    .chain(a.sites_c.iter())
                    .map(|s| state.partial_density[a.component_index[s.assoc_comp]] * s.n)
                    .collect();
                let rho = DVector::from(rho);

                // Helmholtz energy
                self.helmholtz_energy_density_cross_association(&rho, &delta_ab, &delta_cc, None)
                    .unwrap_or_else(|_| D::from(f64::NAN))
            }
        }
    }

    #[expect(clippy::too_many_arguments)]
    fn association_strength<D: DualNum<f64> + Copy>(
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

        let delta_ab = DMatrix::from_fn(p.sites_a.len(), p.sites_b.len(), |i, j| {
            if let Some(par) = &self.parameters_ab[(i, j)] {
                let di = diameter[p.sites_a[i].assoc_comp];
                let dj = diameter[p.sites_b[j].assoc_comp];
                let k = di * dj / (di + dj) * (n2 * n3i);
                n3i * (k * xi * (k / 18.0 + 0.5) + 1.0)
                    * model.association_strength(
                        temperature,
                        p.sites_a[i].assoc_comp,
                        p.sites_b[j].assoc_comp,
                        par,
                    )
            } else {
                D::zero()
            }
        });
        let delta_cc = DMatrix::from_fn(p.sites_c.len(), p.sites_c.len(), |i, j| {
            if let Some(par) = &self.parameters_cc[(i, j)] {
                let di = diameter[p.sites_c[i].assoc_comp];
                let dj = diameter[p.sites_c[j].assoc_comp];
                let k = di * dj / (di + dj) * (n2 * n3i);
                n3i * (k * xi * (k / 18.0 + 0.5) + 1.0)
                    * model.association_strength(
                        temperature,
                        p.sites_c[i].assoc_comp,
                        p.sites_c[j].assoc_comp,
                        par,
                    )
            } else {
                D::zero()
            }
        });
        [delta_ab, delta_cc]
    }

    fn helmholtz_energy_density_ab_analytic<D: DualNum<f64> + Copy>(
        &self,
        parameters: &AssociationParameters<A::Record>,
        state: &StateHD<D>,
        delta: D,
    ) -> D {
        let a = parameters;

        // site densities
        let rhoa =
            state.partial_density[a.component_index[a.sites_a[0].assoc_comp]] * a.sites_a[0].n;
        let rhob =
            state.partial_density[a.component_index[a.sites_b[0].assoc_comp]] * a.sites_b[0].n;

        // fraction of non-bonded association sites
        let sqrt = ((delta * (rhoa - rhob) + 1.0).powi(2) + delta * rhob * 4.0).sqrt();
        let xa = (sqrt + (delta * (rhob - rhoa) + 1.0)).recip() * 2.0;
        let xb = (sqrt + (delta * (rhoa - rhob) + 1.0)).recip() * 2.0;

        rhoa * (xa.ln() - xa * 0.5 + 0.5) + rhob * (xb.ln() - xb * 0.5 + 0.5)
    }

    fn helmholtz_energy_density_cc_analytic<D: DualNum<f64> + Copy>(
        &self,
        parameters: &AssociationParameters<A::Record>,
        state: &StateHD<D>,
        delta: D,
    ) -> D {
        let a = parameters;

        // site density
        let rhoc =
            state.partial_density[a.component_index[a.sites_c[0].assoc_comp]] * a.sites_c[0].n;

        // fraction of non-bonded association sites
        let xc = ((delta * 4.0 * rhoc + 1.0).sqrt() + 1.0).recip() * 2.0;

        rhoc * (xc.ln() - xc * 0.5 + 0.5)
    }

    fn helmholtz_energy_density_cross_association<D: DualNum<f64> + Copy>(
        &self,
        rho: &DVector<D>,
        delta_ab: &DMatrix<D>,
        delta_cc: &DMatrix<D>,
        x0: Option<&mut DVector<f64>>,
    ) -> FeosResult<D> {
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
            None => DVector::from_element(rho.len(), 0.2),
        };

        let delta_ab_re = delta_ab.map(|d| d.re());
        let delta_cc_re = delta_cc.map(|d| d.re());
        let rho_re = rho.map(|r| r.re());
        for k in 0..self.max_iter {
            if Self::newton_step_cross_association(
                &mut x,
                &delta_ab_re,
                &delta_cc_re,
                &rho_re,
                self.tol,
            )? {
                break;
            }
            if k == self.max_iter - 1 {
                return Err(FeosError::NotConverged("Cross association".into()));
            }
        }

        // calculate derivatives
        let mut x_dual = x.map(D::from);
        for _ in 0..D::NDERIV {
            Self::newton_step_cross_association(&mut x_dual, delta_ab, delta_cc, rho, self.tol)?;
        }

        // save monomer fraction
        if let Some(x0) = x0 {
            *x0 = x;
        }

        // Helmholtz energy density
        let f = |x: D| x.ln() - x * 0.5 + 0.5;
        Ok(rho.dot(&x_dual.map(f)))
    }

    fn newton_step_cross_association<D: DualNum<f64> + Copy>(
        x: &mut DVector<D>,
        delta_ab: &DMatrix<D>,
        delta_cc: &DMatrix<D>,
        rho: &DVector<D>,
        tol: f64,
    ) -> FeosResult<bool> {
        let nassoc = x.len();
        // gradient
        let mut g = x.map(|x| x.recip());
        // Hessian
        let mut h: DMatrix<D> = DMatrix::zeros(nassoc, nassoc);

        // split arrays
        let (a, b) = delta_ab.shape();
        let (c, _) = delta_cc.shape();
        let (xa, xc) = x.rows_range_pair(..a + b, a + b..);
        let (xa, xb) = xa.rows_range_pair(..a, a..);
        let (rhoa, rhoc) = rho.rows_range_pair(..a + b, a + b..);
        let (rhoa, rhob) = rhoa.rows_range_pair(..a, a..);

        for i in 0..nassoc {
            // calculate gradients
            let dnx = if i < a {
                let d = delta_ab.row(i).transpose();
                xb.component_mul(&rhob).dot(&d) + 1.0
            } else if i < a + b {
                let d = delta_ab.column(i - a);
                xa.component_mul(&rhoa).dot(&d) + 1.0
            } else {
                let d = delta_cc.column(i - a - b);
                xc.component_mul(&rhoc).dot(&d) + 1.0
            };
            g[i] -= dnx;

            // approximate hessian
            h[(i, i)] = -dnx / x[i];
            if i < a {
                for j in 0..b {
                    h[(i, a + j)] = -delta_ab[(i, j)] * rhob[j];
                }
            } else if i < a + b {
                for j in 0..a {
                    h[(i, j)] = -delta_ab[(j, i - a)] * rhoa[j];
                }
            } else {
                for j in 0..c {
                    h[(i, a + b + j)] -= delta_cc[(i - a - b, j)] * rhoc[j];
                }
            }
        }

        // Newton step
        // avoid stepping to negative values for x (see Michelsen 2006)
        let delta_x = LU::new(h)?.solve(&g);
        x.iter_mut().zip(&delta_x).for_each(|(x, &delta_x)| {
            if delta_x.re() < x.re() * 0.8 {
                *x -= delta_x
            } else {
                *x *= 0.2
            }
        });

        // check convergence
        Ok(g.map(|g| g.re()).norm() < tol)
    }
}

#[cfg(test)]
#[cfg(feature = "pcsaft")]
mod tests_pcsaft {
    use super::*;
    use crate::hard_sphere::HardSphereProperties;
    use crate::pcsaft::PcSaftRecord;
    use crate::pcsaft::parameters::utils::water_parameters;
    use crate::pcsaft::parameters::{PcSaftAssociationRecord, PcSaftPars};
    use approx::assert_relative_eq;
    use feos_core::parameter::{
        AssociationRecord, BinaryAssociationRecord, BinaryRecord, Parameters, PureRecord,
    };
    use nalgebra::{dmatrix, dvector};

    fn pcsaft() -> PcSaftRecord {
        PcSaftRecord::new(1.2, 3.5, 245.0, 2.3, 4.4, None, None, None)
    }

    fn record(
        id: &'static str,
        kappa_ab: f64,
        epsilon_k_ab: f64,
        na: f64,
        nb: f64,
    ) -> AssociationRecord<PcSaftAssociationRecord> {
        let pcsaft = PcSaftAssociationRecord::new(kappa_ab, epsilon_k_ab);
        AssociationRecord::with_id(id.into(), Some(pcsaft), na, nb, 0.0)
    }

    fn binary_record(
        id1: &'static str,
        id2: &'static str,
        kappa_ab: f64,
        epsilon_k_ab: f64,
    ) -> BinaryAssociationRecord<PcSaftAssociationRecord> {
        let pcsaft = PcSaftAssociationRecord::new(kappa_ab, epsilon_k_ab);
        BinaryAssociationRecord {
            id1: id1.into(),
            id2: id2.into(),
            parameters: pcsaft,
        }
    }

    #[derive(Clone, Copy)]
    struct NoRecord;

    #[test]
    fn test_binary_parameters() -> FeosResult<()> {
        let comp1 = vec![record("0", 0.1, 2500., 1.0, 1.0)];
        let comp2 = vec![record("0", 0.2, 1500., 1.0, 1.0)];
        let comp3 = vec![record("0", 0.3, 500., 0.0, 1.0)];
        let comp4 = vec![
            record("0", 0.3, 1000., 1.0, 0.0),
            record("1", 0.3, 2000., 0.0, 1.0),
        ];
        let pure_records = [comp1, comp2, comp3, comp4]
            .map(|r| PureRecord::with_association(Default::default(), 0.0, pcsaft(), r))
            .to_vec();
        let binary = [
            ([0, 1], binary_record("0", "0", 3.5, 1234.)),
            ([0, 2], binary_record("0", "0", 3.5, 3140.)),
            ([1, 3], binary_record("0", "1", 3.5, 3333.)),
        ];
        let binary_records = binary
            .map(|([i, j], br)| BinaryRecord::with_association(i, j, Some(NoRecord), vec![br]))
            .to_vec();
        let params = Parameters::new(pure_records, binary_records)?;
        let assoc: Association<PcSaftPars> = Association::new(&params, 100, 1e-10)?.unwrap();
        println!("{}", assoc.parameters_ab.map(|p| p.unwrap().epsilon_k_ab));
        let epsilon_k_ab = dmatrix![
            2500., 1234., 3140., 2250.;
            1234., 1500., 1000., 3333.;
            1750., 1250., 750., 1500.;
        ];
        assert_eq!(
            assoc.parameters_ab.map(|p| p.unwrap().epsilon_k_ab),
            epsilon_k_ab
        );
        Ok(())
    }

    #[test]
    fn test_induced_association() -> FeosResult<()> {
        let comp1 = vec![record("", 0.1, 2500., 1.0, 1.0)];
        let comp2 = vec![record("", 0.1, -500., 0.0, 1.0)];
        let comp3 = vec![record("", 0.0, 0.0, 0.0, 1.0)];
        let [pr1, pr2, pr3] = [comp1, comp2, comp3]
            .map(|r| PureRecord::with_association(Default::default(), 0.0, pcsaft(), r));
        let br = vec![binary_record("", "", 0.1, 1000.)];

        let params1 = Parameters::new_binary([pr1.clone(), pr2], Some(NoRecord), vec![])?;
        let params2 = Parameters::new_binary([pr1, pr3], Some(NoRecord), br)?;
        let assoc1: Association<PcSaftPars> = Association::new(&params1, 100, 1e-15)?.unwrap();
        let assoc2: Association<PcSaftPars> = Association::new(&params2, 100, 1e-15)?.unwrap();
        println!("{}", assoc1.parameters_ab.map(|p| p.unwrap().epsilon_k_ab));
        println!("{}", assoc2.parameters_ab.map(|p| p.unwrap().epsilon_k_ab));
        println!("{}", assoc1.parameters_ab.map(|p| p.unwrap().kappa_ab));
        println!("{}", assoc2.parameters_ab.map(|p| p.unwrap().kappa_ab));
        assert_eq!(
            assoc1.parameters_ab.map(|p| p.unwrap().epsilon_k_ab),
            assoc2.parameters_ab.map(|p| p.unwrap().epsilon_k_ab)
        );
        assert_eq!(
            assoc1.parameters_ab.map(|p| p.unwrap().kappa_ab),
            assoc2.parameters_ab.map(|p| p.unwrap().kappa_ab)
        );
        Ok(())
    }

    #[test]
    fn helmholtz_energy() {
        let parameters = water_parameters(1.0);
        let params = PcSaftPars::new(&parameters);
        let assoc = Association::new(&parameters, 50, 1e-10).unwrap().unwrap();
        let t = 350.0;
        let v = 41.248289328513216;
        let n = 1.23;
        let s = StateHD::new(t, v, &dvector![n]);
        let d = params.hs_diameter(t);
        let a_rust =
            assoc.helmholtz_energy_density(&params, &parameters.association, &s, &d) * v / n;
        assert_relative_eq!(a_rust, -4.229878997054543, epsilon = 1e-10);
    }

    #[test]
    fn helmholtz_energy_cross() {
        let parameters = water_parameters(1.0);
        let params = PcSaftPars::new(&parameters);
        let assoc = Association::new(&parameters, 50, 1e-10).unwrap().unwrap();
        let t = 350.0;
        let v = 41.248289328513216;
        let n = 1.23;
        let s = StateHD::new(t, v, &dvector![n]);
        let d = params.hs_diameter(t);
        let a_rust =
            assoc.helmholtz_energy_density(&params, &parameters.association, &s, &d) * v / n;
        assert_relative_eq!(a_rust, -4.229878997054543, epsilon = 1e-10);
    }

    #[test]
    fn helmholtz_energy_cross_3b() -> FeosResult<()> {
        let parameters = water_parameters(2.0);
        let params = PcSaftPars::new(&parameters);
        let assoc = Association::new(&parameters, 50, 1e-10).unwrap().unwrap();
        let cross_assoc = Association::new_cross_association(&parameters, 50, 1e-10)
            .unwrap()
            .unwrap();
        let t = 350.0;
        let v = 41.248289328513216;
        let n = 1.23;
        let s = StateHD::new(t, v, &dvector![n]);
        let d = params.hs_diameter(t);
        let a_assoc = assoc.helmholtz_energy_density(&params, &parameters.association, &s, &d);
        let a_cross_assoc =
            cross_assoc.helmholtz_energy_density(&params, &parameters.association, &s, &d);
        assert_relative_eq!(a_assoc, a_cross_assoc, epsilon = 1e-10);
        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "gc_pcsaft")]
mod tests_gc_pcsaft {
    use super::*;
    use crate::gc_pcsaft::{GcPcSaftEosParameters, eos::parameter::test::*};
    use approx::assert_relative_eq;
    use feos_core::ReferenceSystem;
    use nalgebra::dvector;
    use num_dual::Dual64;
    use quantity::{METER, MOL, PASCAL, Pressure};
    use typenum::P3;

    #[test]
    fn test_assoc_propanol() {
        let parameters = propanol();
        let params = GcPcSaftEosParameters::new(&parameters);
        let contrib = Association::new(&parameters, 50, 1e-10).unwrap().unwrap();
        let temperature = 300.0;
        let volume = Dual64::from_re(METER.powi::<P3>().to_reduced()).derivative();
        let moles = Dual64::from_re((1.5 * MOL).to_reduced());
        let molar_volume = volume / moles;
        let state = StateHD::new(
            Dual64::from_re(temperature),
            molar_volume,
            &dvector![Dual64::from_re(1.0)],
        );
        let diameter = params.hs_diameter(state.temperature);
        let pressure = Pressure::from_reduced(
            -(contrib.helmholtz_energy_density(
                &params,
                &parameters.association,
                &state,
                &diameter,
            ) * volume)
                .eps
                * temperature,
        );
        assert_relative_eq!(pressure, -3.6819598891967344 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn test_cross_assoc_propanol() {
        let parameters = propanol();
        let params = GcPcSaftEosParameters::new(&parameters);
        let contrib = Association::new_cross_association(&parameters, 50, 1e-10)
            .unwrap()
            .unwrap();
        let temperature = 300.0;
        let volume = Dual64::from_re(METER.powi::<P3>().to_reduced()).derivative();
        let moles = Dual64::from_re((1.5 * MOL).to_reduced());
        let molar_volume = volume / moles;
        let state = StateHD::new(
            Dual64::from_re(temperature),
            molar_volume,
            &dvector![Dual64::from_re(1.0)],
        );
        let diameter = params.hs_diameter(state.temperature);
        let pressure = Pressure::from_reduced(
            -(contrib.helmholtz_energy_density(
                &params,
                &parameters.association,
                &state,
                &diameter,
            ) * volume)
                .eps
                * temperature,
        );
        assert_relative_eq!(pressure, -3.6819598891967344 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn test_cross_assoc_ethanol_propanol() {
        let parameters = ethanol_propanol(false);
        let params = GcPcSaftEosParameters::new(&parameters);
        let contrib = Association::new(&parameters, 50, 1e-10).unwrap().unwrap();
        let temperature = 300.0;
        let volume = Dual64::from_re(METER.powi::<P3>().to_reduced()).derivative();
        let moles = (dvector![1.5, 2.5] * MOL).to_reduced().map(Dual64::from_re);
        let total_moles = moles.sum();
        let molar_volume = volume / total_moles;
        let molefracs = moles / total_moles;
        let state = StateHD::new(Dual64::from_re(temperature), molar_volume, &molefracs);
        let diameter = params.hs_diameter(state.temperature);
        let pressure = Pressure::from_reduced(
            -(contrib.helmholtz_energy_density(
                &params,
                &parameters.association,
                &state,
                &diameter,
            ) * volume)
                .eps
                * temperature,
        );
        assert_relative_eq!(pressure, -26.105606376765632 * PASCAL, max_relative = 1e-10);
    }
}
