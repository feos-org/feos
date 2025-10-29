//! Generic implementation of the SAFT association contribution
//! that can be used across models.
use crate::hard_sphere::HardSphereProperties;
use feos_core::parameter::AssociationParameters;
use feos_core::{FeosError, FeosResult, StateHD};
use nalgebra::{DMatrix, DVector};
use num_dual::linalg::LU;
use num_dual::*;

#[cfg(feature = "dft")]
mod dft;
#[cfg(feature = "dft")]
pub use dft::AssociationFunctional;

/// Implementation of the association strength in the SAFT association model.
pub trait AssociationStrength: HardSphereProperties {
    type Record;

    fn association_strength<D: DualNum<f64> + Copy>(
        &self,
        parameters: &AssociationParameters<Self::Record>,
        state: &StateHD<D>,
        diameter: &DVector<D>,
        xi: D,
    ) -> [DMatrix<D>; 2];
}

/// Implementation of the SAFT association Helmholtz energy
/// contribution and functional.
#[derive(Clone, Copy)]
pub struct Association {
    max_iter: usize,
    tol: f64,
    force_cross_association: bool,
}

impl Association {
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self {
            max_iter,
            tol,
            force_cross_association: false,
        }
    }

    pub fn new_cross_association(max_iter: usize, tol: f64) -> Self {
        let mut res = Self::new(max_iter, tol);
        res.force_cross_association = true;
        res
    }

    #[inline]
    pub fn helmholtz_energy_density<A: AssociationStrength, D: DualNum<f64> + Copy>(
        &self,
        model: &A,
        parameters: &AssociationParameters<A::Record>,
        state: &StateHD<D>,
        diameter: &DVector<D>,
    ) -> D {
        let a = parameters;

        // association strength
        let [delta_ab, delta_cc] =
            model.association_strength(parameters, state, diameter, D::one());

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

    // #[expect(clippy::too_many_arguments)]
    // fn association_strength<A: AssociationStrength, D: DualNum<f64> + Copy>(
    //     &self,
    //     parameters: &AssociationParameters<A::Record>,
    //     model: &A,
    //     temperature: D,
    //     partial_density: &DVector<D>,
    //     diameter: &DVector<D>,
    // ) -> [DMatrix<D>; 2] {
    //     let p = parameters;

    //     let mut delta_ab = DMatrix::zeros(p.sites_a.len(), p.sites_b.len());
    //     for b in &p.binary_ab {
    //         let [i, j] = [b.id1, b.id2];
    //         delta_ab[(i, j)] = model.association_strength(
    //             temperature,
    //             partial_density,
    //             diameter,
    //             p.sites_a[i].assoc_comp,
    //             p.sites_b[j].assoc_comp,
    //             &b.model_record,
    //         )
    //     }
    //     let mut delta_cc = DMatrix::zeros(p.sites_c.len(), p.sites_c.len());
    //     for b in &p.binary_cc {
    //         let [i, j] = [b.id1, b.id2];
    //         delta_cc[(i, j)] = model.association_strength(
    //             temperature,
    //             partial_density,
    //             diameter,
    //             p.sites_c[i].assoc_comp,
    //             p.sites_c[j].assoc_comp,
    //             &b.model_record,
    //         )
    //     }
    //     [delta_ab, delta_cc]
    // }

    fn helmholtz_energy_density_ab_analytic<A, D: DualNum<f64> + Copy>(
        &self,
        parameters: &AssociationParameters<A>,
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

    fn helmholtz_energy_density_cc_analytic<A, D: DualNum<f64> + Copy>(
        &self,
        parameters: &AssociationParameters<A>,
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
            .map(|([i, j], br)| BinaryRecord::with_association(i, j, Some(()), vec![br]))
            .to_vec();
        let params = Parameters::new(pure_records, binary_records)?;
        let [epsilon_k_ab, kappa_ab] = params
            .collate_ab(|p| [p.epsilon_k_ab, p.kappa_ab])
            .map(|p| p.map(Option::unwrap));
        println!("{epsilon_k_ab}");
        println!("{kappa_ab}");
        let epsilon_k_ab_ref = dmatrix![
            2500., 1234., 3140., 2250.;
            1234., 1500., 1000., 3333.;
            1750., 1250., 750., 1500.;
        ];
        assert_eq!(epsilon_k_ab, epsilon_k_ab_ref);
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

        let params1 = Parameters::new_binary([pr1.clone(), pr2], Some(()), vec![])?;
        let params2 = Parameters::new_binary([pr1, pr3], Some(()), br)?;
        let [epsilon_k_ab1, kappa_ab1] = params1
            .collate_ab(|p| [p.epsilon_k_ab, p.kappa_ab])
            .map(|p| p.map(Option::unwrap));
        let [epsilon_k_ab2, kappa_ab2] = params2
            .collate_ab(|p| [p.epsilon_k_ab, p.kappa_ab])
            .map(|p| p.map(Option::unwrap));
        println!("{epsilon_k_ab1}");
        println!("{epsilon_k_ab2}");
        println!("{kappa_ab1}");
        println!("{kappa_ab2}");
        assert_eq!(epsilon_k_ab1, epsilon_k_ab2);
        assert_eq!(kappa_ab1, kappa_ab2);
        Ok(())
    }

    #[test]
    fn helmholtz_energy() {
        let parameters = water_parameters(1.0);
        let params = PcSaftPars::new(&parameters);
        let assoc = Association::new(50, 1e-10);
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
        let assoc = Association::new(50, 1e-10);
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
        let assoc = Association::new(50, 1e-10);
        let cross_assoc = Association::new_cross_association(50, 1e-10);
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
        let contrib = Association::new(50, 1e-10);
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
        let contrib = Association::new_cross_association(50, 1e-10);
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
        let contrib = Association::new(50, 1e-10);
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
