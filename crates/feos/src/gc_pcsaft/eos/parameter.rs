use crate::association::AssociationStrength;
use crate::gc_pcsaft::record::{GcPcSaftAssociationRecord, GcPcSaftParameters};
use crate::hard_sphere::{HardSphereProperties, MonomerShape};
use feos_core::StateHD;
use feos_core::parameter::AssociationParameters;
use itertools::Itertools;
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use quantity::{JOULE, KB, KELVIN};

/// Parameter set required for the gc-PC-SAFT equation of state.
#[derive(Clone)]
pub struct GcPcSaftEosParameters {
    pub component_index: DVector<usize>,
    pub m: DVector<f64>,
    pub sigma: DVector<f64>,
    pub epsilon_k: DVector<f64>,
    pub bonds: Vec<([usize; 2], f64)>,
    pub dipole_comp: DVector<usize>,
    pub mu2: DVector<f64>,
    pub m_mix: DVector<f64>,
    pub s_ij: DMatrix<f64>,
    pub e_k_ij: DMatrix<f64>,

    pub sigma_ij: DMatrix<f64>,
    pub epsilon_k_ij: DMatrix<f64>,
}

// The gc-PC-SAFT parameters in an easier accessible format.
impl GcPcSaftEosParameters {
    pub fn new(parameters: &GcPcSaftParameters<f64>) -> Self {
        let component_index = parameters.component_index().into();

        let [m, sigma, epsilon_k] = parameters.collate(|pr| [pr.m, pr.sigma, pr.epsilon_k]);
        let m = m.component_mul(&parameters.segment_counts());

        let bonds = parameters
            .bonds
            .iter()
            .map(|b| ([b.id1, b.id2], b.count))
            .collect();

        let mut dipole_comp = Vec::new();
        let mut mu2 = Vec::new();
        let mut m_mix = Vec::new();
        let mut sigma_mix = Vec::new();
        let mut epsilon_k_mix = Vec::new();

        let mut m_i: DVector<f64> = DVector::zeros(parameters.molar_weight.len());
        let mut sigma_i: DVector<f64> = DVector::zeros(parameters.molar_weight.len());
        let mut epsilon_k_i: DVector<f64> = DVector::zeros(parameters.molar_weight.len());
        let mut mu2_i: DVector<f64> = DVector::zeros(parameters.molar_weight.len());
        for p in &parameters.pure {
            m_i[p.component_index] += p.model_record.m * p.count;
            sigma_i[p.component_index] += p.model_record.m * p.model_record.sigma.powi(3) * p.count;
            epsilon_k_i[p.component_index] += p.model_record.m * p.model_record.epsilon_k * p.count;
            mu2_i[p.component_index] += p.model_record.mu.powi(2);
        }
        for (i, &mu2_i) in mu2_i.iter().enumerate() {
            if mu2_i > 0.0 {
                dipole_comp.push(i);
                mu2.push(mu2_i / m_i[i] * (1e-19 * (JOULE / KELVIN / KB).into_value()));
                m_mix.push(m_i[i]);
                sigma_mix.push((sigma_i[i] / m_i[i]).cbrt());
                epsilon_k_mix.push(epsilon_k_i[i] / m_i[i]);
            }
        }

        // Combining rules dispersion
        let [k_ij] = parameters.collate_binary(|&br| [br]);
        let sigma_ij =
            DMatrix::from_fn(sigma.len(), sigma.len(), |i, j| 0.5 * (sigma[i] + sigma[j]));
        let epsilon_k_ij = DMatrix::from_fn(epsilon_k.len(), epsilon_k.len(), |i, j| {
            (epsilon_k[i] * epsilon_k[j]).sqrt()
        })
        .component_mul(&((-k_ij).add_scalar(1.0)));

        // Combining rules polar
        let s_ij = DMatrix::from_fn(dipole_comp.len(), dipole_comp.len(), |i, j| {
            0.5 * (sigma_mix[i] + sigma_mix[j])
        });
        let e_k_ij = DMatrix::from_fn(dipole_comp.len(), dipole_comp.len(), |i, j| {
            (epsilon_k_mix[i] * epsilon_k_mix[j]).sqrt()
        });

        Self {
            component_index,
            m,
            sigma,
            epsilon_k,
            bonds,
            dipole_comp: DVector::from_vec(dipole_comp),
            mu2: DVector::from_vec(mu2),
            m_mix: DVector::from_vec(m_mix),
            s_ij,
            e_k_ij,
            sigma_ij,
            epsilon_k_ij,
        }
    }
}

impl GcPcSaftEosParameters {
    pub fn phi(mut self, phi: &[f64]) -> Self {
        for i in (0..self.m.len()).combinations_with_replacement(2) {
            let (i, j) = (i[0], i[1]);
            self.epsilon_k_ij[(i, j)] *=
                (phi[self.component_index[i]] * phi[self.component_index[j]]).sqrt();
        }
        self
    }
}

impl HardSphereProperties for GcPcSaftEosParameters {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<'_, N> {
        let m = self.m.map(N::from);
        MonomerShape::Heterosegmented([m.clone(), m.clone(), m.clone(), m], &self.component_index)
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> DVector<D> {
        let ti = temperature.recip() * -3.0;
        DVector::from_fn(self.sigma.len(), |i, _| {
            -((ti * self.epsilon_k[i]).exp() * 0.12 - 1.0) * self.sigma[i]
        })
    }
}

impl AssociationStrength for GcPcSaftEosParameters {
    type Record = GcPcSaftAssociationRecord;

    fn association_strength<D: DualNum<f64> + Copy>(
        &self,
        parameters: &AssociationParameters<Self::Record>,
        state: &StateHD<D>,
        diameter: &DVector<D>,
        xi: D,
    ) -> [DMatrix<D>; 2] {
        let p = parameters;
        let t_inv = state.temperature.recip();
        let [zeta2, n3] = self.zeta(state.temperature, &state.partial_density, [2, 3]);
        let n2 = zeta2 * 6.0;
        let n3i = (-n3 + 1.0).recip();

        let mut delta_ab = DMatrix::zeros(p.sites_a.len(), p.sites_b.len());
        for b in &p.binary_ab {
            let [i, j] = [b.id1, b.id2];
            let [comp_i, comp_j] = [p.sites_a[i].assoc_comp, p.sites_b[j].assoc_comp];
            let f_ab_ij = (t_inv * b.model_record.epsilon_k_ab).exp_m1();
            let k_ab_ij =
                b.model_record.kappa_ab * (self.sigma[comp_i] * self.sigma[comp_j]).powf(1.5);

            // g_HS(d)
            let di = diameter[comp_i];
            let dj = diameter[comp_j];
            let k = di * dj / (di + dj) * (n2 * n3i);
            let g_contact = n3i * (k * xi * (k / 18.0 + 0.5) + 1.0);

            delta_ab[(i, j)] = g_contact * f_ab_ij * k_ab_ij;
        }

        let mut delta_cc = DMatrix::zeros(p.sites_c.len(), p.sites_c.len());
        for b in &p.binary_cc {
            let [i, j] = [b.id1, b.id2];
            let [comp_i, comp_j] = [p.sites_c[i].assoc_comp, p.sites_c[j].assoc_comp];
            let f_ab_ij = (t_inv * b.model_record.epsilon_k_ab).exp_m1();
            let k_ab_ij =
                b.model_record.kappa_ab * (self.sigma[comp_i] * self.sigma[comp_j]).powf(1.5);

            // g_HS(d)
            let di = diameter[comp_i];
            let dj = diameter[comp_j];
            let k = di * dj / (di + dj) * (n2 * n3i);
            let g_contact = n3i * (k * xi * (k / 18.0 + 0.5) + 1.0);

            delta_cc[(i, j)] = g_contact * f_ab_ij * k_ab_ij;
        }
        [delta_ab, delta_cc]
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::gc_pcsaft::GcPcSaftRecord;
    use feos_core::parameter::{
        AssociationRecord, BinarySegmentRecord, ChemicalRecord, Identifier, SegmentRecord,
    };

    type Pure = SegmentRecord<GcPcSaftRecord, GcPcSaftAssociationRecord>;
    type Binary = BinarySegmentRecord<f64, GcPcSaftAssociationRecord>;

    fn ch3() -> Pure {
        SegmentRecord::new(
            "CH3".into(),
            15.0,
            GcPcSaftRecord::new(0.77247, 3.6937, 181.49, 0.0, None),
        )
    }

    fn ch2() -> Pure {
        SegmentRecord::new(
            "CH2".into(),
            14.0,
            GcPcSaftRecord::new(0.7912, 3.0207, 157.23, 0.0, None),
        )
    }

    fn oh() -> Pure {
        SegmentRecord::with_association(
            "OH".into(),
            17.0,
            GcPcSaftRecord::new(1.0231, 2.7702, 334.29, 0.0, None),
            vec![AssociationRecord::new(
                Some(GcPcSaftAssociationRecord::new(0.009583, 2575.9)),
                1.0,
                1.0,
                0.0,
            )],
        )
    }

    pub fn ch3_oh() -> Binary {
        BinarySegmentRecord::new("CH3".to_string(), "OH".to_string(), Some(-0.0087))
    }

    pub fn propane() -> GcPcSaftEosParameters {
        let pure = ChemicalRecord::new(
            Identifier::new(Some("74-98-6"), Some("propane"), None, None, None, None),
            vec!["CH3".into(), "CH2".into(), "CH3".into()],
            None,
        );
        let params =
            GcPcSaftParameters::from_segments_hetero(vec![pure], &[ch3(), ch2()], None).unwrap();
        GcPcSaftEosParameters::new(&params)
    }

    pub fn propanol() -> GcPcSaftParameters<f64> {
        let pure = ChemicalRecord::new(
            Identifier::new(Some("71-23-8"), Some("1-propanol"), None, None, None, None),
            vec!["CH3".into(), "CH2".into(), "CH2".into(), "OH".into()],
            None,
        );
        GcPcSaftParameters::from_segments_hetero(vec![pure], &[ch3(), ch2(), oh()], None).unwrap()
    }

    pub fn ethanol_propanol(binary: bool) -> GcPcSaftParameters<f64> {
        let ethanol = ChemicalRecord::new(
            Identifier::new(Some("64-17-5"), Some("ethanol"), None, None, None, None),
            vec!["CH3".into(), "CH2".into(), "OH".into()],
            None,
        );
        let propanol = ChemicalRecord::new(
            Identifier::new(Some("71-23-8"), Some("1-propanol"), None, None, None, None),
            vec!["CH3".into(), "CH2".into(), "CH2".into(), "OH".into()],
            None,
        );
        let binary = if binary { Some(vec![ch3_oh()]) } else { None };
        GcPcSaftParameters::from_segments_hetero(
            vec![ethanol, propanol],
            &[ch3(), ch2(), oh()],
            binary.as_deref(),
        )
        .unwrap()
    }

    #[test]
    fn test_kij() {
        let params = ethanol_propanol(true);
        let identifiers: Vec<_> = params
            .pure
            .iter()
            .map(|r| &r.identifier)
            .enumerate()
            .collect();
        let params = GcPcSaftEosParameters::new(&params);
        let ch3 = identifiers.iter().find(|&&(_, id)| id == "CH3").unwrap();
        let ch2 = identifiers
            .iter()
            .skip(3)
            .find(|&&(_, id)| id == "CH2")
            .unwrap();
        let oh = identifiers
            .iter()
            .skip(3)
            .find(|&&(_, id)| id == "OH")
            .unwrap();
        // println!("{:?}", params.identifiers);
        // println!("{}", params.k_ij);
        // CH3 - CH2
        assert_eq!(
            params.epsilon_k_ij[(ch3.0, ch2.0)],
            (181.49f64 * 157.23).sqrt()
        );
        // CH3 - OH
        assert_eq!(
            params.epsilon_k_ij[(ch3.0, oh.0)],
            (181.49f64 * 334.29).sqrt() * 1.0087
        );
    }
}
