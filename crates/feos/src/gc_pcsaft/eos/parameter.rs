use crate::association::AssociationStrength;
use crate::gc_pcsaft::record::{GcPcSaftAssociationRecord, GcPcSaftParameters, GcPcSaftRecord};
use crate::hard_sphere::{HardSphereProperties, MonomerShape};
use ndarray::{Array1, Array2};
use num_dual::DualNum;
use quantity::{JOULE, KB, KELVIN};

/// Parameter set required for the gc-PC-SAFT equation of state.
#[derive(Clone)]
pub struct GcPcSaftEosParameters {
    pub component_index: Array1<usize>,
    pub m: Array1<f64>,
    pub sigma: Array1<f64>,
    pub epsilon_k: Array1<f64>,
    pub bonds: Vec<([usize; 2], f64)>,
    pub dipole_comp: Array1<usize>,
    pub mu2: Array1<f64>,
    pub m_mix: Array1<f64>,
    pub s_ij: Array2<f64>,
    pub e_k_ij: Array2<f64>,

    pub sigma_ij: Array2<f64>,
    pub epsilon_k_ij: Array2<f64>,
}

// The gc-PC-SAFT parameters in an easier accessible format.
impl GcPcSaftEosParameters {
    pub fn new(parameters: &GcPcSaftParameters<f64>) -> Self {
        let component_index = parameters.component_index();

        let [m, sigma, epsilon_k] = parameters.collate(|pr| [pr.m, pr.sigma, pr.epsilon_k]);
        let m = m * parameters.segment_counts();

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

        let mut m_i: Array1<f64> = Array1::zeros(parameters.molar_weight.len());
        let mut sigma_i: Array1<f64> = Array1::zeros(parameters.molar_weight.len());
        let mut epsilon_k_i: Array1<f64> = Array1::zeros(parameters.molar_weight.len());
        let mut mu2_i: Array1<f64> = Array1::zeros(parameters.molar_weight.len());
        for p in &parameters.pure {
            m_i[p.component_index] += p.model_record.m * p.count;
            sigma_i += p.model_record.m * p.model_record.sigma.powi(3) * p.count;
            epsilon_k_i += p.model_record.m * p.model_record.epsilon_k * p.count;
            mu2_i[p.component_index] += p.model_record.mu.powi(2);
        }
        for (i, mu2_i) in mu2_i.into_iter().enumerate() {
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
            Array2::from_shape_fn([sigma.len(); 2], |(i, j)| 0.5 * (sigma[i] + sigma[j]));
        let epsilon_k_ij = Array2::from_shape_fn([epsilon_k.len(); 2], |(i, j)| {
            (epsilon_k[i] * epsilon_k[j]).sqrt()
        }) * (1.0 - &k_ij);

        // Combining rules polar
        let s_ij = Array2::from_shape_fn([dipole_comp.len(); 2], |(i, j)| {
            0.5 * (sigma_mix[i] + sigma_mix[j])
        });
        let e_k_ij = Array2::from_shape_fn([dipole_comp.len(); 2], |(i, j)| {
            (epsilon_k_mix[i] * epsilon_k_mix[j]).sqrt()
        });

        Self {
            component_index,
            m,
            sigma,
            epsilon_k,
            bonds,
            dipole_comp: Array1::from_vec(dipole_comp),
            mu2: Array1::from_vec(mu2),
            m_mix: Array1::from_vec(m_mix),
            s_ij,
            e_k_ij,
            sigma_ij,
            epsilon_k_ij,
        }
    }
}

impl GcPcSaftEosParameters {
    pub fn phi(mut self, phi: &[f64]) -> Self {
        for ((i, j), e) in self.epsilon_k_ij.indexed_iter_mut() {
            *e *= (phi[self.component_index[i]] * phi[self.component_index[j]]).sqrt();
        }
        self
    }
}

impl HardSphereProperties for GcPcSaftEosParameters {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<N> {
        let m = self.m.mapv(N::from);
        MonomerShape::Heterosegmented([m.clone(), m.clone(), m.clone(), m], &self.component_index)
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D> {
        let ti = temperature.recip() * -3.0;
        Array1::from_shape_fn(self.sigma.len(), |i| {
            -((ti * self.epsilon_k[i]).exp() * 0.12 - 1.0) * self.sigma[i]
        })
    }
}

impl AssociationStrength for GcPcSaftEosParameters {
    type Pure = GcPcSaftRecord;
    type Record = GcPcSaftAssociationRecord;

    fn association_strength<D: DualNum<f64> + Copy>(
        &self,
        temperature: D,
        comp_i: usize,
        comp_j: usize,
        assoc_ij: &Self::Record,
    ) -> D {
        let si = self.sigma[comp_i];
        let sj = self.sigma[comp_j];
        (temperature.recip() * assoc_ij.epsilon_k_ab).exp_m1()
            * assoc_ij.kappa_ab
            * (si * sj).powf(1.5)
    }

    fn combining_rule(
        _: &Self::Pure,
        _: &Self::Pure,
        parameters_i: &Self::Record,
        parameters_j: &Self::Record,
    ) -> Self::Record {
        Self::Record {
            kappa_ab: (parameters_i.kappa_ab * parameters_j.kappa_ab).sqrt(),
            epsilon_k_ab: 0.5 * (parameters_i.epsilon_k_ab + parameters_j.epsilon_k_ab),
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
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
