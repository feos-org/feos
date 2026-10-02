use super::record::{FcSaftAssociationRecord, FcSaftParameters};
use crate::association::AssociationStrength;
use crate::hard_sphere::{HardSphereProperties, MonomerShape};
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use petgraph::graph::{NodeIndex, UnGraph};

/// psi parameter for the dispersion functional of fused chains.
const PSI_FUSED_CHAINS: f64 = 1.5;

/// The FC-SAFT parameters in an easier accessible format.
pub struct FcSaftPars {
    pub component_index: DVector<usize>,
    pub sigma: DVector<f64>,
    pub epsilon_k: DVector<f64>,
    pub bonds: UnGraph<(), f64>,
    pub psi_dft: DVector<f64>,
    pub k_ij: DMatrix<f64>,
    pub sigma_ij: DMatrix<f64>,
    pub epsilon_k_ij: DMatrix<f64>,
}

impl FcSaftPars {
    pub fn new(parameters: &FcSaftParameters) -> Self {
        let component_index = parameters.component_index().into();

        let [sigma, epsilon_k] = parameters.collate(|pr| [pr.sigma, pr.epsilon_k]);
        let [psi_dft] = parameters.collate(|pr| [pr.psi_dft.unwrap_or(PSI_FUSED_CHAINS)]);

        // every segment is a node in the graph, so that single-segment molecules are included
        let n = sigma.len();
        let mut bonds = UnGraph::with_capacity(n, parameters.bonds.len());
        (0..n).for_each(|_| {
            bonds.add_node(());
        });
        for b in &parameters.bonds {
            bonds.add_edge(
                NodeIndex::new(b.id1),
                NodeIndex::new(b.id2),
                b.model_record.bond_length,
            );
        }

        // Combining rules dispersion
        let [k_ij] = parameters.collate_binary(|br| [br.k_ij]);
        let sigma_ij = DMatrix::from_fn(n, n, |i, j| 0.5 * (sigma[i] + sigma[j]));
        let epsilon_k_ij = DMatrix::from_fn(n, n, |i, j| {
            (epsilon_k[i] * epsilon_k[j]).sqrt() * (1.0 - k_ij[(i, j)])
        });

        Self {
            component_index,
            sigma,
            epsilon_k,
            bonds,
            psi_dft,
            k_ij,
            sigma_ij,
            epsilon_k_ij,
        }
    }

    /// The geometry coefficients of the fused spheres: the reduced surface $a_\alpha$
    /// and the reduced volume $v_\alpha$ of every segment.
    pub fn fused_sphere_coefficients<D: DualNum<Primitive = f64> + Copy>(
        &self,
        diameter: &DVector<D>,
    ) -> [DVector<D>; 2] {
        let mut a = DVector::from_element(diameter.len(), D::one());
        let mut v = DVector::from_element(diameter.len(), D::one());
        for e in self.bonds.edge_indices() {
            let (n1, n2) = self.bonds.edge_endpoints(e).unwrap();
            let l12 = self.bonds[e];
            for (i, j) in [(n1.index(), n2.index()), (n2.index(), n1.index())] {
                let (d1, d2) = (diameter[i], diameter[j]);
                let delta12 = (d1.powi(2) - d2.powi(2) + 4.0 * l12.powi(2)) / (8.0 * l12);
                a[i] -= (-delta12 * 2.0 / d1 + 1.0) * 0.5;
                v[i] -= (-delta12 * 3.0 / d1 + (delta12 / d1).powi(3) * 4.0 + 1.0) * 0.5;
            }
        }
        [a, v]
    }
}

impl HardSphereProperties for FcSaftPars {
    fn monomer_shape<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
    ) -> MonomerShape<'_, N> {
        let [a, v] = self.fused_sphere_coefficients(&self.hs_diameter(temperature));
        MonomerShape::Heterosegmented(
            [DVector::from_element(a.len(), N::one()), a.clone(), a, v],
            &self.component_index,
        )
    }

    fn hs_diameter<D: DualNum<Primitive = f64> + Copy>(&self, temperature: D) -> DVector<D> {
        let ti = temperature.recip() * -3.0;
        DVector::from_fn(self.sigma.len(), |i, _| {
            -((ti * self.epsilon_k[i]).exp() * 0.12 - 1.0) * self.sigma[i]
        })
    }
}

impl AssociationStrength for FcSaftPars {
    type Record = FcSaftAssociationRecord;

    fn association_strength_ij<D: DualNum<Primitive = f64> + Copy>(
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
}
