use crate::gc_pcsaft::GcPcSaftParameters;
use nalgebra::{DMatrix, DVector};
use petgraph::graph::{Graph, UnGraph};

/// psi Parameter for heterosegmented DFT (Mairhofer2018)
const PSI_GC_DFT: f64 = 1.5357;

/// Parameter set required for the gc-PC-SAFT Helmholtz energy functional.
pub struct GcPcSaftFunctionalParameters {
    pub component_index: DVector<usize>,
    pub m: DVector<f64>,
    pub sigma: DVector<f64>,
    pub epsilon_k: DVector<f64>,
    pub bonds: UnGraph<(), ()>,
    pub psi_dft: DVector<f64>,
    pub sigma_ij: DMatrix<f64>,
    pub epsilon_k_ij: DMatrix<f64>,
}

impl GcPcSaftFunctionalParameters {
    pub fn new(parameters: &GcPcSaftParameters<()>) -> Self {
        let component_index = parameters.component_index().into();

        let [m, sigma, epsilon_k] = parameters.collate(|pr| [pr.m, pr.sigma, pr.epsilon_k]);
        let [psi_dft] = parameters.collate(|pr| [pr.psi_dft.unwrap_or(PSI_GC_DFT)]);

        let bonds = Graph::from_edges(
            parameters
                .bonds
                .iter()
                .map(|b| (b.id1 as u32, b.id2 as u32)),
        );

        // Combining rules dispersion
        let [k_ij] = parameters.collate_binary(|&br| [br]);
        let sigma_ij =
            DMatrix::from_fn(sigma.len(), sigma.len(), |i, j| 0.5 * (sigma[i] + sigma[j]));
        let epsilon_k_ij = DMatrix::from_fn(epsilon_k.len(), epsilon_k.len(), |i, j| {
            (epsilon_k[i] * epsilon_k[j]).sqrt() * (1.0 - k_ij[(i, j)])
        });
        Self {
            component_index,
            m,
            sigma,
            epsilon_k,
            bonds,
            psi_dft,
            sigma_ij,
            epsilon_k_ij,
        }
    }
}
