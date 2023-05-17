use super::PcSaftParameters;
use crate::hard_sphere::HardSphereProperties;
use feos_core::{HelmholtzEnergyDual, StateHD};
use num_dual::*;
use std::fmt;
use std::sync::Arc;

const VMAX: f64 = 1.1549055629071023;

#[cfg_attr(feature = "python", pyo3::pyclass)]
#[derive(Clone, Debug, Copy)]
pub struct ElasticParameters {
    pub n_linker: f64,
    pub n_chain: f64,
    pub v0: f64,
    pub phi: f64,
}


pub struct Elastic {
    pub parameters: Arc<PcSaftParameters>,
    pub elastic_parameters: ElasticParameters,
}

impl<D: DualNum<f64>> HelmholtzEnergyDual<D> for Elastic {
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let m = self.parameters.m[0];
        let d = self.parameters.hs_diameter(state.temperature)[0];
        let v = state.partial_density[0].recip() * self.elastic_parameters.n_chain;
        let v_max = (d * m * VMAX).powi(3) * self.elastic_parameters.n_linker;
        let v0 = self.elastic_parameters.v0;
        let phi = self.elastic_parameters.phi;
        state.moles[0] * (phi - 2.0) / phi
            * (((v / v0).powf(2.0 / 3.0) - 1.0)
            / (-(v / v_max).powf(2.0 / 3.0) + 1.0)
            * 1.5
            - (v / v0).ln())
    }
}

impl fmt::Display for Elastic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Elastic")
    }
}
