use crate::association::Association;
use crate::fcsaft::FcSaftOptions;
use crate::hard_sphere::{HardSphere, HardSphereProperties};
use feos_core::{Components, Molarweight, Residual};
use ndarray::Array1;
use quantity::{MolarWeight, GRAM, MOL};
use std::f64::consts::FRAC_PI_6;
use std::sync::Arc;

pub(crate) mod dispersion;
pub(crate) mod fused_chain;
mod parameter;
use dispersion::Dispersion;
use fused_chain::FusedChain;
pub use parameter::{FcSaftParameters, FcSaftRecord};

pub struct FcSaft {
    pub parameters: Arc<FcSaftParameters>,
    options: FcSaftOptions,
    hard_sphere: HardSphere<FcSaftParameters>,
    fused_chain: FusedChain,
    dispersion: Dispersion,
    association: Option<Association<FcSaftParameters>>,
    model_params: Option<[[f64; 7]; 4]>,
}

impl FcSaft {
    pub fn new(parameters: Arc<FcSaftParameters>) -> Self {
        Self::with_options(parameters, FcSaftOptions::default(), None)
    }

    pub fn with_options(
        parameters: Arc<FcSaftParameters>,
        options: FcSaftOptions,
        model_params: Option<[[f64; 7]; 4]>,
    ) -> Self {
        let hard_sphere = HardSphere::new(&parameters);
        let fused_chain = FusedChain::new(&parameters);
        let dispersion = Dispersion::new(&parameters, model_params);
        let association = (!parameters.association.is_empty()).then(|| {
            Association::new(
                &parameters,
                &parameters.association,
                options.max_iter_cross_assoc,
                options.tol_cross_assoc,
            )
        });
        Self {
            parameters,
            options,
            hard_sphere,
            fused_chain,
            dispersion,
            association,
            model_params,
        }
    }
}

impl Components for FcSaft {
    fn components(&self) -> usize {
        self.parameters.molarweight.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            Arc::new(self.parameters.subset(component_list)),
            self.options,
            self.model_params,
        )
    }
}

impl Residual for FcSaft {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        let p = &self.parameters;
        let moles_segments: Array1<f64> = p.component_index.iter().map(|&i| moles[i]).collect();
        let [_, v] = p.geometry_coefficients(&p.sigma);
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * p.sigma.mapv(|v| v.powi(3)) * v * moles_segments).sum()
    }

    fn residual_helmholtz_energy_contributions<D: num_dual::DualNum<f64> + Copy>(
        &self,
        state: &feos_core::StateHD<D>,
    ) -> Vec<(String, D)> {
        let mut v = Vec::with_capacity(7);
        let d = self.parameters.hs_diameter(state.temperature);

        v.push((
            self.hard_sphere.to_string(),
            self.hard_sphere.helmholtz_energy(state),
        ));
        v.push((
            self.fused_chain.to_string(),
            self.fused_chain.helmholtz_energy(state),
        ));
        v.push((
            self.dispersion.to_string(),
            self.dispersion.helmholtz_energy(state),
        ));
        if let Some(association) = self.association.as_ref() {
            v.push((
                association.to_string(),
                association.helmholtz_energy(state, &d),
            ))
        }
        v
    }
}

impl Molarweight for FcSaft {
    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}
