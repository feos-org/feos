use feos_core::{FeosResult, ReferenceSystem, StateHD};
use ndarray::*;
use num_dual::DualNum;
use quantity::{Density, Pressure, Temperature};

#[derive(Clone)]
pub struct IdealChainContribution {
    component_index: Array1<usize>,
    m: Array1<f64>,
}

impl IdealChainContribution {
    pub fn new(component_index: &Array1<usize>, m: &Array1<f64>) -> Self {
        Self {
            component_index: component_index.clone(),
            m: m.clone(),
        }
    }

    pub fn name(&self) -> String {
        "Ideal chain".to_string()
    }

    pub fn helmholtz_energy<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> D {
        let segments = self.component_index.len();
        if self.component_index[segments - 1] + 1 != segments {
            return D::zero();
        }

        // calculate segment density
        let density = self.component_index.mapv(|c| state.partial_density[c]);

        // calculate Helmholtz energy
        (&density
            * &(&self.m - 1.0)
            * density.mapv(|r| (r.abs() + D::from(f64::EPSILON)).ln() - 1.0))
        .sum()
            * state.volume
    }

    pub fn helmholtz_energy_density<D, N>(
        &self,
        density: &Array<N, D::Larger>,
    ) -> FeosResult<Array<N, D>>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
        N: DualNum<f64>,
    {
        let mut phi = Array::zeros(density.raw_dim().remove_axis(Axis(0)));
        for (i, rhoi) in density.outer_iter().enumerate() {
            phi += &rhoi.mapv(|rhoi| (rhoi.ln() - 1.0) * (self.m[i] - 1.0) * rhoi);
        }
        Ok(phi)
    }

    pub fn helmholtz_energy_density_units<D>(
        &self,
        temperature: Temperature,
        density: &Density<Array<f64, D::Larger>>,
    ) -> FeosResult<Pressure<Array<f64, D>>>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
    {
        let rho = density.to_reduced();
        let t = temperature.to_reduced();
        Ok(Pressure::from_reduced(
            self.helmholtz_energy_density(&rho)? * t,
        ))
    }
}
