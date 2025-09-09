use feos_core::{FeosResult, ReferenceSystem};
use nalgebra::DVector;
use ndarray::*;
use num_dual::DualNum;
use quantity::{Density, Pressure, Temperature};

#[derive(Clone)]
pub struct IdealChainContribution {
    component_index: DVector<usize>,
    m: DVector<f64>,
}

impl IdealChainContribution {
    pub fn new(component_index: &[usize], m: &[f64]) -> Self {
        Self {
            component_index: DVector::from_column_slice(component_index),
            m: DVector::from_column_slice(m),
        }
    }

    pub fn bulk_helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        partial_density: &DVector<D>,
    ) -> D {
        let segments = self.component_index.len();
        if self.component_index[segments - 1] + 1 != segments {
            return D::zero();
        }

        // calculate segment density
        let density = self.component_index.map(|c| partial_density[c]);

        // calculate Helmholtz energy
        density
            .component_mul(&self.m.add_scalar(-1.0).map(D::from))
            .dot(&density.map(|r| (r.abs() + D::from(f64::EPSILON)).ln() - 1.0))
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
