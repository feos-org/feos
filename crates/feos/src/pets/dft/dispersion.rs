use crate::hard_sphere::HardSphereProperties;
use crate::pets::Pets;
use crate::pets::eos::dispersion::{A, B};
use feos_core::FeosError;
use feos_dft::{FunctionalContribution, WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use nalgebra::DVector;
use ndarray::*;
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_6, PI};

/// psi Parameter for DFT (Heier2018)
const PSI_DFT: f64 = 1.21;
/// psi Parameter for pDGT (not adjusted, yet)
const PSI_PDGT: f64 = 1.21;

#[derive(Clone)]
pub struct AttractiveFunctional<'a> {
    parameters: &'a Pets,
}

impl<'a> AttractiveFunctional<'a> {
    pub fn new(parameters: &'a Pets) -> Self {
        Self { parameters }
    }

    fn att_weight_functions<N: DualNum<f64> + Copy>(
        &self,
        psi: f64,
        temperature: N,
    ) -> WeightFunctionInfo<N> {
        let d = self.parameters.hs_diameter(temperature);
        WeightFunctionInfo::new(DVector::from_fn(d.len(), |i, _| i), false).add(
            WeightFunction::new_scaled(d * N::from(psi), WeightFunctionShape::Theta),
            false,
        )
    }
}

impl<'a> FunctionalContribution for AttractiveFunctional<'a> {
    fn name(&self) -> &'static str {
        "Attractive functional"
    }

    fn weight_functions<N: DualNum<f64> + Copy>(&self, temperature: N) -> WeightFunctionInfo<N> {
        self.att_weight_functions(PSI_DFT, temperature)
    }

    fn weight_functions_pdgt<N: DualNum<f64> + Copy>(
        &self,
        temperature: N,
    ) -> WeightFunctionInfo<N> {
        self.att_weight_functions(PSI_PDGT, temperature)
    }

    fn helmholtz_energy_density<N: DualNum<f64> + Copy>(
        &self,
        temperature: N,
        density: ArrayView2<N>,
    ) -> Result<Array1<N>, FeosError> {
        // auxiliary variables
        let p = &self.parameters;
        let n = p.sigma.len();

        // temperature dependent segment diameter
        let d = p.hs_diameter(temperature);

        // packing fraction
        let eta = density.outer_iter().zip(d.map(|d| d.powi(3)).iter()).fold(
            Array::zeros(density.raw_dim().remove_axis(Axis(0))),
            |acc: Array1<N>, (rho, &d3)| acc + &rho * d3,
        ) * FRAC_PI_6;

        // mixture densities, crosswise interactions of all segments on all chains
        let mut rho1mix: Array1<N> = Array::zeros(eta.raw_dim());
        let mut rho2mix: Array1<N> = Array::zeros(eta.raw_dim());
        for i in 0..n {
            for j in 0..n {
                let eps_ij_t = temperature.recip() * p.epsilon_k_ij[(i, j)];
                let sigma_ij_3 = p.sigma_ij[(i, j)].powi(3);
                rho1mix = rho1mix
                    + (&density.index_axis(Axis(0), i) * &density.index_axis(Axis(0), j))
                        .mapv(|x| x * (eps_ij_t * sigma_ij_3));
                rho2mix = rho2mix
                    + (&density.index_axis(Axis(0), i) * &density.index_axis(Axis(0), j))
                        .mapv(|x| x * (eps_ij_t * eps_ij_t * sigma_ij_3));
            }
        }

        // I1, I2 and C1
        let mut i1: Array1<N> = Array::zeros(eta.raw_dim());
        let mut i2: Array1<N> = Array::zeros(eta.raw_dim());
        let mut eta_i: Array1<N> = Array::ones(eta.raw_dim());
        for i in 0..=6 {
            i1 = i1 + &eta_i * A[i];
            i2 = i2 + &eta_i * B[i];
            eta_i = &eta_i * &eta;
        }
        let c1 =
            eta.mapv(|eta| ((eta * 8.0 - eta.powi(2) * 2.0) / (eta - 1.0).powi(4) + 1.0).recip());

        // Helmholtz energy density
        Ok((-rho1mix * i1 * 2.0 - rho2mix * c1 * i2) * PI)
    }
}
