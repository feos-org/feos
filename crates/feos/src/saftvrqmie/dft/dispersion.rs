use crate::saftvrqmie::eos::dispersion::{Alpha, dispersion_energy_density};
use crate::saftvrqmie::parameters::SaftVRQMiePars;
use feos_core::FeosResult;
use feos_dft::{FunctionalContribution, WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use ndarray::*;
use num_dual::DualNum;

/// psi Parameter for DFT (Sauer2017)
const PSI_DFT: f64 = 1.3862;
/// psi Parameter for pDGT (Rehner2018)
const PSI_PDGT: f64 = 1.3286;

#[derive(Clone)]
pub struct AttractiveFunctional<'a> {
    parameters: &'a SaftVRQMiePars,
}

impl<'a> AttractiveFunctional<'a> {
    pub fn new(parameters: &'a SaftVRQMiePars) -> Self {
        Self { parameters }
    }
}

fn att_weight_functions<N: DualNum<f64> + Copy>(
    p: &SaftVRQMiePars,
    psi: f64,
    temperature: N,
) -> WeightFunctionInfo<N> {
    let d = p.hs_diameter(temperature);
    WeightFunctionInfo::new(Array1::from_shape_fn(d.len(), |i| i), false).add(
        WeightFunction::new_scaled(d * psi, WeightFunctionShape::Theta),
        false,
    )
}

impl<'a> FunctionalContribution for AttractiveFunctional<'a> {
    fn name(&self) -> &'static str {
        "Attractive functional"
    }

    fn weight_functions<N: DualNum<f64> + Copy>(
        &self,
        temperature: N,
    ) -> WeightFunctionInfo<N> {
        att_weight_functions(self.parameters, PSI_DFT, temperature)
    }

    fn weight_functions_pdgt<N: DualNum<f64> + Copy>(
        &self,
        temperature: N,
    ) -> WeightFunctionInfo<N> {
        att_weight_functions(self.parameters, PSI_PDGT, temperature)
    }

    fn helmholtz_energy_density<N: DualNum<f64> + Copy>(
        &self,
        temperature: N,
        density: ArrayView2<N>,
    ) -> FeosResult<Array1<N>> {
        // auxiliary variables
        let p = &self.parameters;
        let n = p.m.len();

        // temperature dependent segment radius // calc & store this in struct
        let s_eff_ij =
            Array2::from_shape_fn((n, n), |(i, j)| p.calc_sigma_eff_ij(i, j, temperature));

        // temperature dependent segment radius // calc & store this in struct
        let d_hs_ij = Array2::from_shape_fn((n, n), |(i, j)| {
            p.hs_diameter_ij(i, j, temperature, s_eff_ij[[i, j]])
        });

        // temperature dependent well depth // calc & store this in struct
        let epsilon_k_eff_ij =
            Array2::from_shape_fn((n, n), |(i, j)| p.calc_epsilon_k_eff_ij(i, j, temperature));

        // temperature dependent well depth // calc & store this in struct
        let dq_ij = Array2::from_shape_fn((n, n), |(i, j)| p.quantum_d_ij(i, j, temperature));

        // alphas .... // calc & store this in struct
        let alpha = Alpha::new(p, &s_eff_ij, &epsilon_k_eff_ij, temperature);

        let phi = density
            .axis_iter(Axis(1))
            .map(|rho_lane| {
                dispersion_energy_density(
                    p,
                    &d_hs_ij,
                    &s_eff_ij,
                    &epsilon_k_eff_ij,
                    &dq_ij,
                    &alpha,
                    &rho_lane.into_owned(),
                    temperature,
                )
            })
            .collect();
        Ok(phi)
    }
}
