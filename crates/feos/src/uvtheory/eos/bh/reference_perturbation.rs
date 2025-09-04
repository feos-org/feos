use super::BarkerHenderson;
use super::hard_sphere::{packing_fraction, packing_fraction_a, packing_fraction_b};
use crate::uvtheory::parameters::UVTheoryPars;
use feos_core::StateHD;
use num_dual::DualNum;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub(super) struct ReferencePerturbation;

impl ReferencePerturbation {
    /// Helmholtz energy for perturbation reference (Mayer-f), eq. 29
    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        parameters: &UVTheoryPars,
        state: &StateHD<D>,
    ) -> D {
        let p = parameters;
        let n = p.sigma.len();
        let x = &state.molefracs;
        let d = BarkerHenderson::diameter_bh(p, state.temperature);
        let eta = packing_fraction(&state.partial_density, &d);
        let eta_a = packing_fraction_a(p, &d, eta);
        let eta_b = packing_fraction_b(p, &d, eta);
        let mut a = D::zero();
        for i in 0..n {
            for j in 0..n {
                let d_ij = (d[i] + d[j]) * 0.5; // (d[i] * p.sigma[i] + d[j] * p.sigma[j]) * 0.5;
                a += x[i]
                    * x[j]
                    * (((-eta_a[(i, j)] * 0.5 + 1.0) / (-eta_a[(i, j)] + 1.0).powi(3))
                        - ((-eta_b[(i, j)] * 0.5 + 1.0) / (-eta_b[(i, j)] + 1.0).powi(3)))
                    * (-d_ij.powi(3) + p.sigma_ij[(i, j)].powi(3))
            }
        }

        -a * state.partial_density.sum().powi(2) * 2.0 / 3.0 * PI
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::uvtheory::{Perturbation, parameters::utils::test_parameters};
    use approx::assert_relative_eq;
    use nalgebra::dvector;

    #[test]
    fn test_delta_a0_bh() {
        // m = 12.0, t = 4.0, rho = 1.0
        let reduced_temperature = 4.0;
        let reduced_density = 1.0;

        let p = test_parameters(24.0, 6.0, 1.0, 1.0, Perturbation::BarkerHenderson);
        let state = StateHD::new(reduced_temperature, 1.0 / reduced_density, &dvector![1.0]);
        let a = ReferencePerturbation.helmholtz_energy_density(&p, &state) / reduced_density;
        assert_relative_eq!(a, -0.0611105573289734, epsilon = 1e-10);
    }
}
