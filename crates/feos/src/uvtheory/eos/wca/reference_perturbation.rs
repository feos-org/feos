use super::WeeksChandlerAndersen;
use super::hard_sphere::{
    dimensionless_diameter_q_wca, packing_fraction, packing_fraction_a, packing_fraction_b,
};
use crate::uvtheory::parameters::*;
use feos_core::StateHD;
use num_dual::DualNum;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub(super) struct ReferencePerturbation;

impl ReferencePerturbation {
    /// Helmholtz energy for perturbation reference (Mayer-f), eq. 29
    pub fn helmholtz_energy<D: DualNum<f64> + Copy>(
        &self,
        parameters: &UVTheoryPars,
        state: &StateHD<D>,
    ) -> D {
        let p = parameters;
        let n = p.sigma.len();
        let x = &state.molefracs;
        let d = WeeksChandlerAndersen::diameter_wca(p, state.temperature);
        //let q = diameter_q_wca(&p, state.temperature);
        let eta = packing_fraction(&state.partial_density, &d);
        let eta_a = packing_fraction_a(p, eta, state.temperature);
        let eta_b = packing_fraction_b(p, eta, state.temperature);
        let mut a = D::zero();

        for i in 0..n {
            for j in 0..n {
                let rs_ij = ((p.rep[i] / p.att[i]).powf(1.0 / (p.rep[i] - p.att[i]))
                    + (p.rep[j] / p.att[j]).powf(1.0 / (p.rep[j] - p.att[j])))
                    * 0.5; // MIXING RULE not clear!!!
                let d_ij = (d[i] + d[j]) * 0.5; // (d[i] * p.sigma[i] + d[j] * p.sigma[j]) * 0.5;

                let t_ij = state.temperature / p.eps_k_ij[[i, j]];
                let rep_ij = p.rep_ij[[i, j]];
                let att_ij = p.att_ij[[i, j]];
                let q_ij = dimensionless_diameter_q_wca(t_ij, D::from(rep_ij), D::from(att_ij))
                    * p.sigma_ij[[i, j]];

                a += x[i]
                    * x[j]
                    * ((-eta_a[[i, j]] * 0.5 + 1.0) / (-eta_a[[i, j]] + 1.0).powi(3)
                        * (-q_ij.powi(3) + (rs_ij * p.sigma_ij[[i, j]]).powi(3))
                        - ((-eta_b[[i, j]] * 0.5 + 1.0) / (-eta_b[[i, j]] + 1.0).powi(3))
                            * (-d_ij.powi(3) + (rs_ij * p.sigma_ij[[i, j]]).powi(3)))
            }
        }

        -a * state.moles.sum().powi(2) * 2.0 / 3.0 / state.volume * PI
    }
}

#[cfg(test)]
#[expect(clippy::excessive_precision)]
mod test {
    use super::*;
    use crate::uvtheory::Perturbation::WeeksChandlerAndersen as WCA;
    use crate::uvtheory::parameters::utils::{test_parameters, test_parameters_mixture};
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_delta_a0_wca_pure() {
        let moles = arr1(&[2.0]);

        // m = 12.0, t = 4.0, rho = 1.0

        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let reduced_volume = moles[0] / reduced_density;

        let p = test_parameters(24.0, 6.0, 1.0, 1.0, WCA);
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());
        let a = ReferencePerturbation.helmholtz_energy(&p, &state) / moles[0];
        assert_relative_eq!(a, 0.258690311450425, epsilon = 1e-10);
    }
    #[test]
    fn test_delta_a0_wca_mixture() {
        let moles = arr1(&[0.40000000000000002, 0.59999999999999998]);
        let reduced_temperature = 1.0;
        let reduced_density = 0.90000000000000002;
        let reduced_volume = (moles[0] + moles[1]) / reduced_density;

        let p = UVTheoryPars::new(
            &test_parameters_mixture(
                arr1(&[12.0, 12.0]),
                arr1(&[6.0, 6.0]),
                arr1(&[1.0, 1.0]),
                arr1(&[1.0, 0.5]),
            ),
            WCA,
        );

        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());
        let a = ReferencePerturbation.helmholtz_energy(&p, &state) / (moles[0] + moles[1]);

        assert_relative_eq!(a, 0.308268896386771, epsilon = 1e-6);
    }
}
