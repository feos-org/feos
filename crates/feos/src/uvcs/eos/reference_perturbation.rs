use super::hard_sphere::{
    diameter_wca, dimensionless_diameter_q_wca, packing_fraction, packing_fraction_a,
    packing_fraction_b,
};
use crate::uvcs::corresponding_states::CorrespondingParameters;
use feos_core::StateHD;
use num_dual::DualNum;
use std::f64::consts::PI;

/// Helmholtz energy for perturbation reference (Mayer-f), eq. 29
pub fn reference_perturbation_helmholtz_energy_density<D: DualNum<f64> + Copy>(
    parameters: &CorrespondingParameters<D>,
    state: &StateHD<D>,
) -> D {
    let p = &parameters;
    let n = p.sigma.len();
    let x = &state.molefracs;
    let d = diameter_wca(p, state.temperature);
    let eta = packing_fraction(&state.partial_density, &d);
    let eta_a = packing_fraction_a(p, eta, state.temperature);
    let eta_b = packing_fraction_b(p, eta, state.temperature);
    let mut a = D::zero();

    for i in 0..n {
        for j in 0..n {
            let rs_ij = (p.rep_ij[(i, j)] / p.att_ij[(i, j)])
                .powd((p.rep_ij[(i, j)] - p.att_ij[(i, j)]).inv());
            // let rs_ij = ((p.rep[i] / p.att[i]).powd((p.rep[i] - p.att[i]).inv())
            //     + (p.rep[j] / p.att[j]).powd((p.rep[j] - p.att[j]).inv()))
            //     * 0.5; // MIXING RULE not clear!!!
            let d_ij = (d[i] + d[j]) * 0.5; // (d[i] * p.sigma[i] + d[j] * p.sigma[j]) * 0.5;

            let t_ij = state.temperature / p.eps_k_ij[(i, j)];
            let rep_ij = p.rep_ij[(i, j)];
            let att_ij = p.att_ij[(i, j)];
            let q_ij = dimensionless_diameter_q_wca(t_ij, rep_ij, att_ij) * p.sigma_ij[(i, j)];

            a += x[i]
                * x[j]
                * ((-eta_a[(i, j)] * 0.5 + 1.0) / (-eta_a[(i, j)] + 1.0).powi(3)
                    * (-q_ij.powi(3) + (rs_ij * p.sigma_ij[(i, j)]).powi(3))
                    - ((-eta_b[(i, j)] * 0.5 + 1.0) / (-eta_b[(i, j)] + 1.0).powi(3))
                        * (-d_ij.powi(3) + (rs_ij * p.sigma_ij[(i, j)]).powi(3)))
        }
    }

    -a * state.partial_density.sum().powi(2) * 2.0 / 3.0 * PI
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::uvcs::{
        UVCSPars,
        parameters::utils::{test_parameters, test_parameters_mixture},
    };
    use approx::assert_relative_eq;
    use nalgebra::dvector;

    #[test]
    fn test_delta_a0_wca_pure() {
        let moles = dvector![2.0];
        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let reduced_volume = moles[0] / reduced_density;

        let _p = test_parameters(24.0, 6.0, 1.0, 1.0);
        let p = UVCSPars::new(&_p);
        let state = StateHD::new(reduced_temperature, reduced_volume, &moles);
        let cp = CorrespondingParameters::new(&p, state.temperature);
        let a = reference_perturbation_helmholtz_energy_density(&cp, &state) / reduced_density;
        assert_relative_eq!(a, 0.258690311450425, epsilon = 1e-10);
    }
    #[test]
    fn test_delta_a0_wca_mixture() {
        let moles = dvector![0.40000000000000002, 0.59999999999999998];
        let reduced_temperature = 1.0;
        let reduced_density = 0.90000000000000002;
        let reduced_volume = (moles[0] + moles[1]) / reduced_density;

        let _p = test_parameters_mixture(
            dvector![12.0, 12.0],
            dvector![6.0, 6.0],
            dvector![1.0, 1.0],
            dvector![1.0, 0.5],
        );
        let p = UVCSPars::new(&_p);

        let state = StateHD::new(reduced_temperature, reduced_volume, &moles);
        let cp = CorrespondingParameters::new(&p, state.temperature);
        let a =
            reference_perturbation_helmholtz_energy_density(&cp, &state) / (moles[0] + moles[1]);

        assert_relative_eq!(a, 0.308268896386771, epsilon = 1e-6);
    }
}
