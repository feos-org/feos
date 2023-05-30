use super::hard_sphere_wca::{
    diameter_wca, dimensionless_diameter_q_wca, packing_fraction, packing_fraction_a_uvb3,
    packing_fraction_b_uvb3,
};
use crate::uvtheory::parameters::*;
use feos_core::{HelmholtzEnergyDual, StateHD};
use num_dual::DualNum;
use std::fmt;
use std::{f64::consts::PI, sync::Arc};

#[derive(Debug, Clone)]
pub struct ReferencePerturbationUVB3 {
    pub parameters: Arc<UVParameters>,
}

impl fmt::Display for ReferencePerturbationUVB3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Reference Perturbation")
    }
}

impl<D: DualNum<f64> + Copy> HelmholtzEnergyDual<D> for ReferencePerturbationUVB3 {
    /// Helmholtz energy for perturbation reference (Mayer-f), eq. 29
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let p = &self.parameters;
        let n = p.sigma.len();
        let x = &state.molefracs;
        let d = diameter_wca(p, state.temperature);
        //let q = diameter_q_wca(&p, state.temperature);
        let eta = packing_fraction(&state.partial_density, &d);
        let eta_a = packing_fraction_a_uvb3(p, eta, state.temperature);
        let eta_b = packing_fraction_b_uvb3(p, eta, state.temperature);
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
mod test {
    use super::*;
    use crate::uvtheory::parameters::utils::test_parameters;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_delta_a0_uvb3_pure() {
        let moles = arr1(&[2.0]);
        // #temp = 2.0, rho = 0.5, nu = 12
        // Hard sphere adhs  1.3491645849732654
        // Delta a0           0.1130778070897391

        let reduced_temperature = 2.0;
        let reduced_density = 0.5;
        let reduced_volume = moles[0] / reduced_density;

        let p = test_parameters(12.0, 6.0, 1.0, 1.0);
        let pt = ReferencePerturbationUVB3 {
            parameters: Arc::new(p),
        };
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());
        let a = pt.helmholtz_energy(&state) / moles[0];
        dbg!(a.re());
        assert_relative_eq!(a, 0.1130778070897391, epsilon = 1e-10);

        // #temp = 3.0, rho = 1.1, nu = 20
        // Hard sphere adhs  5.458989212531771
        //Delta a0  0.3405167374787895
        let reduced_temperature = 3.0;
        let reduced_density = 1.1;
        let reduced_volume = moles[0] / reduced_density;

        let p = test_parameters(20.0, 6.0, 1.0, 1.0);
        let pt = ReferencePerturbationUVB3 {
            parameters: Arc::new(p),
        };
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());
        let a = pt.helmholtz_energy(&state) / moles[0];

        assert_relative_eq!(a, 0.3405167374787895, epsilon = 1e-10);
    }
}
