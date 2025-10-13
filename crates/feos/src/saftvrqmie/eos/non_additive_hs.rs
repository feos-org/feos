use super::TemperatureDependentProperties;
use crate::saftvrqmie::eos::hard_sphere::zeta;
use crate::saftvrqmie::parameters::SaftVRQMiePars;
use feos_core::StateHD;
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use std::f64::consts::PI;

pub struct NonAddHardSphere;

impl NonAddHardSphere {
    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        parameters: &SaftVRQMiePars,
        state: &StateHD<D>,
        properties: &TemperatureDependentProperties<D>,
    ) -> D {
        let p = parameters;
        let n = p.m.len();
        let d_hs_ij = &properties.hs_diameter_ij;

        // Additive hard-sphere diameter
        let d_hs_add_ij = DMatrix::from_fn(n, n, |i, j| (d_hs_ij[(i, i)] + d_hs_ij[(j, j)]) * 0.5);

        let rho_s = DVector::from_fn(n, |i, _| state.partial_density[i] * p.m[i]).sum();
        rho_s * reduced_non_additive_hs_energy(p, d_hs_ij, &d_hs_add_ij, &state.partial_density)
    }
}

pub fn reduced_non_additive_hs_energy<D: DualNum<f64> + Copy>(
    parameters: &SaftVRQMiePars,
    d_hs_ij: &DMatrix<D>,
    d_hs_add_ij: &DMatrix<D>,
    rho: &DVector<D>,
) -> D {
    // auxiliary variables
    let n = rho.len();
    let p = &parameters;
    let d = DVector::from_fn(n, |i, _| d_hs_ij[(i, i)]);
    let zeta = zeta(&p.m, rho, &d);
    let frac_1mz3 = -(zeta[3] - 1.0).recip();
    let g_hs_ij = DMatrix::from_fn(n, n, |i, j| {
        let mu = d[i] * d[j] / (d[i] + d[j]);
        frac_1mz3
            + mu * zeta[2] * frac_1mz3.powi(2) * 3.0
            + (mu * zeta[2]).powi(2) * frac_1mz3.powi(3) * 2.0
    });

    // overall density
    let rho_s = DVector::from_fn(n, |i, _| -> D { rho[i] * p.m[i] }).sum();
    // segment fractions
    let x_s = DVector::from_fn(n, |i, _| -> D { rho[i] * p.m[i] / rho_s });

    DMatrix::from_fn(n, n, |i, j| {
        -rho_s
            * x_s[i]
            * x_s[j]
            * d_hs_add_ij[(i, j)].powi(2)
            * g_hs_ij[(i, j)]
            * (d_hs_add_ij[(i, j)] - d_hs_ij[(i, j)])
            * 2.0
            * PI
    })
    .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::saftvrqmie::parameters::utils::h2_ne_fh;
    use crate::saftvrqmie::parameters::utils::hydrogen_fh;
    use approx::assert_relative_eq;
    use nalgebra::dvector;

    #[test]
    fn test_non_add_hs_helmholtz_energy() {
        let parameters = hydrogen_fh("1");
        let na = 6.02214076e23;
        let t = 26.7060;
        let v = 1.0e26;
        let n = na * 1.1;
        let s = StateHD::new(t, v / n, &dvector![1.0]);
        let properties = TemperatureDependentProperties::new(&parameters, s.temperature);
        let a_rust = NonAddHardSphere.helmholtz_energy_density(&parameters, &s, &properties) * v;
        dbg!(a_rust / na);
        assert_relative_eq!(a_rust / na, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_non_add_hs_helmholtz_energy_mix() {
        let parameters = h2_ne_fh("1");
        let na = 6.02214076e23;
        let t = 30.0;
        let v = 1.0e26;
        let n = dvector![na * 1.1, na * 1.0];
        let s = StateHD::new(t, v / n.sum(), &(&n / n.sum()));
        let properties = TemperatureDependentProperties::new(&parameters, s.temperature);
        let a_rust = NonAddHardSphere.helmholtz_energy_density(&parameters, &s, &properties) * v;
        dbg!(a_rust / na);
        assert_relative_eq!(a_rust / na, 1.7874359117834266E-002, epsilon = 5e-7);
    }
}
