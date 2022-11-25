use crate::saftvrqmie::parameters::SaftVRQMieParameters;
use feos_core::{HelmholtzEnergyDual, StateHD};
use ndarray::*;
use num_dual::DualNum;
use std::f64::consts::TAU;
use std::fmt;
use std::sync::Arc;

const X_K21: [f64; 21] = [
    -0.995657163025808080735527280689003,
    -0.973906528517171720077964012084452,
    -0.930157491355708226001207180059508,
    -0.865063366688984510732096688423493,
    -0.780817726586416897063717578345042,
    -0.679409568299024406234327365114874,
    -0.562757134668604683339000099272694,
    -0.433395394129247190799265943165784,
    -0.294392862701460198131126603103866,
    -0.148874338981631210884826001129720,
    0.000000000000000000000000000000000,
    0.148874338981631210884826001129720,
    0.294392862701460198131126603103866,
    0.433395394129247190799265943165784,
    0.562757134668604683339000099272694,
    0.679409568299024406234327365114874,
    0.780817726586416897063717578345042,
    0.865063366688984510732096688423493,
    0.930157491355708226001207180059508,
    0.973906528517171720077964012084452,
    0.995657163025808080735527280689003,
];

const W_K21: [f64; 21] = [
    0.011694638867371874278064396062192,
    0.032558162307964727478818972459390,
    0.054755896574351996031381300244580,
    0.075039674810919952767043140916190,
    0.093125454583697605535065465083366,
    0.109387158802297641899210590325805,
    0.123491976262065851077958109831074,
    0.134709217311473325928054001771707,
    0.142775938577060080797094273138717,
    0.147739104901338491374841515972068,
    0.149445554002916905664936468389821,
    0.147739104901338491374841515972068,
    0.142775938577060080797094273138717,
    0.134709217311473325928054001771707,
    0.123491976262065851077958109831074,
    0.109387158802297641899210590325805,
    0.093125454583697605535065465083366,
    0.075039674810919952767043140916190,
    0.054755896574351996031381300244580,
    0.032558162307964727478818972459390,
    0.011694638867371874278064396062192,
];

impl SaftVRQMieParameters {
    #[inline]
    pub fn hs_diameter<D: DualNum<f64>>(&self, temperature: D) -> Array1<D> {
        let d = Array1::from_shape_fn(self.m.len(), |i| -> D {
            let sigma_eff = self.calc_sigma_eff_ij(i, i, temperature);
            self.hs_diameter_ij(i, i, temperature, sigma_eff)
        });
        d
    }

    #[inline]
    pub fn hs_diameter_ij<D: DualNum<f64>>(
        &self,
        i: usize,
        j: usize,
        temperature: D,
        sigma_eff: D,
    ) -> D {
        let r0 = self.zero_integrand(i, j, temperature, sigma_eff);
        let mut d_hs = r0;
        for k in 0..21 {
            let width = (sigma_eff - r0) * 0.5;
            let r = width * X_K21[k] + width + r0;
            let u = self.qmie_potential_ij(i, j, r, temperature);
            let f_u = -(-u[0] / temperature).exp() + 1.0;
            d_hs += width * f_u * W_K21[k];
        }
        d_hs
    }

    pub fn zero_integrand<D: DualNum<f64>>(
        &self,
        i: usize,
        j: usize,
        temperature: D,
        sigma_eff: D,
    ) -> D {
        let mut r = sigma_eff * 0.7;
        let mut f = D::zero();
        for _k in 1..20 {
            let u_vec = self.qmie_potential_ij(i, j, r, temperature);
            f = u_vec[0] / temperature + f64::EPSILON.ln();
            if f.re().abs() < 1.0e-12 {
                break;
            }
            let dfdr = u_vec[1] / temperature;
            //let d2fdr2 = u_vec[2] / temperature;
            //let mut correction = f * d2fdr2 * 0.5 / dfdr.powi(2);
            //if correction.re().abs() > 0.25 {
            //    correction = D::zero()
            //}
            //dbg!(correction.re());
            //let dr = -(f / dfdr) * (correction + 1.0);
            let mut dr = -(f / dfdr);
            if dr.re().abs() > 0.5 {
                dr *= 0.5 / dr.re().abs();
            }
            //println!("r {}, dr {}", r.re(), dr.re());
            r += dr;
        }
        if f.re().abs() > 1.0e-12 {
            println!("zero_integrand  calculation failed {}", f.re().abs());
        }
        r
    }

    #[inline]
    pub fn epsilon_k_eff<D: DualNum<f64>>(&self, temperature: D) -> Array1<D> {
        Array1::from_shape_fn(self.m.len(), |i| -> D {
            self.calc_epsilon_k_eff_ij(i, i, temperature)
        })
    }

    pub fn calc_epsilon_k_eff_ij<D: DualNum<f64>>(&self, i: usize, j: usize, temperature: D) -> D {
        let mut r = D::one() * self.sigma_ij[[i, j]];
        let mut u_vec = [D::zero(), D::zero(), D::zero()];
        for _k in 1..20 {
            u_vec = self.qmie_potential_ij(i, j, r, temperature);
            if u_vec[1].re().abs() < 1.0e-12 {
                break;
            }
            r += -u_vec[1] / u_vec[2];
        }
        if u_vec[1].re().abs() > 1.0e-12 {
            println!("calc_epsilon_k_eff_ij calculation failed");
        }
        -u_vec[0]
    }

    #[inline]
    pub fn sigma_eff<D: DualNum<f64>>(&self, temperature: D) -> Array1<D> {
        Array1::from_shape_fn(self.m.len(), |i| -> D {
            self.calc_sigma_eff_ij(i, i, temperature)
        })
    }

    pub fn calc_sigma_eff_ij<D: DualNum<f64>>(&self, i: usize, j: usize, temperature: D) -> D {
        let mut r = D::one() * self.sigma_ij[[i, j]];
        let mut u_vec = [D::zero(), D::zero(), D::zero()];
        for _k in 1..20 {
            u_vec = self.qmie_potential_ij(i, j, r, temperature);
            if u_vec[0].re().abs() < 1.0e-12 {
                break;
            }
            r += -u_vec[0] / u_vec[1];
        }
        if u_vec[0].re().abs() > 1.0e-12 {
            println!("calc_sigma_eff_ij calculation failed");
        }
        r
    }

    #[inline]
    pub fn quantum_d_ij<D: DualNum<f64>>(&self, i: usize, j: usize, temperature: D) -> D {
        let d = quantum_d_mass(self.mass_ij[[i, j]], temperature);
        d
    }

    /// Feynman-Hibbs corrected potential
    pub fn qmie_potential_ij<D: DualNum<f64>>(
        &self,
        i: usize,
        j: usize,
        r: D,
        temperature: D,
    ) -> [D; 3] {
        let lr = self.lambda_r_ij[[i, j]];
        let la = self.lambda_a_ij[[i, j]];
        let s = self.sigma_ij[[i, j]];
        let eps = self.epsilon_k_ij[[i, j]];
        let c = self.c_ij[[i, j]];
        let q1r = lr * (lr - 1.0);
        let q1a = la * (la - 1.0);
        let d = self.quantum_d_ij(i, j, temperature);
        let u = (d
            * (r.powf(lr + 2.0).recip() * q1r * s.powf(lr)
                - r.powf(la + 2.0).recip() * q1a * s.powf(la))
            + r.powf(lr).recip() * s.powf(lr)
            - r.powf(la).recip() * s.powf(la))
            * c
            * eps;

        let u_r = (d
            * (r.powf(lr + 3.0).recip() * -q1r * (lr + 2.0) * s.powf(lr)
                + r.powf(la + 3.0).recip() * q1a * (la + 2.0) * s.powf(la))
            - r.powf(lr + 1.0).recip() * lr * s.powf(lr)
            + r.powf(la + 1.0).recip() * la * s.powf(la))
            * c
            * eps;
        let u_rr = (d
            * (r.powf(lr + 4.0).recip() * q1r * (lr + 2.0) * (lr + 3.0) * s.powf(lr)
                - r.powf(la + 4.0).recip() * q1a * (la + 2.0) * (la + 3.0) * s.powf(la))
            + r.powf(lr + 2.0).recip() * lr * (lr + 1.0) * s.powf(lr)
            - r.powf(la + 2.0).recip() * la * (la + 1.0) * s.powf(la))
            * c
            * eps;
        [u, u_r, u_rr]
    }
}

#[inline]
pub fn quantum_d_mass<D: DualNum<f64>>(mass: f64, temperature: D) -> D {
    const KB: f64 = 1.380649e-23;
    let h = 6.62607015e-34;
    let d = temperature.recip() / KB * (h / TAU).powi(2) / (12.0 * mass) * 1e20;
    // TAU = 2 * PI
    d
}

pub struct HardSphere {
    pub parameters: Arc<SaftVRQMieParameters>,
}

impl<D: DualNum<f64>> HelmholtzEnergyDual<D> for HardSphere {
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let d = self.parameters.hs_diameter(state.temperature);
        let zeta = zeta(&self.parameters.m, &state.partial_density, &d);
        let frac_1mz3 = -(zeta[3] - 1.0).recip();
        let zeta_23 = zeta_23(&self.parameters.m, &state.molefracs, &d);
        state.volume * 6.0 / std::f64::consts::PI
            * (zeta[1] * zeta[2] * frac_1mz3 * 3.0
                + zeta[2].powi(2) * frac_1mz3.powi(2) * zeta_23
                + (zeta[2] * zeta_23.powi(2) - zeta[0]) * (zeta[3] * (-1.0)).ln_1p())
    }
}

impl fmt::Display for HardSphere {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hard Sphere")
    }
}

pub fn zeta<D: DualNum<f64>>(
    m: &Array1<f64>,
    partial_density: &Array1<D>,
    diameter: &Array1<D>,
) -> [D; 4] {
    let mut zeta: [D; 4] = [D::zero(), D::zero(), D::zero(), D::zero()];
    for i in 0..m.len() {
        for (k, z) in zeta.iter_mut().enumerate() {
            *z += partial_density[i]
                * diameter[i].powi(k as i32)
                * (std::f64::consts::PI / 6.0 * m[i]);
        }
    }
    zeta
}

pub fn zeta_23<D: DualNum<f64>>(m: &Array1<f64>, molefracs: &Array1<D>, diameter: &Array1<D>) -> D {
    let mut zeta: [D; 2] = [D::zero(), D::zero()];
    for i in 0..m.len() {
        for (k, z) in zeta.iter_mut().enumerate() {
            *z += molefracs[i] * diameter[i].powi((k + 2) as i32) * m[i];
        }
    }
    zeta[0] / zeta[1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::saftvrqmie::parameters::utils::h2_ne_fh1;
    use crate::saftvrqmie::parameters::utils::hydrogen_fh1;
    use approx::assert_relative_eq;
    use ndarray::arr1;
    use num_dual::Dual2;

    #[test]
    fn test_quantum_d_mass() {
        //quantum_d_mass(10.0);
        let parameters = hydrogen_fh1();
        let temperature = Dual2::from_re(26.7060).derive();
        let r = Dual2::from_re(3.5);
        let u0 = parameters.qmie_potential_ij(0, 0, r, temperature);
        let eps = 1.0e-5;
        let u2 = parameters.qmie_potential_ij(0, 0, r + eps, temperature);
        let u1 = parameters.qmie_potential_ij(0, 0, r - eps, temperature);
        let dudr_num = (u2[0].re() - u1[0].re()) / eps / 2.0;
        let d2udr2_num = (u2[1].re() - u1[1].re()) / eps / 2.0;
        assert!(((dudr_num - u0[1].re()) / u0[1].re()).abs() < 1.0e-9);
        assert!(((d2udr2_num - u0[2].re()) / u0[2].re()).abs() < 1.0e-9);
    }

    #[test]
    fn test_sigma_effective() {
        //quantum_d_mass(10.0);
        let parameters = hydrogen_fh1();
        let temperature = Dual2::from_re(26.7060).derive();
        let sigma_eff = parameters.calc_sigma_eff_ij(0, 0, temperature);
        println!("{}", sigma_eff.re() - 3.2540054024660556);
        assert!((sigma_eff.re() - 3.2540054024660556).abs() < 5.0e-7)
    }

    #[test]
    fn test_eps_div_k_effective() {
        let parameters = hydrogen_fh1();
        let temperature = Dual2::from_re(26.7060).derive();
        let epsilon_k_eff = parameters.calc_epsilon_k_eff_ij(0, 0, temperature);
        println!("{}", epsilon_k_eff.re() - 21.654396207986697);
        assert!((epsilon_k_eff.re() - 21.654396207986697).abs() < 1.0e-6)
    }

    #[test]
    fn test_zero_integrand() {
        let parameters = hydrogen_fh1();
        let temperature = Dual2::from_re(26.706).derive();
        let sigma_eff = parameters.calc_sigma_eff_ij(0, 0, temperature);
        let r0 = parameters.zero_integrand(0, 0, temperature, sigma_eff);
        println!("{}", r0.re() - 2.5265031901173732);
        assert!((r0.re() - 2.5265031901173732).abs() < 5.0e-7)
    }

    #[test]
    fn test_hs_diameter() {
        let parameters = hydrogen_fh1();
        let temperature = Dual2::from_re(26.7060).derive();
        let sigma_eff = parameters.calc_sigma_eff_ij(0, 0, temperature);
        let d_hs = parameters.hs_diameter_ij(0, 0, temperature, sigma_eff);
        assert!((d_hs.re() - 3.1410453883283341).abs() < 5.0e-8);
        assert!((d_hs.v1[0] + 8.4528823966252661e-3).abs() < 1.0e-9);
    }

    #[test]
    fn test_hs_helmholtz_energy() {
        let hs = HardSphere {
            parameters: hydrogen_fh1(),
        };
        let na = 6.02214076e23;
        let t = 26.7060;
        let v = 1.0e26;
        let n = na * 1.1;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a_rust = hs.helmholtz_energy(&s);
        dbg!(a_rust / na);
        assert_relative_eq!(a_rust / na, 0.54586730268029837, epsilon = 5e-7);
    }

    #[test]
    fn test_hs_helmholtz_energy_mix() {
        let hs = HardSphere {
            parameters: h2_ne_fh1(),
        };
        let na = 6.02214076e23;
        let t = 30.0;
        let v = 1.0e26;
        let n = [na * 1.1, na * 1.0];
        let s = StateHD::new(t, v, arr1(&n));
        let a_rust = hs.helmholtz_energy(&s);
        dbg!(a_rust / na);
        // non-additive: 1.8249307925054206
        assert_relative_eq!(a_rust / na, 1.8074833133403905, epsilon = 5e-7);
    }
}
