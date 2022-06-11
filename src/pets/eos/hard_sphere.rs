use crate::pets::parameters::PetsParameters;
use feos_core::{HelmholtzEnergyDual, StateHD};
use ndarray::*;
use num_dual::DualNum;
use std::fmt;
use std::rc::Rc;

impl PetsParameters {
    pub fn hs_diameter<D: DualNum<f64>>(&self, temperature: D) -> Array1<D> {
        let ti = temperature.recip() * -3.052785558;
        Array::from_shape_fn(self.sigma.len(), |i| {
            -((ti * self.epsilon_k[i]).exp() * 0.127112544 - 1.0) * self.sigma[i]
        })
    }
}

#[derive(Debug, Clone)]
pub struct HardSphere {
    pub parameters: Rc<PetsParameters>,
}

impl<D: DualNum<f64>> HelmholtzEnergyDual<D> for HardSphere {
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let d = self.parameters.hs_diameter(state.temperature);
        let zeta = zeta(&state.partial_density, &d);
        let frac_1mz3 = -(zeta[3] - 1.0).recip();
        let zeta_23 = zeta_23(&state.molefracs, &d);

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

pub fn zeta<D: DualNum<f64>>(partial_density: &Array1<D>, diameter: &Array1<D>) -> [D; 4] {
    let mut zeta: [D; 4] = [D::zero(), D::zero(), D::zero(), D::zero()];
    for i in 0..diameter.len() {
        for (k, z) in zeta.iter_mut().enumerate() {
            *z += partial_density[i] * diameter[i].powi(k as i32) * (std::f64::consts::PI / 6.0);
        }
    }
    zeta
}

pub fn zeta_23<D: DualNum<f64>>(molefracs: &Array1<D>, diameter: &Array1<D>) -> D {
    let mut zeta: [D; 2] = [D::zero(), D::zero()];
    for i in 0..diameter.len() {
        for (k, z) in zeta.iter_mut().enumerate() {
            *z += molefracs[i] * diameter[i].powi((k + 2) as i32);
        }
    }
    zeta[0] / zeta[1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pets::parameters::utils::{
        argon_krypton_parameters, argon_parameters, krypton_parameters,
    };
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn helmholtz_energy() {
        let hs = HardSphere {
            parameters: argon_parameters(),
        };
        let t = 250.0;
        let v = 1000.0;
        let n = 1.0;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a_rust = hs.helmholtz_energy(&s);
        assert_relative_eq!(a_rust, 0.410610492598808, epsilon = 1e-10);
    }

    #[test]
    fn mix() {
        let c1 = HardSphere {
            parameters: argon_parameters(),
        };
        let c2 = HardSphere {
            parameters: krypton_parameters(),
        };
        let c12 = HardSphere {
            parameters: argon_krypton_parameters(),
        };
        let t = 250.0;
        let v = 2.5e28;
        let n = 1.0;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a1 = c1.helmholtz_energy(&s);
        let a2 = c2.helmholtz_energy(&s);
        let s1m = StateHD::new(t, v, arr1(&[n, 0.0]));
        let a1m = c12.helmholtz_energy(&s1m);
        let s2m = StateHD::new(t, v, arr1(&[0.0, n]));
        let a2m = c12.helmholtz_energy(&s2m);
        assert_relative_eq!(a1, a1m, epsilon = 1e-14);
        assert_relative_eq!(a2, a2m, epsilon = 1e-14);
    }
}
