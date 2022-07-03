use super::GcPcSaftEosParameters;
use feos_core::{HelmholtzEnergyDual, StateHD};
use ndarray::*;
use num_dual::DualNum;
use std::f64::consts::FRAC_PI_6;
use std::fmt;
use std::rc::Rc;

impl GcPcSaftEosParameters {
    pub fn hs_diameter<D: DualNum<f64>>(&self, temperature: D) -> Array1<D> {
        let ti = temperature.recip() * -3.0;
        Array::from_shape_fn(self.sigma.len(), |i| {
            -((ti * self.epsilon_k[i]).exp() * 0.12 - 1.0) * self.sigma[i]
        })
    }
}

#[derive(Clone)]
pub struct HardSphere {
    pub parameters: Rc<GcPcSaftEosParameters>,
}

impl<D: DualNum<f64>> HelmholtzEnergyDual<D> for HardSphere {
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let diameter = self.parameters.hs_diameter(state.temperature);
        let zeta = self
            .parameters
            .zeta(&diameter, &state.partial_density, [0, 1, 2, 3]);
        let frac_1mz3 = -(zeta[3] - 1.0).recip();
        let zeta_23 = self.parameters.zeta_23(&diameter, &state.molefracs);
        state.volume * 6.0 / std::f64::consts::PI
            * (zeta[1] * zeta[2] * frac_1mz3 * 3.0
                + zeta[2].powi(2) * frac_1mz3.powi(2) * zeta_23
                + (zeta[2] * zeta_23.powi(2) - zeta[0]) * (zeta[3] * (-1.0)).ln_1p())
    }
}

impl fmt::Display for HardSphere {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hard Sphere (GC)")
    }
}

impl GcPcSaftEosParameters {
    pub(crate) fn zeta<D: DualNum<f64>, const N: usize>(
        &self,
        diameter: &Array1<D>,
        partial_density: &Array1<D>,
        k: [i32; N],
    ) -> [D; N] {
        let mut zeta = [D::zero(); N];
        for i in 0..self.m.len() {
            for (z, &k) in zeta.iter_mut().zip(k.iter()) {
                *z += partial_density[self.component_index[i]]
                    * diameter[i].powi(k)
                    * (FRAC_PI_6 * self.m[i]);
            }
        }

        zeta
    }

    fn zeta_23<D: DualNum<f64>>(&self, diameter: &Array1<D>, molefracs: &Array1<D>) -> D {
        let mut zeta: [D; 2] = [D::zero(); 2];
        for i in 0..self.m.len() {
            for (k, z) in zeta.iter_mut().enumerate() {
                *z += molefracs[self.component_index[i]]
                    * diameter[i].powi((k + 2) as i32)
                    * (FRAC_PI_6 * self.m[i]);
            }
        }

        zeta[0] / zeta[1]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gc_pcsaft::eos::parameter::test::*;
    use approx::assert_relative_eq;
    use feos_core::EosUnit;
    use num_dual::Dual64;
    use quantity::si::{METER, MOL, PASCAL};

    #[test]
    fn test_hs_propane() {
        let parameters = propane();
        let contrib = HardSphere {
            parameters: Rc::new(parameters),
        };
        let temperature = 300.0;
        let volume = METER
            .powi(3)
            .to_reduced(EosUnit::reference_volume())
            .unwrap();
        let moles = (1.5 * MOL).to_reduced(EosUnit::reference_moles()).unwrap();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derive(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let pressure =
            -contrib.helmholtz_energy(&state).eps[0] * temperature * EosUnit::reference_pressure();
        assert_relative_eq!(pressure, 1.5285037907989527 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn test_hs_propanol() {
        let parameters = propanol();
        let contrib = HardSphere {
            parameters: Rc::new(parameters),
        };
        let temperature = 300.0;
        let volume = METER
            .powi(3)
            .to_reduced(EosUnit::reference_volume())
            .unwrap();
        let moles = (1.5 * MOL).to_reduced(EosUnit::reference_moles()).unwrap();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derive(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let pressure =
            -contrib.helmholtz_energy(&state).eps[0] * temperature * EosUnit::reference_pressure();
        assert_relative_eq!(pressure, 2.3168212018200243 * PASCAL, max_relative = 1e-10);
    }
}
