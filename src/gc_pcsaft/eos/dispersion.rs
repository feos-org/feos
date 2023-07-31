use super::GcPcSaftEosParameters;
use crate::hard_sphere::HardSphereProperties;
use feos_core::{HelmholtzEnergyDual, StateHD};
use num_dual::DualNum;
use std::f64::consts::PI;
use std::fmt;
use std::sync::Arc;

pub const A0: [f64; 7] = [
    0.91056314451539,
    0.63612814494991,
    2.68613478913903,
    -26.5473624914884,
    97.7592087835073,
    -159.591540865600,
    91.2977740839123,
];
pub const A1: [f64; 7] = [
    -0.30840169182720,
    0.18605311591713,
    -2.50300472586548,
    21.4197936296668,
    -65.2558853303492,
    83.3186804808856,
    -33.7469229297323,
];
pub const A2: [f64; 7] = [
    -0.09061483509767,
    0.45278428063920,
    0.59627007280101,
    -1.72418291311787,
    -4.13021125311661,
    13.7766318697211,
    -8.67284703679646,
];
pub const B0: [f64; 7] = [
    0.72409469413165,
    2.23827918609380,
    -4.00258494846342,
    -21.00357681484648,
    26.8556413626615,
    206.5513384066188,
    -355.60235612207947,
];
pub const B1: [f64; 7] = [
    -0.57554980753450,
    0.69950955214436,
    3.89256733895307,
    -17.21547164777212,
    192.6722644652495,
    -161.8264616487648,
    -165.2076934555607,
];
pub const B2: [f64; 7] = [
    0.09768831158356,
    -0.25575749816100,
    -9.15585615297321,
    20.64207597439724,
    -38.80443005206285,
    93.6267740770146,
    -29.66690558514725,
];

#[derive(Clone)]
pub struct Dispersion {
    pub parameters: Arc<GcPcSaftEosParameters>,
}

impl<D: DualNum<f64> + Copy> HelmholtzEnergyDual<D> for Dispersion {
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        // auxiliary variables
        let p = &self.parameters;
        let n = p.m.len();
        let rho = &state.partial_density;

        // packing fraction
        let eta = p.zeta(state.temperature, &state.partial_density, [3])[0];

        // mean segment number
        let m =
            p.m.iter()
                .zip(p.component_index.iter())
                .map(|(&m, &i)| state.molefracs[i] * m)
                .sum::<D>();

        // mixture densities, crosswise interactions of all segments on all chains
        let mut rho1mix = D::zero();
        let mut rho2mix = D::zero();
        for i in 0..n {
            for j in 0..n {
                let eps_ij = state.temperature.recip() * self.parameters.epsilon_k_ij[(i, j)];
                let sigma_ij = self.parameters.sigma_ij[(i, j)].powi(3);
                let rho1 = rho[p.component_index[i]]
                    * rho[p.component_index[j]]
                    * (eps_ij * p.m[i] * p.m[j] * sigma_ij);
                rho1mix += rho1;
                rho2mix += rho1 * eps_ij;
            }
        }

        // I1, I2 and C1
        let mut i1 = D::zero();
        let mut i2 = D::zero();
        let mut eta_i = D::one();
        let m1 = (m - 1.0) / m;
        let m2 = (m - 2.0) / m * m1;
        for i in 0..=6 {
            i1 += (m2 * A2[i] + m1 * A1[i] + A0[i]) * eta_i;
            i2 += (m2 * B2[i] + m1 * B1[i] + B0[i]) * eta_i;
            eta_i *= eta;
        }
        let c1 = (m * (eta * 8.0 - eta.powi(2) * 2.0) / (eta - 1.0).powi(4)
            + (D::one() - m)
                * (eta * 20.0 - eta.powi(2) * 27.0 + eta.powi(3) * 12.0 - eta.powi(4) * 2.0)
                / ((eta - 1.0) * (eta - 2.0)).powi(2)
            + 1.0)
            .recip();

        // Helmholtz energy
        (-rho1mix * i1 * 2.0 - rho2mix * m * c1 * i2) * PI * state.volume
    }
}

impl fmt::Display for Dispersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dispersion (GC)")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gc_pcsaft::eos::parameter::test::*;
    use approx::assert_relative_eq;
    use feos_core::si::{Pressure, METER, MOL, PASCAL};
    use ndarray::arr1;
    use num_dual::Dual64;
    use typenum::P3;

    #[test]
    fn test_dispersion_propane() {
        let parameters = propane();
        let contrib = Dispersion {
            parameters: Arc::new(parameters),
        };
        let temperature = 300.0;
        let volume = METER.powi::<P3>().to_reduced();
        let moles = (1.5 * MOL).to_reduced();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derivative(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let pressure = Pressure::from_reduced(-contrib.helmholtz_energy(&state).eps * temperature);
        assert_relative_eq!(pressure, -2.846724434944439 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn test_dispersion_propanol() {
        let parameters = propanol();
        let contrib = Dispersion {
            parameters: Arc::new(parameters),
        };
        let temperature = 300.0;
        let volume = METER.powi::<P3>().to_reduced();
        let moles = (1.5 * MOL).to_reduced();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derivative(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let pressure = Pressure::from_reduced(-contrib.helmholtz_energy(&state).eps * temperature);
        assert_relative_eq!(pressure, -5.432173507270732 * PASCAL, max_relative = 1e-10);
    }
}
