use super::GcPcSaftEosParameters;
use crate::hard_sphere::HardSphereProperties;
use feos_core::{HelmholtzEnergyDual, StateHD};
use num_dual::*;
use std::fmt;
use std::sync::Arc;

#[derive(Clone)]
pub struct HardChain {
    pub parameters: Arc<GcPcSaftEosParameters>,
}

impl<D: DualNum<f64>> HelmholtzEnergyDual<D> for HardChain {
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        // temperature dependent segment diameter
        let diameter = self.parameters.hs_diameter(state.temperature);

        // Packing fractions
        let [zeta2, zeta3] =
            self.parameters
                .zeta(state.temperature, &state.partial_density, [2, 3]);

        // Helmholtz energy
        let frac_1mz3 = -(zeta3 - 1.0).recip();
        let c = zeta2 * frac_1mz3 * frac_1mz3;
        self.parameters
            .bonds
            .iter()
            .map(|([i, j], count)| {
                let (di, dj) = (diameter[*i], diameter[*j]);
                let cdij = c * di * dj / (di + dj);
                let g = frac_1mz3 + cdij * 3.0 - cdij * cdij * (zeta3 - 1.0) * 2.0;
                -state.moles[self.parameters.component_index[*i]] * *count * g.ln()
            })
            .sum()
    }
}

impl fmt::Display for HardChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hard Chain (GC)")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gc_pcsaft::eos::parameter::test::*;
    use approx::assert_relative_eq;
    use feos_core::EosUnit;
    use ndarray::arr1;
    use num_dual::Dual64;
    use quantity::si::{METER, MOL, PASCAL};

    #[test]
    fn test_hc_propane() {
        let parameters = propane();
        let contrib = HardChain {
            parameters: Arc::new(parameters),
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
        assert_relative_eq!(
            pressure,
            -7.991735636207462e-1 * PASCAL,
            max_relative = 1e-10
        );
    }

    #[test]
    fn test_hc_propanol() {
        let parameters = propanol();
        let contrib = HardChain {
            parameters: Arc::new(parameters),
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
        assert_relative_eq!(pressure, -1.2831486124723626 * PASCAL, max_relative = 1e-10);
    }
}
