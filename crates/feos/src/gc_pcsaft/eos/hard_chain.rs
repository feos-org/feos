use super::GcPcSaftEosParameters;
use crate::hard_sphere::HardSphereProperties;
use feos_core::StateHD;
use num_dual::*;

pub(super) struct HardChain;

impl HardChain {
    pub(super) fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        parameters: &GcPcSaftEosParameters,
        state: &StateHD<D>,
    ) -> D {
        // temperature dependent segment diameter
        let diameter = parameters.hs_diameter(state.temperature);

        // Packing fractions
        let [zeta2, zeta3] = parameters.zeta(state.temperature, &state.partial_density, [2, 3]);

        // Helmholtz energy
        let frac_1mz3 = -(zeta3 - 1.0).recip();
        let c = zeta2 * frac_1mz3 * frac_1mz3;
        parameters
            .bonds
            .iter()
            .map(|&([i, j], count)| {
                let (di, dj) = (diameter[i], diameter[j]);
                let cdij = c * di * dj / (di + dj);
                let g = frac_1mz3 + cdij * 3.0 - cdij * cdij * (zeta3 - 1.0) * 2.0;
                -state.partial_density[parameters.component_index[i]] * count * g.ln()
            })
            .sum()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gc_pcsaft::eos::parameter::test::*;
    use approx::assert_relative_eq;
    use feos_core::ReferenceSystem;
    use nalgebra::dvector;
    use num_dual::Dual64;
    use quantity::{METER, MOL, PASCAL, Pressure};

    #[test]
    fn test_hc_propane() {
        let parameters = propane();
        let temperature = 300.0;
        let volume = METER.powi::<3>().to_reduced();
        let volume = Dual64::from_re(volume).derivative();
        let moles = (1.5 * MOL).to_reduced();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            volume,
            &dvector![Dual64::from_re(moles)],
        );
        let pressure = Pressure::from_reduced(
            -(HardChain.helmholtz_energy_density(&parameters, &state) * volume).eps * temperature,
        );
        assert_relative_eq!(
            pressure,
            -7.991735636207462e-1 * PASCAL,
            max_relative = 1e-10
        );
    }

    #[test]
    fn test_hc_propanol() {
        let parameters = GcPcSaftEosParameters::new(&propanol());
        let temperature = 300.0;
        let volume = METER.powi::<3>().to_reduced();
        let volume = Dual64::from_re(volume).derivative();
        let moles = (1.5 * MOL).to_reduced();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            volume,
            &dvector![Dual64::from_re(moles)],
        );
        let pressure = Pressure::from_reduced(
            -(HardChain.helmholtz_energy_density(&parameters, &state) * volume).eps * temperature,
        );
        assert_relative_eq!(pressure, -1.2831486124723626 * PASCAL, max_relative = 1e-10);
    }
}
