use super::parameters::PetsParameters;
#[cfg(feature = "dft")]
use crate::hard_sphere::FMTVersion;
use crate::hard_sphere::{HardSphere, HardSphereProperties, MonomerShape};
use feos_core::{Molarweight, ResidualDyn, Subset};
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use quantity::MolarWeight;
use std::f64::consts::FRAC_PI_6;

pub(crate) mod dispersion;

/// Configuration options for the PeTS equation of state and Helmholtz energy functional.
///
/// The maximum packing fraction is used to infer initial values
/// for routines that depend on starting values for the system density.
#[derive(Copy, Clone)]
pub struct PetsOptions {
    /// maximum packing fraction
    pub max_eta: f64,
    /// The version of the FMT functional to use
    #[cfg(feature = "dft")]
    pub fmt_version: FMTVersion,
}

impl Default for PetsOptions {
    fn default() -> Self {
        Self {
            max_eta: 0.5,
            #[cfg(feature = "dft")]
            fmt_version: FMTVersion::WhiteBear,
        }
    }
}

/// PeTS Helmholtz energy model.
pub struct Pets {
    pub parameters: PetsParameters,
    pub options: PetsOptions,
    pub sigma: DVector<f64>,
    pub epsilon_k: DVector<f64>,
    pub sigma_ij: DMatrix<f64>,
    pub epsilon_k_ij: DMatrix<f64>,
}

impl Pets {
    /// PeTS model with default options.
    pub fn new(parameters: PetsParameters) -> Self {
        Self::with_options(parameters, PetsOptions::default())
    }

    /// PeTS model with provided options.
    pub fn with_options(parameters: PetsParameters, options: PetsOptions) -> Self {
        let [sigma, epsilon_k] = parameters.collate(|pr| [pr.sigma, pr.epsilon_k]);

        let n = parameters.pure.len();

        let mut sigma_ij = DMatrix::zeros(n, n);
        let mut epsilon_k_ij = DMatrix::zeros(n, n);
        let [k_ij] = parameters.collate_binary(|b| [b.k_ij]);
        for i in 0..n {
            for j in 0..n {
                epsilon_k_ij[(i, j)] = (epsilon_k[i] * epsilon_k[j]).sqrt() * (1.0 - k_ij[(i, j)]);
                sigma_ij[(i, j)] = 0.5 * (sigma[i] + sigma[j]);
            }
        }

        Self {
            parameters,
            options,
            sigma,
            epsilon_k,
            sigma_ij,
            epsilon_k_ij,
        }
    }
}

// impl Components for Pets {
//     fn components(&self) -> usize {
//         self.parameters.pure.len()
//     }

impl Subset for Pets {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(self.parameters.subset(component_list), self.options)
    }
}

impl ResidualDyn for Pets {
    fn components(&self) -> usize {
        self.parameters.pure.len()
    }

    fn compute_max_density<D: num_dual::DualNum<f64> + Copy>(&self, moles: &DVector<D>) -> D {
        moles.sum() * self.options.max_eta
            / (self.sigma.map(|v| D::from(v.powi(3))).component_mul(moles)).sum()
            / FRAC_PI_6
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &feos_core::StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        vec![
            (
                "Hard Sphere",
                HardSphere.helmholtz_energy_density(self, state),
            ),
            (
                "Dispersion",
                self.dispersion_helmholtz_energy_density(state),
            ),
        ]
    }
}

impl Molarweight for Pets {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

impl HardSphereProperties for Pets {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<'_, N> {
        MonomerShape::Spherical(self.sigma.len())
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> DVector<D> {
        let ti = temperature.recip() * -3.052785558;
        DVector::from_fn(self.sigma.len(), |i, _| {
            -((ti * self.epsilon_k[i]).exp() * 0.127112544 - 1.0) * self.sigma[i]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pets::parameters::utils::{
        argon_krypton_parameters, argon_parameters, krypton_parameters,
    };
    use approx::assert_relative_eq;
    use feos_core::{Contributions, PhaseEquilibrium, State, StateHD};
    use nalgebra::dvector;
    use quantity::{BAR, KELVIN, METER, MOL, RGAS};
    use typenum::P3;

    #[test]
    fn ideal_gas_pressure() {
        let e = &Pets::new(argon_parameters());
        let t = 200.0 * KELVIN;
        let v = 1e-3 * METER.powi::<P3>();
        let n = dvector![1.0] * MOL;
        let s = State::new_nvt(&e, t, v, &n).unwrap();
        let p_ig = s.total_moles * RGAS * t / v;
        assert_relative_eq!(s.pressure(Contributions::IdealGas), p_ig, epsilon = 1e-10);
        assert_relative_eq!(
            s.pressure(Contributions::IdealGas) + s.pressure(Contributions::Residual),
            s.pressure(Contributions::Total),
            epsilon = 1e-10
        );
    }

    #[test]
    fn hard_sphere_mix() {
        let argon = Pets::new(argon_parameters());
        let krypton = Pets::new(krypton_parameters());
        let mix = Pets::new(argon_krypton_parameters());
        let t = 250.0;
        let v = 2.5e28;
        let n = 1.0;
        let s = StateHD::new(t, v, &dvector![n]);
        let a1 = HardSphere.helmholtz_energy_density(&argon, &s);
        let a2 = HardSphere.helmholtz_energy_density(&krypton, &s);
        let s1m = StateHD::new(t, v, &dvector![n, 0.0]);
        let a1m = HardSphere.helmholtz_energy_density(&mix, &s1m);
        let s2m = StateHD::new(t, v, &dvector![0.0, n]);
        let a2m = HardSphere.helmholtz_energy_density(&mix, &s2m);
        assert_relative_eq!(a1, a1m, epsilon = 1e-14);
        assert_relative_eq!(a2, a2m, epsilon = 1e-14);
    }

    #[test]
    fn new_tpn() {
        let e = &Pets::new(argon_parameters());
        let t = 300.0 * KELVIN;
        let p = BAR;
        let m = dvector![1.0] * MOL;
        let s = State::new_npt(&e, t, p, &m, None).unwrap();
        let p_calc = s.pressure(Contributions::Total);
        assert_relative_eq!(p, p_calc, epsilon = 1e-6);
    }

    #[test]
    fn vle_pure_t() {
        let e = &Pets::new(argon_parameters());
        let t = 300.0 * KELVIN;
        let vle = PhaseEquilibrium::pure(&e, t, None, Default::default());
        if let Ok(v) = vle {
            assert_relative_eq!(
                v.vapor().pressure(Contributions::Total),
                v.liquid().pressure(Contributions::Total),
                epsilon = 1e-6
            )
        }
    }

    #[test]
    fn mix_single() {
        let e1 = &Pets::new(argon_parameters());
        let e2 = &Pets::new(krypton_parameters());
        let e12 = &Pets::new(argon_krypton_parameters());
        let t = 300.0 * KELVIN;
        let v = 0.02456883872966545 * METER.powi::<P3>();
        let m1 = dvector![2.0] * MOL;
        let m1m = dvector![2.0, 0.0] * MOL;
        let m2m = dvector![0.0, 2.0] * MOL;
        let s1 = State::new_nvt(&e1, t, v, &m1).unwrap();
        let s2 = State::new_nvt(&e2, t, v, &m1).unwrap();
        let s1m = State::new_nvt(&e12, t, v, &m1m).unwrap();
        let s2m = State::new_nvt(&e12, t, v, &m2m).unwrap();
        assert_relative_eq!(
            s1.pressure(Contributions::Total),
            s1m.pressure(Contributions::Total),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            s2.pressure(Contributions::Total),
            s2m.pressure(Contributions::Total),
            epsilon = 1e-12
        )
    }
}
