use super::parameters::SaftVRQMieParameters;
use feos_core::parameter::{Parameter, ParameterError};
use feos_core::si::*;
use feos_core::{
    Components, EntropyScaling, EosError, EosResult, HelmholtzEnergy, Residual, State,
};
use ndarray::Array1;
use std::convert::TryFrom;
use std::f64::consts::{FRAC_PI_6, PI};
use std::sync::Arc;
use typenum::P2;

pub(crate) mod dispersion;
pub(crate) mod hard_sphere;
pub(crate) mod non_additive_hs;
use dispersion::Dispersion;
use hard_sphere::HardSphere;
use non_additive_hs::NonAddHardSphere;

/// Customization options for the SAFT-VRQ Mie equation of state and functional.
#[derive(Copy, Clone)]
pub struct SaftVRQMieOptions {
    pub max_eta: f64,
    pub inc_nonadd_term: bool,
}

impl Default for SaftVRQMieOptions {
    fn default() -> Self {
        Self {
            max_eta: 0.5,
            inc_nonadd_term: true,
        }
    }
}

/// Order of Feynman-Hibbs potential
#[derive(Copy, Clone)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub enum FeynmanHibbsOrder {
    /// Mie potential
    FH0 = 0,
    /// First order correction
    FH1 = 1,
    /// Second order correction
    FH2 = 2,
}

impl TryFrom<usize> for FeynmanHibbsOrder {
    type Error = ParameterError;

    fn try_from(u: usize) -> Result<Self, Self::Error> {
        match u {
            0 => Ok(Self::FH0),
            1 => Ok(Self::FH1),
            2 => Ok(Self::FH2),
            _ => Err(ParameterError::IncompatibleParameters(format!(
                "failed to parse value '{}' as FeynmanHibbsOrder. Has to be one of '0, 1, or 2'.",
                u
            ))),
        }
    }
}

/// SAFT-VRQ Mie equation of state.
///
/// # Note
/// Currently, only the first-order Feynman-Hibbs term is implemented.
pub struct SaftVRQMie {
    parameters: Arc<SaftVRQMieParameters>,
    options: SaftVRQMieOptions,
    contributions: Vec<Box<dyn HelmholtzEnergy>>,
}

impl SaftVRQMie {
    pub fn new(parameters: Arc<SaftVRQMieParameters>) -> Self {
        Self::with_options(parameters, SaftVRQMieOptions::default())
    }

    pub fn with_options(parameters: Arc<SaftVRQMieParameters>, options: SaftVRQMieOptions) -> Self {
        let mut contributions: Vec<Box<dyn HelmholtzEnergy>> = Vec::with_capacity(4);
        contributions.push(Box::new(HardSphere {
            parameters: parameters.clone(),
        }));
        contributions.push(Box::new(Dispersion {
            parameters: parameters.clone(),
        }));
        if parameters.m.len() > 1 && options.inc_nonadd_term {
            contributions.push(Box::new(NonAddHardSphere {
                parameters: parameters.clone(),
            }));
        }

        Self {
            parameters,
            options,
            contributions,
        }
    }
}

impl Components for SaftVRQMie {
    fn components(&self) -> usize {
        self.parameters.pure_records.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            Arc::new(self.parameters.subset(component_list)),
            self.options,
        )
    }
}

impl Residual for SaftVRQMie {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * &self.parameters.m * self.parameters.sigma.mapv(|v| v.powi(3)) * moles)
                .sum()
    }

    fn contributions(&self) -> &[Box<dyn HelmholtzEnergy>] {
        &self.contributions
    }

    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}

fn omega11(t: f64) -> f64 {
    1.06036 * t.powf(-0.15610)
        + 0.19300 * (-0.47635 * t).exp()
        + 1.03587 * (-1.52996 * t).exp()
        + 1.76474 * (-3.89411 * t).exp()
}

fn omega22(t: f64) -> f64 {
    1.16145 * t.powf(-0.14874) + 0.52487 * (-0.77320 * t).exp() + 2.16178 * (-2.43787 * t).exp()
        - 6.435e-4 * t.powf(0.14874) * (18.0323 * t.powf(-0.76830) - 7.27371).sin()
}

#[inline]
fn chapman_enskog_thermal_conductivity(
    temperature: Temperature,
    molarweight: MolarWeight,
    m: f64,
    sigma: f64,
    epsilon_k: f64,
) -> ThermalConductivity {
    let t = temperature.to_reduced();
    0.083235 * (t * m / (molarweight / (GRAM / MOL)).into_value()).sqrt()
        / sigma.powi(2)
        / omega22(t / epsilon_k)
        * WATT
        / METER
        / KELVIN
}

impl EntropyScaling for SaftVRQMie {
    fn viscosity_reference(
        &self,
        temperature: Temperature,
        _: Volume,
        moles: &Moles<Array1<f64>>,
    ) -> EosResult<Viscosity> {
        let p = &self.parameters;
        let mw = &p.molarweight;
        let x = (moles / moles.sum()).into_value();
        let sigma_eff = p.sigma_eff(temperature.to_reduced());
        let epsilon_k_eff = p.epsilon_k_eff(temperature.to_reduced());
        let ce: Array1<_> = (0..self.components())
            .map(|i| {
                let tr = (temperature / epsilon_k_eff[i] / KELVIN).into_value();
                5.0 / 16.0 * (mw[i] * GRAM / MOL * KB / NAV * temperature / PI).sqrt()
                    / omega22(tr)
                    / (sigma_eff[i] * ANGSTROM).powi::<P2>()
            })
            .collect();
        let mut ce_mix = 0.0 * MILLI * PASCAL * SECOND;
        for i in 0..self.components() {
            let denom: f64 = (0..self.components())
                .map(|j| {
                    x[j] * (1.0
                        + (ce[i] / ce[j]).into_value().sqrt() * (mw[j] / mw[i]).powf(1.0 / 4.0))
                    .powi(2)
                        / (8.0 * (1.0 + mw[i] / mw[j])).sqrt()
                })
                .sum();
            ce_mix += ce[i] * x[i] / denom
        }
        Ok(ce_mix)
    }

    fn viscosity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        let coefficients = self
            .parameters
            .viscosity
            .as_ref()
            .expect("Missing viscosity coefficients.");
        let m = (x * &self.parameters.m).sum();
        let s = s_res / m;
        let pref = (x * &self.parameters.m) / m;
        let a: f64 = (&coefficients.row(0) * x).sum();
        let b: f64 = (&coefficients.row(1) * &pref).sum();
        let c: f64 = (&coefficients.row(2) * &pref).sum();
        let d: f64 = (&coefficients.row(3) * &pref).sum();
        Ok(a + b * s + c * s.powi(2) + d * s.powi(3))
    }

    fn diffusion_reference(
        &self,
        temperature: Temperature,
        volume: Volume,
        moles: &Moles<Array1<f64>>,
    ) -> EosResult<Diffusivity> {
        if self.components() != 1 {
            return Err(EosError::IncompatibleComponents(self.components(), 1));
        }
        let p = &self.parameters;
        let density = moles.sum() / volume;
        let res: Array1<_> = (0..self.components())
            .map(|i| {
                let tr = (temperature / p.epsilon_k[i] / KELVIN).into_value();
                3.0 / 8.0 / (p.sigma[i] * ANGSTROM).powi::<P2>() / omega11(tr) / (density * NAV)
                    * (temperature * RGAS / PI / (p.molarweight[i] * GRAM / MOL)).sqrt()
            })
            .collect();
        Ok(res[0])
    }

    fn diffusion_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        if self.components() != 1 {
            return Err(EosError::IncompatibleComponents(self.components(), 1));
        }
        let coefficients = self
            .parameters
            .diffusion
            .as_ref()
            .expect("Missing diffusion coefficients.");
        let m = (x * &self.parameters.m).sum();
        let s = s_res / m;
        let pref = (x * &self.parameters.m).mapv(|v| v / m);
        let a: f64 = (&coefficients.row(0) * x).sum();
        let b: f64 = (&coefficients.row(1) * &pref).sum();
        let c: f64 = (&coefficients.row(2) * &pref).sum();
        let d: f64 = (&coefficients.row(3) * &pref).sum();
        let e: f64 = (&coefficients.row(4) * &pref).sum();
        Ok(a + b * s - c * (1.0 - s.exp()) * s.powi(2) - d * s.powi(4) - e * s.powi(8))
    }

    // Equation 4 of DOI: 10.1021/acs.iecr.9b04289
    fn thermal_conductivity_reference(
        &self,
        temperature: Temperature,
        volume: Volume,
        moles: &Moles<Array1<f64>>,
    ) -> EosResult<ThermalConductivity> {
        if self.components() != 1 {
            return Err(EosError::IncompatibleComponents(self.components(), 1));
        }
        let p = &self.parameters;
        let mws = self.molar_weight();
        let state = State::new_nvt(&Arc::new(Self::new(p.clone())), temperature, volume, moles)?;
        let res: Array1<_> = (0..self.components())
            .map(|i| {
                let tr = (temperature / p.epsilon_k[i] / KELVIN).into_value();
                let s_res_reduced = (state.residual_molar_entropy() / RGAS).into_value() / p.m[i];
                let ref_ce = chapman_enskog_thermal_conductivity(
                    temperature,
                    mws.get(i),
                    p.m[i],
                    p.sigma[i],
                    p.epsilon_k[i],
                );
                let alpha_visc = (-s_res_reduced / -0.5).exp();
                let ref_ts = (-0.0167141 * tr / p.m[i] + 0.0470581 * (tr / p.m[i]).powi(2))
                    * (p.m[i] * p.m[i] * p.sigma[i].powi(3) * p.epsilon_k[i])
                    * 1e-5
                    * WATT
                    / METER
                    / KELVIN;
                ref_ce + ref_ts * alpha_visc
            })
            .collect();
        Ok(res[0])
    }

    fn thermal_conductivity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        if self.components() != 1 {
            return Err(EosError::IncompatibleComponents(self.components(), 1));
        }
        let coefficients = self
            .parameters
            .thermal_conductivity
            .as_ref()
            .expect("Missing thermal conductivity coefficients");
        let m = (x * &self.parameters.m).sum();
        let s = s_res / m;
        let pref = (x * &self.parameters.m).mapv(|v| v / m);
        let a: f64 = (&coefficients.row(0) * x).sum();
        let b: f64 = (&coefficients.row(1) * &pref).sum();
        let c: f64 = (&coefficients.row(2) * &pref).sum();
        let d: f64 = (&coefficients.row(3) * &pref).sum();
        Ok(a + b * s + c * (1.0 - s.exp()) + d * s.powi(2))
    }
}
