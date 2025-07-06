#[cfg(feature = "dft")]
use crate::hard_sphere::FMTVersion;

use super::parameters::{SaftVRQMieParameters, SaftVRQMiePars};
use feos_core::{
    Components, EntropyScaling, FeosError, FeosResult, Molarweight, ReferenceSystem, Residual,
    StateHD,
};
use ndarray::{Array1, Array2};
use num_dual::{Dual64, DualNum};
use quantity::*;
use std::convert::TryFrom;
use std::f64::consts::{FRAC_PI_6, PI};
use std::fs::File;
use std::io::BufWriter;
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
    #[cfg(feature = "dft")]
    pub fmt_version: FMTVersion,
}

impl Default for SaftVRQMieOptions {
    fn default() -> Self {
        Self {
            max_eta: 0.5,
            inc_nonadd_term: true,
            #[cfg(feature = "dft")]
            fmt_version: FMTVersion::WhiteBear,
        }
    }
}

/// Order of Feynman-Hibbs potential
#[derive(Copy, Clone, PartialEq)]
pub enum FeynmanHibbsOrder {
    /// Mie potential
    FH0 = 0,
    /// First order correction
    FH1 = 1,
    /// Second order correction
    FH2 = 2,
}

impl TryFrom<usize> for FeynmanHibbsOrder {
    type Error = FeosError;

    fn try_from(u: usize) -> Result<Self, Self::Error> {
        match u {
            0 => Ok(Self::FH0),
            1 => Ok(Self::FH1),
            2 => Ok(Self::FH2),
            _ => Err(FeosError::IncompatibleParameters(format!(
                "failed to parse value '{u}' as FeynmanHibbsOrder. Has to be one of '0, 1, or 2'."
            ))),
        }
    }
}

pub(crate) struct TemperatureDependentProperties<D> {
    sigma_eff_ij: Array2<D>,
    epsilon_k_eff_ij: Array2<D>,
    hs_diameter_ij: Array2<D>,
    quantum_d_ij: Array2<D>,
}

impl<D: DualNum<f64> + Copy> TemperatureDependentProperties<D> {
    fn new(parameters: &SaftVRQMiePars, temperature: D) -> Self {
        let n = parameters.m.len();
        let sigma_eff_ij = Array2::from_shape_fn((n, n), |(i, j)| -> D {
            parameters.calc_sigma_eff_ij(i, j, temperature)
        });

        // temperature dependent segment radius
        let hs_diameter_ij = Array2::from_shape_fn((n, n), |(i, j)| -> D {
            parameters.hs_diameter_ij(i, j, temperature, sigma_eff_ij[[i, j]])
        });

        // temperature dependent well depth
        let epsilon_k_eff_ij = Array2::from_shape_fn((n, n), |(i, j)| -> D {
            parameters.calc_epsilon_k_eff_ij(i, j, temperature)
        });

        // temperature dependent well depth
        let quantum_d_ij = Array2::from_shape_fn((n, n), |(i, j)| -> D {
            parameters.quantum_d_ij(i, j, temperature)
        });
        Self {
            sigma_eff_ij,
            epsilon_k_eff_ij,
            hs_diameter_ij,
            quantum_d_ij,
        }
    }
}

/// SAFT-VRQ Mie Helmholtz energy model.
///
/// # Note
/// Currently, only the first-order Feynman-Hibbs term is implemented.
pub struct SaftVRQMie {
    pub parameters: SaftVRQMieParameters,
    pub params: SaftVRQMiePars,
    pub options: SaftVRQMieOptions,
    pub non_additive_hard_sphere: bool,
}

impl SaftVRQMie {
    pub fn new(parameters: SaftVRQMieParameters) -> FeosResult<Self> {
        Self::with_options(parameters, SaftVRQMieOptions::default())
    }

    pub fn with_options(
        parameters: SaftVRQMieParameters,
        options: SaftVRQMieOptions,
    ) -> FeosResult<Self> {
        let params = SaftVRQMiePars::new(&parameters)?;
        let non_additive_hard_sphere = params.m.len() > 1 && options.inc_nonadd_term;

        Ok(Self {
            parameters,
            params,
            options,
            non_additive_hard_sphere,
        })
    }
}

impl Components for SaftVRQMie {
    fn components(&self) -> usize {
        self.parameters.pure.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(self.parameters.subset(component_list), self.options).unwrap()
    }
}

impl Residual for SaftVRQMie {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * &self.params.m * self.params.sigma.mapv(|v| v.powi(3)) * moles).sum()
    }

    fn residual_helmholtz_energy_contributions<D: num_dual::DualNum<f64> + Copy>(
        &self,
        state: &feos_core::StateHD<D>,
    ) -> Vec<(String, D)> {
        let mut v = Vec::with_capacity(7);
        let properties = TemperatureDependentProperties::new(&self.params, state.temperature);

        v.push((
            "Hard Sphere".to_string(),
            HardSphere.helmholtz_energy(&self.params, state, &properties),
        ));
        v.push((
            "Dispersion".to_string(),
            Dispersion.helmholtz_energy(&self.params, state, &properties),
        ));
        if self.non_additive_hard_sphere {
            v.push((
                "Non additive Hard Sphere".to_string(),
                NonAddHardSphere.helmholtz_energy(&self.params, state, &properties),
            ))
        }
        v
    }
}

impl Molarweight for SaftVRQMie {
    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molar_weight.clone()
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
    0.083235 * (t * m / molarweight.convert_to(GRAM / MOL)).sqrt()
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
    ) -> FeosResult<Viscosity> {
        let p = &self.params;
        let mw = &self.parameters.molar_weight;
        let x = (moles / moles.sum()).into_value();
        let ce: Array1<_> = (0..self.components())
            .map(|i| {
                let tr = (temperature / p.epsilon_k[i] / KELVIN).into_value();
                5.0 / 16.0 * (mw.get(i) * KB / NAV * temperature / PI).sqrt()
                    / omega22(tr)
                    / (p.sigma[i] * ANGSTROM).powi::<P2>()
            })
            .collect();
        let mut ce_mix = 0.0 * MILLI * PASCAL * SECOND;
        for i in 0..self.components() {
            let denom: f64 = (0..self.components())
                .map(|j| {
                    x[j] * (1.0
                        + (ce[i] / ce[j]).into_value().sqrt()
                            * (mw.get(j) / mw.get(i)).powf(1.0 / 4.0))
                    .powi(2)
                        / (8.0 * (1.0 + (mw.get(i) / mw.get(j)).into_value())).sqrt()
                })
                .sum();
            ce_mix += ce[i] * x[i] / denom
        }
        Ok(ce_mix)
    }

    fn viscosity_correlation(&self, s_res: f64, x: &Array1<f64>) -> FeosResult<f64> {
        let coefficients = self
            .params
            .viscosity
            .as_ref()
            .expect("Missing viscosity coefficients.");
        let m = (x * &self.params.m).sum();
        let s = s_res / m;
        let pref = (x * &self.params.m) / m;
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
    ) -> FeosResult<Diffusivity> {
        if self.components() != 1 {
            return Err(FeosError::IncompatibleComponents(self.components(), 1));
        }
        let p = &self.params;
        let mw = &self.parameters.molar_weight;
        let density = moles.sum() / volume;
        let res: Array1<_> = (0..self.components())
            .map(|i| {
                let tr = (temperature / p.epsilon_k[i] / KELVIN).into_value();
                3.0 / 8.0 / (p.sigma[i] * ANGSTROM).powi::<P2>() / omega11(tr) / (density * NAV)
                    * (temperature * RGAS / PI / mw.get(i) / p.m[i]).sqrt()
            })
            .collect();
        Ok(res[0])
    }

    fn diffusion_correlation(&self, s_res: f64, x: &Array1<f64>) -> FeosResult<f64> {
        if self.components() != 1 {
            return Err(FeosError::IncompatibleComponents(self.components(), 1));
        }
        let coefficients = self
            .params
            .diffusion
            .as_ref()
            .expect("Missing diffusion coefficients.");
        let m = (x * &self.params.m).sum();
        let s = s_res / m;
        let pref = (x * &self.params.m).mapv(|v| v / m);
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
    ) -> FeosResult<ThermalConductivity> {
        if self.components() != 1 {
            return Err(FeosError::IncompatibleComponents(self.components(), 1));
        }
        let p = &self.params;
        let mws = self.molar_weight();
        let t = Dual64::from(temperature.into_reduced()).derivative();
        let v = Dual64::from(volume.into_reduced());
        let n = moles.to_reduced().mapv(Dual64::from);
        let n_tot = n.sum();
        let state = StateHD::new(t, v, n);
        let s_res = -(self.residual_helmholtz_energy(&state) * t / n_tot).eps;
        let res: Array1<_> = (0..self.components())
            .map(|i| {
                let tr = (temperature / p.epsilon_k[i] / KELVIN).into_value();
                let s_res_reduced = s_res / p.m[i];
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

    fn thermal_conductivity_correlation(&self, s_res: f64, x: &Array1<f64>) -> FeosResult<f64> {
        if self.components() != 1 {
            return Err(FeosError::IncompatibleComponents(self.components(), 1));
        }
        let coefficients = self
            .params
            .thermal_conductivity
            .as_ref()
            .expect("Missing thermal conductivity coefficients");
        let m = (x * &self.params.m).sum();
        let s = s_res / m;
        let pref = (x * &self.params.m).mapv(|v| v / m);
        let a: f64 = (&coefficients.row(0) * x).sum();
        let b: f64 = (&coefficients.row(1) * &pref).sum();
        let c: f64 = (&coefficients.row(2) * &pref).sum();
        let d: f64 = (&coefficients.row(3) * &pref).sum();
        Ok(a + b * s + c * (1.0 - s.exp()) + d * s.powi(2))
    }
}

impl SaftVRQMie {
    /// Generate energy and force tables to be used with LAMMPS' `pair_style table` command.
    ///
    /// For a given `temperature`, `n` values between `r_min` and `r_max` (both including) are tabulated.
    ///
    /// Files for all pure substances and all unique pairs are generated,
    /// where filenames use either the "name" field of the identifier or the index if no name is present.
    ///
    /// # Example
    ///
    /// For a hydrogen-neon mixture at 30 K, three files will be created.
    ///
    /// - "hydrogen_30K.table" for H-H interactions,
    /// - "neon_30K.table" for Ne-Ne interactions,
    /// - "hydrogen_neon_30K.table" for H-Ne interactions.
    pub fn lammps_tables(
        &self,
        temperature: Temperature,
        n: usize,
        r_min: Length,
        r_max: Length,
    ) -> std::io::Result<()> {
        let t = temperature.to_reduced();
        let rs = Array1::linspace(r_min.to_reduced(), r_max.to_reduced(), n);
        let energy_conversion = (KELVIN * RGAS / (KILO * CALORIE / MOL)).into_value();
        let force_conversion = (KELVIN * RGAS / (KILO * CALORIE / MOL)).into_value();

        let n_components = self.params.sigma.len();
        for i in 0..n_components {
            for j in i..n_components {
                let name_i = self.parameters.identifiers[i]
                    .name
                    .clone()
                    .unwrap_or_else(|| i.to_string());
                let name_j = self.parameters.identifiers[j]
                    .name
                    .clone()
                    .unwrap_or_else(|| j.to_string());

                let name = if i == j {
                    name_i
                } else {
                    format!("{name_i}_{name_j}")
                };
                let f = File::create(format!("{name}_{t}K.table"))?;
                let mut stream = BufWriter::new(f);

                std::io::Write::write(
                    &mut stream,
                    b"# DATE: YYYY-MM-DD UNITS: real CONTRIBUTOR: YOUR NAME\n",
                )?;
                std::io::Write::write(
                    &mut stream,
                    format!("# FH1 potential for {name} at T = {temperature}\n").as_bytes(),
                )?;
                std::io::Write::write(&mut stream, format!("FH1_{name}\n").as_bytes())?;
                std::io::Write::write(&mut stream, format!("N {n}\n\n").as_bytes())?;

                for (k, &r) in rs.iter().enumerate() {
                    let [u, du, _] = self.params.qmie_potential_ij(i, j, r, t);
                    std::io::Write::write(
                        &mut stream,
                        format!(
                            "{} {:12.8} {:12.8} {:12.8}\n",
                            k + 1,
                            r,
                            u * energy_conversion,
                            -du * force_conversion
                        )
                        .as_bytes(),
                    )?;
                }
                std::io::Write::flush(&mut stream)?;
            }
        }
        Ok(())
    }
}
