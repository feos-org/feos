#[cfg(feature = "dft")]
use crate::hard_sphere::FMTVersion;

use super::parameters::{SaftVRQMieParameters, SaftVRQMiePars};
use feos_core::{
    FeosError, FeosResult, Molarweight, ReferenceSystem, ResidualDyn, StateHD, Subset,
};
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use quantity::*;
use std::convert::TryFrom;
use std::f64::consts::FRAC_PI_6;
use std::fs::File;
use std::io::BufWriter;

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
#[derive(Copy, Clone, PartialEq, Debug)]
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
    sigma_eff_ij: DMatrix<D>,
    epsilon_k_eff_ij: DMatrix<D>,
    hs_diameter_ij: DMatrix<D>,
    quantum_d_ij: DMatrix<D>,
}

impl<D: DualNum<f64> + Copy> TemperatureDependentProperties<D> {
    fn new(parameters: &SaftVRQMiePars, temperature: D) -> Self {
        let n = parameters.m.len();
        let sigma_eff_ij = DMatrix::from_fn(n, n, |i, j| -> D {
            parameters.calc_sigma_eff_ij(i, j, temperature)
        });

        // temperature dependent segment radius
        let hs_diameter_ij = DMatrix::from_fn(n, n, |i, j| -> D {
            parameters.hs_diameter_ij(i, j, temperature, sigma_eff_ij[(i, j)])
        });

        // temperature dependent well depth
        let epsilon_k_eff_ij = DMatrix::from_fn(n, n, |i, j| -> D {
            parameters.calc_epsilon_k_eff_ij(i, j, temperature)
        });

        // temperature dependent well depth
        let quantum_d_ij = DMatrix::from_fn(n, n, |i, j| -> D {
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

impl Subset for SaftVRQMie {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(self.parameters.subset(component_list), self.options).unwrap()
    }
}

impl ResidualDyn for SaftVRQMie {
    fn components(&self) -> usize {
        self.parameters.pure.len()
    }

    fn compute_max_density<D: num_dual::DualNum<f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        let msigma3 = self
            .params
            .m
            .component_mul(&self.params.sigma.map(|v| v.powi(3)));
        (msigma3.map(D::from).dot(molefracs) * FRAC_PI_6).recip() * self.options.max_eta
    }

    fn reduced_helmholtz_energy_density_contributions<D: num_dual::DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        let mut v = Vec::with_capacity(7);
        let properties = TemperatureDependentProperties::new(&self.params, state.temperature);

        v.push((
            "Hard Sphere",
            HardSphere.helmholtz_energy_density(&self.params, state, &properties),
        ));
        v.push((
            "Dispersion",
            Dispersion.helmholtz_energy_density(&self.params, state, &properties),
        ));
        if self.non_additive_hard_sphere {
            v.push((
                "Non additive Hard Sphere",
                NonAddHardSphere.helmholtz_energy_density(&self.params, state, &properties),
            ))
        }
        v
    }
}

impl Molarweight for SaftVRQMie {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

#[cfg(feature = "ndarray")]
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
        let rs = ndarray::Array1::linspace(r_min.to_reduced(), r_max.to_reduced(), n);
        let energy_conversion = (KELVIN * RGAS / (KILO * CALORIE / MOL)).into_value();
        let force_conversion = (KELVIN * RGAS / (KILO * CALORIE / MOL)).into_value();
        let identifiers = self.parameters.identifiers();

        let n_components = self.params.sigma.len();
        for i in 0..n_components {
            for j in i..n_components {
                let name_i = identifiers[i].name.clone().unwrap_or_else(|| i.to_string());
                let name_j = identifiers[j].name.clone().unwrap_or_else(|| j.to_string());

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
