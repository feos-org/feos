use feos_core::{FeosError, FeosResult, StateHD};
use ndarray::Array1;
use num_dual::DualNum;
use serde::{Deserialize, Serialize};

use std::f64::consts::PI;

use crate::epcsaft::eos::ElectrolytePcSaftVariants;
use crate::epcsaft::parameters::ElectrolytePcSaftParameters;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum PermittivityRecord {
    ExperimentalData {
        data: Vec<(f64, f64)>,
    },
    PerturbationTheory {
        dipole_scaling: f64,
        polarizability_scaling: f64,
        correlation_integral_parameter: f64,
    },
}

impl std::fmt::Display for PermittivityRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PermittivityRecord::ExperimentalData { data } => {
                write!(f, "PermittivityRecord(data={:?}", data)?;
                write!(f, ")")
            }
            PermittivityRecord::PerturbationTheory {
                dipole_scaling,
                polarizability_scaling,
                correlation_integral_parameter,
            } => {
                write!(f, "PermittivityRecord(dipole_scaling={}", dipole_scaling)?;
                write!(f, ", polarizability_scaling={}", polarizability_scaling)?;
                write!(
                    f,
                    ", correlation_integral_parameter={}",
                    correlation_integral_parameter
                )?;
                write!(f, ")")
            }
        }
    }
}

#[derive(Clone)]
pub struct Permittivity<D: DualNum<f64>> {
    pub permittivity: D,
}

impl<D: DualNum<f64> + Copy> Permittivity<D> {
    pub fn new(
        state: &StateHD<D>,
        parameters: &ElectrolytePcSaftParameters,
        epcsaft_variant: &ElectrolytePcSaftVariants,
    ) -> FeosResult<Self> {
        // Set permittivity to an arbitrary value of 1 if system contains no ions
        // Ionic and Born contributions will be zero anyways
        if parameters.nionic == 0 {
            return Ok(Self {
                permittivity: D::one() * 1.,
            });
        }
        let all_comp: Array1<usize> = parameters
            .pure_records
            .iter()
            .enumerate()
            .map(|(i, _pr)| i)
            .collect();

        if let ElectrolytePcSaftVariants::Advanced = epcsaft_variant {
            // check if permittivity is Some for all components
            if parameters
                .permittivity
                .iter()
                .any(|record| record.is_none())
            {
                return Err(FeosError::IncompatibleParameters(
                    "Provide permittivities for all components for ePC-SAFT advanced.".to_string(),
                ));
            }

            // Extract parameters from PermittivityRecords
            let mut mu_scaling: Vec<&f64> = vec![];
            let mut alpha_scaling: Vec<&f64> = vec![];
            let mut ci_param: Vec<&f64> = vec![];
            let mut datas: Vec<Vec<(f64, f64)>> = vec![];

            parameters
                .permittivity
                .iter()
                .for_each(|record| match record.as_ref().unwrap() {
                    PermittivityRecord::PerturbationTheory {
                        dipole_scaling,
                        polarizability_scaling,
                        correlation_integral_parameter,
                    } => {
                        mu_scaling.push(dipole_scaling);
                        alpha_scaling.push(polarizability_scaling);
                        ci_param.push(correlation_integral_parameter);
                    }
                    PermittivityRecord::ExperimentalData { data } => {
                        datas.push(data.clone());
                    }
                });

            if let PermittivityRecord::ExperimentalData { .. } =
                parameters.permittivity[0].as_ref().unwrap()
            {
                let permittivity =
                    Self::from_experimental_data(&datas, state.temperature, &state.molefracs)
                        .permittivity;
                return Ok(Self { permittivity });
            }

            if let PermittivityRecord::PerturbationTheory { .. } =
                parameters.permittivity[0].as_ref().unwrap()
            {
                let permittivity = Self::from_perturbation_theory(
                    state,
                    &mu_scaling,
                    &alpha_scaling,
                    &ci_param,
                    &all_comp,
                )
                .permittivity;
                return Ok(Self { permittivity });
            }
        }

        if let ElectrolytePcSaftVariants::Revised = epcsaft_variant {
            if parameters.nsolvent > 1 {
                return Err(FeosError::IncompatibleParameters(
                    "ePC-SAFT revised cannot be used for more than 1 solvent.".to_string(),
                ));
            };
            let permittivity = match parameters.permittivity[parameters.solvent_comp[0]]
                .as_ref()
                .unwrap()
            {
                PermittivityRecord::ExperimentalData { data } => {
                    Self::pure_from_experimental_data(data, state.temperature).permittivity
                }
                PermittivityRecord::PerturbationTheory {
                    dipole_scaling,
                    polarizability_scaling,
                    correlation_integral_parameter,
                } => {
                    Self::pure_from_perturbation_theory(
                        state,
                        *dipole_scaling,
                        *polarizability_scaling,
                        *correlation_integral_parameter,
                    )
                    .permittivity
                }
            };

            return Ok(Self { permittivity });
        };
        Err(FeosError::IncompatibleParameters(
            "Permittivity computation failed".to_string(),
        ))
    }

    pub fn pure_from_experimental_data(data: &[(f64, f64)], temperature: D) -> Self {
        let permittivity_pure = Self::interpolate(data, temperature).permittivity;
        Self {
            permittivity: permittivity_pure,
        }
    }

    pub fn pure_from_perturbation_theory(
        state: &StateHD<D>,
        dipole_scaling: f64,
        polarizability_scaling: f64,
        correlation_integral_parameter: f64,
    ) -> Self {
        // reciprocal thermodynamic temperature
        let boltzmann = 1.380649e-23;
        let beta = (state.temperature * boltzmann).recip();

        // Density
        // let total_moles = state.moles.sum();
        let density = state.moles.mapv(|n| n / state.volume).sum();

        // dipole density y -> scaled dipole density y_star
        let y_star =
            density * (beta * dipole_scaling * 1e-19 + polarizability_scaling * 3.) * 4. / 9. * PI;

        // correlation integral
        let correlation_integral = ((-y_star).exp() - 1.0) * correlation_integral_parameter + 1.0;

        // dielectric constan
        let permittivity_pure = y_star
            * 3.0
            * (y_star.powi(2) * (correlation_integral * (17. / 16.) - 1.0) + y_star + 1.0)
            + 1.0;

        Self {
            permittivity: permittivity_pure,
        }
    }

    pub fn from_experimental_data(
        data: &[Vec<(f64, f64)>],
        temperature: D,
        molefracs: &Array1<D>,
    ) -> Self {
        let permittivity = data
            .iter()
            .enumerate()
            .map(|(i, d)| Self::interpolate(d, temperature).permittivity * molefracs[i])
            .sum();
        Self { permittivity }
    }

    pub fn from_perturbation_theory(
        state: &StateHD<D>,
        dipole_scaling: &[&f64],
        polarizability_scaling: &[&f64],
        correlation_integral_parameter: &[&f64],
        comp: &Array1<usize>,
    ) -> Self {
        //let nsolvent = comp.len();
        // reciprocal thermodynamic temperature
        let boltzmann = 1.380649e-23;
        let beta = (state.temperature * boltzmann).recip();
        // Determine scaled dipole density and correlation integral parameter of the mixture
        let mut y_star = D::zero();
        let mut correlation_integral_parameter_mixture = D::zero();
        for i in comp.iter() {
            let rho_i = state.partial_density[*i];
            let x_i = state.molefracs[*i];

            y_star +=
                rho_i * (beta * *dipole_scaling[*i] * 1e-19 + polarizability_scaling[*i] * 3.) * 4.
                    / 9.
                    * PI;
            correlation_integral_parameter_mixture += x_i * *correlation_integral_parameter[*i];
        }

        // correlation integral
        let correlation_integral =
            ((-y_star).exp() - 1.0) * correlation_integral_parameter_mixture + 1.0;

        // permittivity
        let permittivity = y_star
            * 3.0
            * (y_star.powi(2) * (correlation_integral * (17. / 16.) - 1.0) + y_star + 1.0)
            + 1.0;

        Self { permittivity }
    }

    /// Structure: &[(temperature, epsilon)]
    /// Assume ordered by temperature
    /// and temperatures are all finite.
    pub fn interpolate(interpolation_points: &[(f64, f64)], temperature: D) -> Self {
        // if there is only one point, return it (means constant permittivity)
        if interpolation_points.len() == 1 {
            return Self {
                permittivity: D::one() * interpolation_points[0].1,
            };
        }

        // find index where temperature could be inserted
        let i = interpolation_points.binary_search_by(|&(ti, _)| {
            ti.partial_cmp(&temperature.re())
                .expect("Unexpected value for temperature in interpolation points.")
        });

        // unwrap
        let i = i.unwrap_or_else(|i| i);
        let n = interpolation_points.len();

        // check cases:
        // 0.   : below lowest temperature
        // >= n : above highest temperature
        // else : regular interpolation

        let (l, u) = match i {
            0 => (interpolation_points[0], interpolation_points[1]),
            i if i >= n => (interpolation_points[n - 2], interpolation_points[n - 1]),
            _ => (interpolation_points[i - 1], interpolation_points[i]),
        };
        let permittivity_pure = (temperature - l.0) / (u.0 - l.0) * (u.1 - l.1) + l.1;

        Self {
            permittivity: permittivity_pure,
        }
    }
}
