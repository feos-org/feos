use feos_core::StateHD;
use ndarray::Array1;
use num_dual::DualNum;
use serde::{Deserialize, Serialize};

use std::f64::consts::PI;

use crate::epcsaft::eos::ElectrolytePcSaftVariants;
use crate::epcsaft::parameters::ElectrolytePcSaftParameters;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum PermittivityRecord {
    ExperimentalData {
        data: Vec<Vec<(f64, f64)>>,
    },
    PerturbationTheory {
        dipole_scaling: Vec<f64>,
        polarizability_scaling: Vec<f64>,
        correlation_integral_parameter: Vec<f64>,
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
                write!(f, "PermittivityRecord(dipole_scaling={}", dipole_scaling[0])?;
                write!(f, ", polarizability_scaling={}", polarizability_scaling[0])?;
                write!(
                    f,
                    ", correlation_integral_parameter={}",
                    correlation_integral_parameter[0]
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
    ) -> Self {
        let n = parameters.pure_records.len();

        // Set permittivity to an arbitrary value of 1 if system contains no ions
        // Ionic and Born contributions will be zero anyways
        if parameters.nionic == 0 {
            return Self {
                permittivity: D::one() * 1.,
            };
        }
        let all_comp: Array1<usize> = parameters
            .pure_records
            .iter()
            .enumerate()
            .map(|(i, _pr)| i)
            .collect();

        if let ElectrolytePcSaftVariants::Advanced = epcsaft_variant {
            let permittivity = match parameters.permittivity.as_ref().unwrap() {
                PermittivityRecord::ExperimentalData { data } => {
                    // Check length of permittivity_record
                    if data.len() != n {
                        panic!("Provide permittivities for all components for ePC-SAFT advanced.")
                    }
                    Self::from_experimental_data(data, state.temperature, &state.molefracs)
                        .permittivity
                }
                PermittivityRecord::PerturbationTheory {
                    dipole_scaling,
                    polarizability_scaling,
                    correlation_integral_parameter,
                } => {
                    // Check length of permittivity_record
                    if dipole_scaling.len() != n {
                        panic!("Provide permittivities for all components for ePC-SAFT advanced.")
                    }
                    Self::from_perturbation_theory(
                        state,
                        dipole_scaling,
                        polarizability_scaling,
                        correlation_integral_parameter,
                        &all_comp,
                    )
                    .permittivity
                }
            };

            return Self { permittivity };
        }
        if let ElectrolytePcSaftVariants::Revised = epcsaft_variant {
            if parameters.nsolvent > 1 {
                panic!(
                "The use of ePC-SAFT revised requires the definition of exactly 1 solvent. Currently specified: {} solvents", parameters.nsolvent
                    )
            };
            let permittivity = match parameters.permittivity.as_ref().unwrap() {
                PermittivityRecord::ExperimentalData { data } => {
                    Self::pure_from_experimental_data(&data[0], state.temperature).permittivity
                }
                PermittivityRecord::PerturbationTheory {
                    dipole_scaling,
                    polarizability_scaling,
                    correlation_integral_parameter,
                } => {
                    // Check length of permittivity_record
                    if dipole_scaling.len() != n {
                        panic!("Provide permittivities for all components for ePC-SAFT advanced.")
                    }
                    Self::pure_from_perturbation_theory(
                        state,
                        &dipole_scaling[0],
                        &polarizability_scaling[0],
                        &correlation_integral_parameter[0],
                    )
                    .permittivity
                }
            };

            return Self { permittivity };
        };
        Self {
            permittivity: D::zero(),
        }
    }

    pub fn pure_from_experimental_data(data: &[(f64, f64)], temperature: D) -> Self {
        let permittivity_pure = Self::interpolate(data.to_vec(), temperature).permittivity;
        Self {
            permittivity: permittivity_pure,
        }
    }

    pub fn pure_from_perturbation_theory(
        state: &StateHD<D>,
        dipole_scaling: &f64,
        polarizability_scaling: &f64,
        correlation_integral_parameter: &f64,
    ) -> Self {
        // reciprocal thermodynamic temperature
        let boltzmann = 1.380649e-23;
        let beta = (state.temperature * boltzmann).recip();

        // Density
        // let total_moles = state.moles.sum();
        let density = state.moles.mapv(|n| n / state.volume).sum();

        // dipole density y -> scaled dipole density y_star
        let y_star = density * (beta * *dipole_scaling * 1e-19 + *polarizability_scaling * 3.) * 4.
            / 9.
            * PI;

        // correlation integral
        let correlation_integral = ((-y_star).exp() - 1.0) * *correlation_integral_parameter + 1.0;

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
            .map(|(i, d)| Self::interpolate(d.to_vec(), temperature).permittivity * molefracs[i])
            .sum();
        Self { permittivity }
    }

    pub fn from_perturbation_theory(
        state: &StateHD<D>,
        dipole_scaling: &[f64],
        polarizability_scaling: &[f64],
        correlation_integral_parameter: &[f64],
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
                rho_i * (beta * dipole_scaling[*i] * 1e-19 + polarizability_scaling[*i] * 3.) * 4.
                    / 9.
                    * PI;
            correlation_integral_parameter_mixture += x_i * correlation_integral_parameter[*i];
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

    pub fn interpolate(interpolation_points: Vec<(f64, f64)>, temperature: D) -> Self {
        let t_interpol = Array1::from_iter(interpolation_points.iter().map(|t| (t.0)));
        let eps_interpol = Array1::from_iter(interpolation_points.iter().map(|e| e.1));

        // Initialize permittivity
        let mut permittivity_pure = D::zero();

        // Check if only 1 data point is given
        if interpolation_points.len() == 1 {
            permittivity_pure = D::one() * eps_interpol[0];
        } else {
            // Check which interval temperature is in
            let temperature = temperature.re();
            for i in 0..(t_interpol.len() - 1) {
                // Temperature is within intervals
                if temperature >= t_interpol[i] && temperature < t_interpol[i + 1] {
                    // Interpolate
                    permittivity_pure = D::one() * eps_interpol[i]
                        + (temperature - t_interpol[i]) * (eps_interpol[i + 1] - eps_interpol[i])
                            / (t_interpol[i + 1] - t_interpol[i]);
                }
            }
            // Temperature is lower than lowest temperature
            if temperature < t_interpol[0] {
                // Extrapolate from eps_0 and eps_1
                permittivity_pure = D::one() * eps_interpol[0]
                    + (temperature - t_interpol[0]) * (eps_interpol[1] - eps_interpol[0])
                        / (t_interpol[1] - t_interpol[0]);
            // Temperature is higher than highest temperature
            } else if temperature >= t_interpol[t_interpol.len() - 1] {
                // extrapolate from last two epsilons
                permittivity_pure = D::one() * eps_interpol[t_interpol.len() - 2]
                    + (temperature - t_interpol[t_interpol.len() - 2])
                        * (eps_interpol[t_interpol.len() - 1] - eps_interpol[t_interpol.len() - 2])
                        / (t_interpol[t_interpol.len() - 1] - t_interpol[t_interpol.len() - 2]);
            }
        }
        Self {
            permittivity: permittivity_pure,
        }
    }
}
