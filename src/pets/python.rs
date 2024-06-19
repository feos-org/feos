use super::parameters::*;
use feos_core::parameter::*;
use feos_core::python::parameter::*;
use feos_core::{impl_binary_record, impl_json_handling, impl_parameter, impl_pure_record};
use numpy::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

/// Create a set of PeTS parameters from records.
#[pyclass(name = "PetsRecord")]
#[derive(Clone)]
pub struct PyPetsRecord(PetsRecord);

#[pymethods]
impl PyPetsRecord {
    #[new]
    #[pyo3(
        text_signature = "(sigma, epsilon_k, viscosity=None, diffusion=None, thermal_conductivity=None)"
    )]
    fn new(
        sigma: f64,
        epsilon_k: f64,
        viscosity: Option<[f64; 4]>,
        diffusion: Option<[f64; 5]>,
        thermal_conductivity: Option<[f64; 4]>,
    ) -> Self {
        Self(PetsRecord::new(
            sigma,
            epsilon_k,
            viscosity,
            diffusion,
            thermal_conductivity,
        ))
    }

    #[getter]
    fn get_sigma(&self) -> f64 {
        self.0.sigma
    }

    #[getter]
    fn get_epsilon_k(&self) -> f64 {
        self.0.epsilon_k
    }

    #[getter]
    fn get_viscosity(&self) -> Option<[f64; 4]> {
        self.0.viscosity
    }

    #[getter]
    fn get_diffusion(&self) -> Option<[f64; 5]> {
        self.0.diffusion
    }

    #[getter]
    fn get_thermal_conductivity(&self) -> Option<[f64; 4]> {
        self.0.thermal_conductivity
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PyPetsRecord);
impl_pure_record!(PetsRecord, PyPetsRecord);

/// Create a record for a binary interaction parameter, i.e. k_ij.
#[pyclass(name = "PetsBinaryRecord")]
#[derive(Clone)]
pub struct PyPetsBinaryRecord(PetsBinaryRecord);
impl_binary_record!(PetsBinaryRecord, PyPetsBinaryRecord);

#[pyclass(name = "PetsParameters")]
#[derive(Clone)]
pub struct PyPetsParameters(pub Arc<PetsParameters>);

#[pymethods]
impl PyPetsParameters {
    // Create a set of PeTS parameters from lists.
    ///
    /// Parameters
    /// ----------
    /// sigma : List[float]
    ///     PeTS segment diameter in units of Angstrom.
    /// epsilon_k : List[float]
    ///     PeTS energy parameter in units of Kelvin.
    /// k_ij: numpy.ndarray[float]
    ///     matrix of binary interaction parameters.
    /// molarweight: List[float], optional
    ///     molar weight in units of Gram per Mol.
    /// viscosity: List[List[float]], optional
    ///     entropy scaling parameters for viscosity.
    /// diffusion: List[List[float]], optional
    ///     entropy scaling parameters for self-diffusion.
    /// thermal_conductivity: List[List[float]], optional
    ///     entropy scaling parameters for thermal conductivity.
    /// Returns
    /// -------
    /// PetsParameters
    #[pyo3(
        text_signature = "(sigma, epsilon_k, k_ij=None, molarweight=None, viscosity=None, diffusion=None, thermal_conductivity=None)"
    )]
    #[staticmethod]
    fn from_lists(
        sigma: Vec<f64>,
        epsilon_k: Vec<f64>,
        k_ij: Option<&Bound<'_, PyArray2<f64>>>,
        molarweight: Option<Vec<f64>>,
        viscosity: Option<Vec<[f64; 4]>>,
        diffusion: Option<Vec<[f64; 5]>>,
        thermal_conductivity: Option<Vec<[f64; 4]>>,
    ) -> PyResult<Self> {
        // Check if all inputs have the same length
        let n = sigma.len();
        let input_length = [
            Some(sigma.len()),
            Some(epsilon_k.len()),
            k_ij.as_ref().map(|v| v.shape()[0]),
            k_ij.as_ref().map(|v| v.shape()[1]),
            molarweight.as_ref().map(|v| v.len()),
            viscosity.as_ref().map(|v| v.len()),
            diffusion.as_ref().map(|v| v.len()),
            thermal_conductivity.as_ref().map(|v| v.len()),
        ]
        .into_iter()
        .flatten()
        .all(|v| v == n);

        if !input_length {
            return Err(PyValueError::new_err(
                "shape of arguments could not be used together.",
            ));
        }

        // Define `PureRecord`s
        let pure_records = (0..n)
            .map(|i| {
                let identifier = Identifier::new(
                    Some(format!("{}", i).as_str()),
                    None,
                    None,
                    None,
                    None,
                    None,
                );
                let model_record = PetsRecord::new(
                    sigma[i],
                    epsilon_k[i],
                    viscosity.as_ref().map(|v| v[i]),
                    diffusion.as_ref().map(|v| v[i]),
                    thermal_conductivity.as_ref().map(|v| v[i]),
                );
                PureRecord::new(
                    identifier,
                    molarweight.as_ref().map_or(1.0, |v| v[i]),
                    model_record,
                )
                // Hier Ideal Gas anstatt None???
            })
            .collect();

        let binary = k_ij.map(|v| v.to_owned_array().mapv(f64::into));

        Ok(Self(Arc::new(PetsParameters::from_records(
            pure_records,
            binary,
        )?)))
    }

    // Create a set of PeTS parameters from values.
    ///
    /// Parameters
    /// ----------
    /// sigma : float
    ///     PeTS segment diameter in units of Angstrom.
    /// epsilon_k : float
    ///     PeTS energy parameter in units of Kelvin.
    /// molarweight: float, optional
    ///     molar weight in units of Gram per Mol.
    /// viscosity: List[float], optional
    ///     entropy scaling parameters for viscosity.
    /// diffusion: List[float], optional
    ///     entropy scaling parameters for self-diffusion.
    /// thermal_conductivity: List[float], optional
    ///     entropy scaling parameters for thermal conductivity.
    /// Returns
    /// -------
    /// PetsParameters
    #[pyo3(
        text_signature = "(sigma, epsilon_k, molarweight=None, viscosity=None, diffusion=None, thermal_conductivity=None)"
    )]
    #[staticmethod]
    fn from_values(
        sigma: f64,
        epsilon_k: f64,
        molarweight: Option<f64>,
        viscosity: Option<[f64; 4]>,
        diffusion: Option<[f64; 5]>,
        thermal_conductivity: Option<[f64; 4]>,
    ) -> PyResult<Self> {
        let pure_record = PureRecord::new(
            Identifier::new(
                Some(format!("{}", 1).as_str()),
                None,
                None,
                None,
                None,
                None,
            ),
            molarweight.map_or(1.0, |v| v),
            PetsRecord::new(sigma, epsilon_k, viscosity, diffusion, thermal_conductivity),
        );
        Ok(Self(Arc::new(PetsParameters::new_pure(pure_record)?)))
    }

    #[getter]
    fn get_k_ij<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.0.k_ij.as_ref().map(|k| k.view().to_pyarray_bound(py))
    }

    fn _repr_markdown_(&self) -> String {
        self.0.to_markdown()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_parameter!(
    PetsParameters,
    PyPetsParameters,
    PyPetsRecord,
    PyPetsBinaryRecord
);

#[pymodule]
pub fn pets(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<IdentifierOption>()?;

    m.add_class::<PyPetsRecord>()?;
    m.add_class::<PyPetsBinaryRecord>()?;
    m.add_class::<PyPureRecord>()?;
    m.add_class::<PyBinaryRecord>()?;
    m.add_class::<PyPetsParameters>()?;
    Ok(())
}
