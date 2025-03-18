use super::eos::permittivity::PermittivityRecord;
use super::parameters::ElectrolytePcSaftParameters;
use super::ElectrolytePcSaftVariants;
use feos_core::parameter::{BinaryRecord, IdentifierOption, Parameter, ParameterError};
use feos_core::python::parameter::*;
use feos_core::*;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

#[pyclass(name = "ElectrolytePcSaftParameters")]
#[derive(Clone)]
pub struct PyElectrolytePcSaftParameters(pub Arc<ElectrolytePcSaftParameters>);

impl_parameter!(ElectrolytePcSaftParameters, PyElectrolytePcSaftParameters);

#[pymethods]
impl PyElectrolytePcSaftParameters {
    fn _repr_markdown_(&self) -> String {
        self.0.to_markdown()
    }
}

/// Class permittivity record
#[pyclass(name = "PermittivityRecord", unsendable)]
#[derive(Clone)]
pub struct PyPermittivityRecord(pub PermittivityRecord);

#[pymethods]
impl PyPermittivityRecord {
    /// from_experimental_data
    ///
    /// Parameters
    /// ----------
    /// interpolation_points : Vec<Vec<(f64, f64)>>
    ///
    /// Returns
    /// -------
    /// PermittivityRecord
    ///
    #[staticmethod]
    #[pyo3(text_signature = "(interpolation_points)")]
    pub fn from_experimental_data(interpolation_points: Vec<(f64, f64)>) -> Self {
        Self(PermittivityRecord::ExperimentalData {
            data: interpolation_points,
        })
    }

    /// from_perturbation_theory
    ///
    /// Parameters
    /// ----------
    /// dipole_scaling : Vec<f64>,
    /// polarizability_scaling : Vec<f64>,
    /// correlation_integral_parameter : Vec<f64>,
    ///
    /// Returns
    /// -------
    /// PermittivityRecord
    ///
    #[staticmethod]
    #[pyo3(
        text_signature = "(dipole_scaling, polarizability_scaling, correlation_integral_parameter)"
    )]
    pub fn from_perturbation_theory(
        dipole_scaling: f64,
        polarizability_scaling: f64,
        correlation_integral_parameter: f64,
    ) -> Self {
        Self(PermittivityRecord::PerturbationTheory {
            dipole_scaling,
            polarizability_scaling,
            correlation_integral_parameter,
        })
    }
}

#[pymodule]
pub fn epcsaft(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ElectrolytePcSaftVariants>()?;
    m.add_class::<PyElectrolytePcSaftParameters>()?;
    m.add_class::<PyPermittivityRecord>()?;
    Ok(())
}
