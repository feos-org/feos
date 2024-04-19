use super::eos::permittivity::PermittivityRecord;
use super::parameters::{
    ElectrolytePcSaftBinaryRecord, ElectrolytePcSaftParameters, ElectrolytePcSaftRecord,
};
use super::ElectrolytePcSaftVariants;
use feos_core::parameter::{
    BinaryRecord, Identifier, IdentifierOption, Parameter, ParameterError, PureRecord,
    SegmentRecord,
};
use feos_core::python::parameter::*;
use feos_core::*;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

// Pure-substance parameters for the ePC-SAFT equation of state.
///
/// Parameters
/// ----------
/// m : float
///     Segment number
/// sigma : float
///     Segment diameter in units of Angstrom.
/// epsilon_k : float
///     Energetic parameter in units of Kelvin.
/// mu : float, optional
///     Dipole moment in units of Debye.
/// q : float, optional
///     Quadrupole moment in units of Debye * Angstrom.
/// kappa_ab : float, optional
///     Association volume parameter.
/// epsilon_k_ab : float, optional
///     Association energy parameter in units of Kelvin.
/// na : float, optional
///     Number of association sites of type A.
/// nb : float, optional
///     Number of association sites of type B.
/// nc : float, optional
///     Number of association sites of type C.
/// z : float, optional
///     Charge of the electrolyte.
/// permittivity_record : PyPermittivityRecord, optional
///     Permittivity record. Defaults to `None`.
#[pyclass(name = "ElectrolytePcSaftRecord")]
#[derive(Clone)]
pub struct PyElectrolytePcSaftRecord(ElectrolytePcSaftRecord);

#[pymethods]
impl PyElectrolytePcSaftRecord {
    #[new]
    #[pyo3(
        text_signature = "(m, sigma, epsilon_k, mu=None, q=None, kappa_ab=None, epsilon_k_ab=None, na=None, nb=None, nc=None, permittivity_record=None)"
    )]
    fn new(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        mu: Option<f64>,
        q: Option<f64>,
        kappa_ab: Option<f64>,
        epsilon_k_ab: Option<f64>,
        na: Option<f64>,
        nb: Option<f64>,
        nc: Option<f64>,
        z: Option<f64>,
        permittivity_record: Option<PyPermittivityRecord>,
    ) -> Self {
        Self(ElectrolytePcSaftRecord::new(
            m,
            sigma,
            epsilon_k,
            mu,
            q,
            kappa_ab,
            epsilon_k_ab,
            na,
            nb,
            nc,
            z,
            permittivity_record.map(|p| p.0),
        ))
    }

    #[getter]
    fn get_m(&self) -> f64 {
        self.0.m
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
    fn get_kappa_ab(&self) -> Option<f64> {
        self.0.association_record.map(|a| a.parameters.kappa_ab)
    }

    #[getter]
    fn get_epsilon_k_ab(&self) -> Option<f64> {
        self.0.association_record.map(|a| a.parameters.epsilon_k_ab)
    }

    #[getter]
    fn get_z(&self) -> Option<f64> {
        self.0.z
    }

    #[getter]
    fn get_na(&self) -> Option<f64> {
        self.0.association_record.map(|a| a.na)
    }

    #[getter]
    fn get_nb(&self) -> Option<f64> {
        self.0.association_record.map(|a| a.nb)
    }

    #[getter]
    fn get_nc(&self) -> Option<f64> {
        self.0.association_record.map(|a| a.nc)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PyElectrolytePcSaftRecord);

impl_pure_record!(ElectrolytePcSaftRecord, PyElectrolytePcSaftRecord);
impl_segment_record!(ElectrolytePcSaftRecord, PyElectrolytePcSaftRecord);

#[pyclass(name = "ElectrolytePcSaftBinaryRecord")]
#[derive(Clone)]
pub struct PyElectrolytePcSaftBinaryRecord(ElectrolytePcSaftBinaryRecord);

#[pymethods]
impl PyElectrolytePcSaftBinaryRecord {
    #[new]
    fn new(k_ij: [f64; 4]) -> Self {
        Self(ElectrolytePcSaftBinaryRecord::new(
            Some(k_ij.to_vec()),
            None,
            None,
        ))
    }

    #[getter]
    fn get_k_ij(&self) -> Vec<f64> {
        self.0.k_ij.clone()
    }

    #[setter]
    fn set_k_ij(&mut self, k_ij: [f64; 4]) {
        self.0.k_ij = k_ij.to_vec()
    }
}

impl_json_handling!(PyElectrolytePcSaftBinaryRecord);

impl_binary_record!(
    ElectrolytePcSaftBinaryRecord,
    PyElectrolytePcSaftBinaryRecord
);

#[pyclass(name = "ElectrolytePcSaftParameters")]
#[derive(Clone)]
pub struct PyElectrolytePcSaftParameters(pub Arc<ElectrolytePcSaftParameters>);

impl_parameter!(
    ElectrolytePcSaftParameters,
    PyElectrolytePcSaftParameters,
    PyElectrolytePcSaftRecord,
    PyElectrolytePcSaftBinaryRecord
);

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
    #[allow(non_snake_case)]
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
    #[allow(non_snake_case)]
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
pub fn epcsaft(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<IdentifierOption>()?;
    m.add_class::<PyChemicalRecord>()?;
    m.add_class::<PySmartsRecord>()?;

    m.add_class::<ElectrolytePcSaftVariants>()?;
    m.add_class::<PyElectrolytePcSaftRecord>()?;
    m.add_class::<PyElectrolytePcSaftBinaryRecord>()?;
    m.add_class::<PyPureRecord>()?;
    m.add_class::<PySegmentRecord>()?;
    m.add_class::<PyBinaryRecord>()?;
    m.add_class::<PyBinarySegmentRecord>()?;
    m.add_class::<PyElectrolytePcSaftParameters>()?;
    m.add_class::<PyPermittivityRecord>()?;
    Ok(())
}
