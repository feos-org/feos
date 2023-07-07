use super::parameters::{NoRecord, UVBinaryRecord, UVParameters, UVRecord};
use super::{Perturbation, VirialOrder};
use feos_core::parameter::{
    BinaryRecord, Identifier, IdentifierOption, Parameter, ParameterError, PureRecord,
};
use feos_core::python::parameter::*;
use feos_core::*;
use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

/// Create a set of UV Theory parameters from records.
#[pyclass(name = "NoRecord")]
#[derive(Clone)]
struct PyNoRecord(NoRecord);

/// Create a set of UV Theory parameters from records.
#[pyclass(name = "UVRecord")]
#[pyo3(text_signature = "(rep, att, sigma, epsilon_k)")]
#[derive(Clone)]
pub struct PyUVRecord(UVRecord);

#[pymethods]
impl PyUVRecord {
    #[new]
    fn new(rep: f64, att: f64, sigma: f64, epsilon_k: f64) -> Self {
        Self(UVRecord::new(rep, att, sigma, epsilon_k))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PyUVRecord);

#[pyclass(name = "UVBinaryRecord")]
#[derive(Clone)]
pub struct PyUVBinaryRecord(UVBinaryRecord);
impl_binary_record!(UVBinaryRecord, PyUVBinaryRecord);

/// Create a set of UV Theory parameters from records.
///
/// Parameters
/// ----------
/// pure_records : List[PureRecord]
///     pure substance records.
/// binary_records : List[BinarySubstanceRecord], optional
///     binary parameter records
/// substances : List[str], optional
///     The substances to use. Filters substances from `pure_records` according to
///     `search_option`.
///     When not provided, all entries of `pure_records` are used.
/// search_option : IdentifierOption, optional, defaults to IdentifierOption.Name
///     Identifier that is used to search binary records.
#[pyclass(name = "UVParameters")]
#[pyo3(text_signature = "(pure_records, binary_records, substances, search_option)")]
#[derive(Clone)]
pub struct PyUVParameters(pub Arc<UVParameters>);

#[pymethods]
impl PyUVParameters {
    /// Create a set of UV Theory parameters from lists.
    ///
    /// Parameters
    /// ----------
    /// rep : List[float]
    ///     repulsive exponents
    /// att : List[float]
    ///     attractive exponents
    /// sigma : List[float]
    ///     Mie diameter in units of Angstrom
    /// epsilon_k : List[float]
    ///     Mie energy parameter in units of Kelvin
    ///
    /// Returns
    /// -------
    /// UVParameters
    #[pyo3(text_signature = "(rep, att, sigma, epsilon_k)")]
    #[staticmethod]
    fn from_lists(rep: Vec<f64>, att: Vec<f64>, sigma: Vec<f64>, epsilon_k: Vec<f64>) -> Self {
        let n = rep.len();
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
                let model_record = UVRecord::new(rep[i], att[i], sigma[i], epsilon_k[i]);
                PureRecord::new(identifier, 1.0, model_record)
            })
            .collect();
        let binary = Array2::from_shape_fn((n, n), |(_, _)| UVBinaryRecord { k_ij: 0.0 });
        Self(Arc::new(UVParameters::from_records(pure_records, binary)))
    }

    /// Create UV Theory parameters for pure substance.
    ///
    /// Parameters
    /// ----------
    /// rep : float
    ///     repulsive exponents
    /// att : float
    ///     attractive exponents
    /// sigma : float
    ///     Mie diameter in units of Angstrom
    /// epsilon_k : float
    ///     Mie energy parameter in units of Kelvin
    ///
    /// Returns
    /// -------
    /// UVParameters
    ///
    /// # Info
    ///
    /// Molar weight is one. No ideal gas contribution is considered.
    #[pyo3(text_signature = "(rep, att, sigma, epsilon_k)")]
    #[staticmethod]
    fn new_simple(rep: f64, att: f64, sigma: f64, epsilon_k: f64) -> Self {
        Self(Arc::new(UVParameters::new_simple(
            rep, att, sigma, epsilon_k,
        )))
    }
}

impl_pure_record!(UVRecord, PyUVRecord);
impl_parameter!(UVParameters, PyUVParameters);

#[pymodule]
pub fn uvtheory(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<IdentifierOption>()?;
    m.add_class::<PyChemicalRecord>()?;

    m.add_class::<Perturbation>()?;
    m.add_class::<VirialOrder>()?;
    m.add_class::<PyUVRecord>()?;
    m.add_class::<PyPureRecord>()?;
    m.add_class::<PyBinaryRecord>()?;
    m.add_class::<PyUVParameters>()?;
    Ok(())
}
