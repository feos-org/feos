use super::parameters::{UVTheoryParameters, UVTheoryRecord};
use super::Perturbation;
use feos_core::parameter::{
    BinaryRecord, Identifier, IdentifierOption, Parameter, ParameterError, PureRecord,
};
use feos_core::python::parameter::*;
use feos_core::*;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

#[pyclass(name = "UVTheoryParameters")]
#[derive(Clone)]
pub struct PyUVTheoryParameters(pub Arc<UVTheoryParameters>);

#[pymethods]
impl PyUVTheoryParameters {
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
    /// UVTheoryParameters
    #[pyo3(text_signature = "(rep, att, sigma, epsilon_k)")]
    #[staticmethod]
    fn from_lists(
        rep: Vec<f64>,
        att: Vec<f64>,
        sigma: Vec<f64>,
        epsilon_k: Vec<f64>,
    ) -> PyResult<Self> {
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
                let model_record = UVTheoryRecord::new(rep[i], att[i], sigma[i], epsilon_k[i]);
                PureRecord::new(identifier, 1.0, model_record)
            })
            .collect();
        Ok(Self(Arc::new(UVTheoryParameters::from_records(
            pure_records,
            None,
        )?)))
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
    /// UVTheoryParameters
    ///
    /// # Info
    ///
    /// Molar weight is one. No ideal gas contribution is considered.
    #[pyo3(text_signature = "(rep, att, sigma, epsilon_k)")]
    #[staticmethod]
    fn new_simple(rep: f64, att: f64, sigma: f64, epsilon_k: f64) -> PyResult<Self> {
        Ok(Self(Arc::new(UVTheoryParameters::new_simple(
            rep, att, sigma, epsilon_k,
        )?)))
    }
}

impl_parameter!(UVTheoryParameters, PyUVTheoryParameters);

#[pymodule]
pub fn uvtheory(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Perturbation>()?;
    m.add_class::<PyUVTheoryParameters>()?;
    Ok(())
}
