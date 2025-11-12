use super::parameters::{QuantumCorrection, UVCSBinaryRecord, UVCSParameters, UVCSRecord};
use feos_core::parameter::{
    BinaryRecord, Identifier, IdentifierOption, Parameter, ParameterError, PureRecord,
};
use feos_core::python::parameter::*;
use feos_core::*;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

#[pyclass(name = "QuantumCorrection")]
#[derive(Clone)]
pub struct PyQuantumCorrection(QuantumCorrection);

#[pymethods]
impl PyQuantumCorrection {
    #[staticmethod]
    #[pyo3(signature = (c_sigma=None, c_epsilon_k=None, c_rep=None))]
    fn feynman_hibbs1(
        c_sigma: Option<[f64; 3]>,
        c_epsilon_k: Option<[f64; 3]>,
        c_rep: Option<[f64; 5]>,
    ) -> Self {
        Self(QuantumCorrection::FeynmanHibbs1 {
            c_sigma,
            c_epsilon_k,
            c_rep,
        })
    }
}

/// Create a set of UV Theory parameters from records.
#[pyclass(name = "UVCSRecord")]
#[derive(Clone)]
pub struct PyUVCSRecord(UVCSRecord);

#[pymethods]
impl PyUVCSRecord {
    #[new]
    #[pyo3(signature = (rep, att, sigma, epsilon_k, quantum_correction=None))]
    fn new(
        rep: f64,
        att: f64,
        sigma: f64,
        epsilon_k: f64,
        quantum_correction: Option<PyQuantumCorrection>,
    ) -> Self {
        Self(UVCSRecord::new(
            rep,
            att,
            sigma,
            epsilon_k,
            quantum_correction.map(|qc| qc.0),
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }

    #[getter]
    fn get_quantum_correction(&self) -> Option<PyQuantumCorrection> {
        if let Some(qc) = self.0.quantum_correction.as_ref() {
            Some(PyQuantumCorrection(qc.clone()))
        } else {
            None
        }
    }
}

/// Create a binary record from k_ij and l_ij values.
#[pyclass(name = "UVCSBinaryRecord")]
#[derive(Clone)]
pub struct PyUVCSBinaryRecord(UVCSBinaryRecord);

#[pymethods]
impl PyUVCSBinaryRecord {
    #[new]
    #[pyo3(text_signature = "(k_ij, l_ij)")]
    fn new(k_ij: f64, l_ij: f64) -> Self {
        Self(UVCSBinaryRecord { k_ij, l_ij })
    }

    #[getter]
    fn get_k_ij(&self) -> f64 {
        self.0.k_ij
    }

    #[getter]
    fn get_l_ij(&self) -> f64 {
        self.0.l_ij
    }

    #[setter]
    fn set_k_ij(&mut self, k_ij: f64) {
        self.0.k_ij = k_ij
    }

    #[setter]
    fn set_l_ij(&mut self, l_ij: f64) {
        self.0.l_ij = l_ij
    }
}

impl_json_handling!(PyUVCSRecord);
impl_binary_record!(UVCSBinaryRecord, PyUVCSBinaryRecord);

#[pyclass(name = "UVCSParameters")]
#[derive(Clone)]
pub struct PyUVCSParameters(pub Arc<UVCSParameters>);

#[pymethods]
impl PyUVCSParameters {
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
    /// UVCSParameters
    #[pyo3(text_signature = "(rep, att, sigma, epsilon_k)")]
    #[staticmethod]
    fn from_lists(
        rep: Vec<f64>,
        att: Vec<f64>,
        sigma: Vec<f64>,
        epsilon_k: Vec<f64>,
        quantum_correction: Vec<Option<PyQuantumCorrection>>,
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
                let model_record = UVCSRecord::new(
                    rep[i],
                    att[i],
                    sigma[i],
                    epsilon_k[i],
                    quantum_correction[i].as_ref().map(|qc| qc.0.clone()),
                );
                PureRecord::new(identifier, 1.0, model_record)
            })
            .collect();
        Ok(Self(Arc::new(UVCSParameters::from_records(
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
    /// UVCSParameters
    ///
    /// # Info
    ///
    /// Molar weight is one. No ideal gas contribution is considered.
    #[pyo3(text_signature = "(rep, att, sigma, epsilon_k)")]
    #[staticmethod]
    fn new_simple(rep: f64, att: f64, sigma: f64, epsilon_k: f64) -> PyResult<Self> {
        Ok(Self(Arc::new(UVCSParameters::new_simple(
            rep, att, sigma, epsilon_k,
        )?)))
    }

    /// Print effective parameters
    fn print_effective_parameters(&self, temperature: f64) -> String {
        self.0.print_effective_parameters(temperature)
    }

    fn _repr_markdown_(&self) -> String {
        self.0.to_markdown()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_pure_record!(UVCSRecord, PyUVCSRecord);
impl_parameter!(
    UVCSParameters,
    PyUVCSParameters,
    PyUVCSRecord,
    PyUVCSBinaryRecord
);

#[pymodule]
pub fn uvcstheory(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<IdentifierOption>()?;

    m.add_class::<PyQuantumCorrection>()?;
    m.add_class::<PyUVCSRecord>()?;
    m.add_class::<PyUVCSBinaryRecord>()?;
    m.add_class::<PyPureRecord>()?;
    m.add_class::<PyBinaryRecord>()?;
    m.add_class::<PyUVCSParameters>()?;
    Ok(())
}
