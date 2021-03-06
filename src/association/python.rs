use super::AssociationRecord;
use feos_core::impl_json_handling;
use feos_core::parameter::ParameterError;
use pyo3::prelude::*;

/// Pure component association parameters
#[pyclass(name = "AssociationRecord", unsendable)]
#[pyo3(text_signature = "(kappa_ab, epsilon_k_ab, na, nb)")]
#[derive(Clone)]
pub struct PyAssociationRecord(pub AssociationRecord);

#[pymethods]
impl PyAssociationRecord {
    #[new]
    fn new(kappa_ab: f64, epsilon_k_ab: f64, na: f64, nb: f64) -> Self {
        Self(AssociationRecord::new(kappa_ab, epsilon_k_ab, na, nb))
    }

    #[getter]
    fn get_kappa_ab(&self) -> f64 {
        self.0.kappa_ab
    }

    #[getter]
    fn get_epsilon_k_ab(&self) -> f64 {
        self.0.epsilon_k_ab
    }

    #[getter]
    fn get_na(&self) -> f64 {
        self.0.na
    }

    #[getter]
    fn get_nb(&self) -> f64 {
        self.0.nb
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PyAssociationRecord);
