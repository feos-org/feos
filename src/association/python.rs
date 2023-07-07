use super::{AssociationRecord, BinaryAssociationRecord};
use feos_core::impl_json_handling;
use feos_core::parameter::ParameterError;
use pyo3::prelude::*;

/// Pure component association parameters
#[pyclass(name = "AssociationRecord")]
#[derive(Clone)]
pub struct PyAssociationRecord(pub AssociationRecord);

#[pymethods]
impl PyAssociationRecord {
    #[new]
    #[pyo3(signature = (kappa_ab, epsilon_k_ab, na=0.0, nb=0.0, nc=0.0))]
    fn new(kappa_ab: f64, epsilon_k_ab: f64, na: f64, nb: f64, nc: f64) -> Self {
        Self(AssociationRecord::new(kappa_ab, epsilon_k_ab, na, nb, nc))
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

    #[getter]
    fn get_nc(&self) -> f64 {
        self.0.nc
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PyAssociationRecord);

/// Binary association parameters
#[pyclass(name = "BinaryAssociationRecord")]
#[derive(Clone)]
pub struct PyBinaryAssociationRecord(pub BinaryAssociationRecord);

#[pymethods]
impl PyBinaryAssociationRecord {
    #[new]
    // #[pyo3(signature = (kappa_ab, epsilon_k_ab, na=0.0, nb=0.0, nc=0.0))]
    fn new(
        kappa_ab: Option<f64>,
        epsilon_k_ab: Option<f64>,
        site_indices: Option<[usize; 2]>,
    ) -> Self {
        Self(BinaryAssociationRecord::new(
            kappa_ab,
            epsilon_k_ab,
            site_indices,
        ))
    }

    #[getter]
    fn get_kappa_ab(&self) -> Option<f64> {
        self.0.kappa_ab
    }

    #[getter]
    fn get_epsilon_k_ab(&self) -> Option<f64> {
        self.0.epsilon_k_ab
    }

    // #[getter]
    // fn get_na(&self) -> f64 {
    //     self.0.na
    // }

    // #[getter]
    // fn get_nb(&self) -> f64 {
    //     self.0.nb
    // }

    // #[getter]
    // fn get_nc(&self) -> f64 {
    //     self.0.nc
    // }

    // fn __repr__(&self) -> PyResult<String> {
    //     Ok(self.0.to_string())
    // }
}

impl_json_handling!(PyBinaryAssociationRecord);
