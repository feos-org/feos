#[cfg(feature = "dft")]
use super::dft::GcPcSaftFunctionalParameters;
use super::eos::GcPcSaftEosParameters;
use super::record::GcPcSaftRecord;
use crate::association::PyAssociationRecord;
use feos_core::parameter::{
    BinaryRecord, IdentifierOption, ParameterError, ParameterHetero, SegmentRecord,
};
use feos_core::python::parameter::{
    PyBinarySegmentRecord, PyChemicalRecord, PyIdentifier, PySmartsRecord,
};
use feos_core::{impl_json_handling, impl_parameter_from_segments, impl_segment_record};
#[cfg(feature = "dft")]
use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;
use std::sync::Arc;

#[cfg(feature = "micelles")]
mod micelles;

#[pyclass(name = "GcPcSaftRecord")]
#[pyo3(text_signature = "(m, sigma, epsilon_k, mu=None, association_record=None, psi_dft=None)")]
#[derive(Clone)]
pub struct PyGcPcSaftRecord(GcPcSaftRecord);

#[pymethods]
impl PyGcPcSaftRecord {
    #[pyo3(signature = (m, sigma, epsilon_k, mu=None, association_record=None, psi_dft=None))]
    #[new]
    fn new(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        mu: Option<f64>,
        association_record: Option<PyAssociationRecord>,
        psi_dft: Option<f64>,
    ) -> Self {
        Self(GcPcSaftRecord::new(
            m,
            sigma,
            epsilon_k,
            mu,
            association_record.map(|r| r.0),
            psi_dft,
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
    fn get_mu(&self) -> Option<f64> {
        self.0.mu
    }

    #[getter]
    fn get_association_record(&self) -> Option<PyAssociationRecord> {
        self.0.association_record.map(PyAssociationRecord)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PyGcPcSaftRecord);

impl_segment_record!(GcPcSaftRecord, PyGcPcSaftRecord);

#[pyclass(name = "GcPcSaftEosParameters")]
#[pyo3(
    text_signature = "(pure_records, segmentbinary_records=None, substances=None, search_option='Name')"
)]
#[derive(Clone)]
pub struct PyGcPcSaftEosParameters(pub Arc<GcPcSaftEosParameters>);

impl_parameter_from_segments!(GcPcSaftEosParameters, PyGcPcSaftEosParameters);

#[pymethods]
impl PyGcPcSaftEosParameters {
    fn phi(&self, phi: Vec<f64>) -> PyResult<Self> {
        Ok(Self(Arc::new((*self.0).clone().phi(&phi)?)))
    }

    fn _repr_markdown_(&self) -> String {
        self.0.to_markdown()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

#[cfg(feature = "dft")]
#[pyclass(name = "GcPcSaftFunctionalParameters")]
#[pyo3(
    text_signature = "(pure_records, segmentbinary_records=None, substances=None, search_option)"
)]
#[derive(Clone)]
pub struct PyGcPcSaftFunctionalParameters(pub Arc<GcPcSaftFunctionalParameters>);

#[cfg(feature = "dft")]
impl_parameter_from_segments!(GcPcSaftFunctionalParameters, PyGcPcSaftFunctionalParameters);

#[cfg(feature = "dft")]
#[pymethods]
impl PyGcPcSaftFunctionalParameters {
    // fn _repr_markdown_(&self) -> String {
    //     self.0.to_markdown()
    // }

    #[getter]
    fn get_graph(&self, py: Python) -> PyResult<PyObject> {
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            "def f(s): 
                import graphviz
                return graphviz.Source(s.replace('\\\\\"', ''))",
            "",
            "",
        )?
        .getattr("f")?
        .into();
        fun.call1(py, (self.0.graph(),))
    }

    #[getter]
    fn get_k_ij<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self.0.k_ij.view().to_pyarray(py)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

#[pymodule]
pub fn gc_pcsaft(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<IdentifierOption>()?;
    m.add_class::<PyChemicalRecord>()?;
    m.add_class::<PySmartsRecord>()?;
    m.add_class::<PyAssociationRecord>()?;

    m.add_class::<PyGcPcSaftRecord>()?;
    m.add_class::<PySegmentRecord>()?;
    m.add_class::<PyBinarySegmentRecord>()?;
    m.add_class::<PyGcPcSaftEosParameters>()?;
    #[cfg(feature = "dft")]
    m.add_class::<PyGcPcSaftFunctionalParameters>()?;
    Ok(())
}
