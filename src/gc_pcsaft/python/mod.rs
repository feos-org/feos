#[cfg(feature = "dft")]
use super::dft::GcPcSaftFunctionalParameters;
use super::eos::GcPcSaftEosParameters;
use super::record::GcPcSaftRecord;
use feos_core::joback::JobackRecord;
use feos_core::parameter::{
    BinaryRecord, IdentifierOption, ParameterError, ParameterHetero, SegmentRecord,
};
use feos_core::python::joback::PyJobackRecord;
use feos_core::python::parameter::{PyBinarySegmentRecord, PyChemicalRecord, PyIdentifier};
use feos_core::{impl_json_handling, impl_parameter_from_segments, impl_segment_record};
#[cfg(feature = "dft")]
use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;
use std::rc::Rc;

#[cfg(feature = "micelles")]
mod micelles;

#[pyclass(name = "GcPcSaftRecord", unsendable)]
#[pyo3(
    text_signature = "(m, sigma, epsilon_k, mu=None, q=None, kappa_ab=None, epsilon_k_ab=None, na=None, nb=None)"
)]
#[derive(Clone)]
pub struct PyGcPcSaftRecord(GcPcSaftRecord);

#[pymethods]
impl PyGcPcSaftRecord {
    #[new]
    fn new(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        mu: Option<f64>,
        kappa_ab: Option<f64>,
        epsilon_k_ab: Option<f64>,
        na: Option<f64>,
        nb: Option<f64>,
        psi_dft: Option<f64>,
    ) -> Self {
        Self(GcPcSaftRecord::new(
            m,
            sigma,
            epsilon_k,
            mu,
            kappa_ab,
            epsilon_k_ab,
            na,
            nb,
            psi_dft,
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PyGcPcSaftRecord);

impl_segment_record!(
    GcPcSaftRecord,
    PyGcPcSaftRecord,
    JobackRecord,
    PyJobackRecord
);

#[pyclass(name = "GcPcSaftEosParameters", unsendable)]
#[pyo3(
    text_signature = "(pure_records, segmentbinary_records=None, substances=None, search_option='Name')"
)]
#[derive(Clone)]
pub struct PyGcPcSaftEosParameters(pub Rc<GcPcSaftEosParameters>);

impl_parameter_from_segments!(GcPcSaftEosParameters, PyGcPcSaftEosParameters);

#[pymethods]
impl PyGcPcSaftEosParameters {
    fn _repr_markdown_(&self) -> String {
        self.0.to_markdown()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

#[cfg(feature = "dft")]
#[pyclass(name = "GcPcSaftFunctionalParameters", unsendable)]
#[pyo3(
    text_signature = "(pure_records, segmentbinary_records=None, substances=None, search_option='Name')"
)]
#[derive(Clone)]
pub struct PyGcPcSaftFunctionalParameters(pub Rc<GcPcSaftFunctionalParameters>);

#[cfg(feature = "dft")]
impl_parameter_from_segments!(GcPcSaftFunctionalParameters, PyGcPcSaftFunctionalParameters);

#[cfg(feature = "dft")]
#[pymethods]
impl PyGcPcSaftFunctionalParameters {
    fn _repr_markdown_(&self) -> String {
        self.0.to_markdown()
    }

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
    m.add_class::<PyChemicalRecord>()?;
    m.add_class::<PyJobackRecord>()?;

    m.add_class::<PyGcPcSaftRecord>()?;
    m.add_class::<PySegmentRecord>()?;
    m.add_class::<PyBinarySegmentRecord>()?;
    m.add_class::<PyGcPcSaftEosParameters>()?;
    m.add_class::<PyGcPcSaftFunctionalParameters>()?;
    Ok(())
}
