#[cfg(feature = "dft")]
use super::dft::GcPcSaftFunctionalParameters;
use super::eos::GcPcSaftEosParameters;
use super::record::GcPcSaftRecord;
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
#[derive(Clone)]
pub struct PyGcPcSaftRecord(GcPcSaftRecord);

#[pymethods]
impl PyGcPcSaftRecord {
    #[new]
    #[pyo3(
        text_signature = "(m, sigma, epsilon_k, mu=None, kappa_ab=None, epsilon_k_ab=None, na=None, nb=None, nc=None, psi_dft=None)"
    )]
    #[expect(clippy::too_many_arguments)]
    fn new(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        mu: Option<f64>,
        kappa_ab: Option<f64>,
        epsilon_k_ab: Option<f64>,
        na: Option<f64>,
        nb: Option<f64>,
        nc: Option<f64>,
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
            nc,
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
    fn get_kappa_ab(&self) -> Option<f64> {
        self.0.association_record.map(|a| a.parameters.kappa_ab)
    }

    #[getter]
    fn get_epsilon_k_ab(&self) -> Option<f64> {
        self.0.association_record.map(|a| a.parameters.epsilon_k_ab)
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

impl_json_handling!(PyGcPcSaftRecord);

impl_segment_record!(GcPcSaftRecord, PyGcPcSaftRecord);

#[pyclass(name = "GcPcSaftEosParameters")]
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
        let fun: Py<PyAny> = PyModule::from_code_bound(
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
    fn get_k_ij<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.0.k_ij.view().to_pyarray_bound(py)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

#[pymodule]
pub fn gc_pcsaft(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<IdentifierOption>()?;
    m.add_class::<PyChemicalRecord>()?;
    m.add_class::<PySmartsRecord>()?;

    m.add_class::<PyGcPcSaftRecord>()?;
    m.add_class::<PySegmentRecord>()?;
    m.add_class::<PyBinarySegmentRecord>()?;
    m.add_class::<PyGcPcSaftEosParameters>()?;
    #[cfg(feature = "dft")]
    m.add_class::<PyGcPcSaftFunctionalParameters>()?;
    Ok(())
}
