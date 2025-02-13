use super::parameter::PyPureRecord;
use crate::cubic::PengRobinsonParameters;
use crate::parameter::{IdentifierOption, Parameter, ParameterError};
use crate::python::parameter::PyBinaryRecord;
use crate::*;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

// /// A pure substance parameter for the Peng-Robinson equation of state.
// #[pyclass(name = "PengRobinsonRecord")]
// #[derive(Clone)]
// pub struct PyPengRobinsonRecord(PengRobinsonRecord);

// #[pymethods]
// impl PyPengRobinsonRecord {
//     #[new]
//     fn new(tc: f64, pc: f64, acentric_factor: f64) -> Self {
//         Self(PengRobinsonRecord::new(tc, pc, acentric_factor))
//     }

//     fn __repr__(&self) -> PyResult<String> {
//         Ok(self.0.to_string())
//     }
// }

// impl_json_handling!(PyPengRobinsonRecord);

// impl_pure_record!(PengRobinsonRecord, PyPengRobinsonRecord);

// impl_binary_record!();

#[pyclass(name = "PengRobinsonParameters")]
#[derive(Clone)]
pub struct PyPengRobinsonParameters(pub Arc<PengRobinsonParameters>);

impl_parameter!(PengRobinsonParameters, PyPengRobinsonParameters);

#[pymethods]
impl PyPengRobinsonParameters {
    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}
