use super::parameter::PyPureRecord;
use crate::cubic::PengRobinsonParameters;
use crate::parameter::{BinaryRecord, IdentifierOption, Parameter, ParameterError};
use crate::python::parameter::PyBinaryRecord;
use crate::*;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

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
