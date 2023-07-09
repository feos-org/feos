use crate::cubic::{PengRobinsonParameters, PengRobinsonRecord};
use crate::parameter::{
    BinaryRecord, Identifier, IdentifierOption, Parameter, ParameterError, PureRecord,
};
use crate::python::parameter::PyIdentifier;
use crate::*;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

/// A pure substance parameter for the Peng-Robinson equation of state.
#[pyclass(name = "PengRobinsonRecord")]
#[derive(Clone)]
pub struct PyPengRobinsonRecord(PengRobinsonRecord);

#[pymethods]
impl PyPengRobinsonRecord {
    #[new]
    fn new(tc: f64, pc: f64, acentric_factor: f64) -> Self {
        Self(PengRobinsonRecord::new(tc, pc, acentric_factor))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PyPengRobinsonRecord);

impl_pure_record!(PengRobinsonRecord, PyPengRobinsonRecord);

impl_binary_record!();

/// Create a set of Peng-Robinson parameters from records.
///
/// Parameters
/// ----------
/// pure_records : List[PureRecord]
///     pure substance records.
/// binary_records : List[BinaryRecord], optional
///     binary parameter records
/// substances : List[str], optional
///     The substances to use. Filters substances from `pure_records` according to
///     `search_option`.
///     When not provided, all entries of `pure_records` are used.
/// search_option : {'Name', 'Cas', 'Inchi', 'IupacName', 'Formula', 'Smiles'}, optional, defaults to 'Name'.
///     Identifier that is used to search substance.
///
/// Returns
/// -------
/// PengRobinsonParameters
#[pyclass(name = "PengRobinsonParameters")]
#[derive(Clone)]
pub struct PyPengRobinsonParameters(pub Arc<PengRobinsonParameters>);

impl_parameter!(
    PengRobinsonParameters,
    PyPengRobinsonParameters,
    PyPengRobinsonRecord,
    f64
);

#[pymethods]
impl PyPengRobinsonParameters {
    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}
