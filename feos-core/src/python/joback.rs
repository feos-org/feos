use std::sync::Arc;

use crate::joback::{JobackBinaryRecord, JobackParameters, JobackRecord};
use crate::parameter::*;
use crate::python::parameter::*;
use crate::{
    impl_binary_record, impl_json_handling, impl_parameter, impl_parameter_from_segments,
    impl_pure_record, impl_segment_record,
};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::convert::{TryFrom, TryInto};

/// Create a set of Joback ideal gas heat capacity parameters
/// for a segment or a pure component.
///
/// The fourth order coefficient `e` is not present in the
/// orginial publication by Joback and Reid but is required
/// for correlations for some pure components that are modeled
/// using the same polynomial approach.
///
/// Parameters
/// ----------
/// a : float
///     zeroth order coefficient
/// b : float
///     first order coefficient
/// c : float
///     second order coefficient
/// d : float
///     third order coefficient
/// e : float
///     fourth order coefficient
///
/// Returns
/// -------
/// JobackRecord
#[pyclass(name = "JobackRecord")]
#[derive(Clone)]
pub struct PyJobackRecord(pub JobackRecord);

#[pymethods]
impl PyJobackRecord {
    #[new]
    fn new(a: f64, b: f64, c: f64, d: f64, e: f64) -> Self {
        Self(JobackRecord::new(a, b, c, d, e))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PyJobackRecord);
impl_pure_record!(JobackRecord, PyJobackRecord);
impl_segment_record!(JobackRecord, PyJobackRecord);

#[pyclass(name = "JobackBinaryRecord")]
#[derive(Clone)]
pub struct PyJobackBinaryRecord(pub JobackBinaryRecord);

impl_binary_record!(JobackBinaryRecord, PyJobackBinaryRecord);

/// Create a set of Joback parameters from records.
///
/// Parameters
/// ----------
/// pure_records : List[PureRecord]
///     pure substance records.
/// substances : List[str], optional
///     The substances to use. Filters substances from `pure_records` according to
///     `search_option`.
///     When not provided, all entries of `pure_records` are used.
/// search_option : {'Name', 'Cas', 'Inchi', 'IupacName', 'Formula', 'Smiles'}, optional, defaults to 'Name'.
///     Identifier that is used to search substance.
///
/// Returns
/// -------
/// JobackParameters
#[pyclass(name = "JobackParameters")]
#[pyo3(text_signature = "(pure_records, substances=None, search_option='Name')")]
#[derive(Clone)]
pub struct PyJobackParameters(pub Arc<JobackParameters>);

impl_parameter!(
    JobackParameters,
    PyJobackParameters,
    PyJobackRecord,
    PyJobackBinaryRecord
);
impl_parameter_from_segments!(JobackParameters, PyJobackParameters);

#[pymethods]
impl PyJobackParameters {
    // fn _repr_markdown_(&self) -> String {
    //     self.0.to_markdown()
    // }
}
