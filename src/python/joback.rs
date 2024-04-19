use std::sync::Arc;

use crate::ideal_gas::{Joback, JobackRecord};
use feos_core::parameter::*;
use feos_core::python::parameter::*;
use feos_core::{
    impl_json_handling, impl_parameter, impl_parameter_from_segments, impl_pure_record,
    impl_segment_record,
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

/// Ideal gas model based on the Joback & Reid GC method.
#[pyclass(name = "Joback")]
#[derive(Clone)]
pub struct PyJoback(pub Arc<Joback>);

impl_parameter!(Joback, PyJoback, PyJobackRecord);
impl_parameter_from_segments!(Joback, PyJoback);

#[pymethods]
impl PyJoback {
    // fn _repr_markdown_(&self) -> String {
    //     self.0.to_markdown()
    // }
}

#[pymodule]
pub fn joback(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<IdentifierOption>()?;
    m.add_class::<PyJobackRecord>()?;
    m.add_class::<PyPureRecord>()?;
    m.add_class::<PySegmentRecord>()?;
    m.add_class::<PyJoback>()
}
