use std::sync::Arc;

use crate::ideal_gas::{Dippr, DipprRecord};
use feos_core::parameter::*;
use feos_core::python::parameter::*;
use feos_core::{impl_json_handling, impl_parameter, impl_pure_record, impl_segment_record};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::convert::{TryFrom, TryInto};

/// Create a set of Dippr ideal gas heat capacity parameters
/// for a segment or a pure component.
///
/// The fourth order coefficient `e` is not present in the
/// orginial publication by Dippr and Reid but is required
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
/// DipprRecord
#[pyclass(name = "DipprRecord")]
#[derive(Clone)]
pub struct PyDipprRecord(pub DipprRecord);

#[pymethods]
impl PyDipprRecord {
    #[staticmethod]
    #[pyo3(signature = (a=0.0, b=0.0, c=0.0, d=0.0, e=0.0, f=0.0, g=0.0))]
    fn eq100(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64) -> Self {
        Self(DipprRecord::eq100(a, b, c, d, e, f, g))
    }

    #[staticmethod]
    fn eq107(a: f64, b: f64, c: f64, d: f64, e: f64) -> Self {
        Self(DipprRecord::eq107(a, b, c, d, e))
    }

    #[staticmethod]
    fn eq127(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64) -> Self {
        Self(DipprRecord::eq127(a, b, c, d, e, f, g))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PyDipprRecord);
impl_pure_record!(DipprRecord, PyDipprRecord);
impl_segment_record!(DipprRecord, PyDipprRecord);

#[pyclass(name = "Dippr")]
#[derive(Clone)]
pub struct PyDippr(pub Arc<Dippr>);

impl_parameter!(Dippr, PyDippr, PyDipprRecord);

#[pymodule]
pub fn dippr(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<IdentifierOption>()?;
    m.add_class::<PyDipprRecord>()?;
    m.add_class::<PyPureRecord>()?;
    m.add_class::<PySegmentRecord>()?;
    m.add_class::<PyDippr>()
}
