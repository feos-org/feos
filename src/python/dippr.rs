use std::sync::Arc;

use crate::ideal_gas::{Dippr, DipprRecord};
use feos_core::parameter::*;
use feos_core::python::parameter::*;
use feos_core::{impl_json_handling, impl_parameter, impl_pure_record, impl_segment_record};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::convert::{TryFrom, TryInto};

/// DIPPR ideal gas heat capacity parameters for a pure component.
#[pyclass(name = "DipprRecord")]
#[derive(Clone)]
pub struct PyDipprRecord(pub DipprRecord);

#[pymethods]
impl PyDipprRecord {
    /// Create a set of parameters for DIPPR eq. # 100.
    ///
    /// Parameters
    /// ----------
    /// coefs : list[float]
    ///     Model parameters.
    ///
    /// Returns
    /// -------
    /// DipprRecord
    #[staticmethod]
    fn eq100(coefs: Vec<f64>) -> Self {
        Self(DipprRecord::eq100(&coefs))
    }

    /// Create a set of parameters for DIPPR eq. # 107.
    ///
    /// Parameters
    /// ----------
    /// a-e : float
    ///     Model parameters.
    ///
    /// Returns
    /// -------
    /// DipprRecord
    #[staticmethod]
    fn eq107(a: f64, b: f64, c: f64, d: f64, e: f64) -> Self {
        Self(DipprRecord::eq107(a, b, c, d, e))
    }

    /// Create a set of parameters for DIPPR eq. # 127.
    ///
    /// Parameters
    /// ----------
    /// a-g : float
    ///     Model parameters.
    ///
    /// Returns
    /// -------
    /// DipprRecord
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
