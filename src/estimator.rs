use crate::eos::{EosVariant, PyEosVariant};
use feos_estimator::*;
use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use quantity::python::PySIArray1;
use quantity::si::SIUnit;
use std::collections::HashMap;
use std::rc::Rc;

impl_estimator!(EosVariant, PyEosVariant);

#[pymodule]
pub fn estimator(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDataSet>()?;
    m.add_class::<PyEstimator>()?;
    m.add_class::<PyLoss>()
}
