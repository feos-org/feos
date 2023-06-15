use feos_core::python::joback::{PyJobackParameters, PyJobackRecord};
use pyo3::prelude::*;

#[pymodule]
pub fn ideal_gas(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyJobackRecord>()?;
    m.add_class::<PyJobackParameters>()
}
