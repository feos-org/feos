use feos_core::python::joback::PyJobackRecord;
use feos_core::python::parameter::*;
use pyo3::prelude::*;

#[pymodule]
pub fn ideal_gas(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyJobackRecord>()
}
