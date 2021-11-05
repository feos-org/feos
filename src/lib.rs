use feos_core::python::{PyInit_cubic, PyInit_user_defined};
use feos_dft::python::PyInit_feos_dft;
use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use quantity::python::PyInit_quantity;

#[pymodule]
pub fn feos(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(quantity))?;

    m.add_wrapped(wrap_pymodule!(user_defined))?;
    m.add_wrapped(wrap_pymodule!(cubic))?;

    m.add_wrapped(wrap_pymodule!(feos_dft))?;

    py.run(
        "\
import sys
sys.modules['feos.si'] = quantity

sys.modules['feos.user_defined'] = user_defined
sys.modules['feos.cubic'] = cubic

sys.modules['feos.fmt'] = feos_dft
    ",
        None,
        Some(m.dict()),
    )?;
    Ok(())
}
