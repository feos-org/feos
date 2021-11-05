use feos_core::python::{PyInit_cubic, PyInit_user_defined};
use feos_dft::python::PyInit_feos_dft;
use feos_gc_pcsaft::python::PyInit_feos_gc_pcsaft;
use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use quantity::python::PyInit_quantity;

#[pymodule]
pub fn feos(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(quantity))?;

    m.add_wrapped(wrap_pymodule!(user_defined))?;
    m.add_wrapped(wrap_pymodule!(cubic))?;

    m.add_wrapped(wrap_pymodule!(feos_dft))?;

    m.add_wrapped(wrap_pymodule!(feos_gc_pcsaft))?;

    py.run(
        "\
import sys
sys.modules['feos.si'] = quantity

sys.modules['feos.user_defined'] = user_defined
sys.modules['feos.cubic'] = cubic

sys.modules['feos.fmt'] = feos_dft

sys.modules['feos.gc_pcsaft'] = feos_gc_pcsaft
sys.modules['feos.gc_pcsaft.eos'] = feos_gc_pcsaft.eos
    ",
        None,
        Some(m.dict()),
    )?;
    Ok(())
}
