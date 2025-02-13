use feos_core::python::cubic::*;
use pyo3::prelude::*;

#[pymodule]
pub fn cubic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPengRobinsonParameters>()?;
    Ok(())
}
