use feos_core::python::cubic::*;
use feos_core::python::parameter::{PyChemicalRecord, PyIdentifier};
use pyo3::prelude::*;

#[pymodule]
pub fn cubic_old(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<PyChemicalRecord>()?;

    m.add_class::<PyPengRobinsonRecord>()?;
    m.add_class::<PyPureRecord>()?;
    m.add_class::<PyBinaryRecord>()?;
    m.add_class::<PyPengRobinsonParameters>()?;
    Ok(())
}
