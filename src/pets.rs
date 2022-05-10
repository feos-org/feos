use feos_core::python::joback::PyJobackRecord;
use feos_core::python::parameter::*;
use feos_pets::python::{PyPetsParameters, PyPetsRecord, PyPureRecord};
use pyo3::prelude::*;

#[pymodule]
pub fn pets(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<PyChemicalRecord>()?;
    m.add_class::<PyJobackRecord>()?;

    m.add_class::<PyPetsRecord>()?;
    m.add_class::<PyPureRecord>()?;
    m.add_class::<PyBinaryRecord>()?;
    m.add_class::<PyPetsParameters>()?;
    Ok(())
}
