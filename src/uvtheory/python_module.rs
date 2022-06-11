use feos_core::python::parameter::*;
use feos_uvtheory::{
    python::{PyPureRecord, PyUVParameters, PyUVRecord},
    Perturbation,
};
use pyo3::prelude::*;

#[pymodule]
pub fn uvtheory(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<PyChemicalRecord>()?;

    m.add_class::<Perturbation>()?;
    m.add_class::<PyUVRecord>()?;
    m.add_class::<PyPureRecord>()?;
    m.add_class::<PyBinaryRecord>()?;
    m.add_class::<PyUVParameters>()?;
    Ok(())
}
