use feos_core::python::joback::PyJobackRecord;
use feos_core::python::parameter::*;
use feos_fcsaft::python::{PyFcSaftParameters, PyFcSaftRecord, PySegmentRecord};
use pyo3::prelude::*;

#[pymodule]
pub fn fcsaft(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<PyChemicalRecord>()?;
    m.add_class::<PyJobackRecord>()?;

    m.add_class::<PyFcSaftRecord>()?;
    m.add_class::<PySegmentRecord>()?;
    m.add_class::<PyBinaryRecord>()?;
    m.add_class::<PyBinarySegmentRecord>()?;
    m.add_class::<PyFcSaftParameters>()?;
    Ok(())
}
