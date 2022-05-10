use feos_core::python::joback::PyJobackRecord;
use feos_core::python::parameter::*;
use feos_gc_pcsaft::python::{
    PyGcPcSaftEosParameters, PyGcPcSaftFunctionalParameters, PyGcPcSaftRecord, PySegmentRecord,
};
use pyo3::prelude::*;

#[pymodule]
pub fn gc_pcsaft(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<PyChemicalRecord>()?;
    m.add_class::<PyJobackRecord>()?;

    m.add_class::<PyGcPcSaftRecord>()?;
    m.add_class::<PySegmentRecord>()?;
    m.add_class::<PyBinaryRecord>()?;
    m.add_class::<PyBinarySegmentRecord>()?;
    m.add_class::<PyGcPcSaftEosParameters>()?;
    m.add_class::<PyGcPcSaftFunctionalParameters>()?;
    Ok(())
}
