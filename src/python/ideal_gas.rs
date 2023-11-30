use feos_core::parameter::IdentifierOption;
use feos_core::python::joback::*;
use feos_core::python::parameter::PyIdentifier;
use pyo3::prelude::*;

#[pymodule]
pub fn ideal_gas(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<IdentifierOption>()?;
    m.add_class::<PyJobackRecord>()?;
    m.add_class::<PyPureRecord>()?;
    m.add_class::<PySegmentRecord>()?;
    m.add_class::<PyJobackParameters>()
}
