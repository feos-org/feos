#[cfg(feature = "epcsaft")]
use crate::epcsaft::python::epcsaft as epcsaft_module;
#[cfg(feature = "pets")]
use crate::pets::python::pets as pets_module;
#[cfg(feature = "saftvrmie")]
use crate::saftvrmie::python::saftvrmie as saftvrmie_module;
#[cfg(feature = "saftvrqmie")]
use crate::saftvrqmie::python::saftvrqmie as saftvrqmie_module;
#[cfg(feature = "uvtheory")]
use crate::uvtheory::python::uvtheory as uvtheory_module;

use feos_core::parameter::{BinarySegmentRecord, ChemicalRecord, Identifier, IdentifierOption};
use feos_core::python::parameter::{
    PyBinaryRecord, PyGcParameters, PyParameters, PyPureRecord, PySegmentRecord, PySmartsRecord,
};
use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use std::ffi::CString;

mod eos;
use eos::eos as eos_module;

#[cfg(feature = "dft")]
mod dft;
#[cfg(feature = "dft")]
use dft::dft as dft_module;

#[pymodule]
pub fn feos(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<Identifier>()?;
    m.add_class::<IdentifierOption>()?;
    m.add_class::<ChemicalRecord>()?;
    m.add_class::<PySmartsRecord>()?;

    m.add_class::<PyPureRecord>()?;
    m.add_class::<PySegmentRecord>()?;
    m.add_class::<PyBinaryRecord>()?;
    m.add_class::<BinarySegmentRecord>()?;
    m.add_class::<PyParameters>()?;
    m.add_class::<PyGcParameters>()?;

    m.add_wrapped(wrap_pymodule!(eos_module))?;
    #[cfg(feature = "dft")]
    m.add_wrapped(wrap_pymodule!(dft_module))?;
    #[cfg(feature = "epcsaft")]
    m.add_wrapped(wrap_pymodule!(epcsaft_module))?;
    #[cfg(feature = "pets")]
    m.add_wrapped(wrap_pymodule!(pets_module))?;
    #[cfg(feature = "uvtheory")]
    m.add_wrapped(wrap_pymodule!(uvtheory_module))?;
    #[cfg(feature = "saftvrqmie")]
    m.add_wrapped(wrap_pymodule!(saftvrqmie_module))?;
    #[cfg(feature = "saftvrmie")]
    m.add_wrapped(wrap_pymodule!(saftvrmie_module))?;

    set_path(m, "feos.eos", "eos")?;
    #[cfg(feature = "estimator")]
    set_path(m, "feos.eos.estimator", "eos.estimator_eos")?;
    #[cfg(feature = "dft")]
    set_path(m, "feos.dft", "dft")?;
    #[cfg(feature = "epcsaft")]
    set_path(m, "feos.epcsaft", "epcsaft")?;
    #[cfg(feature = "pets")]
    set_path(m, "feos.pets", "pets")?;
    #[cfg(feature = "uvtheory")]
    set_path(m, "feos.uvtheory", "uvtheory")?;
    #[cfg(feature = "saftvrqmie")]
    set_path(m, "feos.saftvrqmie", "saftvrqmie")?;
    #[cfg(feature = "saftvrmie")]
    set_path(m, "feos.saftvrmie", "saftvrmie")?;
    Ok(())
}

fn set_path(m: &Bound<'_, PyModule>, path: &str, module: &str) -> PyResult<()> {
    m.py().run(
        &CString::new(format!(
            "\
import sys
sys.modules['{path}'] = {module}
    "
        ))
        .unwrap(),
        None,
        Some(&m.dict()),
    )
}
