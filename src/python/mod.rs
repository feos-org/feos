#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::python::__PYO3_PYMODULE_DEF_GC_PCSAFT;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::python::__PYO3_PYMODULE_DEF_PCSAFT;
#[cfg(feature = "pets")]
use crate::pets::python::__PYO3_PYMODULE_DEF_PETS;
#[cfg(feature = "saftvrqmie")]
use crate::saftvrqmie::python::__PYO3_PYMODULE_DEF_SAFTVRQMIE;
#[cfg(feature = "uvtheory")]
use crate::uvtheory::python::__PYO3_PYMODULE_DEF_UVTHEORY;

use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use quantity::python::__PYO3_PYMODULE_DEF_QUANTITY;

mod cubic;
mod eos;
use cubic::__PYO3_PYMODULE_DEF_CUBIC;
use eos::__PYO3_PYMODULE_DEF_EOS;

#[cfg(feature = "dft")]
mod dft;
#[cfg(feature = "dft")]
use dft::__PYO3_PYMODULE_DEF_DFT;

#[pymodule]
pub fn feos(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_wrapped(wrap_pymodule!(quantity))?;

    m.add_wrapped(wrap_pymodule!(eos))?;
    #[cfg(feature = "dft")]
    m.add_wrapped(wrap_pymodule!(dft))?;
    m.add_wrapped(wrap_pymodule!(cubic))?;
    #[cfg(feature = "pcsaft")]
    m.add_wrapped(wrap_pymodule!(pcsaft))?;
    #[cfg(feature = "gc_pcsaft")]
    m.add_wrapped(wrap_pymodule!(gc_pcsaft))?;
    #[cfg(feature = "pets")]
    m.add_wrapped(wrap_pymodule!(pets))?;
    #[cfg(feature = "uvtheory")]
    m.add_wrapped(wrap_pymodule!(uvtheory))?;
    #[cfg(feature = "saftvrqmie")]
    m.add_wrapped(wrap_pymodule!(saftvrqmie))?;

    set_path(py, m, "feos.si", "quantity")?;
    set_path(py, m, "feos.eos", "eos")?;
    #[cfg(feature = "estimator")]
    set_path(py, m, "feos.eos.estimator", "eos.estimator_eos")?;
    #[cfg(feature = "dft")]
    set_path(py, m, "feos.dft", "dft")?;
    #[cfg(all(feature = "dft", feature = "estimator"))]
    set_path(py, m, "feos.dft.estimator", "dft.estimator_dft")?;
    set_path(py, m, "feos.cubic", "cubic")?;
    #[cfg(feature = "pcsaft")]
    set_path(py, m, "feos.pcsaft", "pcsaft")?;
    #[cfg(feature = "gc_pcsaft")]
    set_path(py, m, "feos.gc_pcsaft", "gc_pcsaft")?;
    #[cfg(feature = "pets")]
    set_path(py, m, "feos.pets", "pets")?;
    #[cfg(feature = "uvtheory")]
    set_path(py, m, "feos.uvtheory", "uvtheory")?;
    #[cfg(feature = "saftvrqmie")]
    set_path(py, m, "feos.saftvrqmie", "saftvrqmie")?;

    py.run(
        "\
import sys
quantity.SINumber.__module__ = 'feos.si'
quantity.SIArray1.__module__ = 'feos.si'
quantity.SIArray2.__module__ = 'feos.si'
quantity.SIArray3.__module__ = 'feos.si'
quantity.SIArray4.__module__ = 'feos.si'
    ",
        None,
        Some(m.dict()),
    )?;
    Ok(())
}

fn set_path(py: Python<'_>, m: &PyModule, path: &str, module: &str) -> PyResult<()> {
    py.run(
        &format!(
            "\
import sys
sys.modules['{path}'] = {module}
    "
        ),
        None,
        Some(m.dict()),
    )
}
