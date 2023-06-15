#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::python::gc_pcsaft as gc_pcsaft_module;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::python::pcsaft as pcsaft_module;
#[cfg(feature = "pets")]
use crate::pets::python::pets as pets_module;
#[cfg(feature = "saftvrqmie")]
use crate::saftvrqmie::python::saftvrqmie as saftvrqmie_module;
#[cfg(feature = "uvtheory")]
use crate::uvtheory::python::uvtheory as uvtheory_module;

use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use quantity::python::quantity as quantity_module;

mod cubic;
mod eos;
mod ideal_gas;
use cubic::cubic as cubic_module;
use eos::eos as eos_module;
use ideal_gas::ideal_gas as ideal_gas_module;

#[cfg(feature = "dft")]
mod dft;
#[cfg(feature = "dft")]
use dft::dft as dft_module;

#[pymodule]
pub fn feos(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_wrapped(wrap_pymodule!(quantity_module))?;

    m.add_wrapped(wrap_pymodule!(eos_module))?;
    #[cfg(feature = "dft")]
    m.add_wrapped(wrap_pymodule!(dft_module))?;
    m.add_wrapped(wrap_pymodule!(ideal_gas_module))?;
    m.add_wrapped(wrap_pymodule!(cubic_module))?;
    #[cfg(feature = "pcsaft")]
    m.add_wrapped(wrap_pymodule!(pcsaft_module))?;
    #[cfg(feature = "gc_pcsaft")]
    m.add_wrapped(wrap_pymodule!(gc_pcsaft_module))?;
    #[cfg(feature = "pets")]
    m.add_wrapped(wrap_pymodule!(pets_module))?;
    #[cfg(feature = "uvtheory")]
    m.add_wrapped(wrap_pymodule!(uvtheory_module))?;
    #[cfg(feature = "saftvrqmie")]
    m.add_wrapped(wrap_pymodule!(saftvrqmie_module))?;

    set_path(py, m, "feos.si", "quantity")?;
    set_path(py, m, "feos.eos", "eos")?;
    #[cfg(feature = "estimator")]
    set_path(py, m, "feos.eos.estimator", "eos.estimator_eos")?;
    #[cfg(feature = "dft")]
    set_path(py, m, "feos.dft", "dft")?;
    #[cfg(all(feature = "dft", feature = "estimator"))]
    set_path(py, m, "feos.dft.estimator", "dft.estimator_dft")?;
    set_path(py, m, "feos.ideal_gas", "ideal_gas")?;
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
