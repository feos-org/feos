#[cfg(feature = "epcsaft")]
use crate::epcsaft::python::epcsaft as epcsaft_module;
#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::python::gc_pcsaft as gc_pcsaft_module;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::python::pcsaft as pcsaft_module;
#[cfg(feature = "pets")]
use crate::pets::python::pets as pets_module;
#[cfg(feature = "saftvrmie")]
use crate::saftvrmie::python::saftvrmie as saftvrmie_module;
#[cfg(feature = "saftvrqmie")]
use crate::saftvrqmie::python::saftvrqmie as saftvrqmie_module;
#[cfg(feature = "uvtheory")]
use crate::uvtheory::python::uvtheory as uvtheory_module;

use pyo3::prelude::*;
use pyo3::wrap_pymodule;

mod cubic;
mod dippr;
mod eos;
mod joback;
use cubic::cubic as cubic_module;
use dippr::dippr as dippr_module;
use eos::eos as eos_module;
use joback::joback as joback_module;

#[cfg(feature = "dft")]
mod dft;
#[cfg(feature = "dft")]
use dft::dft as dft_module;

#[pymodule]
pub fn feos(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // m.add_wrapped(wrap_pymodule!(quantity_module))?;

    m.add_wrapped(wrap_pymodule!(eos_module))?;
    #[cfg(feature = "dft")]
    m.add_wrapped(wrap_pymodule!(dft_module))?;
    m.add_wrapped(wrap_pymodule!(joback_module))?;
    m.add_wrapped(wrap_pymodule!(dippr_module))?;
    m.add_wrapped(wrap_pymodule!(cubic_module))?;
    #[cfg(feature = "pcsaft")]
    m.add_wrapped(wrap_pymodule!(pcsaft_module))?;
    #[cfg(feature = "epcsaft")]
    m.add_wrapped(wrap_pymodule!(epcsaft_module))?;
    #[cfg(feature = "gc_pcsaft")]
    m.add_wrapped(wrap_pymodule!(gc_pcsaft_module))?;
    #[cfg(feature = "pets")]
    m.add_wrapped(wrap_pymodule!(pets_module))?;
    #[cfg(feature = "uvtheory")]
    m.add_wrapped(wrap_pymodule!(uvtheory_module))?;
    #[cfg(feature = "saftvrqmie")]
    m.add_wrapped(wrap_pymodule!(saftvrqmie_module))?;
    #[cfg(feature = "saftvrmie")]
    m.add_wrapped(wrap_pymodule!(saftvrmie_module))?;

    // set_path(m, "feos.si", "quantity")?;
    set_path(m, "feos.eos", "eos")?;
    #[cfg(feature = "estimator")]
    set_path(m, "feos.eos.estimator", "eos.estimator_eos")?;
    #[cfg(feature = "dft")]
    set_path(m, "feos.dft", "dft")?;
    #[cfg(all(feature = "dft", feature = "estimator"))]
    set_path(m, "feos.dft.estimator", "dft.estimator_dft")?;
    set_path(m, "feos.joback", "joback")?;
    set_path(m, "feos.dippr", "dippr")?;
    set_path(m, "feos.cubic", "cubic")?;
    #[cfg(feature = "pcsaft")]
    set_path(m, "feos.pcsaft", "pcsaft")?;
    #[cfg(feature = "epcsaft")]
    set_path(m, "feos.epcsaft", "epcsaft")?;
    #[cfg(feature = "gc_pcsaft")]
    set_path(m, "feos.gc_pcsaft", "gc_pcsaft")?;
    #[cfg(feature = "pets")]
    set_path(m, "feos.pets", "pets")?;
    #[cfg(feature = "uvtheory")]
    set_path(m, "feos.uvtheory", "uvtheory")?;
    #[cfg(feature = "saftvrqmie")]
    set_path(m, "feos.saftvrqmie", "saftvrqmie")?;
    #[cfg(feature = "saftvrmie")]
    set_path(m, "feos.saftvrmie", "saftvrmie")?;

    // re-export si_units within feos. Overriding __module__ is required for pickling.
    m.py().run_bound(
        "\
import sys
import si_units
sys.modules['feos.si'] = si_units
si_units.SIObject.__module__ = 'feos.si'
si_units.Angle.__module__ = 'feos.si'
        ",
        None,
        Some(&m.dict()),
    )?;
    Ok(())
}

fn set_path(m: &Bound<'_, PyModule>, path: &str, module: &str) -> PyResult<()> {
    m.py().run_bound(
        &format!(
            "\
import sys
sys.modules['{path}'] = {module}
    "
        ),
        None,
        Some(&m.dict()),
    )
}
