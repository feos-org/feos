#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]
use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use quantity::python::__PYO3_PYMODULE_DEF_QUANTITY;
mod eos;
use eos::__PYO3_PYMODULE_DEF_EOS;
mod dft;
use dft::__PYO3_PYMODULE_DEF_DFT;
mod cubic;
use cubic::__PYO3_PYMODULE_DEF_CUBIC;
mod pcsaft;
use pcsaft::__PYO3_PYMODULE_DEF_PCSAFT;
mod gc_pcsaft;
use gc_pcsaft::__PYO3_PYMODULE_DEF_GC_PCSAFT;
mod pets;
use pets::__PYO3_PYMODULE_DEF_PETS;
mod uvtheory;
use uvtheory::__PYO3_PYMODULE_DEF_UVTHEORY;
mod estimator;
use estimator::__PYO3_PYMODULE_DEF_ESTIMATOR;

#[pymodule]
pub fn feos(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(quantity))?;
    m.add_wrapped(wrap_pymodule!(eos))?;
    m.add_wrapped(wrap_pymodule!(dft))?;
    m.add_wrapped(wrap_pymodule!(cubic))?;
    m.add_wrapped(wrap_pymodule!(pcsaft))?;
    m.add_wrapped(wrap_pymodule!(gc_pcsaft))?;
    m.add_wrapped(wrap_pymodule!(pets))?;
    m.add_wrapped(wrap_pymodule!(uvtheory))?;
    m.add_wrapped(wrap_pymodule!(estimator))?;
    py.run(
        "\
import sys
quantity.SINumber.__module__ = 'feos.si'
quantity.SIArray1.__module__ = 'feos.si'
quantity.SIArray2.__module__ = 'feos.si'
quantity.SIArray3.__module__ = 'feos.si'
quantity.SIArray4.__module__ = 'feos.si'
sys.modules['feos.si'] = quantity
sys.modules['feos.eos'] = eos
sys.modules['feos.dft'] = dft
sys.modules['feos.cubic'] = cubic
sys.modules['feos.pcsaft'] = pcsaft
sys.modules['feos.gc_pcsaft'] = gc_pcsaft
sys.modules['feos.pets'] = pets
sys.modules['feos.uvtheory'] = uvtheory
sys.modules['feos.estimator'] = estimator
    ",
        None,
        Some(m.dict()),
    )?;
    Ok(())
}
