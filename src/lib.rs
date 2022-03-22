use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use quantity::python::__PYO3_PYMODULE_DEF_QUANTITY;
mod eos;
use eos::__PYO3_PYMODULE_DEF_EOS;
mod pcsaft;
use pcsaft::__PYO3_PYMODULE_DEF_PCSAFT;
mod pets;
use pets::__PYO3_PYMODULE_DEF_PETS;

#[pymodule]
pub fn feos(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(quantity))?;
    m.add_wrapped(wrap_pymodule!(eos))?;
    m.add_wrapped(wrap_pymodule!(pcsaft))?;
    m.add_wrapped(wrap_pymodule!(pets))?;
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
sys.modules['feos.pcsaft'] = pcsaft
sys.modules['feos.pets'] = pets
    ",
        None,
        Some(m.dict()),
    )?;
    Ok(())
}
