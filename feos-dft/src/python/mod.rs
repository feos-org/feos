use crate::{Angle, DEGREES, RADIANS};
use pyo3::prelude::*;

mod adsorption;
mod interface;
mod profile;
mod solvation;
mod solver;

pub use adsorption::PyExternalPotential;
pub use solver::{PyDFTSolver, PyDFTSolverLog};

impl<'py> FromPyObject<'py> for Angle {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let (value, degrees) = ob
            .call_method0("_into_raw_parts")?
            .extract::<(f64, bool)>()?;
        Ok(if degrees {
            value * DEGREES
        } else {
            value * RADIANS
        })
    }
}
