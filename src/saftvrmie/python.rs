use super::parameters::{SaftVRMieBinaryRecord, SaftVRMieParameters, SaftVRMieRecord};
use feos_core::parameter::{
    BinaryRecord, Identifier, IdentifierOption, Parameter, ParameterError, PureRecord,
};
use feos_core::python::parameter::*;
use feos_core::*;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use super::eos::association::AssociationRecord;
use feos_core::impl_json_handling;

/// Pure component association parameters
#[pyclass(name = "SaftVRMieAssociationRecord")]
#[derive(Clone)]
pub struct PySaftVRMieAssociationRecord(pub AssociationRecord);

#[pymethods]
impl PySaftVRMieAssociationRecord {
    #[new]
    #[pyo3(signature = (rc_ab, epsilon_k_ab, na=0.0, nb=0.0, nc=0.0))]
    fn new(rc_ab: f64, epsilon_k_ab: f64, na: f64, nb: f64, nc: f64) -> Self {
        Self(AssociationRecord::new(rc_ab, epsilon_k_ab, na, nb, nc))
    }

    #[getter]
    fn get_rc_ab(&self) -> f64 {
        self.0.rc_ab
    }

    #[getter]
    fn get_epsilon_k_ab(&self) -> f64 {
        self.0.epsilon_k_ab
    }

    #[getter]
    fn get_na(&self) -> f64 {
        self.0.na
    }

    #[getter]
    fn get_nb(&self) -> f64 {
        self.0.nb
    }

    #[getter]
    fn get_nc(&self) -> f64 {
        self.0.nc
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PySaftVRMieAssociationRecord);

/// Pure-substance parameters for the SAFT VR Mie equation of state.
///
/// Parameters
/// ----------
/// m : float
///     Segment number
/// sigma : float
///     Segment diameter in units of Angstrom.
/// epsilon_k : float
///     Energetic parameter in units of Kelvin.
/// lr : float
///     Repulsive Mie exponent.
/// la : float
///     Attractive Mie exponent.
/// rc_ab : float, optional
///     Dimensionless association distance parameter (divided by sigma).
/// epsilon_k_ab : float, optional
///     Association energy parameter in units of Kelvin.
/// na : float, optional
///     Number of association sites of type A.
/// nb : float, optional
///     Number of association sites of type B.
/// nc : float, optional
///     Number of association sites of type C.
/// viscosity : List[float], optional
///     Entropy-scaling parameters for viscosity. Defaults to `None`.
/// diffusion : List[float], optional
///     Entropy-scaling parameters for diffusion. Defaults to `None`.
/// thermal_conductivity : List[float], optional
///     Entropy-scaling parameters for thermal_conductivity. Defaults to `None`.
#[pyclass(name = "SaftVRMieRecord")]
#[derive(Clone)]
pub struct PySaftVRMieRecord(SaftVRMieRecord);

#[pymethods]
impl PySaftVRMieRecord {
    #[new]
    #[pyo3(
        text_signature = "(m, sigma, epsilon_k, lr, la, rc_ab=None, epsilon_k_ab=None, na=None, nb=None, nc=None, viscosity=None, diffusion=None, thermal_conductivity=None)",
        signature = (m, sigma, epsilon_k, lr, la, rc_ab=None, epsilon_k_ab=None, na=None, nb=None, nc=None, viscosity=None, diffusion=None, thermal_conductivity=None)
    )]
    #[expect(clippy::too_many_arguments)]
    fn new(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        lr: f64,
        la: f64,
        rc_ab: Option<f64>,
        epsilon_k_ab: Option<f64>,
        na: Option<f64>,
        nb: Option<f64>,
        nc: Option<f64>,
        viscosity: Option<[f64; 4]>,
        diffusion: Option<[f64; 5]>,
        thermal_conductivity: Option<[f64; 4]>,
    ) -> Self {
        Self(SaftVRMieRecord::new(
            m,
            sigma,
            epsilon_k,
            lr,
            la,
            rc_ab,
            epsilon_k_ab,
            na,
            nb,
            nc,
            viscosity,
            diffusion,
            thermal_conductivity,
        ))
    }

    #[getter]
    fn get_m(&self) -> f64 {
        self.0.m
    }

    #[getter]
    fn get_sigma(&self) -> f64 {
        self.0.sigma
    }

    #[getter]
    fn get_epsilon_k(&self) -> f64 {
        self.0.epsilon_k
    }

    #[getter]
    fn get_lr(&self) -> f64 {
        self.0.lr
    }

    #[getter]
    fn get_la(&self) -> f64 {
        self.0.la
    }

    #[getter]
    fn get_rc_ab(&self) -> Option<f64> {
        self.0.association_record.map(|a| a.rc_ab)
    }

    #[getter]
    fn get_epsilon_k_ab(&self) -> Option<f64> {
        self.0.association_record.map(|a| a.epsilon_k_ab)
    }

    #[getter]
    fn get_na(&self) -> Option<f64> {
        self.0.association_record.map(|a| a.na)
    }

    #[getter]
    fn get_nb(&self) -> Option<f64> {
        self.0.association_record.map(|a| a.nb)
    }

    #[getter]
    fn get_nc(&self) -> Option<f64> {
        self.0.association_record.map(|a| a.nc)
    }

    #[getter]
    fn get_viscosity(&self) -> Option<[f64; 4]> {
        self.0.viscosity
    }

    #[getter]
    fn get_diffusion(&self) -> Option<[f64; 5]> {
        self.0.diffusion
    }

    #[getter]
    fn get_thermal_conductivity(&self) -> Option<[f64; 4]> {
        self.0.thermal_conductivity
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PySaftVRMieRecord);
impl_pure_record!(SaftVRMieRecord, PySaftVRMieRecord);

/// Create a record for a binary interaction parameter.
#[pyclass(name = "SaftVRMieBinaryRecord")]
#[derive(Clone)]
pub struct PySaftVRMieBinaryRecord(SaftVRMieBinaryRecord);

#[pymethods]
impl PySaftVRMieBinaryRecord {
    #[new]
    #[pyo3(text_signature = "(k_ij=None, gamma_ij=None, rc_ab=None, epsilon_k_ab=None)")]
    #[pyo3(signature = (k_ij=None, gamma_ij=None, rc_ab=None, epsilon_k_ab=None))]
    fn new(
        k_ij: Option<f64>,
        gamma_ij: Option<f64>,
        rc_ab: Option<f64>,
        epsilon_k_ab: Option<f64>,
    ) -> Self {
        Self(SaftVRMieBinaryRecord::new(
            k_ij,
            gamma_ij,
            rc_ab,
            epsilon_k_ab,
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PySaftVRMieBinaryRecord);
impl_binary_record!(SaftVRMieBinaryRecord, PySaftVRMieBinaryRecord);

#[pyclass(name = "SaftVRMieParameters")]
#[derive(Clone)]
pub struct PySaftVRMieParameters(pub Arc<SaftVRMieParameters>);

impl_parameter!(
    SaftVRMieParameters,
    PySaftVRMieParameters,
    PySaftVRMieRecord,
    PySaftVRMieBinaryRecord
);

#[pymethods]
impl PySaftVRMieParameters {
    #[getter]
    fn get_k_ij<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.0
            .binary_records
            .as_ref()
            .map(|br| br.map(|br| br.k_ij).view().to_pyarray_bound(py))
    }

    fn _repr_markdown_(&self) -> String {
        self.0.to_markdown()
    }
}

#[pymodule]
pub fn saftvrmie(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<IdentifierOption>()?;

    m.add_class::<PySaftVRMieRecord>()?;
    m.add_class::<PySaftVRMieBinaryRecord>()?;
    m.add_class::<PyPureRecord>()?;
    m.add_class::<PyBinaryRecord>()?;
    m.add_class::<PySaftVRMieParameters>()?;
    Ok(())
}
