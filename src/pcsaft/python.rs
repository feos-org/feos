use super::parameters::{PcSaftBinaryRecord, PcSaftParameters, PcSaftRecord};
use crate::association::PyAssociationRecord;
use feos_core::joback::JobackRecord;
use feos_core::parameter::{
    BinaryRecord, Identifier, IdentifierOption, Parameter, ParameterError, PureRecord,
    SegmentRecord,
};
use feos_core::python::joback::PyJobackRecord;
use feos_core::python::parameter::*;
use feos_core::*;
use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::convert::{TryFrom, TryInto};
use std::rc::Rc;

/// Create a set of PC-Saft parameters from records.
/// 
/// Parameters
/// ----------
/// m : float
///     Number of SAFT segments.
/// sigma : float
///     Segment diameter in units of Angstrom.
/// epsilon_k : float
///     Energetic parameter of segment in units of Kelvin.
/// mu : float, optional
///     Dipole moment. Defaults to None (mu = 0.0).
/// q : float, optional
///     Quadrupole moment. Defaults to None (q = 0.0).
/// association_record : AssociationRecord, optional
///     Parameters for association. Defaluts to None.
/// viscosity : List[float], optional
///     Parameters for entropy scaling of viscosity.
///     Defaults to None (viscosity = [0.0]*4).
/// diffusion : List[float], optional
///     Parameters for entropy scaling of diffusion.
///     Defaults to None (diffusion = [0.0]*5).
/// thermal_conductivity : List[float], optional
///     Parameters for entropy scaling of thermal_conductivity.
///     Defaults to None (thermal_conductivity = [0.0]*4).
/// 
/// Returns
/// -------
/// PcSaftRecord
#[pyclass(name = "PcSaftRecord")]
#[pyo3(
    text_signature = "(m, sigma, epsilon_k, mu=None, q=None, association_record=None, viscosity=None, diffusion=None, thermal_conductivity=None)"
)]
#[derive(Clone)]
pub struct PyPcSaftRecord(PcSaftRecord);

#[pymethods]
impl PyPcSaftRecord {
    #[new]
    fn new(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        mu: Option<f64>,
        q: Option<f64>,
        association_record: Option<PyAssociationRecord>,
        viscosity: Option<[f64; 4]>,
        diffusion: Option<[f64; 5]>,
        thermal_conductivity: Option<[f64; 4]>,
    ) -> Self {
        Self(PcSaftRecord::new(
            m,
            sigma,
            epsilon_k,
            mu,
            q,
            association_record.map(|r| r.0),
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
    fn get_mu(&self) -> Option<f64> {
        self.0.mu
    }

    #[getter]
    fn get_q(&self) -> Option<f64> {
        self.0.q
    }

    #[getter]
    fn get_association_record(&self) -> Option<PyAssociationRecord> {
        self.0.association_record.clone().map(PyAssociationRecord)
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

impl_json_handling!(PyPcSaftRecord);

impl_pure_record!(PcSaftRecord, PyPcSaftRecord, JobackRecord, PyJobackRecord);
impl_segment_record!(PcSaftRecord, PyPcSaftRecord, JobackRecord, PyJobackRecord);

#[pyclass(name = "PcSaftBinaryRecord", unsendable)]
#[pyo3(
    text_signature = "(pure_records, binary_records=None, substances=None, search_option='Name')"
)]
#[derive(Clone)]
pub struct PyPcSaftBinaryRecord(PcSaftBinaryRecord);
impl_binary_record!(PcSaftBinaryRecord, PyPcSaftBinaryRecord);

/// Create a set of PC-SAFT parameters from records.
///
/// Parameters
/// ----------
/// pure_records : List[PureRecord]
///     pure substance records.
/// binary_records : List[BinaryRecord], optional
///     binary saft parameter records
/// substances : List[str], optional
///     The substances to use. Filters substances from `pure_records` according to
///     `search_option`.
///     When not provided, all entries of `pure_records` are used.
/// search_option : {'Name', 'Cas', 'Inchi', 'IupacName', 'Formula', 'Smiles'}, optional, defaults to 'Name'.
///     Identifier that is used to search substance.
///
/// Returns
/// -------
/// PcSaftParameters
#[pyclass(name = "PcSaftParameters", unsendable)]
#[pyo3(
    text_signature = "(pure_records, binary_records=None, substances=None, search_option='Name')"
)]
#[derive(Clone)]
pub struct PyPcSaftParameters(pub Rc<PcSaftParameters>);

impl_parameter!(PcSaftParameters, PyPcSaftParameters);
impl_parameter_from_segments!(PcSaftParameters, PyPcSaftParameters);

#[pymethods]
impl PyPcSaftParameters {
    #[getter]
    fn get_k_ij<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self.0.k_ij.view().to_pyarray(py)
    }

    fn _repr_markdown_(&self) -> String {
        self.0.to_markdown()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

#[pymodule]
pub fn pcsaft(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<IdentifierOption>()?;
    m.add_class::<PyChemicalRecord>()?;
    m.add_class::<PyJobackRecord>()?;
    m.add_class::<PyAssociationRecord>()?;

    m.add_class::<PyPcSaftRecord>()?;
    m.add_class::<PyPureRecord>()?;
    m.add_class::<PySegmentRecord>()?;
    m.add_class::<PyBinaryRecord>()?;
    m.add_class::<PyBinarySegmentRecord>()?;
    m.add_class::<PyPcSaftParameters>()?;
    Ok(())
}
