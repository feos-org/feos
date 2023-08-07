//! Python bindings for the SAFT-VRQ Mie equation of state.
use crate::saftvrqmie::eos::FeynmanHibbsOrder;
use crate::saftvrqmie::parameters::{
    SaftVRQMieBinaryRecord, SaftVRQMieParameters, SaftVRQMieRecord,
};
use feos_core::parameter::{
    BinaryRecord, Identifier, IdentifierOption, Parameter, ParameterError, PureRecord,
};
use feos_core::python::parameter::PyIdentifier;
use feos_core::*;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::{PyIOError, PyTypeError};
use pyo3::prelude::*;
use quantity::python::PySINumber;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

/// Pure-substance parameters for the Saft-VRQ Mie equation of state.
///
/// Parameters
/// ----------
/// m : float
///     Segment number
/// sigma : float
///     Structure parameter of the Mie potential in units of
///     Angstrom.
/// epsilon_k : float
///     Energetic parameter of the Mie potential in units of
///     Kelvin.
/// lr : float
///     Repulsive exponent of the Mie potential.
/// la : float
///     Attractive exponent of the Mie potential.
/// fh : int
///     Feynman-Hibbs order. One of {0, 1, 2}.
///     `fh = 0` disables quantum corrections so that effectively,
///     the SAFT-VR Mie equation of state is used.
/// viscosity : List[float], optional
///     Entropy-scaling parameters for viscosity. Defaults to `None`.
/// diffusion : List[float], optional
///     Entropy-scaling parameters for diffusion. Defaults to `None`.
/// thermal_conductivity : List[float], optional
///     Entropy-scaling parameters for thermal_conductivity. Defaults to `None`.
#[pyclass(name = "SaftVRQMieRecord")]
#[pyo3(
    text_signature = "(m, sigma, epsilon_k, lr, la, fh, viscosity=None, diffusion=None, thermal_conductivity=None)"
)]
#[derive(Clone)]
pub struct PySaftVRQMieRecord(SaftVRQMieRecord);

#[pymethods]
impl PySaftVRQMieRecord {
    #[new]
    fn new(
        m: f64,
        sigma: f64,
        epsilon_k: f64,
        lr: f64,
        la: f64,
        fh: usize,
        viscosity: Option<[f64; 4]>,
        diffusion: Option<[f64; 5]>,
        thermal_conductivity: Option<[f64; 4]>,
    ) -> PyResult<Self> {
        Ok(Self(SaftVRQMieRecord::new(
            m,
            sigma,
            epsilon_k,
            lr,
            la,
            fh,
            viscosity,
            diffusion,
            thermal_conductivity,
        )?))
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

/// Create a set of Saft-VRQ Mie parameters from records.
#[pyclass(name = "SaftVRQMieBinaryRecord")]
#[pyo3(text_signature = "(k_ij, l_ij)")]
#[derive(Clone)]
pub struct PySaftVRQMieBinaryRecord(SaftVRQMieBinaryRecord);

#[pymethods]
impl PySaftVRQMieBinaryRecord {
    #[new]
    fn new(k_ij: f64, l_ij: f64) -> Self {
        Self(SaftVRQMieBinaryRecord { k_ij, l_ij })
    }

    #[getter]
    fn get_k_ij(&self) -> f64 {
        self.0.k_ij
    }

    #[getter]
    fn get_l_ij(&self) -> f64 {
        self.0.l_ij
    }

    #[setter]
    fn set_k_ij(&mut self, k_ij: f64) {
        self.0.k_ij = k_ij
    }

    #[setter]
    fn set_l_ij(&mut self, l_ij: f64) {
        self.0.l_ij = l_ij
    }
}

/// Create a set of SAFT-VRQ Mie parameters from records.
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
/// SaftVRQMieParameters
#[pyclass(name = "SaftVRQMieParameters")]
#[pyo3(
    text_signature = "(pure_records, binary_records=None, substances=None, search_option='Name')"
)]
#[derive(Clone)]
pub struct PySaftVRQMieParameters(pub Arc<SaftVRQMieParameters>);

impl_json_handling!(PySaftVRQMieRecord);
impl_pure_record!(SaftVRQMieRecord, PySaftVRQMieRecord);
impl_binary_record!(SaftVRQMieBinaryRecord, PySaftVRQMieBinaryRecord);
impl_parameter!(
    SaftVRQMieParameters,
    PySaftVRQMieParameters,
    PySaftVRQMieRecord,
    PySaftVRQMieBinaryRecord
);

#[pymethods]
impl PySaftVRQMieParameters {
    #[getter]
    fn get_k_ij<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self.0.k_ij.view().to_pyarray(py)
    }

    #[getter]
    fn get_l_ij<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self.0.l_ij.view().to_pyarray(py)
    }

    /// Generate energy and force tables to be used with LAMMPS' `pair_style table` command.
    ///
    /// Parameters
    /// ----------
    /// temperature : SINumber
    ///     temperature at which the Feynman-Hibbs corrected Mie potential
    ///     is evaluated.
    /// n : int
    ///     total number of points to calculate in the table between r_min and r_max.
    /// r_min : SINumber
    ///     minimum distance (included)
    /// r_max : SINumber
    ///     maximum distance (included)
    ///
    /// Raises
    /// ------
    /// IOError
    ///     if there are issues with writing to a file.
    ///
    /// Notes
    /// -----
    ///
    /// For a given `temperature`, `n` values between `r_min` and `r_max` (both including) are tabulated.
    ///
    /// Files for all pure substances and all unique pairs are generated,
    /// where filenames use either the "name" field of the identifier or the index if no name is present.
    ///
    /// Example
    /// -------
    ///
    /// For a hydrogen-neon mixture at 30 K, three files will be created.
    ///
    /// - "hydrogen_30K.table" for H-H interactions,
    /// - "neon_30K.table" for Ne-Ne interactions,
    /// - "hydrogen_neon_30K.table" for H-Ne interactions.
    #[pyo3(text_signature = "($self, temperature, n, r_min, r_max)")]
    fn lammps_tables(
        &self,
        temperature: PySINumber,
        n: usize,
        r_min: PySINumber,
        r_max: PySINumber,
    ) -> PyResult<()> {
        self.0
            .lammps_tables(
                temperature.try_into()?,
                n,
                r_min.try_into()?,
                r_max.try_into()?,
            )
            .map_err(PyIOError::new_err)
    }

    fn _repr_markdown_(&self) -> String {
        self.0.to_markdown()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

#[pymodule]
pub fn saftvrqmie(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<IdentifierOption>()?;
    m.add_class::<FeynmanHibbsOrder>()?;

    m.add_class::<PySaftVRQMieRecord>()?;
    m.add_class::<PyPureRecord>()?;
    m.add_class::<PyBinaryRecord>()?;
    m.add_class::<PySaftVRQMieParameters>()?;
    Ok(())
}
