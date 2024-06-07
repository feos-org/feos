use super::alpha::{
    Alpha, GeneralizedTwu, MathiasCopeman, PengRobinson1976, PengRobinson1978, PengRobinson2019,
    RedlichKwong1972, RedlichKwong2019, Soave, Twu,
};
use super::mixing_rules::{MixingRule, Quadratic};
use super::parameters::*;
use feos_core::parameter::*;
use feos_core::python::parameter::*;
use feos_core::{impl_binary_record, impl_json_handling, impl_parameter, impl_pure_record};
use numpy::{prelude::*, PyReadonlyArray1};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

#[pyclass(name = "Alpha")]
#[derive(Clone)]
pub struct PyAlpha(pub Alpha);

#[pymethods]
impl PyAlpha {
    /// Generalized Soave alpha function $\alpha(T) = \left[1 + m (1 - \sqrt{T_r})\right]^2$.
    ///
    /// with
    ///
    /// $m = \sum_i=0^N m_i \omega^i$
    ///
    /// Parameters
    /// ----------
    /// m : numpy.ndarray[float]
    ///     Polynomial coefficients for polynomial of acentric factors.
    ///     The number of coefficients specifies the polynomial degree.
    ///
    /// Returns
    /// -------
    /// Alpha
    #[staticmethod]
    fn soave(m: PyReadonlyArray1<'_, f64>) -> Self {
        Self(Soave::new(m.as_array().to_owned()).into())
    }

    /// Soave alpha function parameterized in 1972 for Redlich-Kwong EoS.
    ///
    /// $\alpha(T) = \left[1 + m_\text{RK} (1 - \sqrt{T_r})\right]^2$
    /// using
    /// $m_\text{RK} = 0.480 + 1.574 \omega - 0.176 \omega^2$
    #[staticmethod]
    fn redlich_kwong1972() -> Self {
        Self(RedlichKwong1972.into())
    }

    /// Soave alpha function parameterized in 1976 for Peng Robinson EoS.
    ///
    /// $\alpha(T) = \left[1 + m_\text{PR} (1 - \sqrt{T_r})\right]^2$
    /// using
    /// $m_\text{PR} = 0.37464 + 1.54226 \omega - 0.26992 \omega^2$
    #[staticmethod]
    fn peng_robinson1976() -> Self {
        Self(PengRobinson1976.into())
    }

    /// Soave alpha function parameterized in 1978 for Peng Robinson EoS.
    ///
    /// Uses different polynomials depending on acentric factor.
    #[staticmethod]
    fn peng_robinson1978() -> Self {
        Self(PengRobinson1978.into())
    }

    /// Soave alpha function parameterized in 2019 for Redlich Kwong EoS.
    ///
    /// $\alpha(T) = \left[1 + m_\text{RK} (1 - \sqrt{T_r})\right]^2$
    /// using
    /// $m_\text{RK} = x + x \omega - x \omega^2$
    #[staticmethod]
    fn redlich_kwong2019() -> Self {
        Self(RedlichKwong2019.into())
    }

    /// Soave alpha function parameterized in 2019 for Peng Robinson EoS.
    ///
    /// $\alpha(T) = \left[1 + m_\text{PR} (1 - \sqrt{T_r})\right]^2$
    /// using
    /// $m_\text{PR} = x + x \omega - x \omega^2$
    #[staticmethod]
    fn peng_robinson2019() -> Self {
        Self(PengRobinson2019.into())
    }

    /// Mathias and Compeman alpha function (1983).
    ///
    /// $\alpha(T) = \left[1 + \sum_{i=1}^3 c_i (1 - \sqrt{T_r})^i\right]^2$
    /// with $c_i$ as substance specific adjustable parameters.
    ///
    /// Parameters
    /// ----------
    /// c : List[[float; 3]]
    ///     The three `c` parameters for each substance.
    ///
    /// Returns
    /// -------
    /// Alpha
    #[staticmethod]
    fn mathias_copeman(c: Vec<[f64; 3]>) -> Self {
        Self(MathiasCopeman(c).into())
    }

    /// Generalized Twu alpha function (1995) parameterized for PR.
    ///
    /// Returns
    /// -------
    /// Alpha
    #[staticmethod]
    fn peng_robinson_generalized_twu() -> Self {
        Self(GeneralizedTwu::peng_robinson().into())
    }

    /// Generalized Twu alpha function (1995) parameterized for RK.
    ///
    /// Returns
    /// -------
    /// Alpha
    #[staticmethod]
    fn redlich_kwong_generalized_twu() -> Self {
        Self(GeneralizedTwu::redlich_kwong().into())
    }

    /// Generalized Twu alpha function (1991) parameterized for RK.
    ///
    /// If no `n` parameters are provided, the model of Twu (1988) is constructed.
    ///
    /// Parameters
    /// ----------
    /// m : List[float]
    /// l : List[float]
    /// n : List[float], optional
    ///     Defaults to 2.0 for each component (Twu 1988).
    ///
    /// Returns
    /// -------
    /// Alpha
    #[staticmethod]
    fn twu(l: Vec<f64>, m: Vec<f64>, n: Option<Vec<f64>>) -> Self {
        Self(Twu::new(l, m, n).into())
    }
}

/// Create a mixing rule
#[pyclass(name = "MixingRule")]
#[derive(Clone)]
pub struct PyMixingRule(pub MixingRule);

#[pymethods]
impl PyMixingRule {
    #[staticmethod]
    fn quadratic() -> Self {
        Self(Quadratic.into())
    }
}

/// Create a set of cubic parameters from records.
#[pyclass(name = "CubicRecord")]
#[derive(Clone)]
pub struct PyCubicRecord(CubicRecord);

#[pymethods]
impl PyCubicRecord {
    #[new]
    #[pyo3(text_signature = "(tc, pc, acentric_factor)")]
    fn new(tc: f64, pc: f64, acentric_factor: f64) -> Self {
        Self(CubicRecord::new(tc, pc, acentric_factor))
    }

    #[getter]
    fn get_tc(&self) -> f64 {
        self.0.tc
    }

    #[getter]
    fn get_pc(&self) -> f64 {
        self.0.pc
    }

    #[getter]
    fn get_acentric_factor(&self) -> f64 {
        self.0.acentric_factor
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PyCubicRecord);
impl_pure_record!(CubicRecord, PyCubicRecord);

#[pyclass(name = "CubicBinaryRecord")]
#[derive(Clone)]
pub struct PyCubicBinaryRecord(CubicBinaryRecord);
impl_binary_record!(CubicBinaryRecord, PyCubicBinaryRecord);

#[pyclass(name = "CubicParameters")]
#[derive(Clone)]
pub struct PyCubicParameters(pub Arc<CubicParameters>);

#[pymethods]
impl PyCubicParameters {
    // Create a set of cubic parameters from values.
    ///
    /// Parameters
    /// ----------
    /// tc : float
    ///     critical temperature in units of Kelvin.
    /// pc : float
    ///     critical pressure in units of Pascal.
    /// acentric_factor: float
    ///     acentric factor
    /// molarweight: float, optional
    ///     molar weight in units of Gram per Mol.
    /// Returns
    /// -------
    /// CubicParameters
    #[pyo3(text_signature = "(tc, pc, acentric_factor, molarweight=None)")]
    #[staticmethod]
    fn from_values(
        tc: f64,
        pc: f64,
        acentric_factor: f64,
        molarweight: Option<f64>,
    ) -> PyResult<Self> {
        let pure_record = PureRecord::new(
            Identifier::new(
                Some(format!("{}", 1).as_str()),
                None,
                None,
                None,
                None,
                None,
            ),
            molarweight.map_or(1.0, |v| v),
            CubicRecord::new(tc, pc, acentric_factor),
        );
        Ok(Self(Arc::new(CubicParameters::new_pure(pure_record)?)))
    }

    #[getter]
    fn get_k_ij<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.0
            .binary_records
            .as_ref()
            .map(|br| br.map(|br| br.k_ij).view().to_pyarray_bound(py))
    }

    #[getter]
    fn get_l_ij<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.0
            .binary_records
            .as_ref()
            .map(|br| br.map(|br| br.l_ij).view().to_pyarray_bound(py))
    }

    fn _repr_markdown_(&self) -> String {
        // self.0.to_markdown()
        todo!()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_parameter!(
    CubicParameters,
    PyCubicParameters,
    PyCubicRecord,
    PyCubicBinaryRecord
);

#[pymodule]
pub fn cubic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<IdentifierOption>()?;
    m.add_class::<PyChemicalRecord>()?;

    m.add_class::<PyCubicRecord>()?;
    m.add_class::<PyPureRecord>()?;
    m.add_class::<PyBinaryRecord>()?;
    m.add_class::<PyCubicParameters>()?;
    m.add_class::<PyAlpha>()?;
    m.add_class::<PyMixingRule>()?;
    Ok(())
}
