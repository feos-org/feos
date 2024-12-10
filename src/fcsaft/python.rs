use crate::fcsaft::heterosegmented::{FcSaftParameters, FcSaftRecord};
use feos_core::parameter::{BinaryRecord, IdentifierOption, ParameterError, SegmentRecord};
use feos_core::python::parameter::*;
use feos_core::{impl_json_handling, impl_segment_record};
use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use std::sync::Arc;

#[pyclass(name = "FcSaftRecord")]
#[derive(Clone)]
pub struct PyFcSaftRecord(FcSaftRecord);

#[pymethods]
impl PyFcSaftRecord {
    #[new]
    #[pyo3(
        text_signature = "(sigma, epsilon_k, kappa_ab=None, epsilon_k_ab=None, na=None, nb=None, nc=None, psi_dft=None)",
        signature = (sigma, epsilon_k, kappa_ab=None, epsilon_k_ab=None, na=None, nb=None, nc=None, psi_dft=None)
    )]
    #[expect(clippy::too_many_arguments)]
    fn new(
        sigma: f64,
        epsilon_k: f64,
        kappa_ab: Option<f64>,
        epsilon_k_ab: Option<f64>,
        na: Option<f64>,
        nb: Option<f64>,
        nc: Option<f64>,
        psi_dft: Option<f64>,
    ) -> Self {
        Self(FcSaftRecord::new(
            sigma,
            epsilon_k,
            kappa_ab,
            epsilon_k_ab,
            na,
            nb,
            nc,
            psi_dft,
        ))
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
    fn get_kappa_ab(&self) -> Option<f64> {
        self.0.association_record.map(|a| a.parameters.kappa_ab)
    }

    #[getter]
    fn get_epsilon_k_ab(&self) -> Option<f64> {
        self.0.association_record.map(|a| a.parameters.epsilon_k_ab)
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
    fn get_psi_dft(&self) -> Option<f64> {
        self.0.psi_dft
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

impl_json_handling!(PyFcSaftRecord);

impl_segment_record!(FcSaftRecord, PyFcSaftRecord);

#[pyclass(name = "FcSaftParameters")]
#[derive(Clone)]
pub struct PyFcSaftParameters(pub Arc<FcSaftParameters>);

#[pymethods]
impl PyFcSaftParameters {
    #[staticmethod]
    #[pyo3(
        text_signature = "(identifier, segments, sigma, epsilon_k, bond_length, molarweight, psi_dft=None)",
        signature = (identifier, segments, sigma, epsilon_k, bond_length, molarweight, psi_dft=None)
    )]
    fn new_pure_homosegmented(
        identifier: PyIdentifier,
        segments: usize,
        sigma: f64,
        epsilon_k: f64,
        bond_length: f64,
        molarweight: f64,
        psi_dft: Option<f64>,
    ) -> Result<Self, ParameterError> {
        Ok(Self(Arc::new(FcSaftParameters::new_pure_homosegmented(
            identifier.0,
            segments,
            sigma,
            epsilon_k,
            bond_length,
            molarweight,
            psi_dft,
        )?)))
    }

    /// Creates parameters from segment records.
    ///
    /// Parameters
    /// ----------
    /// chemical_records : [ChemicalRecord]
    ///     A list of pure component parameters.
    /// segment_records : [SegmentRecord]
    ///     A list of records containing the parameters of
    ///     all individual segments.
    /// bond_records : [BinarySegmentRecord]
    ///     A list of bond lengths.
    /// binary_segment_records : [BinarySegmentRecord], optional
    ///     A list of binary segment-segment parameters.
    #[staticmethod]
    #[pyo3(
        text_signature = "(chemical_records, segment_records, bond_records, binary_segment_records=None)",
        signature = (chemical_records, segment_records, bond_records, binary_segment_records=None)
    )]
    fn from_segments(
        chemical_records: Vec<PyChemicalRecord>,
        segment_records: Vec<PySegmentRecord>,
        bond_records: Vec<PyBinarySegmentRecord>,
        binary_segment_records: Option<Vec<PyBinarySegmentRecord>>,
    ) -> Result<Self, ParameterError> {
        Ok(Self(Arc::new(<FcSaftParameters>::from_segments(
            chemical_records.into_iter().map(|cr| cr.0).collect(),
            segment_records.into_iter().map(|sr| sr.0).collect(),
            bond_records.into_iter().map(|br| br.0).collect(),
            binary_segment_records.map(|r| {
                r.into_iter()
                    .map(|r| BinaryRecord {
                        id1: r.0.id1,
                        id2: r.0.id2,
                        model_record: r.0.model_record,
                    })
                    .collect()
            }),
        )?)))
    }

    /// Creates parameters using segments from json file.
    ///
    /// Parameters
    /// ----------
    /// substances : List[str]
    ///     The substances to search.
    /// pure_path : str
    ///     Path to file containing pure substance parameters.
    /// segments_path : str
    ///     Path to file containing segment parameters.
    /// bonds_path : str
    ///     Path to file containing bond parameters.
    /// binary_path : str, optional
    ///     Path to file containing binary segment-segment parameters.
    /// identifier_option : str, optional, defaults to "Name"
    ///     Identifier that is used to search substance.
    ///     One of 'Name', 'Cas', 'Inchi', 'IupacName', 'Formula', 'Smiles'
    #[staticmethod]
    #[pyo3(
        signature = (substances, pure_path, segments_path, bonds_path, binary_path=None, identifier_option=IdentifierOption::Name),
        text_signature = "(substances, pure_path, segments_path, bonds_path, binary_path=None, identifier_option='Name')"
    )]
    fn from_json_segments(
        substances: Vec<PyBackedStr>,
        pure_path: String,
        segments_path: String,
        bonds_path: String,
        binary_path: Option<String>,
        identifier_option: IdentifierOption,
    ) -> Result<Self, ParameterError> {
        let substances: Vec<_> = substances.iter().map(|s| &**s).collect();
        Ok(Self(Arc::new(<FcSaftParameters>::from_json_segments(
            &substances,
            pure_path,
            segments_path,
            bonds_path,
            binary_path,
            identifier_option,
        )?)))
    }
}

#[pymethods]
impl PyFcSaftParameters {
    // fn _repr_markdown_(&self) -> String {
    //     self.0.to_markdown()
    // }

    #[getter]
    fn get_graph(&self, py: Python) -> PyResult<PyObject> {
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            c"def f(s): 
                import graphviz
                return graphviz.Source(s.replace('\\\\\"', ''))",
            c"",
            c"",
        )?
        .getattr("f")?
        .into();
        fun.call1(py, (self.0.graph(),))
    }

    #[getter]
    fn get_k_ij<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.0.k_ij.view().to_pyarray(py)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}

#[pymodule]
pub fn fcsaft(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIdentifier>()?;
    m.add_class::<IdentifierOption>()?;
    m.add_class::<PyChemicalRecord>()?;
    m.add_class::<PySmartsRecord>()?;

    m.add_class::<PyFcSaftRecord>()?;
    m.add_class::<PySegmentRecord>()?;
    m.add_class::<PyBinaryRecord>()?;
    m.add_class::<PyBinarySegmentRecord>()?;
    m.add_class::<PyFcSaftParameters>()?;
    Ok(())
}
