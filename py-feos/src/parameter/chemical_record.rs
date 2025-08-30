use super::fragmentation::{fragment_molecule, PySmartsRecord};
use super::identifier::{PyIdentifier, PyIdentifierOption};
use crate::error::PyFeosError;
use feos_core::parameter::{ChemicalRecord, Identifier};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass(name = "ChemicalRecord")]
#[derive(Deserialize, Serialize, Debug, Clone)]
pub(crate) struct PyChemicalRecord(ChemicalRecord);

impl From<ChemicalRecord> for PyChemicalRecord {
    fn from(value: ChemicalRecord) -> Self {
        Self(value)
    }
}

impl From<PyChemicalRecord> for ChemicalRecord {
    fn from(value: PyChemicalRecord) -> Self {
        value.0
    }
}

#[pymethods]
impl PyChemicalRecord {
    #[new]
    #[pyo3(text_signature = "(identifier, segments, bonds=None)", signature = (identifier, segments, bonds=None))]
    fn py_new(
        identifier: PyIdentifier,
        segments: Vec<String>,
        bonds: Option<Vec<[usize; 2]>>,
    ) -> Self {
        Self(ChemicalRecord::new(identifier.into(), segments, bonds))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }

    /// Read a list of `ChemicalRecord`s from a JSON file.
    ///
    /// Parameters
    /// ----------
    /// substances : list[str]
    ///     List of component identifiers.
    /// path : str
    ///     Path to file containing the segment records.
    /// identifier_option : IdentifierOption
    ///     The type of identifier used in the substance list.
    ///
    /// Returns
    /// -------
    /// [SegmentRecord]
    #[staticmethod]
    pub fn from_json(
        substances: Vec<String>,
        file: &str,
        identifier_option: PyIdentifierOption,
    ) -> PyResult<Vec<Self>> {
        Ok(
            ChemicalRecord::from_json(&substances, file, identifier_option.into())
                .map_err(PyFeosError::from)?
                .into_iter()
                .map(|r| r.into())
                .collect(),
        )
    }

    #[staticmethod]
    pub fn from_smiles(
        identifier: &Bound<'_, PyAny>,
        smarts: Vec<PySmartsRecord>,
    ) -> PyResult<Self> {
        let py = identifier.py();
        let identifier = if let Ok(smiles) = identifier.extract::<String>() {
            Identifier::new(None, None, None, Some(&smiles), None, None)
        } else if let Ok(identifier) = identifier.extract::<PyIdentifier>() {
            identifier.into()
        } else {
            return Err(PyErr::new::<PyValueError, _>(
                "`identifier` must be a SMILES code or `Identifier` object.".to_string(),
            ));
        };
        let smiles = identifier
            .smiles
            .as_ref()
            .expect("Missing SMILES in `Identifier`");
        let (segments, bonds) = fragment_molecule(py, smiles, smarts)?;
        let segments = segments.into_iter().map(|s| s.to_owned()).collect();
        Ok(Self(ChemicalRecord::new(identifier, segments, Some(bonds))))
    }

    /// Creates record from json string.
    #[staticmethod]
    fn from_json_str(json: &str) -> PyResult<Self> {
        Ok(serde_json::from_str(json).map_err(PyFeosError::from)?)
    }

    /// Creates a json string from record.
    fn to_json_str(&self) -> PyResult<String> {
        Ok(serde_json::to_string(&self).map_err(PyFeosError::from)?)
    }
}
