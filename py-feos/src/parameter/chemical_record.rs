use super::{
    fragmentation::{fragment_molecule, PySmartsRecord},
    identifier::PyIdentifier,
};
use feos_core::parameter::{ChemicalRecord, Identifier};
use pyo3::{exceptions::PyValueError, prelude::*};
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
        Self(ChemicalRecord::new(identifier.0, segments, bonds))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
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
            identifier.0
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

    // /// Creates record from json string.
    // #[staticmethod]
    // fn from_json_str(json: &str) -> FeosResult<Self> {
    //     Ok(serde_json::from_str(json)?)
    // }

    // /// Creates a json string from record.
    // fn to_json_str(&self) -> Result<String, ParameterError> {
    //     Ok(serde_json::to_string(&self)?)
    // }
}
