use feos_core::parameter::{Identifier, IdentifierOption};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass(name = "IdentifierOption", eq, eq_int)]
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum PyIdentifierOption {
    Cas,
    Name,
    IupacName,
    Smiles,
    Inchi,
    Formula,
}

impl From<IdentifierOption> for PyIdentifierOption {
    fn from(value: IdentifierOption) -> Self {
        use IdentifierOption::*;
        match value {
            Cas => Self::Cas,
            Name => Self::Name,
            IupacName => Self::IupacName,
            Smiles => Self::Smiles,
            Inchi => Self::Inchi,
            Formula => Self::Formula,
        }
    }
}

impl From<PyIdentifierOption> for IdentifierOption {
    fn from(value: PyIdentifierOption) -> Self {
        use PyIdentifierOption::*;
        match value {
            Cas => Self::Cas,
            Name => Self::Name,
            IupacName => Self::IupacName,
            Smiles => Self::Smiles,
            Inchi => Self::Inchi,
            Formula => Self::Formula,
        }
    }
}

#[pyclass(name = "Identifier")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyIdentifier(pub Identifier);

#[pymethods]
impl PyIdentifier {
    #[new]
    #[pyo3(
        text_signature = "(cas=None, name=None, iupac_name=None, smiles=None, inchi=None, formula=None)",
        signature = (cas=None, name=None, iupac_name=None, smiles=None, inchi=None, formula=None)
    )]
    fn py_new(
        cas: Option<&str>,
        name: Option<&str>,
        iupac_name: Option<&str>,
        smiles: Option<&str>,
        inchi: Option<&str>,
        formula: Option<&str>,
    ) -> Self {
        Self(Identifier::new(
            cas, name, iupac_name, smiles, inchi, formula,
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.0.to_string())
    }
}
