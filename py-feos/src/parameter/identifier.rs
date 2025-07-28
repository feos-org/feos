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

#[pyclass(name = "Identifier", get_all, set_all)]
#[derive(Debug, Clone)]
pub struct PyIdentifier {
    cas: Option<String>,
    name: Option<String>,
    iupac_name: Option<String>,
    smiles: Option<String>,
    inchi: Option<String>,
    formula: Option<String>,
}

impl From<PyIdentifier> for Identifier {
    fn from(value: PyIdentifier) -> Self {
        Self {
            cas: value.cas,
            name: value.name,
            iupac_name: value.iupac_name,
            smiles: value.smiles,
            inchi: value.inchi,
            formula: value.formula,
        }
    }
}

impl From<Identifier> for PyIdentifier {
    fn from(value: Identifier) -> Self {
        Self {
            cas: value.cas,
            name: value.name,
            iupac_name: value.iupac_name,
            smiles: value.smiles,
            inchi: value.inchi,
            formula: value.formula,
        }
    }
}

#[pymethods]
impl PyIdentifier {
    #[new]
    #[pyo3(
        text_signature = "(cas=None, name=None, iupac_name=None, smiles=None, inchi=None, formula=None)",
        signature = (cas=None, name=None, iupac_name=None, smiles=None, inchi=None, formula=None)
    )]
    fn py_new(
        cas: Option<String>,
        name: Option<String>,
        iupac_name: Option<String>,
        smiles: Option<String>,
        inchi: Option<String>,
        formula: Option<String>,
    ) -> Self {
        Self {
            cas,
            name,
            iupac_name,
            smiles,
            inchi,
            formula,
        }
    }

    fn __repr__(&self) -> String {
        Identifier::from(self.clone()).to_string()
    }
}
