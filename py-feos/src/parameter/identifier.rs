use feos_core::parameter::{Identifier, IdentifierOption};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Identifier to match on while reading parameters from files.
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

/// Different common identifiers for chemicals.
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

    #[getter]
    fn get_cas(&self) -> Option<String> {
        self.0.cas.clone()
    }

    #[getter]
    fn get_name(&self) -> Option<String> {
        self.0.name.clone()
    }

    #[getter]
    fn get_iupac_name(&self) -> Option<String> {
        self.0.iupac_name.clone()
    }

    #[getter]
    fn get_smiles(&self) -> Option<String> {
        self.0.smiles.clone()
    }

    #[getter]
    fn get_inchi(&self) -> Option<String> {
        self.0.inchi.clone()
    }

    #[getter]
    fn get_formula(&self) -> Option<String> {
        self.0.formula.clone()
    }

    fn __repr__(&self) -> String {
        self.0.to_string()
    }
}
