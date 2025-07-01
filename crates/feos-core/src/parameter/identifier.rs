use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};

/// Possible variants to identify a substance.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum IdentifierOption {
    Cas,
    Name,
    IupacName,
    Smiles,
    Inchi,
    Formula,
}

impl fmt::Display for IdentifierOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let str = match self {
            IdentifierOption::Cas => "CAS",
            IdentifierOption::Name => "name",
            IdentifierOption::IupacName => "IUPAC name",
            IdentifierOption::Smiles => "SMILES",
            IdentifierOption::Inchi => "InChI",
            IdentifierOption::Formula => "formula",
        };
        write!(f, "{str}")
    }
}

/// A collection of identifiers for a chemical structure or substance.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Identifier {
    /// CAS number
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cas: Option<String>,
    /// Commonly used english name
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// IUPAC name
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iupac_name: Option<String>,
    /// SMILES key
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub smiles: Option<String>,
    /// InchI key
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inchi: Option<String>,
    /// Chemical formula
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formula: Option<String>,
}

impl Identifier {
    /// Create a new identifier.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use feos_core::parameter::Identifier;
    /// let methanol = Identifier::new(
    ///     Some("67-56-1"),
    ///     Some("methanol"),
    ///     Some("methanol"),
    ///     Some("CO"),
    ///     Some("InChI=1S/CH4O/c1-2/h2H,1H3"),
    ///     Some("CH4O")
    /// );
    pub fn new(
        cas: Option<&str>,
        name: Option<&str>,
        iupac_name: Option<&str>,
        smiles: Option<&str>,
        inchi: Option<&str>,
        formula: Option<&str>,
    ) -> Identifier {
        Identifier {
            cas: cas.map(Into::into),
            name: name.map(Into::into),
            iupac_name: iupac_name.map(Into::into),
            smiles: smiles.map(Into::into),
            inchi: inchi.map(Into::into),
            formula: formula.map(Into::into),
        }
    }

    pub fn as_str(&self, option: IdentifierOption) -> Option<&str> {
        match option {
            IdentifierOption::Cas => self.cas.as_deref(),
            IdentifierOption::Name => self.name.as_deref(),
            IdentifierOption::IupacName => self.iupac_name.as_deref(),
            IdentifierOption::Smiles => self.smiles.as_deref(),
            IdentifierOption::Inchi => self.inchi.as_deref(),
            IdentifierOption::Formula => self.formula.as_deref(),
        }
    }

    // returns the first available identifier in a somewhat arbitrary
    // prioritization. Used for readable outputs.
    pub fn as_readable_str(&self) -> Option<&str> {
        self.name
            .as_deref()
            .or(self.iupac_name.as_deref())
            .or(self.smiles.as_deref())
            .or(self.cas.as_deref())
            .or(self.inchi.as_deref())
            .or(self.formula.as_deref())
    }
}

impl std::fmt::Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ids = Vec::new();
        if let Some(n) = &self.cas {
            ids.push(format!("cas={n}"));
        }
        if let Some(n) = &self.name {
            ids.push(format!("name={n}"));
        }
        if let Some(n) = &self.iupac_name {
            ids.push(format!("iupac_name={n}"));
        }
        if let Some(n) = &self.smiles {
            ids.push(format!("smiles={n}"));
        }
        if let Some(n) = &self.inchi {
            ids.push(format!("inchi={n}"));
        }
        if let Some(n) = &self.formula {
            ids.push(format!("formula={n}"));
        }
        write!(f, "Identifier({})", ids.join(", "))
    }
}

impl PartialEq for Identifier {
    fn eq(&self, other: &Self) -> bool {
        self.cas == other.cas
    }
}
impl Eq for Identifier {}

impl Hash for Identifier {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.cas.hash(state);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_fmt() {
        let id = Identifier::new(None, Some("acetone"), None, Some("CC(=O)C"), None, None);
        assert_eq!(id.to_string(), "Identifier(name=acetone, smiles=CC(=O)C)");
    }
}
