use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;

use crate::error::PyFeosError;

/// SMARTS code, required to fragmentize molecules into segments.
#[derive(Clone, Serialize, Deserialize)]
#[pyclass(name = "SmartsRecord")]
pub(crate) struct PySmartsRecord {
    group: String,
    smarts: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    max: Option<usize>,
}

#[pymethods]
impl PySmartsRecord {
    #[new]
    #[pyo3(text_signature = "(group, smarts, max=None)", signature = (group, smarts, max=None))]
    fn new(group: String, smarts: String, max: Option<usize>) -> Self {
        Self { group, smarts, max }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.to_string())
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

    /// Read a list of [SmartsRecord]s from a JSON file.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to file containing the SMARTS records.
    ///
    /// Returns
    /// -------
    /// [SmartsRecord]
    #[staticmethod]
    #[pyo3(text_signature = "(path)")]
    pub fn from_json(path: &str) -> PyResult<Vec<Self>> {
        Ok(
            serde_json::from_reader(BufReader::new(File::open(path)?))
                .map_err(PyFeosError::from)?,
        )
    }
}

impl std::fmt::Display for PySmartsRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SmartsRecord(group={}, smarts={}",
            self.group, self.smarts
        )?;
        if let Some(max) = self.max {
            write!(f, ", max={max}")?;
        }
        write!(f, ")")
    }
}

pub(crate) fn fragment_molecule(
    py: Python<'_>,
    smiles: &str,
    smarts: Vec<PySmartsRecord>,
) -> PyResult<(Vec<String>, Vec<[usize; 2]>)> {
    let chem = py.import("rdkit.Chem")?;
    let mol = chem.call_method1("MolFromSmiles", (smiles,))?;
    let atoms = mol.call_method0("GetNumHeavyAtoms")?.extract::<usize>()?;

    // find the location of all fragment using the given smarts
    let mut matches: HashMap<_, _> = smarts
        .into_iter()
        .map(|s| {
            let m = chem.call_method1("MolFromSmarts", (s.smarts,))?;
            let matches = mol
                .call_method1("GetSubstructMatches", (m,))?
                .extract::<Vec<Bound<'_, PyAny>>>()?;
            let mut matches: Vec<_> = matches
                .into_iter()
                .map(|m| m.extract::<Vec<usize>>())
                .collect::<PyResult<_>>()?;
            // Instead of just throwing an error at this point, just try to continue with the first max
            // occurrences. For some cases (the ethers) this just means that the symetry of C-O-C is broken.
            // If a necessary segment is eliminated the error will be thrown later.
            if let Some(max) = s.max {
                if matches.len() > max {
                    matches = matches[..max].to_vec();
                }
            }
            Ok((s.group, matches))
        })
        .collect::<PyResult<_>>()?;

    // Filter small segments that are covered by larger segments (also only required by the weird
    // ether groups of Sauer et al.)
    let large_segments: HashSet<_> = matches
        .values()
        .flatten()
        .filter(|m| m.len() > 1)
        .flatten()
        .copied()
        .collect();
    matches
        .iter_mut()
        .for_each(|(_, m)| m.retain(|m| !(m.len() == 1 && large_segments.contains(&m[0]))));

    let bonds = mol.call_method0("GetBonds")?;
    let builtins = py.import("builtins")?;
    let bonds = builtins
        .call_method1("list", (bonds,))?
        .extract::<Vec<Bound<'_, PyAny>>>()?;
    let bonds: Vec<_> = bonds
        .into_iter()
        .map(|b| {
            Ok([
                b.call_method0("GetBeginAtomIdx")?.extract::<usize>()?,
                b.call_method0("GetEndAtomIdx")?.extract::<usize>()?,
            ])
        })
        .collect::<PyResult<_>>()?;

    convert_matches(atoms, matches, bonds)
}

pub(crate) fn convert_matches(
    atoms: usize,
    matches: HashMap<String, Vec<Vec<usize>>>,
    bonds: Vec<[usize; 2]>,
) -> PyResult<(Vec<String>, Vec<[usize; 2]>)> {
    // check if every atom is captured by exactly one fragment
    let identified_atoms: Vec<_> = matches
        .values()
        .flat_map(|v| v.iter().flat_map(|l| l.iter()))
        .collect();
    let unique_atoms: HashSet<_> = identified_atoms.iter().collect();
    if unique_atoms.len() == identified_atoms.len() && unique_atoms.len() == atoms {
        // Translate the atom indices to segment indices (some segments contain more than one atom)
        let mut segment_indices: Vec<_> = matches
            .into_iter()
            .flat_map(|(group, l)| {
                l.into_iter().map(move |mut k| {
                    k.sort();
                    (k, group.clone())
                })
            })
            .collect();
        segment_indices.sort();

        let segment_map: Vec<_> = segment_indices
            .iter()
            .enumerate()
            .flat_map(|(i, (k, _))| k.iter().map(move |_| i))
            .collect();
        let segments: Vec<_> = segment_indices.into_iter().map(|(_, g)| g).collect();

        let bonds: Vec<_> = bonds
            .into_iter()
            .map(|[a, b]| [segment_map[a], segment_map[b]])
            .filter(|[a, b]| a != b)
            .collect();
        return Ok((segments, bonds));
    }

    Err(PyErr::new::<PyValueError, _>(
        "Molecule cannot be built from groups!",
    ))
}
