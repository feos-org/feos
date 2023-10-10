use super::parameter::{PyChemicalRecord, PyIdentifier};
use crate::parameter::{ChemicalRecord, Identifier};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

#[pymethods]
impl PyChemicalRecord {
    #[staticmethod]
    fn from_smiles<'py>(
        py: Python<'py>,
        identifier: &PyAny,
        smarts: HashMap<&'py str, (&str, Option<usize>)>,
    ) -> PyResult<Self> {
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
}

fn fragment_molecule<'py>(
    py: Python<'py>,
    smiles: &str,
    smarts: HashMap<&'py str, (&str, Option<usize>)>,
) -> PyResult<(Vec<&'py str>, Vec<[usize; 2]>)> {
    let chem = py.import("rdkit.Chem")?;
    let mol = chem.call_method1("MolFromSmiles", (smiles,))?;
    let atoms = mol.call_method0("GetNumHeavyAtoms")?.extract::<usize>()?;

    // find the location of all fragment using the given smarts
    let mut matches: HashMap<_, _> = smarts
        .iter()
        .map(|(&g, &(s, max))| {
            let m = chem.call_method1("MolFromSmarts", (s,))?;
            let matches = mol
                .call_method1("GetSubstructMatches", (m,))?
                .extract::<Vec<&PyAny>>()?;
            let mut matches: Vec<_> = matches
                .into_iter()
                .map(|m| m.extract::<Vec<usize>>())
                .collect::<PyResult<_>>()?;
            // Instead of just throwing an error at this point, just try to continue with the first max
            // occurrences. For some cases (the ethers) this just means that the symetry of C-O-C is broken.
            // If a necessary segment is eliminated the error will be thrown later.
            if let Some(max) = max {
                if matches.len() > max {
                    matches = matches[..max].to_vec();
                }
            }
            Ok((g, matches))
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
        .extract::<Vec<&PyAny>>()?;
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

fn convert_matches(
    atoms: usize,
    matches: HashMap<&str, Vec<Vec<usize>>>,
    bonds: Vec<[usize; 2]>,
) -> PyResult<(Vec<&str>, Vec<[usize; 2]>)> {
    // check if every atom is captured by exatly one fragment
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
                    (k, group)
                })
            })
            .collect();
        segment_indices.sort();

        let segment_map: Vec<_> = segment_indices
            .iter()
            .enumerate()
            .flat_map(|(i, (k, _))| k.iter().map(move |_| i))
            .collect();
        // bonds = [(segment_map[a], segment_map[b]) for a, b in bonds]
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
