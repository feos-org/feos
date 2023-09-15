use super::PcSaftRecord;
use super::{python::PyPcSaftParameters, PcSaftParameters};
use feos_core::parameter::{ChemicalRecord, Identifier, Parameter, SegmentRecord};
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[pymethods]
impl PyPcSaftParameters {
    #[staticmethod]
    fn from_smiles(py: Python<'_>, smiles: Vec<&str>) -> PyResult<Self> {
        let chemical_records = smiles
            .into_iter()
            .map(|s| {
                let (segments, bonds) = fragment_to_sauer_groups(py, s)?;
                let segments = segments.into_iter().map(|s| s.to_owned()).collect();
                Ok(ChemicalRecord::new(
                    Identifier::new(None, None, None, Some(s), None, None),
                    segments,
                    Some(bonds),
                ))
            })
            .collect::<PyResult<_>>()?;
        let params = PcSaftParameters::from_segments(chemical_records, SEGMENTS.clone(), None)?;
        Ok(Self(Arc::new(params)))
    }
}

fn fragment_molecule(
    py: Python<'_>,
    smiles: &str,
) -> PyResult<(Vec<&'static str>, Vec<[usize; 2]>)> {
    let chem = py.import("rdkit.Chem")?;
    let mol = chem.call_method1("MolFromSmiles", (smiles,))?;
    let atoms = mol.call_method0("GetNumHeavyAtoms")?.extract::<usize>()?;

    // find the location of all fragment using the given smarts
    let matches: HashMap<_, _> = SMARTS
        .iter()
        .map(|(&g, &s)| {
            let m = chem.call_method1("MolFromSmarts", (s,))?;
            let matches = mol
                .call_method1("GetSubstructMatches", (m,))?
                .extract::<Vec<&PyAny>>()?;
            let matches: Vec<_> = matches
                .into_iter()
                .map(|m| m.extract::<Vec<usize>>())
                .collect::<PyResult<_>>()?;
            Ok((g, matches))
        })
        .collect::<PyResult<_>>()?;

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

fn fragment_to_sauer_groups(
    py: Python<'_>,
    smiles: &str,
) -> PyResult<(Vec<&'static str>, Vec<[usize; 2]>)> {
    let (segments, bonds) = fragment_molecule(py, smiles)?;
    let chem = py.import("rdkit.Chem")?;
    let mol = chem.call_method1("MolFromSmiles", (smiles,))?;
    chem.call_method1("GetSymmSSSR", (mol,))?;
    let rings = mol
        .call_method0("GetRingInfo")?
        .call_method0("NumRings")?
        .extract::<usize>()?;

    if rings > 1 {
        panic!("Invalid molecule: Only molecules with up to 1 ring are allowed!")
    }

    let ethers = segments.iter().filter(|&&g| g == "O").count();
    match ethers {
        0 => Ok((segments, bonds)),
        1 => {
            let index = segments.iter().position(|&g| g == "O").unwrap();
            let [_, right] = bonds.iter().copied().find(|[a, _]| *a == index).unwrap();
            let [left, _] = bonds.iter().copied().find(|[_, b]| *b == index).unwrap();
            let right = (right, segments[right]);
            let left = (left, segments[left]);
            let (index2, new_group) = if left.1 == "CH3" {
                (left.0, "OCH3")
            } else if right.1 == "CH3" {
                (right.0, "OCH3")
            } else if left.1 == "CH2" {
                (left.0, "OCH2")
            } else if right.1 == "CH2" {
                (right.0, "OCH2")
            } else {
                panic!("Invalid molecule: The ether is not compatible with the groups by Sauer et al.!")
            };
            let mut matches = HashMap::new();
            matches.insert(new_group, vec![vec![index, index2]]);
            let atoms = segments.len();
            for (i, group) in segments.into_iter().enumerate() {
                if i == index || i == index2 {
                    continue;
                }
                matches.entry(group).or_insert(vec![]).push(vec![i]);
            }
            convert_matches(atoms, matches, bonds)
        }
        _ => panic!("Invalid molecule: More than one ether group!"),
    }
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

    panic!("Invalid molecule: Molecule cannot be built from groups!")
}

static SMARTS: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();
    // NON-POLAR, NON-ASSOCIATING ALIPHATIC FUNCTIONAL GROUPS
    // alkanes -CH3 : no ethane, no methanol, no methylamine
    m.insert("CH3", "[CH3;!$([CH3][CH3]);!$([CH3][OH]);!$([CH3][NH2])]");
    // alkanes -CH2- : no ring
    m.insert("CH2", "[CX4H2;!R]");
    // alkanes >CH- : no ring
    m.insert(">CH", "[CX4H1;!R]");
    // alkanes >C< : no ring
    m.insert(">C<", "[CX4H0;!R]");
    // alkenes =CH2 : no ring, no ethylene, no formaldehyde
    m.insert("=CH2", "[CX3H2;!R;!$([CH2]=[CH2]);!$([CH2]=O)]");
    // alkenes =CH- : no ring, no aldehyde
    m.insert("=CH", "[CX3H1;!R;!$([CX3H1]=O)]");
    // alkenes =C< : no ring, no ketone
    m.insert("=C<", "[CX3H0;!R;!$([CX3H0]=O)]");
    // terminal alkynes -C≡CH
    m.insert("C≡CH", "[CH0]#[CH1]");

    // AROMATIC AND CYCLIC GROUPS
    // aromats =CH- : ring
    m.insert("CH_arom", "[cH1;R]");
    // aromats =C< : ring
    m.insert("C_arom", "[cH0;R]");
    // cyclopentanes -CH2- : ring of size 5
    m.insert("CH2_pent", "[CX4H2;r5]");
    // cyclopentanes >CH- : ring of size 5
    m.insert("CH_pent", "[CX4H1;r5]");
    // cyclohexanes -CH2- : ring of size 6
    m.insert("CH2_hex", "[CX4H2;r6]");
    // cyclohexanes >CH- : ring of size 6
    m.insert("CH_hex", "[CX4H1;r6]");

    // POLAR GROUPS
    // aldehydes -CH=O : neighbor has to be C
    m.insert("CH=O", "[$([CH1][C,c])]=O");
    // ketones >C=O : both neighbors have to be C, no rings
    m.insert(">C=O", "[$([C]([C,c])([C,c]));!R]=O");
    // ethers -O- : both neighbors have to be C, no rings, no formates/esters
    m.insert("O", "[$([O]([C,c])[C,c]);!R;!$([O][C,c]=O)]");
    // formates -OCH=O, neighbor has to be C
    m.insert("HCOO", "[CH1](=O)[$([OH0][C,c])]");
    // esters -OC(=O)- : both neighbors have to be C, no rings
    m.insert("COO", "[$([CH0][C,c]);!R](=O)[$([OH0]([C,c])[C,c])]");

    // ASSOCIATING GROUPS
    // primary alcohols -OH : neighbor has to be CH2
    m.insert("OH", "[$([OH][CX4H2])]");
    // primary amines -NH2 : neighbor has to be C, no methylamine
    m.insert("NH2", "[$([NH2][C,c]);!$([NH2][CH3])]");
    m
});

static SEGMENTS: Lazy<Vec<SegmentRecord<PcSaftRecord>>> = Lazy::new(|| {
    vec![
        SegmentRecord::new(
            "CH3".into(),
            15.0345,
            PcSaftRecord::new(
                0.61198, 3.7202, 229.90, None, None, None, None, None, None, None, None, None, None,
            ),
        ),
        SegmentRecord::new(
            "CH2".into(),
            14.02658,
            PcSaftRecord::new(
                0.45606, 3.8900, 239.01, None, None, None, None, None, None, None, None, None, None,
            ),
        ),
        SegmentRecord::new(
            ">CH".into(),
            13.01854,
            PcSaftRecord::new(
                0.14304, 4.8597, 347.64, None, None, None, None, None, None, None, None, None, None,
            ),
        ),
        SegmentRecord::new(
            ">C<".into(),
            12.0107,
            PcSaftRecord::new(
                -0.66997, -1.7878, 107.68, None, None, None, None, None, None, None, None, None,
                None,
            ),
        ),
        SegmentRecord::new(
            "=CH2".into(),
            14.02658,
            PcSaftRecord::new(
                0.36939, 4.0264, 289.49, None, None, None, None, None, None, None, None, None, None,
            ),
        ),
        SegmentRecord::new(
            "=CH".into(),
            13.01854,
            PcSaftRecord::new(
                0.56361, 3.5519, 216.69, None, None, None, None, None, None, None, None, None, None,
            ),
        ),
        SegmentRecord::new(
            "=C<".into(),
            12.0107,
            PcSaftRecord::new(
                0.86367, 3.1815, 156.31, None, None, None, None, None, None, None, None, None, None,
            ),
        ),
        SegmentRecord::new(
            "C≡CH".into(),
            25.02924,
            PcSaftRecord::new(
                1.3279, 2.9421, 223.05, None, None, None, None, None, None, None, None, None, None,
            ),
        ),
        SegmentRecord::new(
            "CH2_hex".into(),
            14.02658,
            PcSaftRecord::new(
                0.39496, 3.9126, 289.03, None, None, None, None, None, None, None, None, None, None,
            ),
        ),
        SegmentRecord::new(
            "CH_hex".into(),
            13.01854,
            PcSaftRecord::new(
                0.02880, 8.9779, 1306.7, None, None, None, None, None, None, None, None, None, None,
            ),
        ),
        SegmentRecord::new(
            "CH2_pent".into(),
            14.02658,
            PcSaftRecord::new(
                0.46742, 3.7272, 267.16, None, None, None, None, None, None, None, None, None, None,
            ),
        ),
        SegmentRecord::new(
            "CH_pent".into(),
            13.01854,
            PcSaftRecord::new(
                0.03314, 7.7190, 1297.7, None, None, None, None, None, None, None, None, None, None,
            ),
        ),
        SegmentRecord::new(
            "CH_arom".into(),
            13.01854,
            PcSaftRecord::new(
                0.42335, 3.7270, 274.41, None, None, None, None, None, None, None, None, None, None,
            ),
        ),
        SegmentRecord::new(
            "C_arom".into(),
            12.0107,
            PcSaftRecord::new(
                0.15371, 3.9622, 527.20, None, None, None, None, None, None, None, None, None, None,
            ),
        ),
        SegmentRecord::new(
            "CH=O".into(),
            29.01754,
            PcSaftRecord::new(
                1.5774,
                2.8035,
                242.99,
                Some(2.4556),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        ),
        SegmentRecord::new(
            ">C=O".into(),
            28.0097,
            PcSaftRecord::new(
                1.2230,
                2.8124,
                249.04,
                Some(3.2432),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        ),
        SegmentRecord::new(
            "OCH3".into(),
            31.03322,
            PcSaftRecord::new(
                1.6539,
                3.0697,
                196.05,
                Some(1.3866),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        ),
        SegmentRecord::new(
            "OCH2".into(),
            30.02538,
            PcSaftRecord::new(
                1.1349,
                3.2037,
                187.13,
                Some(2.7440),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        ),
        SegmentRecord::new(
            "HCOO".into(),
            45.01654,
            PcSaftRecord::new(
                1.7525,
                2.9043,
                229.63,
                Some(2.7916),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        ),
        SegmentRecord::new(
            "COO".into(),
            44.0087,
            PcSaftRecord::new(
                1.5063,
                2.8166,
                222.52,
                Some(3.1652),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        ),
        SegmentRecord::new(
            "OH".into(),
            17.00734,
            PcSaftRecord::new(
                0.40200,
                3.2859,
                488.66,
                None,
                None,
                Some(0.006825),
                Some(2517.0),
                Some(1.0),
                Some(1.0),
                None,
                None,
                None,
                None,
            ),
        ),
        SegmentRecord::new(
            "NH2".into(),
            16.02238,
            PcSaftRecord::new(
                0.40558,
                3.6456,
                467.59,
                None,
                None,
                Some(0.026662),
                Some(1064.6),
                Some(1.0),
                Some(1.0),
                None,
                None,
                None,
                None,
            ),
        ),
    ]
});
