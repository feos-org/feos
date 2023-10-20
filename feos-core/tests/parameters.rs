use feos_core::parameter::*;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct MyPureModel {
    a: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
struct MyBinaryModel {
    b: f64,
}

impl TryFrom<f64> for MyBinaryModel {
    type Error = &'static str;
    fn try_from(f: f64) -> Result<Self, Self::Error> {
        Ok(Self { b: f })
    }
}

struct MyParameter {
    pure_records: Vec<PureRecord<MyPureModel>>,
    binary_records: Option<Array2<MyBinaryModel>>,
}

impl Parameter for MyParameter {
    type Pure = MyPureModel;
    type Binary = MyBinaryModel;
    fn from_records(
        pure_records: Vec<PureRecord<MyPureModel>>,
        binary_records: Option<Array2<MyBinaryModel>>,
    ) -> Result<Self, ParameterError> {
        Ok(Self {
            pure_records,
            binary_records,
        })
    }

    fn records(&self) -> (&[PureRecord<MyPureModel>], Option<&Array2<MyBinaryModel>>) {
        (&self.pure_records, self.binary_records.as_ref())
    }
}

#[test]
fn from_records() -> Result<(), ParameterError> {
    let pr_json = r#"
        [
            {
                "identifier": {
                    "cas": "123-4-5"
                },
                "molarweight": 16.0426,
                "model_record": {
                    "a": 0.1
                }
            },
            {
                "identifier": {
                    "cas": "678-9-1"
                },
                "molarweight": 32.08412,
                "model_record": {
                    "a": 0.2
                }
            }
        ]
        "#;
    let br_json = r#"
        [
            {
                "id1": {
                    "cas": "123-4-5"
                },
                "id2": {
                    "cas": "678-9-1"
                },
                "model_record": {
                    "b": 12.0
                }
            }
        ]
        "#;
    let pure_records = serde_json::from_str(pr_json).expect("Unable to parse json.");
    let binary_records: Vec<_> = serde_json::from_str(br_json).expect("Unable to parse json.");
    let binary_matrix = MyParameter::binary_matrix_from_records(
        &pure_records,
        &binary_records,
        IdentifierOption::Cas,
    );
    let p = MyParameter::from_records(pure_records, binary_matrix)?;

    assert_eq!(p.pure_records[0].identifier.cas, Some("123-4-5".into()));
    assert_eq!(p.pure_records[1].identifier.cas, Some("678-9-1".into()));
    assert_eq!(p.binary_records.unwrap()[[0, 1]].b, 12.0);
    Ok(())
}

#[test]
fn from_json_duplicates_input() {
    let pure_records = PureRecord::<MyPureModel>::from_json(
        &["123-4-5", "123-4-5"],
        "tests/test_parameters2.json",
        IdentifierOption::Cas,
    );
    assert!(matches!(
        pure_records,
        Err(ParameterError::IncompatibleParameters(t))
        if t == "A substance was defined more than once."
    ));
}

#[test]
fn from_multiple_json_files_duplicates() {
    let my_parameters = MyParameter::from_multiple_json(
        &[
            (vec!["123-4-5"], "tests/test_parameters1.json"),
            (vec!["678-9-1", "123-4-5"], "tests/test_parameters2.json"),
        ],
        None,
        IdentifierOption::Cas,
    );
    assert!(matches!(
        my_parameters,
        Err(ParameterError::IncompatibleParameters(t))
        if t == "A substance was defined more than once."
    ));

    let my_parameters = MyParameter::from_multiple_json(
        &[
            (vec!["123-4-5"], "tests/test_parameters1.json"),
            (vec!["678-9-1"], "tests/test_parameters2.json"),
        ],
        None,
        IdentifierOption::Cas,
    )
    .unwrap();

    // test_parameters1: a = 0.5
    // test_parameters2: a = 0.1 or 0.3
    assert_eq!(my_parameters.pure_records[0].model_record.a, 0.5);
}

#[test]
fn from_multiple_json_files() {
    let p = MyParameter::from_multiple_json(
        &[
            (vec!["678-9-1"], "tests/test_parameters2.json"),
            (vec!["123-4-5"], "tests/test_parameters1.json"),
        ],
        Some("tests/test_parameters_binary.json"),
        IdentifierOption::Cas,
    )
    .unwrap();

    // test_parameters1: a = 0.5
    // test_parameters2: a = 0.1 or 0.3
    assert_eq!(p.pure_records[1].model_record.a, 0.5);
    let br = p.binary_records.as_ref().unwrap();
    assert_eq!(br[[0, 1]].b, 12.0);
    assert_eq!(br[[1, 0]].b, 12.0);
}

#[test]
fn from_records_missing_binary() -> Result<(), ParameterError> {
    let pr_json = r#"
        [
            {
                "identifier": {
                    "cas": "123-4-5"
                },
                "molarweight": 16.0426,
                "model_record": {
                    "a": 0.1
                }
            },
            {
                "identifier": {
                    "cas": "678-9-1"
                },
                "molarweight": 32.08412,
                "model_record": {
                    "a": 0.2
                }
            }
        ]
        "#;
    let br_json = r#"
        [
            {
                "id1": {
                    "cas": "123-4-5"
                },
                "id2": {
                    "cas": "000-00-0"
                },
                "model_record": {
                    "b": 12.0
                }
            }
        ]
        "#;
    let pure_records = serde_json::from_str(pr_json).expect("Unable to parse json.");
    let binary_records: Vec<_> = serde_json::from_str(br_json).expect("Unable to parse json.");
    let binary_matrix = MyParameter::binary_matrix_from_records(
        &pure_records,
        &binary_records,
        IdentifierOption::Cas,
    );
    let p = MyParameter::from_records(pure_records, binary_matrix)?;

    assert_eq!(p.pure_records[0].identifier.cas, Some("123-4-5".into()));
    assert_eq!(p.pure_records[1].identifier.cas, Some("678-9-1".into()));
    let br = p.binary_records.as_ref().unwrap();
    assert_eq!(br[[0, 1]], MyBinaryModel::default());
    assert_eq!(br[[0, 1]].b, 0.0);
    Ok(())
}

#[test]
fn from_records_correct_binary_order() -> Result<(), ParameterError> {
    let pr_json = r#"
        [
            {
                "identifier": {
                    "cas": "000-0-0"
                },
                "molarweight": 32.08412,
                "model_record": {
                    "a": 0.2
                }
            },
            {
                "identifier": {
                    "cas": "123-4-5"
                },
                "molarweight": 16.0426,
                "model_record": {
                    "a": 0.1
                }
            },
            {
                "identifier": {
                    "cas": "678-9-1"
                },
                "molarweight": 32.08412,
                "model_record": {
                    "a": 0.2
                }
            }
        ]
        "#;
    let br_json = r#"
        [
            {
                "id1": {
                    "cas": "123-4-5"
                },
                "id2": {
                    "cas": "678-9-1"
                },
                "model_record": {
                    "b": 12.0
                }
            }
        ]
        "#;
    let pure_records = serde_json::from_str(pr_json).expect("Unable to parse json.");
    let binary_records: Vec<_> = serde_json::from_str(br_json).expect("Unable to parse json.");
    let binary_matrix = MyParameter::binary_matrix_from_records(
        &pure_records,
        &binary_records,
        IdentifierOption::Cas,
    );
    let p = MyParameter::from_records(pure_records, binary_matrix)?;

    assert_eq!(p.pure_records[0].identifier.cas, Some("000-0-0".into()));
    assert_eq!(p.pure_records[1].identifier.cas, Some("123-4-5".into()));
    assert_eq!(p.pure_records[2].identifier.cas, Some("678-9-1".into()));
    let br = p.binary_records.as_ref().unwrap();
    assert_eq!(br[[0, 1]], MyBinaryModel::default());
    assert_eq!(br[[1, 0]], MyBinaryModel::default());
    assert_eq!(br[[0, 2]], MyBinaryModel::default());
    assert_eq!(br[[2, 0]], MyBinaryModel::default());
    assert_eq!(br[[2, 1]].b, 12.0);
    assert_eq!(br[[1, 2]].b, 12.0);
    Ok(())
}
