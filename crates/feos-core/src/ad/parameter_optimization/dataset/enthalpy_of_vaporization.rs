use crate::{ParametersAD, PropertiesAD};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use super::Dataset;

#[derive(Deserialize, Serialize)]
pub struct EnthalpyOfVaporizationRecord {
    pub temperature_k: f64,
    pub dh_vap_j_mol: f64,
}

#[derive(Clone)]
pub struct EnthalpyOfVaporizationDataset {
    inputs: Array2<f64>,
    target: Array1<f64>,
    name: Option<String>,
}

impl EnthalpyOfVaporizationDataset {
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl Dataset<1> for EnthalpyOfVaporizationDataset {
    type Record = EnthalpyOfVaporizationRecord;

    fn from_records(records: Vec<Self::Record>) -> Self {
        let n = records.len();
        let inputs = Array2::from_shape_fn((n, 1), |(i, _)| records[i].temperature_k);
        let target = Array1::from_iter(records.iter().map(|r| r.dh_vap_j_mol));
        Self {
            inputs,
            target,
            name: None,
        }
    }

    fn inputs(&self) -> ArrayView2<'_, f64> {
        self.inputs.view()
    }
    fn target(&self) -> ArrayView1<'_, f64> {
        self.target.view()
    }
    fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("enthalpy of vaporization")
    }
    fn input_names() -> &'static [&'static str] {
        &["temperature_k"]
    }
    fn target_name() -> &'static str {
        "dh_vap_j_mol"
    }

    fn call_model<T: ParametersAD<1>, const P: usize>(
        names: [String; P],
        parameters: ArrayView2<f64>,
        inputs: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        T::enthalpy_of_vaporization_parallel(names, parameters, inputs)
    }
}
