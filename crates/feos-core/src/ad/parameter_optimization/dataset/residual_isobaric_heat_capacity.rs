use crate::{ParametersAD, PropertiesAD};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use super::Dataset;

#[derive(Deserialize, Serialize)]
pub struct ResidualIsobaricHeatCapacityRecord {
    pub temperature_k: f64,
    pub pressure_pa: f64,
    pub cp_res_j_molk: f64,
}

#[derive(Clone)]
pub struct ResidualIsobaricHeatCapacityDataset {
    inputs: Array2<f64>,
    target: Array1<f64>,
    name: Option<String>,
}

impl ResidualIsobaricHeatCapacityDataset {
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl Dataset<1> for ResidualIsobaricHeatCapacityDataset {
    type Record = ResidualIsobaricHeatCapacityRecord;

    fn from_records(records: Vec<Self::Record>) -> Self {
        let n = records.len();
        let inputs = Array2::from_shape_fn((n, 2), |(i, j)| match j {
            0 => records[i].temperature_k,
            _ => records[i].pressure_pa,
        });
        let target = Array1::from_iter(records.iter().map(|r| r.cp_res_j_molk));
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
        self.name
            .as_deref()
            .unwrap_or("residual isobaric heat capacity")
    }
    fn input_names() -> &'static [&'static str] {
        &["temperature_k", "pressure_pa"]
    }
    fn target_name() -> &'static str {
        "cp_res_j_molk"
    }

    fn call_model<T: ParametersAD<1>, const P: usize>(
        names: [String; P],
        parameters: ArrayView2<f64>,
        inputs: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        T::residual_isobaric_heat_capacity_parallel(names, parameters, inputs)
    }
}
