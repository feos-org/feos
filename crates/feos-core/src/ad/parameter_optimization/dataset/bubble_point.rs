use crate::{ParametersAD, PropertiesAD};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use super::Dataset;

/// The pressure column doubles as the initial guess passed to the VLE solver.
#[derive(Deserialize, Serialize)]
pub struct BubblePointRecord {
    pub temperature_k: f64,
    pub liquid_molefrac_1: f64,
    pub bubble_pressure_pa: f64,
}

#[derive(Clone)]
pub struct BubblePointDataset {
    inputs: Array2<f64>,
    target: Array1<f64>,
    name: Option<String>,
}

impl BubblePointDataset {
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl Dataset<2> for BubblePointDataset {
    type Record = BubblePointRecord;

    fn from_records(records: Vec<Self::Record>) -> Self {
        let n = records.len();
        let inputs = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => records[i].temperature_k,
            1 => records[i].liquid_molefrac_1,
            _ => records[i].bubble_pressure_pa,
        });
        let target = Array1::from_iter(records.iter().map(|r| r.bubble_pressure_pa));
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
        self.name.as_deref().unwrap_or("bubble point pressure")
    }
    fn input_names() -> &'static [&'static str] {
        &["temperature_k", "liquid_molefrac_1"]
    }
    fn target_name() -> &'static str {
        "bubble_pressure_pa"
    }

    fn call_model<T: ParametersAD<2>, const P: usize>(
        names: [String; P],
        parameters: ArrayView2<f64>,
        inputs: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        T::bubble_point_pressure_parallel(names, parameters, inputs)
    }
}
