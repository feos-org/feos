use crate::{ParametersAD, PropertiesAD};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use super::Dataset;

#[derive(Deserialize, Serialize)]
pub struct LiquidDensityRecord {
    pub temperature_k: f64,
    pub pressure_pa: f64,
    pub liquid_density_kmol_m3: f64,
}

#[derive(Clone)]
pub struct LiquidDensityDataset {
    inputs: Array2<f64>,
    target: Array1<f64>,
    name: Option<String>,
}

impl LiquidDensityDataset {
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl Dataset<1> for LiquidDensityDataset {
    type Record = LiquidDensityRecord;

    fn from_records(records: Vec<Self::Record>) -> Self {
        let n = records.len();
        let inputs = Array2::from_shape_fn((n, 2), |(i, j)| match j {
            0 => records[i].temperature_k,
            _ => records[i].pressure_pa,
        });
        let target = Array1::from_iter(records.iter().map(|r| r.liquid_density_kmol_m3));
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
        self.name.as_deref().unwrap_or("liquid density")
    }
    fn input_names() -> &'static [&'static str] {
        &["temperature_k", "pressure_pa"]
    }
    fn target_name() -> &'static str {
        "liquid_density_kmol_m3"
    }

    fn call_model<T: ParametersAD<1>, const P: usize>(
        names: [String; P],
        parameters: ArrayView2<f64>,
        inputs: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        T::liquid_density_parallel(names, parameters, inputs)
    }
}
