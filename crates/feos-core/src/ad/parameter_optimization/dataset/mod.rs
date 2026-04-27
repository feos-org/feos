mod bubble_point;
mod dew_point;
mod enthalpy_of_vaporization;
mod equilibrium_liquid_density;
mod liquid_density;
mod residual_isobaric_heat_capacity;
mod vapor_pressure;

pub use bubble_point::{BubblePointDataset, BubblePointRecord};
pub use dew_point::{DewPointDataset, DewPointRecord};
pub use enthalpy_of_vaporization::{EnthalpyOfVaporizationDataset, EnthalpyOfVaporizationRecord};
pub use equilibrium_liquid_density::{
    EquilibriumLiquidDensityDataset, EquilibriumLiquidDensityRecord,
};
pub use liquid_density::{LiquidDensityDataset, LiquidDensityRecord};
pub use residual_isobaric_heat_capacity::{
    ResidualIsobaricHeatCapacityDataset, ResidualIsobaricHeatCapacityRecord,
};
pub use vapor_pressure::{VaporPressureDataset, VaporPressureRecord};

use crate::ParametersAD;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::{io, path::Path};

/// Per-dataset evaluation result: inputs, experimental target, model
/// prediction, and derived statistics at a given set of parameters.
///
/// Produced by [`crate::Regressor::evaluate_datasets`].
/// Fully serializable and convertable into any tabular format (CSV, JSON, DataFrame).
#[derive(Debug, Serialize)]
pub struct DatasetResult {
    /// Dataset name (default property name or user-supplied).
    pub name: String,
    /// Input column names and their values, in the order they appear in the record struct.
    pub inputs: Vec<(&'static str, Vec<f64>)>,
    /// Name of the target property column.
    pub target_name: &'static str,
    /// Experimental target values.
    pub target: Vec<f64>,
    /// Values at the given parameters predicted by the model.
    /// `NaN` for points where calculations did not converge.
    pub predicted: Vec<f64>,
    /// Whether the calculation converged for each point.
    pub converged: Vec<bool>,
    /// Relative deviation `(predicted − target) / target`.
    /// `NaN` for non-converged points.
    pub relative_deviation: Vec<f64>,
}

/// Container for experimental data that can be evaluated
/// with models that implement [`ParametersAD<N>`].
pub trait Dataset<const N: usize> {
    /// Record of the property (row in a CSV file).
    type Record: DeserializeOwned;

    /// Dataset from a vector of records.
    fn from_records(records: Vec<Self::Record>) -> Self;

    /// Parse a CSV file and build the dataset.
    ///
    /// Column headers must match the field names of [`Self::Record`].
    fn from_csv(path: &Path) -> Result<Self, csv::Error>
    where
        Self: Sized,
    {
        let records = csv::Reader::from_path(path)?
            .deserialize()
            .collect::<Result<Vec<Self::Record>, _>>()?;
        Ok(Self::from_records(records))
    }

    /// Parse CSV data from any source that implements [`std::io::Read`].
    ///
    /// Column headers must match the field names of [`Self::Record`].
    fn from_reader<R: io::Read>(reader: R) -> Result<Self, csv::Error>
    where
        Self: Sized,
    {
        let records = csv::Reader::from_reader(reader)
            .deserialize()
            .collect::<Result<Vec<Self::Record>, _>>()?;
        Ok(Self::from_records(records))
    }

    /// Inputs for (parallel) model evaluation, shape `[n_points, k]`.
    fn inputs(&self) -> ArrayView2<'_, f64>;

    /// Target values, shape `[n_points]`.
    fn target(&self) -> ArrayView1<'_, f64>;

    /// Property name used for logging and diagnostics.
    fn name(&self) -> &str;

    /// Names of the input columns, in the order they appear in [`Self::inputs`].
    fn input_names() -> &'static [&'static str]
    where
        Self: Sized;

    /// Name of the target property column.
    fn target_name() -> &'static str
    where
        Self: Sized;

    /// Method for model evaluation of this property.
    fn call_model<T: ParametersAD<N>, const P: usize>(
        names: [String; P],
        parameters: ArrayView2<f64>,
        inputs: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>);

    /// Evaluate the property and its parameter gradients at the given parameters.
    ///
    /// - `param_names`: names of the `P` parameters being differentiated.
    /// - `params`: the full parameter vector; only entries listed in `param_names` are seeded.
    ///
    /// Returns `(predicted, gradients, converged)`:
    /// - `predicted`: shape `[n_points]`, in SI units.
    /// - `gradients`: shape `[n_points, P]`.
    /// - `converged`: shape `[n_points]`.
    fn compute<T: ParametersAD<N>>(
        &self,
        param_names: &[String],
        params: &[f64],
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        let n = self.inputs().nrows();
        let parameters = Array2::from_shape_fn((n, params.len()), |(_, j)| params[j]);

        fn to_const<const P: usize>(names: &[String]) -> [String; P] {
            names.to_vec().try_into().expect("parameter count mismatch")
        }

        match param_names.len() {
            1 => Self::call_model::<T, 1>(to_const(param_names), parameters.view(), self.inputs()),
            2 => Self::call_model::<T, 2>(to_const(param_names), parameters.view(), self.inputs()),
            3 => Self::call_model::<T, 3>(to_const(param_names), parameters.view(), self.inputs()),
            4 => Self::call_model::<T, 4>(to_const(param_names), parameters.view(), self.inputs()),
            5 => Self::call_model::<T, 5>(to_const(param_names), parameters.view(), self.inputs()),
            6 => Self::call_model::<T, 6>(to_const(param_names), parameters.view(), self.inputs()),
            7 => Self::call_model::<T, 7>(to_const(param_names), parameters.view(), self.inputs()),
            8 => Self::call_model::<T, 8>(to_const(param_names), parameters.view(), self.inputs()),
            n => panic!("too many parameters: {n} (max 8)"),
        }
    }
}

macro_rules! impl_dataset_enum {
    ($n:literal, $enum:ident, [ $( ($variant:ident, $ty:ty) ),+ $(,)? ]) => {
        impl $enum {
            pub fn target(&self) -> ArrayView1<'_, f64> {
                match self { $( Self::$variant(d) => d.target(), )+ }
            }
            pub fn name(&self) -> &str {
                match self { $( Self::$variant(d) => d.name(), )+ }
            }
            pub fn inputs(&self) -> ArrayView2<'_, f64> {
                match self { $( Self::$variant(d) => d.inputs(), )+ }
            }
            pub fn input_names(&self) -> &'static [&'static str] {
                match self { $( Self::$variant(_) => <$ty>::input_names(), )+ }
            }
            pub fn target_name(&self) -> &'static str {
                match self { $( Self::$variant(_) => <$ty>::target_name(), )+ }
            }
            pub fn compute<T: ParametersAD<$n>>(
                &self,
                param_names: &[String],
                params: &[f64],
            ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
                match self { $( Self::$variant(d) => d.compute::<T>(param_names, params), )+ }
            }
        }
    };
}

/// Collection of pure-component datasets.
///
/// Wraps any [`Dataset<1>`] implementor so that mixed-property lists can be
/// stored in a `Vec<PureDataset>` without requiring trait objects.
#[derive(Clone)]
pub enum PureDataset {
    VaporPressure(VaporPressureDataset),
    LiquidDensity(LiquidDensityDataset),
    EquilibriumLiquidDensity(EquilibriumLiquidDensityDataset),
    EnthalpyOfVaporization(EnthalpyOfVaporizationDataset),
    ResidualIsobaricHeatCapacity(ResidualIsobaricHeatCapacityDataset),
}

impl_dataset_enum!(
    1,
    PureDataset,
    [
        (VaporPressure, VaporPressureDataset),
        (LiquidDensity, LiquidDensityDataset),
        (EquilibriumLiquidDensity, EquilibriumLiquidDensityDataset),
        (EnthalpyOfVaporization, EnthalpyOfVaporizationDataset),
        (
            ResidualIsobaricHeatCapacity,
            ResidualIsobaricHeatCapacityDataset
        ),
    ]
);

/// Collection of binary mixture datasets.
///
/// Wraps any [`Dataset<2>`] implementor so that mixed-property lists can be
/// stored in a `Vec<BinaryDataset>` without requiring trait objects.
#[derive(Clone)]
pub enum BinaryDataset {
    BubblePoint(BubblePointDataset),
    DewPoint(DewPointDataset),
}

impl_dataset_enum!(
    2,
    BinaryDataset,
    [
        (BubblePoint, BubblePointDataset),
        (DewPoint, DewPointDataset),
    ]
);

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn csv(s: &str) -> Cursor<&[u8]> {
        Cursor::new(s.as_bytes())
    }

    #[test]
    fn vapor_pressure_from_reader() {
        let data = "\
temperature_k,vapor_pressure_pa
300.0,3540.0
350.0,41682.0
400.0,245600.0
";
        let ds = VaporPressureDataset::from_reader(csv(data)).unwrap();

        assert_eq!(ds.target().len(), 3);
        assert_eq!(ds.inputs().nrows(), 3);
        assert_eq!(ds.inputs().ncols(), 1);
        assert_eq!(ds.inputs()[[0, 0]], 300.0);
        assert_eq!(ds.inputs()[[1, 0]], 350.0);
        assert_eq!(ds.inputs()[[2, 0]], 400.0);
        assert_eq!(ds.target()[0], 3540.0);
        assert_eq!(ds.target()[1], 41682.0);
        assert_eq!(ds.target()[2], 245600.0);
        assert_eq!(ds.name(), "vapor pressure");
    }

    #[test]
    fn vapor_pressure_from_records() {
        let records = vec![
            VaporPressureRecord {
                temperature_k: 300.0,
                vapor_pressure_pa: 3540.0,
            },
            VaporPressureRecord {
                temperature_k: 350.0,
                vapor_pressure_pa: 41682.0,
            },
        ];
        let ds = VaporPressureDataset::from_records(records);
        assert_eq!(ds.inputs()[[0, 0]], 300.0);
        assert_eq!(ds.target()[1], 41682.0);
    }

    #[test]
    fn liquid_density_from_reader() {
        let data = "\
temperature_k,pressure_pa,liquid_density_kmol_m3
300.0,101325.0,15.2
320.0,200000.0,14.8
";
        let ds = LiquidDensityDataset::from_reader(csv(data)).unwrap();

        assert_eq!(ds.inputs().nrows(), 2);
        assert_eq!(ds.inputs().ncols(), 2);
        assert_eq!(ds.inputs()[[0, 0]], 300.0);
        assert_eq!(ds.inputs()[[0, 1]], 101325.0);
        assert_eq!(ds.inputs()[[1, 0]], 320.0);
        assert_eq!(ds.inputs()[[1, 1]], 200000.0);
        assert_eq!(ds.target()[0], 15.2);
        assert_eq!(ds.target()[1], 14.8);
        assert_eq!(ds.name(), "liquid density");
    }

    #[test]
    fn liquid_density_from_records() {
        let records = vec![LiquidDensityRecord {
            temperature_k: 300.0,
            pressure_pa: 101325.0,
            liquid_density_kmol_m3: 15.2,
        }];
        let ds = LiquidDensityDataset::from_records(records);
        assert_eq!(ds.inputs()[[0, 1]], 101325.0);
        assert_eq!(ds.target()[0], 15.2);
    }

    #[test]
    fn equilibrium_liquid_density_from_reader() {
        let data = "\
temperature_k,liquid_density_kmol_m3
290.0,15.5
310.0,14.9
330.0,14.1
";
        let ds = EquilibriumLiquidDensityDataset::from_reader(csv(data)).unwrap();

        assert_eq!(ds.inputs().ncols(), 1);
        assert_eq!(ds.inputs().nrows(), 3);
        assert_eq!(ds.inputs()[[2, 0]], 330.0);
        assert_eq!(ds.target()[2], 14.1);
        assert_eq!(ds.name(), "equilibrium liquid density");
    }

    #[test]
    fn bubble_point_from_reader() {
        let data = "\
temperature_k,liquid_molefrac_1,bubble_pressure_pa
300.0,0.3,500000.0
320.0,0.5,800000.0
";
        let ds = BubblePointDataset::from_reader(csv(data)).unwrap();

        assert_eq!(ds.inputs().ncols(), 3);
        assert_eq!(ds.inputs().nrows(), 2);
        assert_eq!(ds.inputs()[[0, 0]], 300.0);
        assert_eq!(ds.inputs()[[0, 1]], 0.3);
        assert_eq!(ds.inputs()[[0, 2]], 500000.0);
        assert_eq!(ds.target()[0], 500000.0);
        assert_eq!(ds.target()[1], 800000.0);
        assert_eq!(ds.name(), "bubble point pressure");
    }

    #[test]
    fn dew_point_from_reader() {
        let data = "\
temperature_k,vapor_molefrac_1,dew_pressure_pa
310.0,0.7,400000.0
330.0,0.9,700000.0
";
        let ds = DewPointDataset::from_reader(csv(data)).unwrap();

        assert_eq!(ds.inputs().ncols(), 3);
        assert_eq!(ds.inputs()[[0, 0]], 310.0);
        assert_eq!(ds.inputs()[[0, 1]], 0.7);
        assert_eq!(ds.inputs()[[0, 2]], 400000.0);
        assert_eq!(ds.target()[1], 700000.0);
        assert_eq!(ds.name(), "dew point pressure");
    }

    #[test]
    fn dew_point_pressure_doubles_as_initial_guess() {
        let records = vec![DewPointRecord {
            temperature_k: 310.0,
            vapor_molefrac_1: 0.7,
            dew_pressure_pa: 400000.0,
        }];
        let ds = DewPointDataset::from_records(records);
        assert_eq!(ds.inputs()[[0, 2]], ds.target()[0]);
    }

    #[test]
    fn missing_column_returns_error() {
        let data = "temperature_k\n300.0\n";
        let result = VaporPressureDataset::from_reader(csv(data));
        assert!(result.is_err());
    }

    #[test]
    fn wrong_type_returns_error() {
        let data = "temperature_k,vapor_pressure_pa\n300.0,not_a_number\n";
        let result = VaporPressureDataset::from_reader(csv(data));
        assert!(result.is_err());
    }
}
