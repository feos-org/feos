use crate::{ParametersAD, PropertiesAD};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::{io, path::Path};

/// Per-dataset evaluation result: inputs, experimental target, model
/// prediction, and derived statistics at a given set of parameters.
///
/// Produced by [`crate::Regressor::evaluate_datasets`].
/// Fully serializable and convertablec into any tabular format (CSV, JSON, DataFrame).
#[derive(Debug, Serialize)]
pub struct DatasetResult {
    /// Dataset name (default property name or user-supplied).
    pub name: String,
    /// Input column names and their values.
    /// In the order they appear in record struct of Dataset.
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
    /// Record of the property (row in a CSV file)
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

    /// Parse CSV data from sources that implement [`std::io::Read`].
    ///
    /// Useful for reading from in memory buffers, network streams,
    /// or compressed sources.
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
    ///
    /// These match the non-target field names of [`Self::Record`].
    fn input_names() -> &'static [&'static str]
    where
        Self: Sized;

    /// Name of the target property column.
    ///
    /// Matches the target field name of [`Self::Record`] (e.g.
    /// `"vapor_pressure_pa"` for [`VaporPressureDataset`]).
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
    /// - `param_names`: names of the `P` model parameters that are differentiated.
    ///   Length determines the compile-time `P` through runtime dispatch.
    /// - `params`: the full parameter vector (not just the fitted subset).
    ///   `seed_derivatives` seeds only the entries listed in `param_names`.
    ///
    /// Returns `(predicted, gradients, converged)`:
    /// - `predicted`: shape `[n_points]`, computed property values in SI units.
    /// - `gradients`: shape `[n_points, P]`, ∂property/∂param for each point.
    /// - `converged`: shape `[n_points]`, `true` where the property calculation
    ///   succeeded.
    ///
    /// Non-converged state points yield NaN for predictions and zero for gradients.
    fn compute<T: ParametersAD<N>>(
        &self,
        param_names: &[String],
        params: &[f64],
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        let n = self.inputs().nrows();

        // Repeat parameters for each data point.
        let parameters = Array2::from_shape_fn((n, params.len()), |(_, j)| params[j]);

        // Translate slice to const array for the const-generic call.
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

/// Record for vapor pressure data.
///
/// Expected column headers: `temperature_k`, `vapor_pressure_pa`.
#[derive(Deserialize, Serialize)]
pub struct VaporPressureRecord {
    pub temperature_k: f64,
    pub vapor_pressure_pa: f64,
}

/// Vapor pressure dataset for pure substances.
///
/// Inputs: `[[T1], [T2], ...]` (temperature in K)
/// Target: vapor pressure in Pa
#[derive(Clone)]
pub struct VaporPressureDataset {
    inputs: Array2<f64>,
    target: Array1<f64>,
    name: Option<String>,
}

impl Dataset<1> for VaporPressureDataset {
    type Record = VaporPressureRecord;

    fn from_records(records: Vec<Self::Record>) -> Self {
        let n = records.len();
        let inputs = Array2::from_shape_fn((n, 1), |(i, _)| records[i].temperature_k);
        let target = Array1::from_iter(records.iter().map(|r| r.vapor_pressure_pa));
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
        self.name.as_deref().unwrap_or("vapor pressure")
    }

    fn input_names() -> &'static [&'static str] {
        &["temperature_k"]
    }

    fn target_name() -> &'static str {
        "vapor_pressure_pa"
    }

    fn call_model<T: ParametersAD<1>, const P: usize>(
        names: [String; P],
        parameters: ArrayView2<f64>,
        inputs: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        T::vapor_pressure_parallel(names, parameters, inputs)
    }
}

/// Liquid density record at given temperature and pressure.
///
/// Expected column headers: `temperature_k`, `pressure_pa`, `liquid_density_molm3`.
#[derive(Deserialize, Serialize)]
pub struct LiquidDensityRecord {
    pub temperature_k: f64,
    pub pressure_pa: f64,
    pub liquid_density_molm3: f64,
}

/// Liquid density dataset for pure substances at given temperature and pressure.
///
/// Inputs: `[[T1, p1], [T2, p2], ...]` (K, Pa)
/// Target: molar density in kmol/m³
#[derive(Clone)]
pub struct LiquidDensityDataset {
    inputs: Array2<f64>,
    target: Array1<f64>,
    name: Option<String>,
}

impl Dataset<1> for LiquidDensityDataset {
    type Record = LiquidDensityRecord;

    fn from_records(records: Vec<Self::Record>) -> Self {
        let n = records.len();
        let inputs = Array2::from_shape_fn((n, 2), |(i, j)| match j {
            0 => records[i].temperature_k,
            _ => records[i].pressure_pa,
        });
        let target = Array1::from_iter(records.iter().map(|r| r.liquid_density_molm3));
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
        "liquid_density_molm3"
    }

    fn call_model<T: ParametersAD<1>, const P: usize>(
        names: [String; P],
        parameters: ArrayView2<f64>,
        inputs: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        T::liquid_density_parallel(names, parameters, inputs)
    }
}

/// Equilibrium liquid density record (saturated liquid).
///
/// Expected column headers: `temperature_k`, `liquid_density_molm3`.
#[derive(Deserialize, Serialize)]
pub struct EquilibriumLiquidDensityRecord {
    pub temperature_k: f64,
    pub liquid_density_molm3: f64,
}

/// Liquid density (at equilibrium) dataset for pure substances.
///
/// Inputs: `[[T1], [T2], ...]` (temperature in K)
/// Target: molar density in kmol/m³
#[derive(Clone)]
pub struct EquilibriumLiquidDensityDataset {
    inputs: Array2<f64>,
    target: Array1<f64>,
    name: Option<String>,
}

impl Dataset<1> for EquilibriumLiquidDensityDataset {
    type Record = EquilibriumLiquidDensityRecord;

    fn from_records(records: Vec<Self::Record>) -> Self {
        let n = records.len();
        let inputs = Array2::from_shape_fn((n, 1), |(i, _)| records[i].temperature_k);
        let target = Array1::from_iter(records.iter().map(|r| r.liquid_density_molm3));
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
        self.name.as_deref().unwrap_or("equilibrium liquid density")
    }

    fn input_names() -> &'static [&'static str] {
        &["temperature_k"]
    }

    fn target_name() -> &'static str {
        "liquid_density_molm3"
    }

    fn call_model<T: ParametersAD<1>, const P: usize>(
        names: [String; P],
        parameters: ArrayView2<f64>,
        inputs: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        T::equilibrium_liquid_density_parallel(names, parameters, inputs)
    }
}

/// Bubble point pressure record for binary mixtures.
///
/// Expected column headers: `temperature_k`, `liquid_molefrac_1`,
/// `bubble_pressure_pa`. The pressure column doubles as the initial guess
/// passed to the VLE solver.
#[derive(Deserialize, Serialize)]
pub struct BubblePointRecord {
    pub temperature_k: f64,
    pub liquid_molefrac_1: f64,
    pub bubble_pressure_pa: f64,
}

/// Bubble point pressure dataset for binary mixtures.
///
/// Inputs: `[[T1, x11, p1], ...]` (K, -, Pa)
/// Target: bubble point pressure in Pa
#[derive(Clone)]
pub struct BubblePointDataset {
    inputs: Array2<f64>,
    target: Array1<f64>,
    name: Option<String>,
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

/// Dew point pressure record for binary mixtures.
///
/// Expected column headers: `temperature_k`, `vapor_molefrac_1`,
/// `dew_pressure_pa`. The pressure column doubles as the initial guess
/// passed to the VLE solver.
#[derive(Deserialize, Serialize)]
pub struct DewPointRecord {
    pub temperature_k: f64,
    pub vapor_molefrac_1: f64,
    pub dew_pressure_pa: f64,
}

/// Dew point pressure dataset for binary mixtures.
///
/// Inputs: `[[T1, y11, p1], ...]` (K, -, Pa)
/// Target: dew point pressure in Pa
#[derive(Clone)]
pub struct DewPointDataset {
    inputs: Array2<f64>,
    target: Array1<f64>,
    name: Option<String>,
}

impl Dataset<2> for DewPointDataset {
    type Record = DewPointRecord;

    fn from_records(records: Vec<Self::Record>) -> Self {
        let n = records.len();
        let inputs = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => records[i].temperature_k,
            1 => records[i].vapor_molefrac_1,
            _ => records[i].dew_pressure_pa,
        });
        let target = Array1::from_iter(records.iter().map(|r| r.dew_pressure_pa));
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
        self.name.as_deref().unwrap_or("dew point pressure")
    }

    fn input_names() -> &'static [&'static str] {
        &["temperature_k", "vapor_molefrac_1"]
    }

    fn target_name() -> &'static str {
        "dew_pressure_pa"
    }

    fn call_model<T: ParametersAD<2>, const P: usize>(
        names: [String; P],
        parameters: ArrayView2<f64>,
        inputs: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        T::dew_point_pressure_parallel(names, parameters, inputs)
    }
}

macro_rules! impl_with_name {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl $ty {
                /// Override the dataset name used in solver diagnostics and results.
                ///
                /// The name must be unique within a solver — [`Solver::new`] returns
                /// [`FittingError::DuplicateDatasetName`] if two datasets share the
                /// same name (whether default or user-supplied).
                pub fn with_name(mut self, name: impl Into<String>) -> Self {
                    self.name = Some(name.into());
                    self
                }
            }
        )+
    };
}

impl_with_name!(
    VaporPressureDataset,
    LiquidDensityDataset,
    EquilibriumLiquidDensityDataset,
    BubblePointDataset,
    DewPointDataset,
);

/// Collection of pure-component datasets.
///
/// Wraps any [`Dataset<1>`] implementor so that mixed-property lists can be
/// stored in a `Vec<PureDataset>` without requiring trait objects.
#[derive(Clone)]
pub enum PureDataset {
    VaporPressure(VaporPressureDataset),
    LiquidDensity(LiquidDensityDataset),
    EquilibriumLiquidDensity(EquilibriumLiquidDensityDataset),
}

impl PureDataset {
    pub fn target(&self) -> ArrayView1<'_, f64> {
        match self {
            Self::VaporPressure(d) => d.target(),
            Self::LiquidDensity(d) => d.target(),
            Self::EquilibriumLiquidDensity(d) => d.target(),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::VaporPressure(d) => d.name(),
            Self::LiquidDensity(d) => d.name(),
            Self::EquilibriumLiquidDensity(d) => d.name(),
        }
    }

    pub fn inputs(&self) -> ArrayView2<'_, f64> {
        match self {
            Self::VaporPressure(d) => d.inputs(),
            Self::LiquidDensity(d) => d.inputs(),
            Self::EquilibriumLiquidDensity(d) => d.inputs(),
        }
    }

    pub fn input_names(&self) -> &'static [&'static str] {
        match self {
            Self::VaporPressure(_) => VaporPressureDataset::input_names(),
            Self::LiquidDensity(_) => LiquidDensityDataset::input_names(),
            Self::EquilibriumLiquidDensity(_) => EquilibriumLiquidDensityDataset::input_names(),
        }
    }

    pub fn target_name(&self) -> &'static str {
        match self {
            Self::VaporPressure(_) => VaporPressureDataset::target_name(),
            Self::LiquidDensity(_) => LiquidDensityDataset::target_name(),
            Self::EquilibriumLiquidDensity(_) => EquilibriumLiquidDensityDataset::target_name(),
        }
    }

    pub fn compute<T: ParametersAD<1>>(
        &self,
        param_names: &[String],
        params: &[f64],
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        match self {
            Self::VaporPressure(d) => d.compute::<T>(param_names, params),
            Self::LiquidDensity(d) => d.compute::<T>(param_names, params),
            Self::EquilibriumLiquidDensity(d) => d.compute::<T>(param_names, params),
        }
    }
}

/// Collection of binary mixture datasets.
///
/// Wraps any [`Dataset<2>`] implementor so that mixed-property lists can be
/// stored in a `Vec<BinaryDataset>` without requiring trait objects.
#[derive(Clone)]
pub enum BinaryDataset {
    BubblePoint(BubblePointDataset),
    DewPoint(DewPointDataset),
}

impl BinaryDataset {
    pub fn target(&self) -> ArrayView1<'_, f64> {
        match self {
            Self::BubblePoint(d) => d.target(),
            Self::DewPoint(d) => d.target(),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::BubblePoint(d) => d.name(),
            Self::DewPoint(d) => d.name(),
        }
    }

    pub fn inputs(&self) -> ArrayView2<'_, f64> {
        match self {
            Self::BubblePoint(d) => d.inputs(),
            Self::DewPoint(d) => d.inputs(),
        }
    }

    pub fn input_names(&self) -> &'static [&'static str] {
        match self {
            Self::BubblePoint(_) => BubblePointDataset::input_names(),
            Self::DewPoint(_) => DewPointDataset::input_names(),
        }
    }

    pub fn target_name(&self) -> &'static str {
        match self {
            Self::BubblePoint(_) => BubblePointDataset::target_name(),
            Self::DewPoint(_) => DewPointDataset::target_name(),
        }
    }

    pub fn compute<T: ParametersAD<2>>(
        &self,
        param_names: &[String],
        params: &[f64],
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        match self {
            Self::BubblePoint(d) => d.compute::<T>(param_names, params),
            Self::DewPoint(d) => d.compute::<T>(param_names, params),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // Helper: build a Cursor from a string literal.
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

        // temperatures land in the single input column
        assert_eq!(ds.inputs()[[0, 0]], 300.0);
        assert_eq!(ds.inputs()[[1, 0]], 350.0);
        assert_eq!(ds.inputs()[[2, 0]], 400.0);

        // experimental values
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
temperature_k,pressure_pa,liquid_density_molm3
300.0,101325.0,15.2
320.0,200000.0,14.8
";
        let ds = LiquidDensityDataset::from_reader(csv(data)).unwrap();

        assert_eq!(ds.inputs().nrows(), 2);
        assert_eq!(ds.inputs().ncols(), 2);

        // column 0: temperature, column 1: pressure
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
            liquid_density_molm3: 15.2,
        }];
        let ds = LiquidDensityDataset::from_records(records);
        assert_eq!(ds.inputs()[[0, 1]], 101325.0);
        assert_eq!(ds.target()[0], 15.2);
    }

    #[test]
    fn equilibrium_liquid_density_from_reader() {
        let data = "\
temperature_k,liquid_density_molm3
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

        // column 0: T, column 1: x1, column 2: pressure (initial guess)
        assert_eq!(ds.inputs()[[0, 0]], 300.0);
        assert_eq!(ds.inputs()[[0, 1]], 0.3);
        assert_eq!(ds.inputs()[[0, 2]], 500000.0);

        // experimental = bubble pressure
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
        // CSV is missing the vapor_pressure_pa column entirely.
        let data = "temperature_k\n300.0\n";
        let result = VaporPressureDataset::from_reader(csv(data));
        assert!(result.is_err());
    }

    #[test]
    fn wrong_type_returns_error() {
        // pressure is not a float
        let data = "temperature_k,vapor_pressure_pa\n300.0,not_a_number\n";
        let result = VaporPressureDataset::from_reader(csv(data));
        assert!(result.is_err());
    }
}
