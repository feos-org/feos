use std::{io, path::Path};

use nalgebra::U2;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::Residual;
use crate::ad::properties::{BubblePointPressure, DewPointPressure, Property};

use super::{Dataset, DatasetAD, DatasetRecord, DatasetStorage, ParametersAD};

/// The pressure column doubles as the initial guess passed to the VLE solver.
#[derive(Deserialize, Serialize)]
pub struct BubblePointRecord {
    pub temperature_k: f64,
    pub liquid_molefrac_1: f64,
    pub bubble_pressure_pa: f64,
}

impl DatasetRecord for BubblePointRecord {
    const N_INPUTS: usize = 3;

    fn input(&self, column: usize) -> f64 {
        match column {
            0 => self.temperature_k,
            1 => self.liquid_molefrac_1,
            2 => self.bubble_pressure_pa,
            _ => unreachable!("invalid bubble point input column"),
        }
    }

    fn target(&self) -> f64 {
        self.bubble_pressure_pa
    }
}

/// The pressure column doubles as the initial guess passed to the VLE solver.
#[derive(Deserialize, Serialize)]
pub struct DewPointRecord {
    pub temperature_k: f64,
    pub vapor_molefrac_1: f64,
    pub dew_pressure_pa: f64,
}

impl DatasetRecord for DewPointRecord {
    const N_INPUTS: usize = 3;

    fn input(&self, column: usize) -> f64 {
        match column {
            0 => self.temperature_k,
            1 => self.vapor_molefrac_1,
            2 => self.dew_pressure_pa,
            _ => unreachable!("invalid dew point input column"),
        }
    }

    fn target(&self) -> f64 {
        self.dew_pressure_pa
    }
}

/// Expand a list of binary-mixture property entries into:
/// - the [`BinaryProperty`] enum, metadata and dispatch methods,
/// - constructors,
/// - [`BinaryDataset::from_csv`] and [`BinaryDataset::from_reader`] match arms.
macro_rules! binary_properties {
    ($(
        $variant:ident {
            record:       $record:ty,
            property:     $prop:ty,
            default_name: $default:expr,
            input_names:  $inputs:expr,
            target_name:  $target:expr,
            constructor:  $ctor:ident,
        }
    ),* $(,)?) => {
        /// Binary-mixture properties.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum BinaryProperty {
            $($variant,)*
        }

        impl BinaryProperty {
            pub fn default_name(self) -> &'static str {
                match self { $(Self::$variant => $default,)* }
            }

            pub fn input_names(self) -> &'static [&'static str] {
                match self { $(Self::$variant => $inputs,)* }
            }

            pub fn target_name(self) -> &'static str {
                match self { $(Self::$variant => $target,)* }
            }

            fn evaluate_ad<T: ParametersAD<U2>, const P: usize>(
                self,
                names: [String; P],
                parameters: ArrayView2<f64>,
                inputs: ArrayView2<f64>,
            ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
                match self {
                    $(Self::$variant => <$prop>::evaluate_parallel_ad::<T, P>(names, parameters, inputs),)*
                }
            }

            fn evaluate<E>(self, eos: &E, inputs: ArrayView2<f64>) -> (Array1<f64>, Array1<bool>)
            where
                E: Residual + Sync,
            {
                match self {
                    $(Self::$variant => <$prop>::evaluate_parallel(eos, inputs),)*
                }
            }
        }

        impl BinaryDataset {
            $(
                pub fn $ctor(records: Vec<$record>) -> Self {
                    Self {
                        property: BinaryProperty::$variant,
                        storage: DatasetStorage::from_records(records),
                    }
                }
            )*

            pub fn from_csv(property: BinaryProperty, path: &Path) -> Result<Self, csv::Error> {
                let storage = match property {
                    $(BinaryProperty::$variant => DatasetStorage::from_csv::<$record>(path)?,)*
                };
                Ok(Self { property, storage })
            }

            pub fn from_reader(
                property: BinaryProperty,
                reader: impl io::Read,
            ) -> Result<Self, csv::Error> {
                let storage = match property {
                    $(BinaryProperty::$variant => DatasetStorage::from_reader::<$record>(reader)?,)*
                };
                Ok(Self { property, storage })
            }
        }
    };
}

binary_properties! {
    BubblePointPressure {
        record:       BubblePointRecord,
        property:     BubblePointPressure,
        default_name: "bubble point pressure",
        input_names:  &["temperature_k", "liquid_molefrac_1"],
        target_name:  "bubble_pressure_pa",
        constructor:  bubble_point_pressure,
    },
    DewPointPressure {
        record:       DewPointRecord,
        property:     DewPointPressure,
        default_name: "dew point pressure",
        input_names:  &["temperature_k", "vapor_molefrac_1"],
        target_name:  "dew_pressure_pa",
        constructor:  dew_point_pressure,
    },
}

/// Binary-mixture dataset: shared data storage plus a property tag.
#[derive(Clone)]
pub struct BinaryDataset {
    property: BinaryProperty,
    storage: DatasetStorage,
}

impl BinaryDataset {
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.storage.set_name(name.into());
        self
    }

    pub fn property(&self) -> BinaryProperty {
        self.property
    }

    pub fn inputs(&self) -> ArrayView2<'_, f64> {
        self.storage.inputs()
    }

    pub fn target(&self) -> ArrayView1<'_, f64> {
        self.storage.target()
    }

    pub fn name(&self) -> &str {
        self.storage.name().unwrap_or(self.property.default_name())
    }

    pub fn input_names(&self) -> &'static [&'static str] {
        self.property.input_names()
    }

    pub fn target_name(&self) -> &'static str {
        self.property.target_name()
    }
}

impl Dataset for BinaryDataset {
    fn inputs(&self) -> ArrayView2<'_, f64> {
        self.inputs()
    }

    fn target(&self) -> ArrayView1<'_, f64> {
        self.target()
    }

    fn name(&self) -> &str {
        self.name()
    }

    fn input_names(&self) -> &'static [&'static str] {
        self.input_names()
    }

    fn target_name(&self) -> &'static str {
        self.target_name()
    }

    fn evaluate<E: Residual + Sync>(&self, eos: &E) -> (Array1<f64>, Array1<bool>) {
        self.property.evaluate(eos, self.inputs())
    }
}

impl DatasetAD<2> for BinaryDataset {
    fn evaluate_ad_const<T: ParametersAD<U2>, const P: usize>(
        &self,
        names: [String; P],
        parameters: ArrayView2<f64>,
        inputs: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
        self.property.evaluate_ad::<T, P>(names, parameters, inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn csv(s: &str) -> Cursor<&[u8]> {
        Cursor::new(s.as_bytes())
    }

    #[test]
    fn bubble_point_from_reader() {
        let data = "\
temperature_k,liquid_molefrac_1,bubble_pressure_pa
300.0,0.3,500000.0
320.0,0.5,800000.0
";
        let ds =
            BinaryDataset::from_reader(BinaryProperty::BubblePointPressure, csv(data)).unwrap();

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
        let ds = BinaryDataset::from_reader(BinaryProperty::DewPointPressure, csv(data)).unwrap();

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
        let ds = BinaryDataset::dew_point_pressure(records);
        assert_eq!(ds.inputs()[[0, 2]], ds.target()[0]);
    }
}
