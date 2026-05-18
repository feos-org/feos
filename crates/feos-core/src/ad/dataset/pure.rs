use std::{io, path::Path};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::ad::properties::*;
use crate::{ParametersAD, Residual};

use super::{Dataset, DatasetAD, DatasetStorage};

/// Expand a list of pure-component property entries into:
/// - the [`PureProperty`] enum, metadata and dispatch methods,
/// - constructors,
/// - [`PureDataset::from_csv`] and [`PureDataset::from_reader`].
///
/// Adding a new property:
/// - write the property file (record, `*_ad`, `*_parallel`, `*_parallel_ad`)
/// - add entry here.
macro_rules! pure_properties {
    ($(
        $variant:ident {
            record:       $record:ty,
            default_name: $default:expr,
            input_names:  $inputs:expr,
            target_name:  $target:expr,
            ad_fn:        $ad_fn:ident,
            eval_fn:      $eval_fn:ident,
            constructor:  $ctor:ident,
        }
    ),* $(,)?) => {
        /// Pure-component properties.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum PureProperty {
            $($variant,)*
        }

        impl PureProperty {
            pub fn default_name(self) -> &'static str {
                match self { $(Self::$variant => $default,)* }
            }

            pub fn input_names(self) -> &'static [&'static str] {
                match self { $(Self::$variant => $inputs,)* }
            }

            pub fn target_name(self) -> &'static str {
                match self { $(Self::$variant => $target,)* }
            }

            fn evaluate_ad<T: ParametersAD<1>, const P: usize>(
                self,
                names: [String; P],
                parameters: ArrayView2<f64>,
                inputs: ArrayView2<f64>,
            ) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
                match self {
                    $(Self::$variant => $ad_fn::<T, P>(names, parameters, inputs),)*
                }
            }

            fn evaluate<E: Residual + Sync>(self, eos: &E, inputs: ArrayView2<f64>) -> (Array1<f64>, Array1<bool>)
            {
                match self {
                    $(Self::$variant => $eval_fn(eos, inputs),)*
                }
            }
        }

        impl PureDataset {
            $(
                pub fn $ctor(records: Vec<$record>) -> Self {
                    Self {
                        property: PureProperty::$variant,
                        storage: DatasetStorage::from_records(records),
                    }
                }
            )*

            pub fn from_csv(property: PureProperty, path: &Path) -> Result<Self, csv::Error> {
                let storage = match property {
                    $(PureProperty::$variant => DatasetStorage::from_csv::<$record>(path)?,)*
                };
                Ok(Self { property, storage })
            }

            pub fn from_reader(
                property: PureProperty,
                reader: impl io::Read,
            ) -> Result<Self, csv::Error> {
                let storage = match property {
                    $(PureProperty::$variant => DatasetStorage::from_reader::<$record>(reader)?,)*
                };
                Ok(Self { property, storage })
            }
        }
    };
}

pure_properties! {
    VaporPressure {
        record:       VaporPressureRecord,
        default_name: "vapor pressure",
        input_names:  &["temperature_k"],
        target_name:  "vapor_pressure_pa",
        ad_fn:        vapor_pressure_parallel_ad,
        eval_fn:      vapor_pressure_parallel,
        constructor:  vapor_pressure,
    },
    LiquidDensity {
        record:       LiquidDensityRecord,
        default_name: "liquid density",
        input_names:  &["temperature_k", "pressure_pa"],
        target_name:  "liquid_density_kmol_m3",
        ad_fn:        liquid_density_parallel_ad,
        eval_fn:      liquid_density_parallel,
        constructor:  liquid_density,
    },
    EquilibriumLiquidDensity {
        record:       EquilibriumLiquidDensityRecord,
        default_name: "equilibrium liquid density",
        input_names:  &["temperature_k"],
        target_name:  "liquid_density_kmol_m3",
        ad_fn:        equilibrium_liquid_density_parallel_ad,
        eval_fn:      equilibrium_liquid_density_parallel,
        constructor:  equilibrium_liquid_density,
    },
    EnthalpyOfVaporization {
        record:       EnthalpyOfVaporizationRecord,
        default_name: "enthalpy of vaporization",
        input_names:  &["temperature_k"],
        target_name:  "dh_vap_j_mol",
        ad_fn:        enthalpy_of_vaporization_parallel_ad,
        eval_fn:      enthalpy_of_vaporization_parallel,
        constructor:  enthalpy_of_vaporization,
    },
    ResidualIsobaricHeatCapacity {
        record:       ResidualIsobaricHeatCapacityRecord,
        default_name: "residual isobaric heat capacity",
        input_names:  &["temperature_k", "pressure_pa"],
        target_name:  "cp_res_j_molk",
        ad_fn:        residual_isobaric_heat_capacity_parallel_ad,
        eval_fn:      residual_isobaric_heat_capacity_parallel,
        constructor:  residual_isobaric_heat_capacity,
    },
}

/// Pure-component dataset: shared data storage plus a property tag.
#[derive(Clone)]
pub struct PureDataset {
    property: PureProperty,
    storage: DatasetStorage,
}

impl PureDataset {
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.storage.set_name(name.into());
        self
    }

    pub fn property(&self) -> PureProperty {
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

impl Dataset for PureDataset {
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

impl DatasetAD<1> for PureDataset {
    fn evaluate_ad_const<T: ParametersAD<1>, const P: usize>(
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
    fn vapor_pressure_from_reader() {
        let data = "\
temperature_k,vapor_pressure_pa
300.0,3540.0
350.0,41682.0
400.0,245600.0
";
        let ds = PureDataset::from_reader(PureProperty::VaporPressure, csv(data)).unwrap();

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
        let ds = PureDataset::vapor_pressure(records);
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
        let ds = PureDataset::from_reader(PureProperty::LiquidDensity, csv(data)).unwrap();

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
        let ds = PureDataset::liquid_density(records);
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
        let ds =
            PureDataset::from_reader(PureProperty::EquilibriumLiquidDensity, csv(data)).unwrap();

        assert_eq!(ds.inputs().ncols(), 1);
        assert_eq!(ds.inputs().nrows(), 3);
        assert_eq!(ds.inputs()[[2, 0]], 330.0);
        assert_eq!(ds.target()[2], 14.1);
        assert_eq!(ds.name(), "equilibrium liquid density");
    }

    #[test]
    fn missing_column_returns_error() {
        let data = "temperature_k\n300.0\n";
        let result = PureDataset::from_reader(PureProperty::VaporPressure, csv(data));
        assert!(result.is_err());
    }

    #[test]
    fn wrong_type_returns_error() {
        let data = "temperature_k,vapor_pressure_pa\n300.0,not_a_number\n";
        let result = PureDataset::from_reader(PureProperty::VaporPressure, csv(data));
        assert!(result.is_err());
    }
}
