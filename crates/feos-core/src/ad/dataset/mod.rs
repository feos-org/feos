mod binary;
mod pure;

use std::{io, path::Path, sync::Arc};

use nalgebra::Const;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::de::DeserializeOwned;

use crate::Residual;
use crate::ad::Gradient;

use super::ParametersAD;

pub use binary::*;
pub use pure::*;

/// Shared numerical data for all datasets.
struct DatasetData {
    inputs: Array2<f64>,
    target: Array1<f64>,
}

/// Shared representation for all datasets.
#[derive(Clone)]
struct DatasetStorage {
    data: Arc<DatasetData>,
    name: Option<String>,
}

impl DatasetStorage {
    fn from_records<R: DatasetRecord>(records: Vec<R>) -> Self {
        let n = records.len();
        let inputs = Array2::from_shape_fn((n, R::N_INPUTS), |(i, j)| records[i].input(j));
        let target = Array1::from_iter(records.iter().map(DatasetRecord::target));
        Self {
            data: Arc::new(DatasetData { inputs, target }),
            name: None,
        }
    }

    fn from_csv<R: DatasetRecord>(path: &Path) -> Result<Self, csv::Error> {
        let records = csv::Reader::from_path(path)?
            .deserialize()
            .collect::<Result<Vec<R>, _>>()?;
        Ok(Self::from_records(records))
    }

    fn from_reader<R: DatasetRecord>(reader: impl io::Read) -> Result<Self, csv::Error> {
        let records = csv::Reader::from_reader(reader)
            .deserialize()
            .collect::<Result<Vec<R>, _>>()?;
        Ok(Self::from_records(records))
    }

    fn inputs(&self) -> ArrayView2<'_, f64> {
        self.data.inputs.view()
    }

    fn target(&self) -> ArrayView1<'_, f64> {
        self.data.target.view()
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }
}

/// A record that can be collected into a dataset.
pub trait DatasetRecord: DeserializeOwned {
    /// Number of columns for inputs.
    const N_INPUTS: usize;

    /// Value of EoS input column.
    fn input(&self, column: usize) -> f64;

    /// Target value.
    fn target(&self) -> f64;
}

/// Dataset that can be evaluated by an equation of state.
pub trait Dataset {
    /// Inputs for EoS evaluation, shape `[n_points, k]`.
    fn inputs(&self) -> ArrayView2<'_, f64>;

    /// Target values, shape `[n_points]`.
    fn target(&self) -> ArrayView1<'_, f64>;

    /// Property name.
    ///
    /// Used for logging and diagnostics.
    fn name(&self) -> &str;

    /// Names of independent input columns.
    fn input_names(&self) -> &'static [&'static str];

    /// Name of the target property.
    fn target_name(&self) -> &'static str;

    /// Evaluate this dataset's property with an equation of state.
    ///
    /// Returns `(predicted, converged)`:
    /// - `predicted`: shape `[n_points]`, in SI units; `NaN` where the
    ///   underlying solver did not converge.
    /// - `converged`: shape `[n_points]`.
    fn evaluate<E: Residual + Sync>(&self, eos: &E) -> (Array1<f64>, Array1<bool>);
}

/// Build [`GRADIENT_SLOTS`] and [`DatasetAD`] trait from a single list of 'slots'.
///
/// For each slot in [`GRADIENT_SLOTS`] a compile-time constant `P` variant is generated.
macro_rules! define_dataset_ad {
    ($($p:literal),+ $(,)?) => {
        /// Compile-time gradient slot supported by [`DatasetAD::evaluate_ad`].
        pub const GRADIENT_SLOTS: &[usize] = &[$($p),+];

        /// Dataset that supports parameter-gradient evaluation
        /// for equations of state implementing [`ParametersAD<N>`].
        pub trait DatasetAD<const N: usize>: Dataset {
            /// Evaluate the property and its `P` parameter gradients.
            fn evaluate_ad_const<T: ParametersAD<Const<N>>, const P: usize>(
                &self,
                names: [String; P],
                parameters: &[f64],
                inputs: ArrayView2<f64>,
            ) -> (Array1<f64>, Array2<f64>, Array1<bool>)
            where
                T::Lifted<Gradient<P>>: Sync;

            /// Evaluate the property and its parameter gradients at the given parameters.
            ///
            /// - `param_names`: names of the `P` parameters being differentiated.
            /// - `params`: the full parameter vector. Only entries listed in `param_names` are seeded.
            ///
            /// This function dispatches the const-P methods at run-time.
            fn evaluate_ad<T: ParametersAD<Const<N>>>(
                &self,
                param_names: &[String],
                parameters: &[f64],
            ) -> (Array1<f64>, Array2<f64>, Array1<bool>)
            where
                $(T::Lifted<Gradient<$p>>: Sync,)*
            {
                fn to_const<const P: usize>(names: &[String]) -> [String; P] {
                    names.to_vec().try_into().expect("parameter count mismatch")
                }

                match param_names.len() {
                    $(
                        $p => self.evaluate_ad_const::<T, $p>(
                            to_const(param_names),
                            parameters,
                            self.inputs().view(),
                        ),
                    )+
                    p => unreachable!(
                        "parameter count {p} is not a member of GRADIENT_SLOTS={:?}",
                        GRADIENT_SLOTS,
                    ),
                }
            }
        }
    };
}

// We define the number of slots here.
//
// Note: might be good to investigate whether a smaller list makes sense here.
// LLVM vectorises across entries in DualSVec. We might see no perf. difference
// when using e.g. 3 vs 4 slots (even if only 3 are needed by the user).
// If the number of monomophised variants ever gets problematic, we could reduce it that way.
define_dataset_ad!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
