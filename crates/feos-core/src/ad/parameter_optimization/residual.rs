use ndarray::{Array2, ArrayView1, Zip};

pub enum ResidualFunction {
    /// Computes the difference between the prediction and target values.
    Difference,
    /// Computes the log difference between the prediction and target values.
    LogDifference,
    /// Computes the relative difference between the prediction and target values.
    RelativeDifference,
}

impl ResidualFunction {
    #[inline(always)]
    pub fn residual(&self, prediction: f64, target: f64) -> f64 {
        match self {
            Self::Difference => prediction - target,
            Self::LogDifference => prediction.ln() - target.ln(),
            Self::RelativeDifference => (prediction - target) / target,
        }
    }

    /// Transforms the jacobian in-place based on the residual function.
    ///
    /// Non-converged entries are skipped.
    pub fn jacobian_transform(
        &self,
        prediction: ArrayView1<f64>,
        target: ArrayView1<f64>,
        converged: ArrayView1<bool>,
        jacobian: &mut Array2<f64>,
    ) {
        let iter = Zip::from(jacobian.rows_mut())
            .and(prediction)
            .and(target)
            .and(converged);

        match self {
            Self::Difference => {}
            Self::LogDifference => {
                iter.for_each(|mut row, &f, &_y, &conv| {
                    if conv {
                        row.mapv_inplace(|g| g / f);
                    }
                });
            }
            Self::RelativeDifference => {
                iter.for_each(|mut row, &_f, &y, &conv| {
                    if conv {
                        row.mapv_inplace(|g| g / y);
                    }
                });
            }
        }
    }
}
