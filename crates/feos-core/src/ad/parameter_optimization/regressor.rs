use std::collections::HashMap;
use std::marker::PhantomData;

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, MinimizationReport};
use nalgebra::{DMatrix, DVector, Dyn, Owned};
use ndarray::{Array1, Array2, Zip, s};
use thiserror::Error;

use crate::ParametersAD;

use super::dataset::{BinaryDataset, DatasetResult, PureDataset};
use super::loss::LossFunction;
use super::residual::ResidualFunction;

/// Error kinds for parameter fitting and data handling.
#[derive(Debug, Error)]
pub enum ParameterOptimizationError {
    #[error("at least one dataset must be provided")]
    EmptyDatasets,

    #[error("unknown parameter '{name}', valid names: {valid:?}")]
    UnknownParameter {
        name: String,
        valid: Vec<&'static str>,
    },

    #[error("missing initial value for parameter '{name}'")]
    MissingParameter { name: String },

    #[error(
        "parameter '{name}' has initial value 0.0; \
         the solver normalizes by dividing through the initial value, \
         so zero freezes the parameter. Use a small non-zero starting \
         value instead."
    )]
    ZeroParameter { name: String },

    #[error("parameter '{name}' appears more than once")]
    DuplicateParameter { name: String },

    #[error(
        "parameter '{name}' is not differentiable; \
         differentiable parameters: {differentiable:?}"
    )]
    NonDifferentiableParameter {
        name: String,
        differentiable: Vec<&'static str>,
    },

    #[error(
        "dataset name '{name}' is used more than once; \
         please assign a unique name to each dataset"
    )]
    DuplicateDatasetName { name: String },

    #[error("weights length {got} does not match number of datasets {expected}")]
    WeightsLengthMismatch { got: usize, expected: usize },

    #[error("all dataset weights must be positive, got: {weights:?}")]
    NonPositiveWeight { weights: Vec<f64> },
}

/// Variants for how to deal with non-converged calculations.
#[derive(Debug, Clone)]
pub enum NonConvergenceStrategy {
    /// Exclude non-converged points from the optimisation entirely.
    /// Both the residual and the Jacobian row are set to zero.
    Ignore,

    /// Assign a fixed dimensionless relative residual to non-converged points.
    /// E.g. `Penalty(10.0)` corresponds to a 1000 % relative error.
    /// The Jacobian row is zero which creates a residual/Jacobian mismatch
    /// might be an issue for the trust region when convergence changes discontinuously.
    Penalty(f64),
}

impl Default for NonConvergenceStrategy {
    fn default() -> Self {
        Self::Penalty(10.0)
    }
}

/// Levenberg-Marquardt solver hyperparameters.
#[derive(Debug, Clone)]
pub struct RegressorConfig {
    /// Terminate when actual and predicted relative reductions in the
    /// objective function are both <= `ftol`. Default: 1e-8.
    pub ftol: f64,
    /// Terminate when the relative change in parameters between
    /// iterations is <= `xtol`. Default: 1e-8.
    pub xtol: f64,
    /// Terminate when the residual vector and every Jacobian column are nearly
    /// orthogonal. Set to `0.0` to disable. Default: 1e-8.
    pub gtol: f64,
    /// Factor for the initial trust-region step bound.
    /// Should lie in `[0.1, 100]`. Small values prevent large
    /// initial steps that can cause (e.g. VLE) solvers to diverge.
    /// Default: 0.1.
    pub stepbound: f64,
    /// Maximum number of residual evaluations = `patience (n_params + 1)`.
    /// Default: 100.
    pub patience: usize,
    /// Rescale parameters internally using the running maximum of each
    /// Jacobian column norm. Default: true.
    pub scale_diag: bool,
    /// Strategy for non-converged calculations.
    /// Default: [`NonConvergenceStrategy::Penalty(10.0)`].
    pub strategy: NonConvergenceStrategy,
}

impl Default for RegressorConfig {
    fn default() -> Self {
        Self {
            ftol: 1e-8,
            xtol: 1e-8,
            gtol: 1e-8,
            stepbound: 0.1,
            patience: 100,
            scale_diag: true,
            strategy: NonConvergenceStrategy::default(),
        }
    }
}

/// Post-fit diagnostics produced by [`Regressor::to_result`].
///
/// For model evaluation against the stored data (predictions, per-dataset
/// comparison, Fisher information), use [`Regressor::evaluate_datasets`] by
/// passing [`RegressorResult::all_parameters`] back to the regressor.
pub struct RegressorResult {
    /// Optimal physical parameters in canonical order.
    /// Includes fitted and non-fitted parameters.
    pub optimal_params: Vec<f64>,
    /// Names of the parameters that were optimised.
    pub fitted_param_names: Vec<String>,
    /// All canonical parameter names in order as defined by the model.
    pub all_param_names: Vec<String>,
    /// Per-dataset average absolute relative deviation in percent.
    /// `None` if no data points converged for that dataset.
    pub aad_per_dataset: Vec<Option<f64>>,
    /// Dataset names in order.
    pub dataset_names: Vec<String>,
    /// Per-dataset `(n_converged, n_total)`.
    pub convergence_stats: Vec<(usize, usize)>,
    /// Whether LM reported successful convergence.
    pub converged: bool,
    /// Description of why the optimiser stopped.
    pub termination_reason: String,
    /// Number of residual evaluations performed by LM.
    pub n_evaluations: usize,
    /// Final LM objective.
    pub objective_function: f64,
    /// Wall-clock time spent inside LM.
    pub elapsed: std::time::Duration,
}

impl RegressorResult {
    /// All parameters as `{name: value}`.
    ///
    /// Fitted parameters at their optimal values, all others at their initial values.
    /// Pass this directly to [`Regressor::evaluate_datasets`] to get per-dataset
    /// predictions at the fitted optimum.
    pub fn all_parameters(&self) -> HashMap<String, f64> {
        self.all_param_names
            .iter()
            .zip(self.optimal_params.iter())
            .map(|(name, &val)| (name.clone(), val))
            .collect()
    }

    /// Optimal parameters for the fitted subset as `{name: value}`.
    pub fn optimal_params_dict(&self) -> HashMap<String, f64> {
        self.fitted_param_names
            .iter()
            .filter_map(|name| {
                self.all_param_names
                    .iter()
                    .position(|n| n == name)
                    .map(|idx| (name.clone(), self.optimal_params[idx]))
            })
            .collect()
    }
}

pub struct DatasetCache {
    /// Number of datapoints in this dataset.
    pub datapoints: usize,
    /// Relative weight assigned by the user for this dataset.
    pub user_weight: f64,
    /// Effective per-residual scale factor: `sqrt(user_weight / datapoints)`.
    ///
    /// Multiplying each residual by this value makes the dataset's contribution
    /// to the LM objective equal to `user_weight * MSE`, so equal user weights
    /// yield equal objective contributions regardless of dataset size.
    pub dataset_weight: f64,
    /// Predicted values for this dataset.
    pub predicted: Array1<f64>,
    /// Gradients of the model w.r.t. model parameters.
    pub gradients: Array2<f64>,
    /// Whether the optimization converged for each datapoint.
    pub converged: Array1<bool>,
    /// Residuals of the model for this dataset defined via `ResidualFunction`.
    pub residuals: Array1<f64>,
    /// Weight weight defined via `LossFunction`.
    pub loss_w_sqrt: Array1<f64>,
}

/// Levenberg-Marquardt fitting problem.
pub struct Regressor<T, D> {
    /// Collections of experimental data.
    pub(crate) datasets: Vec<D>,
    /// Caching for each dataset.
    pub(crate) caches: Vec<DatasetCache>,
    /// Parameter names in canonical order.
    pub(crate) param_names: Vec<String>,
    /// Indices of parameters that are adjusted.
    pub(crate) fitted_indices: Vec<usize>,
    /// All parameters.
    pub(crate) base_params: Vec<f64>,
    /// Scaling factor to switch between EoS input and
    /// optimizer input.
    pub(crate) normalization: Vec<f64>,
    /// Residual function applied to predictions vs. targets.
    pub(crate) residual_fn: ResidualFunction,
    /// Loss function applied to residuals.
    pub(crate) loss_fn: LossFunction,
    /// Strategy for dealing with non-converged calculations.
    pub(crate) strategy: NonConvergenceStrategy,
    /// Parameter storage for the solver.
    pub(crate) params: DVector<f64>,
    pub(crate) _phantom: PhantomData<T>,
}

pub type PureRegressor<T> = Regressor<T, PureDataset>;
pub type BinaryRegressor<T> = Regressor<T, BinaryDataset>;

/// Empty placeholder needed by [`DynRegressor::fit`] for mem::take.
impl<T, D> Default for Regressor<T, D> {
    fn default() -> Self {
        Self {
            datasets: Vec::new(),
            caches: Vec::new(),
            param_names: Vec::new(),
            fitted_indices: Vec::new(),
            base_params: Vec::new(),
            normalization: Vec::new(),
            residual_fn: ResidualFunction::RelativeDifference,
            loss_fn: LossFunction::default(),
            strategy: NonConvergenceStrategy::default(),
            params: DVector::zeros(0),
            _phantom: PhantomData,
        }
    }
}

impl<T, D> Regressor<T, D> {
    /// Set the residual function (default: [`ResidualFunction::RelativeDifference`]).
    pub fn with_residual_fn(mut self, residual_fn: ResidualFunction) -> Self {
        self.residual_fn = residual_fn;
        self
    }

    /// Set per-dataset weights (default: 1.0 for all datasets).
    ///
    /// Each weight scales that dataset's contribution to the mean squared residual
    /// in the objective. Two datasets with equal weights contribute equally to the
    /// objective regardless of their number of data points.
    ///
    /// Weights are not normalized; only their ratios matter.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterOptimizationError::WeightsLengthMismatch`] if `weights.len()` does not
    /// match the number of datasets, or [`ParameterOptimizationError::NonPositiveWeight`] if any
    /// weight is not positive.
    pub fn with_weights(mut self, weights: Vec<f64>) -> Result<Self, ParameterOptimizationError> {
        if weights.len() != self.datasets.len() {
            return Err(ParameterOptimizationError::WeightsLengthMismatch {
                got: weights.len(),
                expected: self.datasets.len(),
            });
        }
        if weights.iter().any(|&w| w <= 0.0) {
            return Err(ParameterOptimizationError::NonPositiveWeight { weights });
        }

        // write weights to caches
        self.caches
            .iter_mut()
            .zip(weights.iter())
            .for_each(|(cache, &w)| {
                cache.user_weight = w;
                cache.dataset_weight = (w / cache.datapoints as f64).sqrt();
            });
        Ok(self)
    }

    /// Optimal physical parameters with non-fitted entries at their initial values.
    pub fn optimal_params(&self) -> Vec<f64> {
        let mut result = self.base_params.clone();
        for (j, &idx) in self.fitted_indices.iter().enumerate() {
            result[idx] = self.params[j] * self.normalization[j];
        }
        result
    }

    /// Predicted values across all datasets at current parameters.
    ///
    /// Non-converged entries are `NaN`; use [`Regressor::convergence_mask`] to filter.
    pub fn predicted(&self) -> Array1<f64> {
        let n_total: usize = self.caches.iter().map(|c| c.datapoints).sum();
        let mut out = Array1::zeros(n_total);
        let mut offset = 0;
        for c in &self.caches {
            let n = c.datapoints;
            out.slice_mut(s![offset..offset + n]).assign(&c.predicted);
            offset += n;
        }
        out
    }

    /// Convergence mask across all datasets.
    pub fn convergence_mask(&self) -> Array1<bool> {
        let n_total: usize = self.caches.iter().map(|c| c.datapoints).sum();
        let mut out = Array1::from_elem(n_total, false);
        let mut offset = 0;
        for c in &self.caches {
            let n = c.datapoints;
            out.slice_mut(s![offset..offset + n]).assign(&c.converged);
            offset += n;
        }
        out
    }

    /// Per-dataset `(n_converged, n_total)`.
    pub fn convergence_stats(&self) -> Vec<(usize, usize)> {
        self.caches
            .iter()
            .map(|c| (c.converged.iter().filter(|&&c| c).count(), c.datapoints))
            .collect()
    }
}

macro_rules! impl_regressor {
    ($n:literal, $dataset:ty) => {
        impl<T: ParametersAD<$n>> Regressor<T, $dataset> {
            /// Construct a new solver.
            ///
            /// - `datasets`: one or more property datasets.
            /// - `params`: full initial parameter set keyed by name.
            /// - `fit`: names of the parameters to optimise.
            pub fn new(
                datasets: Vec<$dataset>,
                params: HashMap<String, f64>,
                fit: &[&str],
            ) -> Result<Self, ParameterOptimizationError> {
                if datasets.is_empty() {
                    return Err(ParameterOptimizationError::EmptyDatasets);
                }

                // Reject duplicate dataset names
                let mut seen_names = std::collections::HashSet::new();
                for d in &datasets {
                    if !seen_names.insert(d.name().to_string()) {
                        return Err(ParameterOptimizationError::DuplicateDatasetName {
                            name: d.name().to_string(),
                        });
                    }
                }

                // Get parameter names from model in canonical order (expected by model)
                let canonical = T::parameter_names();
                let differentiable = T::differentiable_parameters();

                // Collect indices of parameters that are fitted
                let mut fitted_indices = Vec::with_capacity(fit.len());
                for &name in fit {
                    if fitted_indices.iter().any(|&prev| canonical[prev] == name) {
                        return Err(ParameterOptimizationError::DuplicateParameter {
                            name: name.to_string(),
                        });
                    }
                    let i = canonical.iter().position(|&n| n == name).ok_or_else(|| {
                        ParameterOptimizationError::UnknownParameter {
                            name: name.to_string(),
                            valid: canonical.clone(),
                        }
                    })?;
                    if !differentiable.contains(&name) {
                        return Err(ParameterOptimizationError::NonDifferentiableParameter {
                            name: name.to_string(),
                            differentiable: differentiable.clone(),
                        });
                    }
                    fitted_indices.push(i);
                }

                // Get values for all parameters in canonical order.
                let mut full_params = Vec::with_capacity(canonical.len());
                for &name in &canonical {
                    let v = params.get(name).ok_or_else(|| {
                        ParameterOptimizationError::MissingParameter {
                            name: name.to_string(),
                        }
                    })?;
                    full_params.push(*v);
                }

                let param_names: Vec<String> = fit.iter().map(|s| s.to_string()).collect();
                let p = param_names.len();

                // We normalize parameters in optimizer state to be in the order of 1.
                // Hence, scaling factors are the inital parameters.
                let normalization: Vec<f64> =
                    fitted_indices.iter().map(|&i| full_params[i]).collect();

                // Initial fitting parameters with 0.0 are rejected
                // since we divide for normalization.
                for (j, &idx) in fitted_indices.iter().enumerate() {
                    if normalization[j] == 0.0 {
                        return Err(ParameterOptimizationError::ZeroParameter {
                            name: canonical[idx].to_string(),
                        });
                    }
                }

                let caches = datasets
                    .iter()
                    .map(|ds| DatasetCache {
                        datapoints: ds.target().len(),
                        user_weight: 1.0,
                        dataset_weight: 1.0 / (ds.target().len() as f64).sqrt(),
                        predicted: Array1::zeros(ds.target().len()),
                        gradients: Array2::zeros((ds.target().len(), p)),
                        converged: Array1::from_elem(ds.target().len(), false),
                        residuals: Array1::zeros(ds.target().len()),
                        loss_w_sqrt: Array1::zeros(ds.target().len()),
                    })
                    .collect();

                let mut solver = Self {
                    datasets,
                    param_names,
                    fitted_indices,
                    base_params: full_params,
                    normalization,
                    residual_fn: ResidualFunction::RelativeDifference,
                    loss_fn: LossFunction::default(),
                    strategy: NonConvergenceStrategy::default(),
                    params: DVector::from_element(p, 1.0f64),
                    caches,
                    _phantom: PhantomData,
                };
                solver.update_cache();
                Ok(solver)
            }

            /// Set the loss function.
            pub fn with_loss(mut self, loss_fn: LossFunction) -> Self {
                self.loss_fn = loss_fn;
                self
            }

            /// Run Levenberg-Marquardt to convergence.
            pub fn fit(mut self, config: RegressorConfig) -> (Self, MinimizationReport<f64>) {
                // overwrite strategy or nonconverging states
                self.strategy = config.strategy.clone();
                LevenbergMarquardt::new()
                    .with_ftol(config.ftol)
                    .with_xtol(config.xtol)
                    .with_gtol(config.gtol)
                    .with_stepbound(config.stepbound)
                    .with_patience(config.patience)
                    .with_scale_diag(config.scale_diag)
                    .minimize(self)
            }

            /// Evaluate all datasets at the given full parameters (EoS ready).
            ///
            /// Returns `(predicted, gradients, converged)` concatenated across
            /// all datasets. Gradients are w.r.t. the fitted parameters only.
            pub fn predict(&self, params: &[f64]) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
                let p = self.param_names.len();
                let n_total: usize = self.datasets.iter().map(|d| d.target().len()).sum();
                let mut pred = Array1::zeros(n_total);
                let mut grad = Array2::zeros((n_total, p));
                let mut conv = Array1::from_elem(n_total, false);
                let mut offset = 0;
                for d in &self.datasets {
                    let (dp, dg, dc) = d.compute::<T>(&self.param_names, params);
                    let n = dp.len();
                    pred.slice_mut(s![offset..offset + n]).assign(&dp);
                    grad.slice_mut(s![offset..offset + n, ..]).assign(&dg);
                    conv.slice_mut(s![offset..offset + n]).assign(&dc);
                    offset += n;
                }
                (pred, grad, conv)
            }

            /// Evaluate each dataset individually and return structured
            /// per-dataset results.
            ///
            /// Pass [`RegressorResult::all_parameters`] as `params` to inspect the
            /// model at the fitted optimum.
            pub fn evaluate_datasets(&self, params: &[f64]) -> Vec<DatasetResult> {
                self.datasets
                    .iter()
                    .map(|dataset| {
                        let (pred, _grad, conv) = dataset.compute::<T>(&self.param_names, params);
                        let target = dataset.target();
                        let n = target.len();
                        let inputs = dataset
                            .input_names()
                            .iter()
                            .enumerate()
                            .map(|(col, &name)| {
                                let col_data: Vec<f64> =
                                    (0..n).map(|row| dataset.inputs()[[row, col]]).collect();
                                (name, col_data)
                            })
                            .collect();
                        let relative_deviation: Vec<f64> = (0..n)
                            .map(|i| {
                                if conv[i] {
                                    (pred[i] - target[i]) / target[i]
                                } else {
                                    f64::NAN
                                }
                            })
                            .collect();
                        DatasetResult {
                            name: dataset.name().to_string(),
                            inputs,
                            target_name: dataset.target_name(),
                            target: target.to_vec(),
                            predicted: pred.to_vec(),
                            converged: conv.to_vec(),
                            relative_deviation,
                        }
                    })
                    .collect()
            }

            /// Extract [`RegressorResult`] from the current solver state and a [`MinimizationReport`].
            pub fn to_result(&self, report: &MinimizationReport<f64>) -> RegressorResult {
                RegressorResult {
                    optimal_params: self.optimal_params(),
                    fitted_param_names: self.param_names.clone(),
                    all_param_names: T::parameter_names().iter().map(|s| s.to_string()).collect(),
                    aad_per_dataset: self.aad_per_dataset(),
                    dataset_names: self.dataset_names().iter().map(|s| s.to_string()).collect(),
                    convergence_stats: self.convergence_stats(),
                    converged: report.termination.was_successful(),
                    termination_reason: format!("{:?}", report.termination),
                    n_evaluations: report.number_of_evaluations,
                    objective_function: report.objective_function,
                    elapsed: std::time::Duration::ZERO,
                }
            }

            /// Target values across all datasets.
            pub fn target(&self) -> Array1<f64> {
                let n_total: usize = self.datasets.iter().map(|d| d.target().len()).sum();
                let mut out = Array1::zeros(n_total);
                let mut offset = 0;
                for dataset in &self.datasets {
                    let exp = dataset.target();
                    let n = exp.len();
                    out.slice_mut(s![offset..offset + n]).assign(&exp);
                    offset += n;
                }
                out
            }

            /// Names of all datasets.
            pub fn dataset_names(&self) -> Vec<&str> {
                self.datasets.iter().map(|d| d.name()).collect()
            }

            /// Per-dataset average absolute relative deviation in percent.
            pub fn aad_per_dataset(&self) -> Vec<Option<f64>> {
                self.datasets
                    .iter()
                    .zip(self.caches.iter())
                    .map(|(ds, c)| {
                        let target = ds.target();
                        let pred = &c.predicted;
                        let conv = &c.converged;
                        let n_conv = conv.iter().filter(|&&c| c).count();
                        if n_conv == 0 {
                            return None;
                        }
                        Some(
                            pred.iter()
                                .zip(target.iter())
                                .zip(conv.iter())
                                .filter(|(_, c)| **c)
                                .map(|((p, e), _)| ((p - e) / e).abs())
                                .sum::<f64>()
                                / n_conv as f64
                                * 100.0,
                        )
                    })
                    .collect()
            }

            fn residuals_impl(&self) -> Option<DVector<f64>> {
                let n_total: usize = self.caches.iter().map(|c| c.datapoints).sum();
                if n_total == 0 {
                    return None;
                }

                let penalty = match self.strategy {
                    NonConvergenceStrategy::Ignore => 0.0,
                    NonConvergenceStrategy::Penalty(p) => p,
                };

                let mut out = DVector::zeros(n_total);
                let mut offset = 0;

                for cache in self.caches.iter() {
                    let dataset_weight = cache.dataset_weight;
                    let conv = &cache.converged;
                    let w_sqrt = &cache.loss_w_sqrt;

                    for i in 0..cache.datapoints {
                        out[offset + i] = if conv[i] {
                            dataset_weight * cache.residuals[i] * w_sqrt[i]
                        } else {
                            dataset_weight * penalty
                        };
                    }
                    offset += cache.datapoints;
                }
                Some(out)
            }

            /// Jacobian for LM iteration.
            ///
            /// Needs derivatives that account for
            /// - the residual function
            /// - using normalized parameters
            /// - using (optional) non L2 loss
            /// - using weights for whole parameter sets.
            fn jacobian_impl(&self) -> Option<DMatrix<f64>> {
                let n_total: usize = self.datasets.iter().map(|d| d.target().len()).sum();
                let p = self.params.len();
                if n_total == 0 || p == 0 {
                    return None;
                }

                let mut jac = DMatrix::zeros(n_total, p);
                let mut offset = 0;

                for (dataset, cache) in self.datasets.iter().zip(self.caches.iter()) {
                    let target = dataset.target();
                    let conv = &cache.converged;
                    let w_sqrt = &cache.loss_w_sqrt;
                    let dataset_weight = cache.dataset_weight;
                    let n = target.len();

                    let mut grad = cache.gradients.clone();
                    self.residual_fn.jacobian_transform(
                        cache.predicted.view(),
                        target,
                        conv.view(),
                        &mut grad,
                    );

                    let mut jac_block = jac.rows_mut(offset, n);
                    for i in 0..n {
                        if !conv[i] {
                            continue;
                        }
                        let total_weight = dataset_weight * w_sqrt[i];
                        let grad_row = grad.row(i);
                        let mut jac_row = jac_block.row_mut(i);
                        for j in 0..p {
                            jac_row[j] = total_weight * grad_row[j];
                        }
                    }
                    offset += n;
                }

                // Apply parameter scaling.
                for (mut col, &norm) in jac.column_iter_mut().zip(self.normalization.iter()) {
                    for val in col.iter_mut() {
                        *val *= norm;
                    }
                }

                Some(jac)
            }

            /// Populates the caches when parameters are set in LM iteration.
            ///
            /// Using the cache, we only have to compute the residual and gradients once.
            /// Scaling factors and weights are applied in the residual and jacobian functions.
            fn update_cache(&mut self) {
                // precompute full parameter vector
                let mut full_params = self.base_params.clone();
                for (j, &idx) in self.fitted_indices.iter().enumerate() {
                    full_params[idx] = self.params[j] * self.normalization[j];
                }

                let residual_fn = &self.residual_fn;
                let loss_fn = &self.loss_fn;

                for (dataset, cache) in self.datasets.iter().zip(self.caches.iter_mut()) {
                    let (pred, grad, conv) = dataset.compute::<T>(&self.param_names, &full_params);
                    let target = dataset.target();

                    Zip::from(&mut cache.residuals)
                        .and(&mut cache.loss_w_sqrt)
                        .and(target)
                        .and(&pred)
                        .and(&conv)
                        .for_each(|r, w, &t, &p, &c| {
                            if c {
                                *r = residual_fn.residual(p, t);
                                *w = loss_fn.irls_weight_sqrt(*r);
                            } else {
                                *r = 0.0;
                                *w = 0.0;
                            }
                        });

                    cache.predicted = pred;
                    cache.gradients = grad;
                    cache.converged = conv;
                }
            }
        }

        impl<T: ParametersAD<$n>> LeastSquaresProblem<f64, Dyn, Dyn> for Regressor<T, $dataset> {
            type ParameterStorage = Owned<f64, Dyn>;
            type ResidualStorage = Owned<f64, Dyn>;
            type JacobianStorage = Owned<f64, Dyn, Dyn>;

            fn set_params(&mut self, params: &DVector<f64>) {
                self.params.copy_from(params);
                self.update_cache();
            }
            fn params(&self) -> DVector<f64> {
                self.params.clone()
            }
            fn residuals(&self) -> Option<DVector<f64>> {
                self.residuals_impl()
            }
            fn jacobian(&self) -> Option<DMatrix<f64>> {
                self.jacobian_impl()
            }
        }
    };
}

impl_regressor!(1, PureDataset);
impl_regressor!(2, BinaryDataset);

/// Object-safe interface for type-erased regressors.
///
/// A `Box<dyn DynRegressor>` can hold any `Regressor<T, D>`.
///
/// This is useful when models are created at runtime (used in py-feos).
///
/// # Example
///
/// ```ignore
/// let mut reg: Box<dyn DynRegressor> = match model_name {
///     "eos_1" => Box::new(PureRegressor::<Eos1>::new(ds, params, &fit)?),
///     "eos_2" => Box::new(PureRegressor::<Eos2>::new(ds, params, &fit)?),
///     _ => panic!("unknown model"),
/// };
/// let result = reg.fit(RegressorConfig::default(), None);
/// println!("AAD: {:?}", result.aad_per_dataset);
/// ```
pub trait DynRegressor: Send + Sync {
    /// Evaluate all datasets at the given physical parameters.
    ///
    /// Returns `(predicted, gradients, converged)` concatenated across
    /// all datasets.  Gradients are w.r.t. the fitted parameters only.
    fn predict(&self, params: &[f64]) -> (Array1<f64>, Array2<f64>, Array1<bool>);

    /// Run Levenberg-Marquardt, return a [`RegressorResult`].
    ///
    /// Does not consume the solver. This is different from LM.
    fn fit(&mut self, config: RegressorConfig, loss: Option<LossFunction>) -> RegressorResult;

    /// Concatenated target values across all datasets.
    fn target(&self) -> Array1<f64>;

    /// All canonical parameter names in order.
    fn all_param_names(&self) -> Vec<String>;

    /// Names of the parameters being optimised.
    fn fitted_param_names(&self) -> Vec<String>;

    /// Dataset names in order.
    fn dataset_names(&self) -> Vec<String>;

    /// Current physical parameters in canonical order.
    ///
    /// Before [`fit`](Self::fit) is called this returns the initial values.
    /// After fitting it returns the optimal values for the fitted parameters
    /// and the initial values for the fixed ones.
    fn optimal_params(&self) -> Vec<f64>;

    /// Evaluate each dataset at the given full parameters and return
    /// structured per-dataset results.
    ///
    /// See [`Regressor::evaluate_datasets`] for details.
    fn evaluate_datasets(&self, params: &[f64]) -> Vec<DatasetResult>;
}

macro_rules! impl_dyn_regressor {
    ($n:literal, $dataset:ty) => {
        impl<T> DynRegressor for Regressor<T, $dataset>
        where
            T: ParametersAD<$n> + Send + Sync + 'static,
        {
            fn predict(&self, params: &[f64]) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
                self.predict(params)
            }

            fn fit(
                &mut self,
                config: RegressorConfig,
                loss: Option<LossFunction>,
            ) -> RegressorResult {
                let mut solver = std::mem::take(self);
                if let Some(l) = loss {
                    solver = solver.with_loss(l);
                }
                let t0 = std::time::Instant::now();
                let (solved, report) = solver.fit(config);
                let elapsed = t0.elapsed();
                let mut result = solved.to_result(&report);
                result.elapsed = elapsed;
                *self = solved;
                result
            }

            fn target(&self) -> Array1<f64> {
                self.target()
            }

            fn all_param_names(&self) -> Vec<String> {
                T::parameter_names().iter().map(|s| s.to_string()).collect()
            }

            fn fitted_param_names(&self) -> Vec<String> {
                self.param_names.clone()
            }

            fn dataset_names(&self) -> Vec<String> {
                self.dataset_names()
                    .iter()
                    .map(|s: &&str| s.to_string())
                    .collect()
            }

            fn optimal_params(&self) -> Vec<f64> {
                self.optimal_params()
            }

            fn evaluate_datasets(&self, params: &[f64]) -> Vec<DatasetResult> {
                self.evaluate_datasets(params)
            }
        }
    };
}

impl_dyn_regressor!(1, PureDataset);
impl_dyn_regressor!(2, BinaryDataset);
