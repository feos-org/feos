use std::collections::HashMap;
use std::marker::PhantomData;

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, MinimizationReport};
use nalgebra::{DMatrix, DVector, Dyn, Owned};
use ndarray::{Array1, Array2, s};
use thiserror::Error;

use crate::ParametersAD;

use super::dataset::{BinaryDataset, DatasetResult, PureDataset};
use super::loss::LossFunction;

/// Error kinds for parameter fitting and data handling.
#[derive(Debug, Error)]
pub enum FittingError {
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

    /// Scale the penalty by `factor × max |r_rel|` of currently converged
    /// points. Non-converged points always look proportionally worse than
    /// the worst converged one. Falls back to `factor` when no points have
    /// converged.
    AdaptivePenalty(f64),
}

impl Default for NonConvergenceStrategy {
    fn default() -> Self {
        Self::Penalty(10.0)
    }
}

/// Levenberg-Marquardt solver hyperparameters.
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// Terminate when the actual and predicted relative reductions in the
    /// objective function are both ≤ `ftol`. Default: 1e-8.
    pub ftol: f64,
    /// Terminate when the relative change in parameters between
    /// iterations is ≤ `xtol`. Default: 1e-8.
    pub xtol: f64,
    /// Terminate when the residual vector and every Jacobian column are nearly
    /// orthogonal, i.e. the cosine of the angle between `r` and each `J[:,j]`
    /// is ≤ `gtol`. This is a scale-invariant first-order optimality check.
    /// Set to `0.0` to disable. Default: 1e-8.
    pub gtol: f64,
    /// Factor for the initial trust-region step bound.
    /// Should lie in `[0.1, 100]`. Smaller values are more conservative and
    /// prevent the large initial steps that can cause VLE solvers to diverge.
    /// Default: 0.1.
    pub stepbound: f64,
    /// Maximum number of residual evaluations = `patience × (n_params + 1)`.
    /// Default: 100.
    pub patience: usize,
    /// Rescale parameters internally using the running maximum of each
    /// Jacobian column norm. Default: true.
    pub scale_diag: bool,
    /// Strategy for non-converged calculations.
    /// Default: [`NonConvergenceStrategy::Penalty(10.0)`].
    pub strategy: NonConvergenceStrategy,
}

impl Default for FitConfig {
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
/// passing [`FitResult::all_parameters`] back to the regressor.
pub struct FitResult {
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

impl FitResult {
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

/// Levenberg-Marquardt fitting problem.
pub struct Regressor<T, D> {
    /// Collections of experimental data.
    pub(crate) datasets: Vec<D>,
    /// A single weight per dataset applied
    /// to all residuals and Jacobian entries for that set.
    pub(crate) weights: Vec<f64>,
    /// Parameter names in canonical order.
    pub(crate) param_names: Vec<String>,
    /// Indices of parameters that are adjusted.
    pub(crate) fitted_indices: Vec<usize>,
    /// All parameters.
    pub(crate) base_params: Vec<f64>,
    /// Scaling factor to switch between EoS input and
    /// optimizer input.
    pub(crate) normalization: Vec<f64>,
    /// Loss function applied to residuals.
    pub(crate) loss: LossFunction,
    /// Strategy for dealing with non-converged calculations.
    pub(crate) strategy: NonConvergenceStrategy,
    /// Parameter storage for the solver.
    pub(crate) params: DVector<f64>,
    /// Cached predictions, i.e. EoS evaluations. Per dataset.
    pub(crate) cached_predicted: Vec<Array1<f64>>,
    /// Cached gradients of EoS evaluations w.r.t. EoS parameters.
    /// Per Dataset.
    pub(crate) cached_gradients: Vec<Array2<f64>>,
    /// Cached convergence-status of each data point. Per Dataset.
    pub(crate) cached_converged: Vec<Array1<bool>>,
    /// Cached IRLS weight.
    pub(crate) cached_loss_w_sqrt: Vec<Array1<f64>>,
    /// Penalty for non-converged points, updated by `update_cache`.
    /// For `Penalty(p)` this is always `p`; for `AdaptivePenalty(f)` it
    /// tracks `f × max |r_rel|` over the most recent converged points.
    pub(crate) cached_penalty: f64,
    pub(crate) _phantom: PhantomData<T>,
}

pub type PureRegressor<T> = Regressor<T, PureDataset>;
pub type BinaryRegressor<T> = Regressor<T, BinaryDataset>;

/// Empty placeholder needed by [`DynSolver::fit`] for mem::take.
impl<T, D> Default for Regressor<T, D> {
    fn default() -> Self {
        Self {
            datasets: Vec::new(),
            weights: Vec::new(),
            param_names: Vec::new(),
            fitted_indices: Vec::new(),
            base_params: Vec::new(),
            normalization: Vec::new(),
            loss: LossFunction::default(),
            strategy: NonConvergenceStrategy::default(),
            params: DVector::zeros(0),
            cached_predicted: Vec::new(),
            cached_gradients: Vec::new(),
            cached_converged: Vec::new(),
            cached_loss_w_sqrt: Vec::new(),
            cached_penalty: 0.0,
            _phantom: PhantomData,
        }
    }
}

impl<T, D> Regressor<T, D> {
    /// Set per-dataset weights (default 1.0 for all).
    ///
    /// Weights are not normalized in this function.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len()` does not match the number of datasets, or if
    /// any weight is not positive.
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        assert_eq!(weights.len(), self.datasets.len());
        assert!(
            weights.iter().all(|&w| w > 0.0),
            "all dataset weights must be positive, got: {weights:?}"
        );
        self.weights = weights;
        self
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
        let n_total: usize = self.cached_predicted.iter().map(|p| p.len()).sum();
        let mut out = Array1::zeros(n_total);
        let mut offset = 0;
        for pred in &self.cached_predicted {
            let n = pred.len();
            out.slice_mut(s![offset..offset + n]).assign(pred);
            offset += n;
        }
        out
    }

    /// Convergence mask across all datasets.
    pub fn convergence_mask(&self) -> Array1<bool> {
        let n_total: usize = self.cached_converged.iter().map(|c| c.len()).sum();
        let mut out = Array1::from_elem(n_total, false);
        let mut offset = 0;
        for conv in &self.cached_converged {
            let n = conv.len();
            out.slice_mut(s![offset..offset + n]).assign(conv);
            offset += n;
        }
        out
    }

    /// Per-dataset `(n_converged, n_total)`.
    pub fn convergence_stats(&self) -> Vec<(usize, usize)> {
        self.cached_converged
            .iter()
            .map(|conv| (conv.iter().filter(|&&c| c).count(), conv.len()))
            .collect()
    }
}

macro_rules! impl_solver {
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
            ) -> Result<Self, FittingError> {
                if datasets.is_empty() {
                    return Err(FittingError::EmptyDatasets);
                }

                // Reject duplicate dataset names
                let mut seen_names = std::collections::HashSet::new();
                for d in &datasets {
                    if !seen_names.insert(d.name().to_string()) {
                        return Err(FittingError::DuplicateDatasetName {
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
                        return Err(FittingError::DuplicateParameter {
                            name: name.to_string(),
                        });
                    }
                    let i = canonical.iter().position(|&n| n == name).ok_or_else(|| {
                        FittingError::UnknownParameter {
                            name: name.to_string(),
                            valid: canonical.clone(),
                        }
                    })?;
                    if !differentiable.contains(&name) {
                        return Err(FittingError::NonDifferentiableParameter {
                            name: name.to_string(),
                            differentiable: differentiable.clone(),
                        });
                    }
                    fitted_indices.push(i);
                }

                // Get values for all parameters in canonical order.
                let mut full_params = Vec::with_capacity(canonical.len());
                for &name in &canonical {
                    let v = params
                        .get(name)
                        .ok_or_else(|| FittingError::MissingParameter {
                            name: name.to_string(),
                        })?;
                    full_params.push(*v);
                }

                let param_names: Vec<String> = fit.iter().map(|s| s.to_string()).collect();
                let p = param_names.len();
                let n_ds = datasets.len();

                // We normalize parameters in optimizer state to be in the order of 1.
                // Hence, scaling factors are the inital parameters.
                let normalization: Vec<f64> =
                    fitted_indices.iter().map(|&i| full_params[i]).collect();

                // Initial fitting parameters with 0.0 are rejected
                // since we divide for normalization.
                for (j, &idx) in fitted_indices.iter().enumerate() {
                    if normalization[j] == 0.0 {
                        return Err(FittingError::ZeroParameter {
                            name: canonical[idx].to_string(),
                        });
                    }
                }

                let mut solver = Self {
                    datasets,
                    weights: vec![1.0; n_ds],
                    param_names,
                    fitted_indices,
                    base_params: full_params,
                    normalization,
                    loss: LossFunction::default(),
                    strategy: NonConvergenceStrategy::default(),
                    params: DVector::from_element(p, 1.0f64),
                    cached_predicted: vec![Array1::zeros(0); n_ds],
                    cached_gradients: vec![Array2::zeros((0, p)); n_ds],
                    cached_converged: vec![Array1::from_elem(0, false); n_ds],
                    cached_loss_w_sqrt: vec![Array1::zeros(0); n_ds],
                    cached_penalty: 0.0,
                    _phantom: PhantomData,
                };
                solver.update_cache();
                Ok(solver)
            }

            /// Set the loss function.
            pub fn with_loss(mut self, loss: LossFunction) -> Self {
                self.loss = loss;
                self.recompute_loss_weights();
                self
            }

            /// Run Levenberg-Marquardt to convergence.
            pub fn fit(mut self, config: FitConfig) -> (Self, MinimizationReport<f64>) {
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
            /// Pass [`FitResult::all_parameters`] as `params` to inspect the
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

            /// Extract [`FitResult`] from the current solver state and a [`MinimizationReport`].
            pub fn to_result(&self, report: &MinimizationReport<f64>) -> FitResult {
                FitResult {
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

            /// Target values across all datasets in residual-vector order.
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
                    .enumerate()
                    .map(|(k, dataset)| {
                        let exp = dataset.target();
                        let pred = &self.cached_predicted[k];
                        let conv = &self.cached_converged[k];
                        let n_conv = conv.iter().filter(|&&c| c).count();
                        if n_conv == 0 {
                            return None;
                        }
                        Some(
                            pred.iter()
                                .zip(exp.iter())
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

            /// Unweighted relative Jacobian in physical parameter space
            ///
            /// Shape: `[n_converged, P]`
            /// Row: `(1/exp_i) · ∂property_i/∂x_physical`
            /// FIM: `J^T J`
            ///
            /// The Jacobian is in physical parameter space so that `J^T J` can
            /// used without any rescaling. Includes converged points only.
            /// The Jacobian used during LM iterations is scaled according to
            /// parameter normalization and loss definition.
            pub fn fim_jacobian(&self) -> Option<DMatrix<f64>> {
                let n_conv: usize = self
                    .cached_converged
                    .iter()
                    .flat_map(|c| c.iter())
                    .filter(|&&c| c)
                    .count();
                if n_conv == 0 {
                    return None;
                }

                let p = self.params.len();
                let mut jac = DMatrix::zeros(n_conv, p);

                let converged_pts: Vec<(usize, usize)> = self
                    .datasets
                    .iter()
                    .enumerate()
                    .flat_map(|(k, dataset)| {
                        (0..dataset.target().len())
                            .filter(move |&i| self.cached_converged[k][i])
                            .map(move |i| (k, i))
                    })
                    .collect();

                for j in 0..p {
                    for (row, &(k, i)) in converged_pts.iter().enumerate() {
                        let exp = self.datasets[k].target();
                        jac[(row, j)] = self.cached_gradients[k][[i, j]] / exp[i];
                    }
                }
                Some(jac)
            }

            /// Penalty value for non-converged points under the current strategy.
            fn recompute_loss_weights(&mut self) {
                for (k, dataset) in self.datasets.iter().enumerate() {
                    let exp = dataset.target();
                    let pred = &self.cached_predicted[k];
                    let conv = &self.cached_converged[k];
                    self.cached_loss_w_sqrt[k] = Array1::from_iter((0..exp.len()).map(|i| {
                        if conv[i] {
                            self.loss.irls_weight_sqrt((pred[i] - exp[i]) / exp[i])
                        } else {
                            0.0
                        }
                    }));
                }
            }

            fn residuals_impl(&self) -> Option<DVector<f64>> {
                let n_total: usize = self.datasets.iter().map(|d| d.target().len()).sum();
                if n_total == 0 {
                    return None;
                }

                let mut out = DVector::zeros(n_total);
                let mut offset = 0;
                for (k, dataset) in self.datasets.iter().enumerate() {
                    let exp = dataset.target();
                    let pred = &self.cached_predicted[k];
                    let conv = &self.cached_converged[k];
                    let w_sqrt = &self.cached_loss_w_sqrt[k];
                    let weight = self.weights[k];
                    for i in 0..exp.len() {
                        out[offset + i] = if conv[i] {
                            weight * w_sqrt[i] * (pred[i] - exp[i]) / exp[i]
                        } else {
                            weight * self.cached_penalty
                        };
                    }
                    offset += exp.len();
                }
                Some(out)
            }

            /// Jacobian for LM iteration.
            ///
            /// Needs derivatives that account for
            /// - using relative deviation
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

                // This part is a bit ugly
                // We have to convert between ndarray and nalgebra memory layouts.
                // Results from EoS calls: ndarray
                // Format needed by LM: nalgebra
                for (k, dataset) in self.datasets.iter().enumerate() {
                    let exp = dataset.target();
                    let conv = &self.cached_converged[k];
                    let w_sqrt = &self.cached_loss_w_sqrt[k];
                    let weight = self.weights[k];
                    let grad = &self.cached_gradients[k];
                    let n = exp.len();

                    let mut jac_block = jac.rows_mut(offset, n);

                    // Lenthy zip - but no additional allocations!
                    let row_iter = exp
                        .iter()
                        .zip(conv.iter())
                        .zip(w_sqrt.iter())
                        .zip(grad.outer_iter())
                        .zip(jac_block.row_iter_mut());

                    for ((((&e, &c), &w), grad_row), mut jac_row) in row_iter {
                        if !c {
                            continue;
                        }
                        // factor covers:
                        // - dataset weight
                        // - division by experimental value
                        // - irls weight (1.0 for L2 loss)
                        let factor = weight * w / e;
                        for (jac_val, &g) in jac_row.iter_mut().zip(grad_row.iter()) {
                            *jac_val = factor * g;
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
                // Compute physical parameters: x_physical[j] = x_opt[j] * normalization[j],
                // embedded into the full M-dimensional parameter vector.
                let mut full_params = self.base_params.clone();
                for (j, &idx) in self.fitted_indices.iter().enumerate() {
                    full_params[idx] = self.params[j] * self.normalization[j];
                }

                for (k, dataset) in self.datasets.iter().enumerate() {
                    let (pred, grad, conv) = dataset.compute::<T>(&self.param_names, &full_params);
                    let exp = dataset.target();
                    self.cached_loss_w_sqrt[k] = Array1::from_iter((0..exp.len()).map(|i| {
                        if conv[i] {
                            self.loss.irls_weight_sqrt((pred[i] - exp[i]) / exp[i])
                        } else {
                            0.0
                        }
                    }));
                    self.cached_predicted[k] = pred;
                    self.cached_gradients[k] = grad;
                    self.cached_converged[k] = conv;
                }

                // Update cached_penalty from the freshly computed predictions.
                // For AdaptivePenalty this is a cheap O(n) scan over hot cache;
                // for the other variants it is a direct read of the strategy.
                self.cached_penalty = match &self.strategy {
                    NonConvergenceStrategy::Ignore => 0.0,
                    NonConvergenceStrategy::Penalty(p) => *p,
                    NonConvergenceStrategy::AdaptivePenalty(factor) => {
                        let mut max_r = 0.0f64;
                        let mut any_converged = false;
                        for (k, dataset) in self.datasets.iter().enumerate() {
                            let exp = dataset.target();
                            let pred = &self.cached_predicted[k];
                            let conv = &self.cached_converged[k];
                            for i in 0..exp.len() {
                                if conv[i] {
                                    any_converged = true;
                                    max_r = max_r.max(((pred[i] - exp[i]) / exp[i]).abs());
                                }
                            }
                        }
                        if any_converged {
                            max_r * factor
                        } else {
                            *factor
                        }
                    }
                };
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

impl_solver!(1, PureDataset);
impl_solver!(2, BinaryDataset);

/// Object-safe interface for type-erased solvers.
///
/// A `Box<dyn DynSolver>` can hold any concrete `Regressor<T, D>` without
/// exposing the EoS type.
///
/// This is useful when models are created at runtime (used in py-feos).
///
/// # Example
///
/// ```ignore
/// let mut solver: Box<dyn DynSolver> = match model_name {
///     "eos_1" => Box::new(PureRegressor::<Eos1>::new(ds, params, &fit)?),
///     "eos_2" => Box::new(PureRegressor::<Eos2>::new(ds, params, &fit)?),
///     _ => panic!("unknown model"),
/// };
/// let result = solver.fit(FitConfig::default(), None);
/// println!("AAD: {:?}", result.aad_per_dataset);
/// ```
pub trait DynSolver: Send + Sync {
    /// Evaluate all datasets at the given full physical parameters.
    ///
    /// Returns `(predicted, gradients, converged)` concatenated across
    /// all datasets.  Gradients are w.r.t. the fitted parameters only.
    fn predict(&self, params: &[f64]) -> (Array1<f64>, Array2<f64>, Array1<bool>);

    /// Run Levenberg-Marquardt optimisation and return a [`FitResult`].
    ///
    /// Does not consume the solver. This is different from LM.
    fn fit(&mut self, config: FitConfig, loss: Option<LossFunction>) -> FitResult;

    /// Concatenated target values across all datasets.
    fn target(&self) -> Array1<f64>;

    /// All canonical parameter names in order.
    fn all_param_names(&self) -> Vec<String>;

    /// Names of the parameters being optimised.
    fn fitted_param_names(&self) -> Vec<String>;

    /// Dataset names in order.
    fn dataset_names(&self) -> Vec<String>;

    /// Evaluate each dataset at the given full parameters and return
    /// structured per-dataset results.
    ///
    /// See [`Regressor::evaluate_datasets`] for details.
    fn evaluate_datasets(&self, params: &[f64]) -> Vec<DatasetResult>;
}

macro_rules! impl_dyn_solver {
    ($n:literal, $dataset:ty) => {
        impl<T> DynSolver for Regressor<T, $dataset>
        where
            T: ParametersAD<$n> + Send + Sync + 'static,
        {
            fn predict(&self, params: &[f64]) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
                self.predict(params)
            }

            fn fit(&mut self, config: FitConfig, loss: Option<LossFunction>) -> FitResult {
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

            fn evaluate_datasets(&self, params: &[f64]) -> Vec<DatasetResult> {
                self.evaluate_datasets(params)
            }
        }
    };
}

impl_dyn_solver!(1, PureDataset);
impl_dyn_solver!(2, BinaryDataset);
