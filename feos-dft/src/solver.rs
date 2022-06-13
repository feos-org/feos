use feos_core::{log_iter, EosError, EosResult, Verbosity};
use ndarray::prelude::*;
use num_dual::linalg::{norm, LU};
use std::collections::VecDeque;
use std::fmt;

const DEFAULT_PARAMS_PICARD: SolverParameter = SolverParameter {
    solver: DFTAlgorithm::PicardIteration(1.0),
    log: false,
    max_iter: 500,
    tol: 1e-11,
    beta: 0.15,
};
const DEFAULT_PARAMS_ANDERSON_LOG: SolverParameter = SolverParameter {
    solver: DFTAlgorithm::AndersonMixing(100),
    log: true,
    max_iter: 50,
    tol: 1e-5,
    beta: 0.15,
};
const DEFAULT_PARAMS_ANDERSON: SolverParameter = SolverParameter {
    solver: DFTAlgorithm::AndersonMixing(100),
    log: false,
    max_iter: 150,
    tol: 1e-11,
    beta: 0.15,
};

#[derive(Clone, Copy)]
struct SolverParameter {
    solver: DFTAlgorithm,
    log: bool,
    max_iter: usize,
    tol: f64,
    beta: f64,
}

#[derive(Clone, Copy)]
enum DFTAlgorithm {
    PicardIteration(f64),
    AndersonMixing(usize),
}

/// Settings for the DFT solver.
#[derive(Clone)]
pub struct DFTSolver {
    parameters: Vec<SolverParameter>,
    pub verbosity: Verbosity,
}

impl Default for DFTSolver {
    fn default() -> Self {
        Self {
            parameters: vec![DEFAULT_PARAMS_ANDERSON_LOG, DEFAULT_PARAMS_ANDERSON],
            verbosity: Verbosity::None,
        }
    }
}

impl DFTSolver {
    /// Create a new empty `DFTSolver` object.
    pub fn new(verbosity: Verbosity) -> Self {
        Self {
            parameters: Vec::new(),
            verbosity,
        }
    }

    /// Add a Picard iteration to the solver.
    pub fn picard_iteration(mut self, max_rel: Option<f64>) -> Self {
        let mut algorithm = DEFAULT_PARAMS_PICARD;
        if let Some(max_rel) = max_rel {
            algorithm.solver = DFTAlgorithm::PicardIteration(max_rel);
        }
        self.parameters.push(algorithm);
        self
    }

    /// Add Anderson mixing to the solver.
    pub fn anderson_mixing(mut self, mmax: Option<usize>) -> Self {
        let mut algorithm = DEFAULT_PARAMS_ANDERSON;
        if let Some(mmax) = mmax {
            algorithm.solver = DFTAlgorithm::AndersonMixing(mmax);
        }
        self.parameters.push(algorithm);
        self
    }

    /// Iterate the logarithm of the density profile in the last solver.
    pub fn log(mut self) -> Self {
        self.parameters.last_mut().unwrap().log = true;
        self
    }

    /// Set the maximum number of iterations for the last solver.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.parameters.last_mut().unwrap().max_iter = max_iter;
        self
    }

    /// Set the tolerance for the last solver.
    pub fn tol(mut self, tol: f64) -> Self {
        self.parameters.last_mut().unwrap().tol = tol;
        self
    }

    /// Set the damping factor for the last solver
    pub fn beta(mut self, beta: f64) -> Self {
        self.parameters.last_mut().unwrap().beta = beta;
        self
    }

    pub(crate) fn solve<F>(&self, x: &mut Array1<f64>, residual: &mut F) -> EosResult<(bool, usize)>
    where
        F: FnMut(&Array1<f64>, ArrayViewMut1<f64>, bool) -> EosResult<()>,
    {
        log_iter!(self.verbosity, "solver               | iter | residual ");
        let mut converged = false;
        let mut iterations = 0;
        for algorithm in &self.parameters {
            let (c, i) = algorithm.solve(x, residual, self.verbosity)?;
            converged = c;
            iterations += i;
        }
        Ok((converged, iterations))
    }
}

impl SolverParameter {
    fn solve<F>(
        &self,
        x: &mut Array1<f64>,
        residual: &mut F,
        verbosity: Verbosity,
    ) -> EosResult<(bool, usize)>
    where
        F: FnMut(&Array1<f64>, ArrayViewMut1<f64>, bool) -> EosResult<()>,
    {
        match self.solver {
            DFTAlgorithm::PicardIteration(max_rel) => {
                self.solve_picard(max_rel, x, residual, verbosity)
            }
            DFTAlgorithm::AndersonMixing(mmax) => self.solve_anderson(mmax, x, residual, verbosity),
        }
    }

    fn solve_picard<F>(
        &self,
        max_rel: f64,
        x: &mut Array1<f64>,
        residual: &mut F,
        verbosity: Verbosity,
    ) -> EosResult<(bool, usize)>
    where
        F: FnMut(&Array1<f64>, ArrayViewMut1<f64>, bool) -> EosResult<()>,
    {
        log_iter!(verbosity, "{:-<43}", "");
        let mut resm = Array::zeros(x.raw_dim());

        for k in 1..=self.max_iter {
            // calculate residual
            residual(x, resm.view_mut(), self.log)?;

            // calculate beta
            let mut beta_min: Option<f64> = None;
            let beta = (max_rel * (&*x / &resm).mapv(f64::abs)).mapv(|beta| {
                if beta < self.beta {
                    beta_min = Some(beta_min.map_or(beta, |b| b.min(beta)));
                    beta
                } else {
                    self.beta
                }
            });

            // update solution
            if self.log {
                *x *= &(&resm * (-beta)).mapv(f64::exp);
            } else {
                *x -= &(&resm * beta);
            }

            // check for convergence
            let res = norm(&resm) / (resm.len() as f64).sqrt();
            log_iter!(
                verbosity,
                "Picard iteration {:3} | {:>4} | {:.6e} | {}",
                if self.log { "log" } else { "" },
                k,
                res,
                beta_min.unwrap_or(self.beta)
            );

            if res.is_nan() {
                return Err(EosError::IterationFailed(String::from("Picard Iteration")));
            }
            if res < self.tol && beta_min.is_none() {
                return Ok((true, k));
            }
        }
        Ok((false, self.max_iter))
    }

    fn solve_anderson<F>(
        &self,
        mmax: usize,
        x: &mut Array1<f64>,
        residual: &mut F,
        verbosity: Verbosity,
    ) -> EosResult<(bool, usize)>
    where
        F: FnMut(&Array1<f64>, ArrayViewMut1<f64>, bool) -> EosResult<()>,
    {
        log_iter!(verbosity, "{:-<43}", "");
        let mut resm = VecDeque::with_capacity(mmax);
        let mut xm = VecDeque::with_capacity(mmax);
        let mut r;
        let mut alpha;

        for k in 1..=self.max_iter {
            // drop old values
            if resm.len() == mmax {
                resm.pop_front();
                xm.pop_front();
            }
            let m = resm.len() + 1;

            // calculate residual
            let mut res = Array::zeros(x.raw_dim());
            residual(x, res.view_mut(), self.log)?;
            resm.push_back(res);

            // save x value
            if self.log {
                xm.push_back(x.mapv(f64::ln));
            } else {
                xm.push_back(x.clone());
            }

            // calculate alpha
            r = Array::from_shape_fn((m + 1, m + 1), |(i, j)| match (i == m, j == m) {
                (false, false) => (&resm[i] * &resm[j]).sum(),
                (true, true) => 0.0,
                _ => 1.0,
            });
            alpha = Array::zeros(m + 1);
            alpha[m] = 1.0;
            alpha = LU::new(r)?.solve(&alpha);
            // r.solveh_inplace(&mut alpha)?;

            // update solution
            x.fill(0.0);
            for i in 0..m {
                *x += &(alpha[i] * (&xm[i] - &(self.beta * &resm[i])));
            }
            if self.log {
                x.mapv_inplace(f64::exp);
            } else {
                x.mapv_inplace(f64::abs);
            }

            // check for convergence
            let resv = &resm[m - 1];
            let res = norm(resv) / (resv.len() as f64).sqrt();
            log_iter!(
                verbosity,
                "Anderson mixing {:3}  | {:>4} | {:.6e} ",
                if self.log { "log" } else { "" },
                k,
                res
            );

            if res.is_nan() {
                return Err(EosError::IterationFailed(String::from("Anderson Mixing")));
            }
            if res < self.tol {
                return Ok((true, k));
            }
        }
        Ok((false, self.max_iter))
    }
}

impl fmt::Display for DFTAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::PicardIteration(max_rel) => write!(f, "Picard Iteration (max_rel={})", max_rel),
            Self::AndersonMixing(mmax) => write!(f, "Anderson Mixing (mmax={})", mmax),
        }
    }
}

impl fmt::Debug for DFTAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::PicardIteration(_) => write!(f, "Picard Iteration"),
            Self::AndersonMixing(_) => write!(f, "Anderson Mixing"),
        }
    }
}

impl fmt::Display for SolverParameter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} max_iter: {}, tol: {}, beta: {}",
            self.solver, self.max_iter, self.tol, self.beta
        )
    }
}

impl fmt::Display for DFTSolver {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for algorithm in &self.parameters {
            writeln!(f, "{algorithm}")?;
        }
        Ok(())
    }
}

impl DFTSolver {
    pub fn _repr_markdown_(&self) -> String {
        let mut res =
            String::from("|solver|log|max_iter|tol|beta|mmax|max_rel|\n|-|:-:|-:|-:|-:|-:|-:|");
        for algorithm in &self.parameters {
            let (mmax, max_rel) = match algorithm.solver {
                DFTAlgorithm::PicardIteration(max_rel) => (String::new(), max_rel.to_string()),
                DFTAlgorithm::AndersonMixing(mmax) => (mmax.to_string(), String::new()),
            };
            res += &fmt::format(format_args!(
                "\n|{:?}|{}|{}|{:e}|{}|{}|{}|",
                algorithm.solver,
                if algorithm.log { "x" } else { "" },
                algorithm.max_iter,
                algorithm.tol,
                algorithm.beta,
                mmax,
                max_rel
            ));
        }
        res
    }
}
