use crate::{DFTProfile, HelmholtzEnergyFunctional};
use feos_core::{FeosError, FeosResult, Verbosity, log_iter, log_result};
use indexmap::IndexMap;
use nalgebra::{DMatrix, DVector};
use ndarray::prelude::*;
use quantity::{MILLI, SECOND, Time};
use std::collections::VecDeque;
use std::fmt;
use std::time::{Duration, Instant};

mod euler_lagrange;
mod newton;
mod tape;

pub(crate) use newton::{NonlinearSystem, newton_krylov, solve_linearized};
pub use tape::{Evaluate, LocalFunction, Tape};
pub(crate) use tape::{flatten, unflatten};

/// Damping coefficient of the successive substitution for X in the Picard iteration. In the bulk,
/// the error is reduced by the factor 1 - alpha (2 - X) ≈ X/2 per step, for a contact-shell kernel
/// at least by about 0.4.
const ASSOCIATION_DAMPING: f64 = 0.5;

const DEFAULT_PARAMS_PICARD: PicardIteration = PicardIteration {
    log: false,
    max_iter: 500,
    tol: 1e-11,
    damping_coefficient: None,
};
const DEFAULT_PARAMS_ANDERSON: AndersonMixing = AndersonMixing {
    log: false,
    max_iter: 150,
    tol: 1e-11,
    damping_coefficient: 0.15,
    mmax: 100,
};
const DEFAULT_PARAMS_NEWTON: Newton = Newton {
    max_iter: 50,
    max_iter_gmres: 200,
    tol: 1e-11,
};

#[derive(Clone, Copy, Debug)]
struct PicardIteration {
    log: bool,
    max_iter: usize,
    tol: f64,
    damping_coefficient: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct AndersonMixing {
    log: bool,
    max_iter: usize,
    tol: f64,
    damping_coefficient: f64,
    mmax: usize,
}

#[derive(Clone, Copy, Debug)]
struct Newton {
    max_iter: usize,
    max_iter_gmres: usize,
    tol: f64,
}

#[derive(Clone, Copy)]
enum DFTAlgorithm {
    PicardIteration(PicardIteration),
    AndersonMixing(AndersonMixing),
    Newton(Newton),
}

/// Settings for the DFT solver.
#[derive(Clone)]
pub struct DFTSolver {
    algorithms: Vec<DFTAlgorithm>,
    pub verbosity: Verbosity,
}

impl Default for DFTSolver {
    fn default() -> Self {
        Self {
            algorithms: vec![DFTAlgorithm::Newton(DEFAULT_PARAMS_NEWTON)],
            verbosity: Default::default(),
        }
    }
}

impl DFTSolver {
    pub fn new(verbosity: Option<Verbosity>) -> Self {
        Self {
            algorithms: vec![],
            verbosity: verbosity.unwrap_or_default(),
        }
    }

    pub fn picard_iteration(
        mut self,
        log: Option<bool>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        damping_coefficient: Option<f64>,
    ) -> Self {
        let mut params = DEFAULT_PARAMS_PICARD;
        params.log = log.unwrap_or(params.log);
        params.max_iter = max_iter.unwrap_or(params.max_iter);
        params.tol = tol.unwrap_or(params.tol);
        params.damping_coefficient = damping_coefficient;
        self.algorithms.push(DFTAlgorithm::PicardIteration(params));
        self
    }

    pub fn anderson_mixing(
        mut self,
        log: Option<bool>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        damping_coefficient: Option<f64>,
        mmax: Option<usize>,
    ) -> Self {
        let mut params = DEFAULT_PARAMS_ANDERSON;
        params.log = log.unwrap_or(params.log);
        params.max_iter = max_iter.unwrap_or(params.max_iter);
        params.tol = tol.unwrap_or(params.tol);
        params.damping_coefficient = damping_coefficient.unwrap_or(params.damping_coefficient);
        params.mmax = mmax.unwrap_or(params.mmax);
        self.algorithms.push(DFTAlgorithm::AndersonMixing(params));
        self
    }

    pub fn newton(
        mut self,
        max_iter: Option<usize>,
        max_iter_gmres: Option<usize>,
        tol: Option<f64>,
    ) -> Self {
        let mut params = DEFAULT_PARAMS_NEWTON;
        params.max_iter = max_iter.unwrap_or(params.max_iter);
        params.max_iter_gmres = max_iter_gmres.unwrap_or(params.max_iter_gmres);
        params.tol = tol.unwrap_or(params.tol);
        self.algorithms.push(DFTAlgorithm::Newton(params));
        self
    }
}

/// A log that stores the residuals and execution time of DFT solvers.
#[derive(Clone)]
pub struct DFTSolverLog {
    start_time: Instant,
    residual: Vec<f64>,
    time: Vec<Duration>,
    solver: Vec<&'static str>,
    profile: IndexMap<&'static str, Duration>,
}

impl Default for DFTSolverLog {
    fn default() -> Self {
        Self::new()
    }
}

impl DFTSolverLog {
    pub(crate) fn new() -> Self {
        Self {
            start_time: Instant::now(),
            residual: Vec::new(),
            time: Vec::new(),
            solver: Vec::new(),
            profile: IndexMap::new(),
        }
    }

    fn add_residual(
        &mut self,
        solver: &'static str,
        iteration: usize,
        residual: f64,
        verbosity: Verbosity,
    ) {
        if iteration == 0 {
            log_iter!(verbosity, "{:-<59}", "");
        }
        self.solver.push(solver);
        self.residual.push(residual);
        let time = self.start_time.elapsed();
        self.time.push(self.start_time.elapsed());
        log_iter!(
            verbosity,
            "{:22} | {:>4} | {:7.3} | {:.6e}",
            solver,
            iteration,
            time.as_secs_f64() * SECOND,
            residual,
        );
    }

    pub fn time_function<F: FnOnce() -> O, O>(&mut self, key: &'static str, f: F) -> O {
        let start = Instant::now();
        let output = f();
        *self.profile.entry(key).or_insert(Duration::ZERO) += start.elapsed();
        output
    }

    pub fn residual(&self) -> ArrayView1<'_, f64> {
        (&self.residual).into()
    }

    pub fn time(&self) -> Time<Array1<f64>> {
        self.time.iter().map(|t| t.as_secs_f64() * SECOND).collect()
    }

    pub fn solver(&self) -> &[&'static str] {
        &self.solver
    }

    pub fn profile(&self) -> IndexMap<&'static str, Time> {
        self.profile
            .iter()
            .map(|(&k, t)| (k, t.as_secs_f64() * SECOND))
            .collect()
    }
}

impl fmt::Display for DFTSolverLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_time = self.time.last().unwrap_or(&Duration::ZERO).as_secs_f64() * SECOND;
        let (unit, u) = if total_time > SECOND {
            (SECOND, "s")
        } else {
            (MILLI * SECOND, "ms")
        };
        let max_len = self.profile.keys().map(|k| k.len()).max().unwrap_or(0) + 1;
        let mut time_sum = 0.0 * SECOND;
        for (&k, &t) in self.profile.iter() {
            let t = t.as_secs_f64() * SECOND;
            time_sum += t;
            writeln!(
                f,
                "Time spent calculating {:max_len$} {:6.2} {}  ({:5.2} %)",
                format!("{k}:"),
                t.convert_into(unit),
                u,
                t.convert_into(total_time) * 100.
            )?;
        }
        writeln!(f, "                       {:max_len$} {:-<20}", "", "")?;
        writeln!(
            f,
            "                       {:max_len$} {:6.2} {}  ({:5.2} %)",
            "",
            time_sum.convert_into(unit),
            u,
            time_sum.convert_into(total_time) * 100.
        )
    }
}

impl<D: Dimension + 'static, F: HelmholtzEnergyFunctional> DFTProfile<D, F>
where
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    pub(crate) fn call_solver(
        &self,
        rho: &mut Array<f64, D::Larger>,
        non_bonded_fractions: &mut Array<f64, D::Larger>,
        solver: &DFTSolver,
        solver_log: &mut DFTSolverLog,
        debug: bool,
    ) -> FeosResult<()> {
        log_iter!(
            solver.verbosity,
            "solver                 | iter |    time    | residual "
        );
        let mut converged = false;
        let mut iterations = 0;
        for algorithm in &solver.algorithms {
            let (conv, iter) = match algorithm {
                DFTAlgorithm::PicardIteration(picard) => self.solve_picard(
                    *picard,
                    rho,
                    non_bonded_fractions,
                    solver_log,
                    solver.verbosity,
                ),
                DFTAlgorithm::AndersonMixing(anderson) => self.solve_anderson(
                    *anderson,
                    rho,
                    non_bonded_fractions,
                    solver_log,
                    solver.verbosity,
                ),
                DFTAlgorithm::Newton(newton) => self.solve_newton(
                    *newton,
                    rho,
                    non_bonded_fractions,
                    solver_log,
                    solver.verbosity,
                ),
            }?;
            converged = conv;
            iterations += iter;
        }
        if converged {
            log_result!(solver.verbosity, "DFT solved in {} iterations", iterations);
        } else if debug {
            log_result!(
                solver.verbosity,
                "DFT not converged in {} iterations",
                iterations
            );
        } else {
            return Err(FeosError::NotConverged(String::from("DFT")));
        }
        Ok(())
    }

    fn solve_picard(
        &self,
        picard: PicardIteration,
        rho: &mut Array<f64, D::Larger>,
        non_bonded_fractions: &mut Array<f64, D::Larger>,
        solver_log: &mut DFTSolverLog,
        verbosity: Verbosity,
    ) -> FeosResult<(bool, usize)> {
        let solver = if picard.log {
            "Picard iteration (log)"
        } else {
            "Picard iteration"
        };

        for k in 0..picard.max_iter {
            // calculate residual
            let residual =
                self.euler_lagrange_equation(rho, non_bonded_fractions, picard.log, solver_log)?;
            solver_log.add_residual(solver, k, residual.norm, verbosity);

            // check for convergence
            if residual.norm < picard.tol {
                return Ok((true, k));
            }

            // apply line search (on the density residual at fixed X) or constant damping
            let res = residual.density;
            let damping_coefficient = picard.damping_coefficient.map_or_else(
                || {
                    self.line_search(
                        rho,
                        non_bonded_fractions,
                        &res,
                        residual.density_norm,
                        picard.log,
                        solver_log,
                    )
                },
                Ok,
            )?;

            // update the fractions of non-bonded association sites (damped successive
            // substitution, ln X <- ln X + alpha (ln X_projected - ln X))
            *non_bonded_fractions *= &residual
                .mass_action
                .mapv(|r| (ASSOCIATION_DAMPING * r).exp());

            // update solution
            if picard.log {
                *rho *= &(&res * damping_coefficient).mapv(f64::exp);
            } else {
                *rho += &(&res * damping_coefficient);
            }
        }
        Ok((false, picard.max_iter))
    }

    fn line_search(
        &self,
        rho: &Array<f64, D::Larger>,
        non_bonded_fractions: &Array<f64, D::Larger>,
        delta_rho: &Array<f64, D::Larger>,
        res0: f64,
        logarithm: bool,
        solver_log: &mut DFTSolverLog,
    ) -> FeosResult<f64> {
        let mut alpha = 2.0;

        // reduce step until a feasible solution is found
        for _ in 0..8 {
            alpha *= 0.5;

            // calculate full step
            let rho_new = if logarithm {
                rho * (alpha * delta_rho).mapv(f64::exp)
            } else {
                rho + alpha * delta_rho
            };
            let Ok(residual) =
                self.euler_lagrange_equation(&rho_new, non_bonded_fractions, logarithm, solver_log)
            else {
                continue;
            };
            let res2 = residual.density_norm;
            if res2 > res0 {
                continue;
            }

            // calculate intermediate step
            let rho_new = if logarithm {
                rho * (0.5 * alpha * delta_rho).mapv(f64::exp)
            } else {
                rho + 0.5 * alpha * delta_rho
            };
            let Ok(residual) =
                self.euler_lagrange_equation(&rho_new, non_bonded_fractions, logarithm, solver_log)
            else {
                continue;
            };
            let res1 = residual.density_norm;

            // estimate minimum
            let mut alpha_opt = if res2 - 2.0 * res1 + res0 != 0.0 {
                alpha * 0.25 * (res2 - 4.0 * res1 + 3.0 * res0) / (res2 - 2.0 * res1 + res0)
            } else {
                continue;
            };

            // prohibit negative steps
            if alpha_opt <= 0.0 {
                alpha_opt = if res1 < res2 { 0.5 * alpha } else { alpha };
            }

            // prohibit too large steps
            if alpha_opt > alpha {
                alpha_opt = alpha;
            }
            alpha = alpha_opt;
            break;
        }
        Ok(alpha)
    }

    fn solve_anderson(
        &self,
        anderson: AndersonMixing,
        rho: &mut Array<f64, D::Larger>,
        non_bonded_fractions: &mut Array<f64, D::Larger>,
        solver_log: &mut DFTSolverLog,
        verbosity: Verbosity,
    ) -> FeosResult<(bool, usize)> {
        let solver = if anderson.log {
            "Anderson mixing (log)"
        } else {
            "Anderson mixing"
        };

        let mut resm = VecDeque::with_capacity(anderson.mmax);
        let mut rhom = VecDeque::with_capacity(anderson.mmax);
        let mut xm = VecDeque::with_capacity(anderson.mmax);

        for k in 0..anderson.max_iter {
            // drop old values
            if resm.len() == anderson.mmax {
                resm.pop_front();
                rhom.pop_front();
                xm.pop_front();
            }
            let m = resm.len() + 1;

            // calculate residual
            let residual =
                self.euler_lagrange_equation(rho, non_bonded_fractions, anderson.log, solver_log)?;
            solver_log.add_residual(solver, k, residual.norm, verbosity);

            // check for convergence
            if residual.norm < anderson.tol {
                return Ok((true, k));
            }

            // save residuals and unknowns (the residual for ln X is that of successive substitution)
            resm.push_back((residual.density, residual.mass_action));
            if anderson.log {
                rhom.push_back(rho.mapv(f64::ln));
            } else {
                rhom.push_back(rho.clone());
            }
            xm.push_back(non_bonded_fractions.mapv(f64::ln));

            // calculate alpha
            solver_log.time_function("Anderson mixing", || {
                let r = DMatrix::from_fn(m + 1, m + 1, |i, j| match (i == m, j == m) {
                    (false, false) => {
                        let (resi, resxi) = &resm[i];
                        let (resj, resxj) = &resm[j];
                        (resi * resj).sum() + (resxi * resxj).sum()
                    }
                    (true, true) => 0.0,
                    _ => 1.0,
                });
                let mut alpha = DVector::zeros(m + 1);
                alpha[m] = 1.0;
                let alpha = r.lu().solve(&alpha);
                let alpha =
                    alpha.ok_or(FeosError::Error("alpha matrix is not invertible".into()))?;

                // update solution
                rho.fill(0.0);
                non_bonded_fractions.fill(0.0);
                for i in 0..m {
                    let (resi, resxi) = &resm[i];
                    *rho += &(alpha[i] * (&rhom[i] + &(anderson.damping_coefficient * resi)));
                    *non_bonded_fractions +=
                        &(alpha[i] * (&xm[i] + &(anderson.damping_coefficient * resxi)));
                }
                if anderson.log {
                    rho.mapv_inplace(f64::exp);
                } else {
                    rho.mapv_inplace(f64::abs);
                }
                non_bonded_fractions.mapv_inplace(f64::exp);
                Ok::<_, FeosError>(())
            })?;
        }
        Ok((false, anderson.max_iter))
    }

    fn solve_newton(
        &self,
        newton: Newton,
        rho: &mut Array<f64, D::Larger>,
        non_bonded_fractions: &mut Array<f64, D::Larger>,
        solver_log: &mut DFTSolverLog,
        verbosity: Verbosity,
    ) -> FeosResult<(bool, usize)> {
        let mut x = Self::stack_variables(rho, non_bonded_fractions);

        let result = newton_krylov(
            self,
            &mut x,
            newton.max_iter,
            newton.max_iter_gmres,
            newton.tol,
            solver_log,
            verbosity,
        );

        *rho = self.density_rows(&x).mapv(f64::exp);
        *non_bonded_fractions = self.association_rows(&x).mapv(f64::exp);
        result
    }
}

impl fmt::Display for DFTAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::PicardIteration(picard) => write!(f, "{picard:?}"),
            Self::AndersonMixing(anderson) => write!(f, "{anderson:?}"),
            Self::Newton(newton) => write!(f, "{newton:?}"),
        }
    }
}

impl fmt::Display for DFTSolver {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "DFTSolver(")?;
        for algorithm in &self.algorithms {
            writeln!(f, "    {algorithm}")?;
        }
        writeln!(f, ")")
    }
}

impl DFTSolver {
    pub fn _repr_markdown_(&self) -> String {
        let mut res = String::from("|solver|max_iter|tol|\n|-|-:|-:|");
        for algorithm in &self.algorithms {
            let (solver, max_iter, tol) = match algorithm {
                DFTAlgorithm::PicardIteration(picard) => (
                    format!(
                        "Picard iteration ({}{})",
                        if picard.log { "log, " } else { "" },
                        match picard.damping_coefficient {
                            None => "line search".into(),
                            Some(damping_coefficient) =>
                                format!("damping_coefficient={damping_coefficient}"),
                        }
                    ),
                    picard.max_iter,
                    picard.tol,
                ),
                DFTAlgorithm::AndersonMixing(anderson) => (
                    format!(
                        "Anderson mixing ({}damping_coefficient={}, mmax={})",
                        if anderson.log { "log, " } else { "" },
                        anderson.damping_coefficient,
                        anderson.mmax
                    ),
                    anderson.max_iter,
                    anderson.tol,
                ),
                DFTAlgorithm::Newton(newton) => (
                    format!("Newton (max_iter_gmres={})", newton.max_iter_gmres),
                    newton.max_iter,
                    newton.tol,
                ),
            };
            res += &format!("\n|{solver}|{max_iter}|{tol:e}|");
        }
        res
    }
}
