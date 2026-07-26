use crate::Convolver;
use crate::{DFTProfile, HelmholtzEnergyFunctional};
use feos_core::{FeosError, FeosResult, ReferenceSystem, Verbosity, log_iter, log_result};
use indexmap::IndexMap;
use nalgebra::{DMatrix, DVector};
use ndarray::RemoveAxis;
use ndarray::prelude::*;
use quantity::{MILLI, SECOND, Time};
use std::collections::VecDeque;
use std::fmt;
use std::ops::AddAssign;
use std::time::{Duration, Instant};

const DEFAULT_PARAMS_PICARD: PicardIteration = PicardIteration {
    log: false,
    max_iter: 500,
    tol: 1e-11,
    damping_coefficient: None,
};
const DEFAULT_PARAMS_ANDERSON_LOG: AndersonMixing = AndersonMixing {
    log: true,
    max_iter: 50,
    tol: 1e-5,
    damping_coefficient: 0.15,
    mmax: 100,
};
const DEFAULT_PARAMS_ANDERSON: AndersonMixing = AndersonMixing {
    log: false,
    max_iter: 150,
    tol: 1e-11,
    damping_coefficient: 0.15,
    mmax: 100,
};
const DEFAULT_PARAMS_NEWTON: Newton = Newton {
    log: false,
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
    log: bool,
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
            algorithms: vec![
                DFTAlgorithm::AndersonMixing(DEFAULT_PARAMS_ANDERSON_LOG),
                DFTAlgorithm::AndersonMixing(DEFAULT_PARAMS_ANDERSON),
            ],
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
        log: Option<bool>,
        max_iter: Option<usize>,
        max_iter_gmres: Option<usize>,
        tol: Option<f64>,
    ) -> Self {
        let mut params = DEFAULT_PARAMS_NEWTON;
        params.log = log.unwrap_or(params.log);
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
        &mut self,
        rho: &mut Array<f64, D::Larger>,
        solver: &DFTSolver,
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
                DFTAlgorithm::PicardIteration(picard) => {
                    self.solve_picard(*picard, rho, solver.verbosity)
                }
                DFTAlgorithm::AndersonMixing(anderson) => {
                    self.solve_anderson(*anderson, rho, solver.verbosity)
                }
                DFTAlgorithm::Newton(newton) => self.solve_newton(*newton, rho, solver.verbosity),
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
        &mut self,
        picard: PicardIteration,
        rho: &mut Array<f64, D::Larger>,
        verbosity: Verbosity,
    ) -> FeosResult<(bool, usize)> {
        let solver = if picard.log {
            "Picard iteration (log)"
        } else {
            "Picard iteration"
        };

        for k in 0..picard.max_iter {
            // calculate residual
            let (res, res_norm, _, _, _) = self.euler_lagrange_equation(&*rho, picard.log)?;
            self.solver_log.add_residual(solver, k, res_norm, verbosity);

            // check for convergence
            if res_norm < picard.tol {
                return Ok((true, k));
            }

            // apply line search or constant damping
            let damping_coefficient = picard
                .damping_coefficient
                .map_or_else(|| self.line_search(rho, &res, res_norm, picard.log), Ok)?;

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
        &mut self,
        rho: &Array<f64, D::Larger>,
        delta_rho: &Array<f64, D::Larger>,
        res0: f64,
        logarithm: bool,
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
            let Ok((_, res2, _, _, _)) = self.euler_lagrange_equation(&rho_new, logarithm) else {
                continue;
            };
            if res2 > res0 {
                continue;
            }

            // calculate intermediate step
            let rho_new = if logarithm {
                rho * (0.5 * alpha * delta_rho).mapv(f64::exp)
            } else {
                rho + 0.5 * alpha * delta_rho
            };
            let Ok((_, res1, _, _, _)) = self.euler_lagrange_equation(&rho_new, logarithm) else {
                continue;
            };

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
        &mut self,
        anderson: AndersonMixing,
        rho: &mut Array<f64, D::Larger>,
        verbosity: Verbosity,
    ) -> FeosResult<(bool, usize)> {
        let solver = if anderson.log {
            "Anderson mixing (log)"
        } else {
            "Anderson mixing"
        };

        let mut resm = VecDeque::with_capacity(anderson.mmax);
        let mut rhom = VecDeque::with_capacity(anderson.mmax);

        for k in 0..anderson.max_iter {
            // drop old values
            if resm.len() == anderson.mmax {
                resm.pop_front();
                rhom.pop_front();
            }
            let m = resm.len() + 1;

            // calculate residual
            let (res, res_norm, _, _, _) = self.euler_lagrange_equation(&*rho, anderson.log)?;
            self.solver_log.add_residual(solver, k, res_norm, verbosity);

            // check for convergence
            if res_norm < anderson.tol {
                return Ok((true, k));
            }

            // save residual and x value
            resm.push_back((res, res_norm));
            if anderson.log {
                rhom.push_back(rho.mapv(f64::ln));
            } else {
                rhom.push_back(rho.clone());
            }

            // calculate alpha
            self.solver_log.time_function("Anderson mixing", || {
                let r = DMatrix::from_fn(m + 1, m + 1, |i, j| match (i == m, j == m) {
                    (false, false) => {
                        let (resi, _) = &resm[i];
                        let (resj, _) = &resm[j];
                        (resi * resj).sum()
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
                for i in 0..m {
                    let rhoi = &rhom[i];
                    let (resi, _) = &resm[i];
                    *rho += &(alpha[i] * (rhoi + &(anderson.damping_coefficient * resi)));
                }
                if anderson.log {
                    rho.mapv_inplace(f64::exp);
                } else {
                    rho.mapv_inplace(f64::abs);
                }
                Ok::<_, FeosError>(())
            })?;
        }
        Ok((false, anderson.max_iter))
    }

    fn solve_newton(
        &mut self,
        newton: Newton,
        rho: &mut Array<f64, D::Larger>,
        verbosity: Verbosity,
    ) -> FeosResult<(bool, usize)> {
        let solver = if newton.log { "Newton (log)" } else { "Newton" };
        for k in 0..newton.max_iter {
            // calculate initial residual
            let (res, res_norm, exp_dfdrho, z, rho_p) =
                self.euler_lagrange_equation(rho, newton.log)?;
            self.solver_log.add_residual(solver, k, res_norm, verbosity);

            // check convergence
            if res_norm < newton.tol {
                return Ok((true, k));
            }

            // calculate second partial derivatives once
            let second_partial_derivatives =
                self.solver_log
                    .time_function("second partial derivatives", || {
                        self.bulk.eos.second_partial_derivatives(
                            self.bulk.temperature.into_reduced(),
                            rho,
                            self.convolver.as_ref(),
                        )
                    })?;

            // define rhs function
            let rhs = |delta_rho: &_| {
                let mut delta_functional_derivative = Self::delta_functional_derivative(
                    delta_rho,
                    &second_partial_derivatives,
                    self.convolver.as_ref(),
                );
                delta_functional_derivative
                    .outer_iter_mut()
                    .zip(self.bulk.eos.m().iter())
                    .for_each(|(mut q, &m)| q /= m);
                let delta_i = self.bulk.eos.delta_bond_integrals(
                    self.bulk.temperature.into_reduced(),
                    &exp_dfdrho,
                    &delta_functional_derivative,
                    self.convolver.as_ref(),
                );
                let mut delta_exp_dfdrho = delta_functional_derivative - delta_i;
                let delta_z = -self
                    .grid
                    .integrate_reduced_comp(&(&delta_exp_dfdrho * &exp_dfdrho));

                let delta_fugacity = self.specification.delta_fugacity(&z, &delta_z);
                delta_exp_dfdrho
                    .outer_iter_mut()
                    .zip(delta_fugacity.iter())
                    .for_each(|(mut z, &f)| {
                        z -= f;
                    });

                let rho = if newton.log { &*rho } else { &rho_p };
                delta_rho + delta_exp_dfdrho * rho
            };

            // update solution
            let lhs = if newton.log { &*rho * res } else { res };
            *rho += &Self::gmres(
                rhs,
                &lhs,
                newton.max_iter_gmres,
                newton.tol * 1e-2,
                &mut self.solver_log,
            )?;
            rho.mapv_inplace(f64::abs);
        }

        Ok((false, newton.max_iter))
    }

    pub(crate) fn gmres<R>(
        rhs: R,
        r0: &Array<f64, D::Larger>,
        max_iter: usize,
        tol: f64,
        solver_log: &mut DFTSolverLog,
    ) -> FeosResult<Array<f64, D::Larger>>
    where
        R: Fn(&Array<f64, D::Larger>) -> Array<f64, D::Larger>,
    {
        // allocate vectors and arrays
        let mut v = Vec::with_capacity(max_iter);
        let mut h: Array2<f64> = Array::zeros([max_iter + 1; 2]);
        let mut c: Array1<f64> = Array::zeros(max_iter + 1);
        let mut s: Array1<f64> = Array::zeros(max_iter + 1);
        let mut gamma: Array1<f64> = Array::zeros(max_iter + 1);

        gamma[0] = (r0 * r0).sum().sqrt();
        v.push(r0 / gamma[0]);

        let mut iter = 0;
        for j in 0..max_iter {
            // calculate q=Av_j
            let mut q = solver_log.time_function("Jacobian vector product (GMRES)", || rhs(&v[j]));

            // calculate h_ij
            v.iter()
                .enumerate()
                .for_each(|(i, v_i)| h[(i, j)] = (v_i * &q).sum());

            // calculate w_j (stored in q)
            v.iter()
                .enumerate()
                .for_each(|(i, v_i)| q -= &(h[(i, j)] * v_i));
            h[(j + 1, j)] = (&q * &q).sum().sqrt();

            // update h_ij and h_i+1j
            if j > 0 {
                for i in 0..=j - 1 {
                    let temp = c[i + 1] * h[(i, j)] + s[i + 1] * h[(i + 1, j)];
                    h[(i + 1, j)] = -s[i + 1] * h[(i, j)] + c[i + 1] * h[(i + 1, j)];
                    h[(i, j)] = temp;
                }
            }

            // update auxiliary variables
            let beta = (h[(j, j)] * h[(j, j)] + h[(j + 1, j)] * h[(j + 1, j)]).sqrt();
            s[j + 1] = h[(j + 1, j)] / beta;
            c[j + 1] = h[(j, j)] / beta;
            h[(j, j)] = beta;
            gamma[j + 1] = -s[j + 1] * gamma[j];
            gamma[j] *= c[j + 1];

            // check for convergence
            if gamma[j + 1].abs() >= tol && j + 1 < max_iter {
                v.push(q / h[(j + 1, j)]);
                iter += 1;
            } else {
                break;
            }
        }
        // calculate solution vector
        let mut x = Array::zeros(r0.raw_dim());
        let mut y = Array::zeros(iter + 1);
        for i in (0..=iter).rev() {
            y[i] = (gamma[i] - (i + 1..=iter).map(|k| h[(i, k)] * y[k]).sum::<f64>()) / h[(i, i)];
        }
        v.iter().zip(y).for_each(|(v, y)| x += &(y * v));
        Ok(x)
    }

    pub(crate) fn delta_functional_derivative(
        delta_density: &Array<f64, D::Larger>,
        second_partial_derivatives: &[Array<f64, <D::Larger as Dimension>::Larger>],
        convolver: &dyn Convolver<f64, D>,
    ) -> Array<f64, D::Larger> {
        let delta_weighted_densities = convolver.weighted_densities(delta_density);
        let delta_partial_derivatives: Vec<_> = second_partial_derivatives
            .iter()
            .zip(delta_weighted_densities)
            .map(|(pd2, wd)| {
                let mut delta_partial_derivatives =
                    Array::zeros(pd2.raw_dim().remove_axis(Axis(0)));
                let n = wd.shape()[0];
                for i in 0..n {
                    for j in 0..n {
                        delta_partial_derivatives
                            .index_axis_mut(Axis(0), i)
                            .add_assign(
                                &(&pd2.index_axis(Axis(0), i).index_axis(Axis(0), j)
                                    * &wd.index_axis(Axis(0), j)),
                            );
                    }
                }
                delta_partial_derivatives
            })
            .collect();
        convolver.functional_derivative(&delta_partial_derivatives)
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
                    format!(
                        "Newton ({}max_iter_gmres={})",
                        if newton.log { "log, " } else { "" },
                        newton.max_iter_gmres
                    ),
                    newton.max_iter,
                    newton.tol,
                ),
            };
            res += &format!("\n|{solver}|{max_iter}|{tol:e}|");
        }
        res
    }
}
