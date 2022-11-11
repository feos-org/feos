use crate::{DFTProfile, HelmholtzEnergyFunctional};
use feos_core::{log_iter, EosResult, EosUnit, Verbosity};
use ndarray::prelude::*;
use ndarray::RemoveAxis;
use num_dual::linalg::LU;
use quantity::si::{SIArray1, SECOND};
use std::collections::VecDeque;
use std::fmt;
use std::ops::AddAssign;
use std::time::{Duration, Instant};

const DEFAULT_PARAMS_PICARD: PicardIteration = PicardIteration {
    log: false,
    max_iter: 500,
    tol: 1e-11,
    beta: None,
};
const DEFAULT_PARAMS_ANDERSON_LOG: AndersonMixing = AndersonMixing {
    log: true,
    max_iter: 50,
    tol: 1e-5,
    beta: 0.15,
    mmax: 100,
};
const DEFAULT_PARAMS_ANDERSON: AndersonMixing = AndersonMixing {
    log: false,
    max_iter: 150,
    tol: 1e-11,
    beta: 0.15,
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
    beta: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct AndersonMixing {
    log: bool,
    max_iter: usize,
    tol: f64,
    beta: f64,
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
        beta: Option<f64>,
    ) -> Self {
        let mut params = DEFAULT_PARAMS_PICARD;
        params.log = log.unwrap_or(params.log);
        params.max_iter = max_iter.unwrap_or(params.max_iter);
        params.tol = tol.unwrap_or(params.tol);
        params.beta = beta;
        self.algorithms.push(DFTAlgorithm::PicardIteration(params));
        self
    }

    pub fn anderson_mixing(
        mut self,
        log: Option<bool>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        beta: Option<f64>,
        mmax: Option<usize>,
    ) -> Self {
        let mut params = DEFAULT_PARAMS_ANDERSON;
        params.log = log.unwrap_or(params.log);
        params.max_iter = max_iter.unwrap_or(params.max_iter);
        params.tol = tol.unwrap_or(params.tol);
        params.beta = beta.unwrap_or(params.beta);
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

#[derive(Clone)]
pub struct DFTSolverLog {
    converged: bool,
    pub residual: Array1<f64>,
    pub time: SIArray1,
    pub damping: Option<Array1<f64>>,
}

impl DFTSolverLog {
    fn new(
        converged: bool,
        residual: Vec<f64>,
        time: Vec<Duration>,
        damping: Option<Vec<f64>>,
    ) -> Self {
        Self {
            converged,
            residual: Array1::from_vec(residual),
            time: time.into_iter().map(|d| d.as_secs_f64() * SECOND).collect(),
            damping: damping.map(Array1::from_vec),
        }
    }
}

impl<U: EosUnit, D: Dimension, F: HelmholtzEnergyFunctional> DFTProfile<U, D, F>
where
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    pub(crate) fn call_solver(
        &mut self,
        rho: &mut Array<f64, D::Larger>,
        rho_bulk: &mut Array1<f64>,
        solver: DFTSolver,
    ) -> EosResult<(bool, usize)> {
        let mut converged = false;
        let mut iterations = 0;
        for algorithm in solver.algorithms {
            let log = match algorithm {
                DFTAlgorithm::PicardIteration(params) => {
                    self.solve_picard(params, rho, rho_bulk, solver.verbosity)
                }
                DFTAlgorithm::AndersonMixing(params) => {
                    self.solve_anderson(params, rho, rho_bulk, solver.verbosity)
                }
                DFTAlgorithm::Newton(params) => {
                    self.solve_newton(params, rho, rho_bulk, solver.verbosity)
                }
            }?;
            converged = log.converged;
            iterations += log.residual.len();
            self.solver_log.push(log);
        }

        Ok((converged, iterations))
    }

    fn solve_picard(
        &self,
        picard: PicardIteration,
        rho: &mut Array<f64, D::Larger>,
        rho_bulk: &mut Array1<f64>,
        verbosity: Verbosity,
    ) -> EosResult<DFTSolverLog> {
        log_iter!(verbosity, "{:-<43}", "");
        let mut residual = Vec::new();
        let mut time = Vec::new();
        let mut damping = Vec::new();
        let start_time = Instant::now();

        for k in 0..picard.max_iter {
            // calculate residual
            let (res, res_bulk, res_norm) =
                self.euler_lagrange_equation(&*rho, &*rho_bulk, picard.log)?;
            time.push(start_time.elapsed());
            residual.push(res_norm);

            // check for convergence
            log_iter!(
                verbosity,
                "Picard iteration {:3} | {:>4} | {:.6e}",
                if picard.log { "log" } else { "" },
                k,
                res_norm,
            );

            if res_norm < picard.tol {
                return Ok(DFTSolverLog::new(true, residual, time, Some(damping)));
            }

            // apply line search or constant damping
            let beta = picard.beta.map_or_else(
                || self.line_search(rho, &res, rho_bulk, res_norm, picard.log, verbosity),
                Ok,
            )?;
            damping.push(beta);

            // update solution
            if picard.log {
                *rho *= &(&res * (-beta)).mapv(f64::exp);
                *rho_bulk *= &(&res_bulk * (-beta)).mapv(f64::exp);
            } else {
                *rho -= &(&res * beta);
                *rho_bulk -= &(&res_bulk * beta);
            }
        }
        Ok(DFTSolverLog::new(false, residual, time, Some(damping)))
    }

    fn line_search(
        &self,
        rho: &Array<f64, D::Larger>,
        delta_rho: &Array<f64, D::Larger>,
        rho_bulk: &Array1<f64>,
        res0: f64,
        log: bool,
        verbosity: Verbosity,
    ) -> EosResult<f64> {
        let mut alpha = 2.0;

        // reduce step until a feasible solution is found
        for _ in 0..8 {
            alpha *= 0.5;

            // calculate full step
            let rho_new = if log {
                rho * ((-alpha) * delta_rho).mapv(f64::exp)
            } else {
                rho - alpha * delta_rho
            };
            let Ok((_, _, res2)) =
                self.euler_lagrange_equation(&rho_new, rho_bulk, log) else {
                    continue;
            };
            if res2 > res0 {
                continue;
            }

            // calculate intermediate step
            let rho_new = if log {
                rho * ((-0.5 * alpha) * delta_rho).mapv(f64::exp)
            } else {
                rho - 0.5 * alpha * delta_rho
            };
            let Ok((_, _, res1)) =
                self.euler_lagrange_equation(&rho_new, rho_bulk, log) else {
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
        log_iter!(verbosity, "    Line search      | {} | ", alpha);
        Ok(alpha)
    }

    fn solve_anderson(
        &self,
        anderson: AndersonMixing,
        rho: &mut Array<f64, D::Larger>,
        rho_bulk: &mut Array1<f64>,
        verbosity: Verbosity,
    ) -> EosResult<DFTSolverLog> {
        log_iter!(verbosity, "{:-<43}", "");
        let mut residual = Vec::new();
        let mut time = Vec::new();
        let mut damping = Vec::new();
        let start_time = Instant::now();

        let mut resm = VecDeque::with_capacity(anderson.mmax);
        let mut rhom = VecDeque::with_capacity(anderson.mmax);
        let mut r;
        let mut alpha;

        for k in 1..=anderson.max_iter {
            // drop old values
            if resm.len() == anderson.mmax {
                resm.pop_front();
                rhom.pop_front();
            }
            let m = resm.len() + 1;

            // calculate residual
            resm.push_back(self.euler_lagrange_equation(&*rho, &*rho_bulk, anderson.log)?);

            // save x value
            if anderson.log {
                rhom.push_back((rho.mapv(f64::ln), rho_bulk.mapv(f64::ln)));
            } else {
                rhom.push_back((rho.clone(), rho_bulk.clone()));
            }

            // calculate alpha
            r = Array::from_shape_fn((m + 1, m + 1), |(i, j)| match (i == m, j == m) {
                (false, false) => {
                    let (resi, resi_bulk, _) = &resm[i];
                    let (resj, resj_bulk, _) = &resm[j];
                    (resi * resj).sum() + (resi_bulk * resj_bulk).sum()
                }
                (true, true) => 0.0,
                _ => 1.0,
            });
            alpha = Array::zeros(m + 1);
            alpha[m] = 1.0;
            alpha = LU::new(r)?.solve(&alpha);

            // update solution
            rho.fill(0.0);
            rho_bulk.fill(0.0);
            for i in 0..m {
                let (rhoi, rhoi_bulk) = &rhom[i];
                let (resi, resi_bulk, _) = &resm[i];
                *rho += &(alpha[i] * (rhoi - &(anderson.beta * resi)));
                *rho_bulk += &(alpha[i] * (rhoi_bulk - &(anderson.beta * resi_bulk)));
            }
            if anderson.log {
                rho.mapv_inplace(f64::exp);
                rho_bulk.mapv_inplace(f64::exp);
            } else {
                rho.mapv_inplace(f64::abs);
                rho_bulk.mapv_inplace(f64::abs);
            }

            // check for convergence
            let (res, _, res_norm) = &resm[m - 1];
            time.push(start_time.elapsed());
            residual.push(*res_norm);
            damping.push(anderson.beta);
            log_iter!(
                verbosity,
                "Anderson mixing {:3}  | {:>4} | {:.6e} ",
                if anderson.log { "log" } else { "" },
                k,
                res
            );
            if *res_norm < anderson.tol {
                return Ok(DFTSolverLog::new(true, residual, time, Some(damping)));
            }
        }
        Ok(DFTSolverLog::new(false, residual, time, Some(damping)))
    }

    fn solve_newton(
        &self,
        newton: Newton,
        rho: &mut Array<f64, D::Larger>,
        rho_bulk: &mut Array1<f64>,
        verbosity: Verbosity,
    ) -> EosResult<DFTSolverLog> {
        let mut residual = Vec::new();
        let mut time = Vec::new();
        let start_time = Instant::now();

        for k in 0..100 {
            // calculate initial residual
            let (res, _, res_norm) = self.euler_lagrange_equation(rho, rho_bulk, false)?;
            time.push(start_time.elapsed());
            residual.push(res_norm);

            // check convegrence
            if res_norm < newton.tol {
                return Ok(DFTSolverLog::new(true, residual, time, None));
            }

            // update solution
            let (x, iter) = self.gmres(newton, rho, &res)?;
            *rho += &x;
            log_iter!(verbosity, "Newton   | {k:>4} | {res_norm:.6e} | {iter} ");
        }

        Ok(DFTSolverLog::new(false, residual, time, None))
    }

    fn gmres(
        &self,
        newton: Newton,
        density: &Array<f64, D::Larger>,
        r0: &Array<f64, D::Larger>,
    ) -> EosResult<(Array<f64, D::Larger>, usize)> {
        // calculate second partial derivatives once
        let second_partial_derivatives = self.second_partial_derivatives(density)?;
        let exp_dfdrho = r0 - density;

        // allocate vectors and arrays
        let mut v = Vec::with_capacity(newton.max_iter);
        let mut h: Array2<f64> = Array::zeros([newton.max_iter + 1; 2]);
        let mut c: Array1<f64> = Array::zeros(newton.max_iter + 1);
        let mut s: Array1<f64> = Array::zeros(newton.max_iter + 1);
        let mut gamma: Array1<f64> = Array::zeros(newton.max_iter + 1);

        gamma[0] = (r0 * r0).sum().sqrt();
        v.push(r0 / gamma[0]);

        let mut iter = 0;
        for j in 0..newton.max_iter {
            // calculate q=Av_j
            let delta_n = self.convolver.weighted_densities(&v[j]);
            let mut q =
                self.functional_derivative_derivative(&second_partial_derivatives, &delta_n);
            q.outer_iter_mut()
                .zip(self.dft.m().iter())
                .for_each(|(mut q, &m)| q /= m);
            q = q * &exp_dfdrho - &v[j];

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
            if gamma[j + 1].abs() >= newton.tol && j + 1 < newton.max_iter {
                v.push(q / h[(j + 1, j)]);
                iter += 1;
            } else {
                break;
            }
        }
        // calculate solution vector
        let mut x = Array::zeros(density.raw_dim());
        let mut y = Array::zeros(iter + 1);
        for i in (0..=iter).rev() {
            y[i] = (gamma[i] - (i + 1..=iter).map(|k| h[(i, k)] * y[k]).sum::<f64>()) / h[(i, i)];
        }
        v.iter().zip(y.into_iter()).for_each(|(v, y)| x += &(y * v));
        Ok((x, iter))
    }

    fn second_partial_derivatives(
        &self,
        density: &Array<f64, D::Larger>,
    ) -> EosResult<Vec<Array<f64, <D::Larger as Dimension>::Larger>>> {
        let temperature = self.temperature.to_reduced(U::reference_temperature())?;
        let contributions = self.dft.contributions();
        let weighted_densities = self.convolver.weighted_densities(density);
        let mut second_partial_derivatives = Vec::with_capacity(contributions.len());
        for (c, wd) in contributions.iter().zip(&weighted_densities) {
            let nwd = wd.shape()[0];
            let ngrid = wd.len() / nwd;
            let mut phi = Array::zeros(density.raw_dim().remove_axis(Axis(0)));
            let mut pd = Array::zeros(wd.raw_dim());
            let dim = wd.shape();
            let dim: Vec<_> = std::iter::once(&nwd).chain(dim).cloned().collect();
            let mut pd2 = Array::zeros(dim).into_dimensionality().unwrap();
            c.second_partial_derivatives(
                temperature,
                wd.view().into_shape((nwd, ngrid)).unwrap(),
                phi.view_mut().into_shape(ngrid).unwrap(),
                pd.view_mut().into_shape((nwd, ngrid)).unwrap(),
                pd2.view_mut().into_shape((nwd, nwd, ngrid)).unwrap(),
            )?;
            second_partial_derivatives.push(pd2);
        }
        Ok(second_partial_derivatives)
    }

    fn functional_derivative_derivative(
        &self,
        second_partial_derivatives: &[Array<f64, <D::Larger as Dimension>::Larger>],
        weighted_densities: &[Array<f64, D::Larger>],
    ) -> Array<f64, D::Larger> {
        let partial_derivatives: Vec<_> = second_partial_derivatives
            .iter()
            .zip(weighted_densities)
            .map(|(pd2, wd)| {
                let mut partial_derivatives_derivatives =
                    Array::zeros(pd2.raw_dim().remove_axis(Axis(0)));
                let n = wd.shape()[0];
                for i in 0..n {
                    for j in 0..n {
                        partial_derivatives_derivatives
                            .index_axis_mut(Axis(0), i)
                            .add_assign(
                                &(&pd2.index_axis(Axis(0), i).index_axis(Axis(0), j)
                                    * &wd.index_axis(Axis(0), j)),
                            );
                    }
                }
                partial_derivatives_derivatives
            })
            .collect();
        self.convolver.functional_derivative(&partial_derivatives)
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
                        match picard.beta {
                            None => "line search".into(),
                            Some(beta) => format!("beta={beta}"),
                        }
                    ),
                    picard.max_iter,
                    picard.tol,
                ),
                DFTAlgorithm::AndersonMixing(anderson) => (
                    format!(
                        "Anderson mixing ({}beta={}, mmax={})",
                        if anderson.log { "log, " } else { "" },
                        anderson.beta,
                        anderson.mmax
                    ),
                    anderson.max_iter,
                    anderson.tol,
                ),
                DFTAlgorithm::Newton(newton) => (
                    format!("Newton (FFT, max_iter_gmres={})", newton.max_iter_gmres),
                    newton.max_iter,
                    newton.tol,
                ),
            };
            res += &format!("\n|{}|{}|{:e}|", solver, max_iter, tol);
        }
        res
    }
}
