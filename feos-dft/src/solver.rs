use crate::{DFTProfile, HelmholtzEnergyFunctional, WeightFunction, WeightFunctionShape};
use feos_core::si::{Time, SECOND};
use feos_core::{log_iter, log_result, EosError, EosResult, Verbosity};
use nalgebra::{DMatrix, DVector};
use ndarray::prelude::*;
use ndarray::RemoveAxis;
use petgraph::graph::Graph;
use petgraph::visit::EdgeRef;
use petgraph::Directed;
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
    verbosity: Verbosity,
    start_time: Instant,
    residual: Vec<f64>,
    time: Vec<Duration>,
    solver: Vec<&'static str>,
}

impl DFTSolverLog {
    pub(crate) fn new(verbosity: Verbosity) -> Self {
        log_iter!(
            verbosity,
            "solver                 | iter |    time    | residual "
        );
        Self {
            verbosity,
            start_time: Instant::now(),
            residual: Vec::new(),
            time: Vec::new(),
            solver: Vec::new(),
        }
    }

    fn add_residual(&mut self, solver: &'static str, iteration: usize, residual: f64) {
        if iteration == 0 {
            log_iter!(self.verbosity, "{:-<59}", "");
        }
        self.solver.push(solver);
        self.residual.push(residual);
        let time = self.start_time.elapsed();
        self.time.push(self.start_time.elapsed());
        log_iter!(
            self.verbosity,
            "{:22} | {:>4} | {:7.3} | {:.6e}",
            solver,
            iteration,
            time.as_secs_f64() * SECOND,
            residual,
        );
    }

    pub fn residual(&self) -> ArrayView1<f64> {
        (&self.residual).into()
    }

    pub fn time(&self) -> Time<Array1<f64>> {
        self.time.iter().map(|t| t.as_secs_f64() * SECOND).collect()
    }

    pub fn solver(&self) -> &[&'static str] {
        &self.solver
    }
}

impl<D: Dimension, F: HelmholtzEnergyFunctional> DFTProfile<D, F>
where
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    pub(crate) fn call_solver(
        &mut self,
        rho: &mut Array<f64, D::Larger>,
        rho_bulk: &mut Array1<f64>,
        solver: &DFTSolver,
        debug: bool,
    ) -> EosResult<()> {
        let mut converged = false;
        let mut iterations = 0;
        let mut log = DFTSolverLog::new(solver.verbosity);
        for algorithm in &solver.algorithms {
            let (conv, iter) = match algorithm {
                DFTAlgorithm::PicardIteration(picard) => {
                    self.solve_picard(*picard, rho, rho_bulk, &mut log)
                }
                DFTAlgorithm::AndersonMixing(anderson) => {
                    self.solve_anderson(*anderson, rho, rho_bulk, &mut log)
                }
                DFTAlgorithm::Newton(newton) => self.solve_newton(*newton, rho, rho_bulk, &mut log),
            }?;
            converged = conv;
            iterations += iter;
        }
        self.solver_log = Some(log);
        if converged {
            log_result!(solver.verbosity, "DFT solved in {} iterations", iterations);
        } else if debug {
            log_result!(
                solver.verbosity,
                "DFT not converged in {} iterations",
                iterations
            );
        } else {
            return Err(EosError::NotConverged(String::from("DFT")));
        }
        Ok(())
    }

    fn solve_picard(
        &self,
        picard: PicardIteration,
        rho: &mut Array<f64, D::Larger>,
        rho_bulk: &mut Array1<f64>,
        log: &mut DFTSolverLog,
    ) -> EosResult<(bool, usize)> {
        let solver = if picard.log {
            "Picard iteration (log)"
        } else {
            "Picard iteration"
        };

        for k in 0..picard.max_iter {
            // calculate residual
            let (res, res_bulk, res_norm, _, _) =
                self.euler_lagrange_equation(&*rho, &*rho_bulk, picard.log)?;
            log.add_residual(solver, k, res_norm);

            // check for convergence
            if res_norm < picard.tol {
                return Ok((true, k));
            }

            // apply line search or constant damping
            let damping_coefficient = picard.damping_coefficient.map_or_else(
                || self.line_search(rho, &res, rho_bulk, res_norm, picard.log),
                Ok,
            )?;

            // update solution
            if picard.log {
                *rho *= &(&res * damping_coefficient).mapv(f64::exp);
                *rho_bulk *= &(&res_bulk * damping_coefficient).mapv(f64::exp);
            } else {
                *rho += &(&res * damping_coefficient);
                *rho_bulk += &(&res_bulk * damping_coefficient);
            }
        }
        Ok((false, picard.max_iter))
    }

    fn line_search(
        &self,
        rho: &Array<f64, D::Larger>,
        delta_rho: &Array<f64, D::Larger>,
        rho_bulk: &Array1<f64>,
        res0: f64,
        logarithm: bool,
    ) -> EosResult<f64> {
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
            let Ok((_, _, res2, _, _)) =
                self.euler_lagrange_equation(&rho_new, rho_bulk, logarithm)
            else {
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
            let Ok((_, _, res1, _, _)) =
                self.euler_lagrange_equation(&rho_new, rho_bulk, logarithm)
            else {
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
        // log_iter!(verbosity, "    Line search      | {} | ", alpha);
        Ok(alpha)
    }

    fn solve_anderson(
        &self,
        anderson: AndersonMixing,
        rho: &mut Array<f64, D::Larger>,
        rho_bulk: &mut Array1<f64>,
        log: &mut DFTSolverLog,
    ) -> EosResult<(bool, usize)> {
        let solver = if anderson.log {
            "Anderson mixing (log)"
        } else {
            "Anderson mixing"
        };

        let mut resm = VecDeque::with_capacity(anderson.mmax);
        let mut rhom = VecDeque::with_capacity(anderson.mmax);
        let mut r;
        let mut alpha;

        for k in 0..anderson.max_iter {
            // drop old values
            if resm.len() == anderson.mmax {
                resm.pop_front();
                rhom.pop_front();
            }
            let m = resm.len() + 1;

            // calculate residual
            let (res, res_bulk, res_norm, _, _) =
                self.euler_lagrange_equation(&*rho, &*rho_bulk, anderson.log)?;
            log.add_residual(solver, k, res_norm);

            // check for convergence
            if res_norm < anderson.tol {
                return Ok((true, k));
            }

            // save residual and x value
            resm.push_back((res, res_bulk, res_norm));
            if anderson.log {
                rhom.push_back((rho.mapv(f64::ln), rho_bulk.mapv(f64::ln)));
            } else {
                rhom.push_back((rho.clone(), rho_bulk.clone()));
            }

            // calculate alpha
            r = DMatrix::from_fn(m + 1, m + 1, |i, j| match (i == m, j == m) {
                (false, false) => {
                    let (resi, resi_bulk, _) = &resm[i];
                    let (resj, resj_bulk, _) = &resm[j];
                    (resi * resj).sum() + (resi_bulk * resj_bulk).sum()
                }
                (true, true) => 0.0,
                _ => 1.0,
            });
            alpha = DVector::zeros(m + 1);
            alpha[m] = 1.0;
            let alpha = r.lu().solve(&alpha);
            let alpha = alpha.ok_or(EosError::Error("alpha matrix is not invertible".into()))?;

            // update solution
            rho.fill(0.0);
            rho_bulk.fill(0.0);
            for i in 0..m {
                let (rhoi, rhoi_bulk) = &rhom[i];
                let (resi, resi_bulk, _) = &resm[i];
                *rho += &(alpha[i] * (rhoi + &(anderson.damping_coefficient * resi)));
                *rho_bulk +=
                    &(alpha[i] * (rhoi_bulk + &(anderson.damping_coefficient * resi_bulk)));
            }
            if anderson.log {
                rho.mapv_inplace(f64::exp);
                rho_bulk.mapv_inplace(f64::exp);
            } else {
                rho.mapv_inplace(f64::abs);
                rho_bulk.mapv_inplace(f64::abs);
            }
        }
        Ok((false, anderson.max_iter))
    }

    fn solve_newton(
        &self,
        newton: Newton,
        rho: &mut Array<f64, D::Larger>,
        rho_bulk: &mut Array1<f64>,
        log: &mut DFTSolverLog,
    ) -> EosResult<(bool, usize)> {
        let solver = if newton.log { "Newton (log)" } else { "Newton" };
        for k in 0..newton.max_iter {
            // calculate initial residual
            let (res, _, res_norm, exp_dfdrho, rho_p) =
                self.euler_lagrange_equation(rho, rho_bulk, newton.log)?;
            log.add_residual(solver, k, res_norm);

            // check convergence
            if res_norm < newton.tol {
                return Ok((true, k));
            }

            // calculate second partial derivatives once
            let second_partial_derivatives = self.second_partial_derivatives(rho)?;

            // define rhs function
            let rhs = |delta_rho: &_| {
                let mut delta_functional_derivative =
                    self.delta_functional_derivative(delta_rho, &second_partial_derivatives);
                delta_functional_derivative
                    .outer_iter_mut()
                    .zip(self.dft.m().iter())
                    .for_each(|(mut q, &m)| q /= m);
                let delta_i = self.delta_bond_integrals(&exp_dfdrho, &delta_functional_derivative);
                let rho = if newton.log { &*rho } else { &rho_p };
                delta_rho + (delta_functional_derivative - delta_i) * rho
            };

            // update solution
            let lhs = if newton.log { &*rho * res } else { res };
            *rho += &Self::gmres(rhs, &lhs, newton.max_iter_gmres, newton.tol * 1e-2, log)?;
            rho.mapv_inplace(f64::abs);
            rho_bulk.mapv_inplace(f64::abs);
        }

        Ok((false, newton.max_iter))
    }

    pub(crate) fn gmres<R>(
        rhs: R,
        r0: &Array<f64, D::Larger>,
        max_iter: usize,
        tol: f64,
        log: &mut DFTSolverLog,
    ) -> EosResult<Array<f64, D::Larger>>
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
        log.add_residual("GMRES", 0, gamma[0]);

        let mut iter = 0;
        for j in 0..max_iter {
            // calculate q=Av_j
            let mut q = rhs(&v[j]);

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
            log.add_residual("GMRES", j + 1, gamma[j + 1].abs());
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
        v.iter().zip(y.into_iter()).for_each(|(v, y)| x += &(y * v));
        Ok(x)
    }

    pub(crate) fn second_partial_derivatives(
        &self,
        density: &Array<f64, D::Larger>,
    ) -> EosResult<Vec<Array<f64, <D::Larger as Dimension>::Larger>>> {
        let temperature = self.temperature.to_reduced();
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

    pub(crate) fn delta_functional_derivative(
        &self,
        delta_density: &Array<f64, D::Larger>,
        second_partial_derivatives: &[Array<f64, <D::Larger as Dimension>::Larger>],
    ) -> Array<f64, D::Larger> {
        let delta_weighted_densities = self.convolver.weighted_densities(delta_density);
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
        self.convolver
            .functional_derivative(&delta_partial_derivatives)
    }

    pub(crate) fn delta_bond_integrals(
        &self,
        exponential: &Array<f64, D::Larger>,
        delta_functional_derivative: &Array<f64, D::Larger>,
    ) -> Array<f64, D::Larger> {
        let temperature = self.temperature.to_reduced();

        // calculate weight functions
        let bond_lengths = self.dft.bond_lengths(temperature).into_edge_type();
        let mut bond_weight_functions = bond_lengths.map(
            |_, _| (),
            |_, &l| WeightFunction::new_scaled(arr1(&[l]), WeightFunctionShape::Delta),
        );
        for n in bond_lengths.node_indices() {
            for e in bond_lengths.edges(n) {
                bond_weight_functions.add_edge(
                    e.target(),
                    e.source(),
                    WeightFunction::new_scaled(arr1(&[*e.weight()]), WeightFunctionShape::Delta),
                );
            }
        }

        let mut i_graph: Graph<_, Option<Array<f64, D>>, Directed> =
            bond_weight_functions.map(|_, _| (), |_, _| None);
        let mut delta_i_graph: Graph<_, Option<Array<f64, D>>, Directed> =
            bond_weight_functions.map(|_, _| (), |_, _| None);

        let bonds = i_graph.edge_count();
        let mut calc = 0;

        // go through the whole graph until every bond has been calculated
        while calc < bonds {
            let mut edge_id = None;
            let mut i1 = None;
            let mut delta_i1 = None;

            // find the first bond that can be calculated
            'nodes: for node in i_graph.node_indices() {
                for edge in i_graph.edges(node) {
                    // skip already calculated bonds
                    if edge.weight().is_some() {
                        continue;
                    }

                    // if all bonds from the neighboring segment are calculated calculate the bond
                    let edges = i_graph
                        .edges(edge.target())
                        .filter(|e| e.target().index() != node.index());
                    let delta_edges = delta_i_graph
                        .edges(edge.target())
                        .filter(|e| e.target().index() != node.index());
                    if edges.clone().all(|e| e.weight().is_some()) {
                        edge_id = Some(edge.id());
                        let i0 = edges.fold(
                            exponential
                                .index_axis(Axis(0), edge.target().index())
                                .to_owned(),
                            |acc: Array<f64, _>, e| acc * e.weight().as_ref().unwrap(),
                        );
                        let delta_i0 = delta_edges.fold(
                            -&delta_functional_derivative
                                .index_axis(Axis(0), edge.target().index()),
                            |acc: Array<f64, _>, delta_e| acc + delta_e.weight().as_ref().unwrap(),
                        ) * &i0;
                        i1 = Some(
                            self.convolver
                                .convolve(i0, &bond_weight_functions[edge.id()]),
                        );
                        delta_i1 = Some(
                            (self
                                .convolver
                                .convolve(delta_i0, &bond_weight_functions[edge.id()])
                                / i1.as_ref().unwrap())
                            .mapv(|x| if x.is_finite() { x } else { 0.0 }),
                        );
                        break 'nodes;
                    }
                }
            }
            if let Some(edge_id) = edge_id {
                i_graph[edge_id] = i1;
                delta_i_graph[edge_id] = delta_i1;
                calc += 1;
            } else {
                panic!("Cycle in molecular structure detected!")
            }
        }

        let mut delta_i = Array::zeros(exponential.raw_dim());
        for node in delta_i_graph.node_indices() {
            for edge in delta_i_graph.edges(node) {
                delta_i
                    .index_axis_mut(Axis(0), node.index())
                    .add_assign(edge.weight().as_ref().unwrap());
            }
        }

        delta_i
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
