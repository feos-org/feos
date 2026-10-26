//! Newton-Krylov solver for systems of equations that are written once against a [Tape], which
//! provides both the residual and its Jacobian-vector products.
use super::DFTSolverLog;
use super::tape::{Evaluate, Record, Tape};
use feos_core::{FeosError, FeosResult, Verbosity};
use ndarray::{Array, Array1, Array2, Dimension};

/// Relative tolerance of the GMRES solver in the Newton-Krylov solver (inexact Newton).
const NEWTON_GMRES_RTOL: f64 = 1e-3;
/// Maximum number of step halvings in the line search of the Newton-Krylov solver.
const NEWTON_MAX_LINE_SEARCH: usize = 20;

/// A system of equations $r(x)=0$ written once against a [Tape].
pub(crate) trait NonlinearSystem<D: Dimension> {
    /// Residual for the unknowns `x` (values), or its Jacobian-vector product for the tangent `x`
    /// (replay).
    ///
    /// Computation times of parts of the residual can be added to `log`.
    fn residual<T: Tape>(
        &self,
        tape: &mut T,
        x: &Array<f64, D>,
        log: &mut DFTSolverLog,
    ) -> Result<Array<f64, D>, T::Error<FeosError>>;

    /// Norm of the residual that the line search reduces (root mean square of all entries).
    fn norm(&self, residual: &Array<f64, D>) -> f64 {
        (residual.mapv(|r| r * r).sum() / residual.len() as f64).sqrt()
    }

    /// Norm of the residual for the convergence check (by default the same as [Self::norm]).
    fn convergence_norm(&self, _x: &Array<f64, D>, residual: &Array<f64, D>) -> f64 {
        self.norm(residual)
    }

    /// Initial step length of the line search for the Newton step `delta` (at most 1).
    fn step_length(&self, _x: &Array<f64, D>, _delta: &Array<f64, D>) -> f64 {
        1.0
    }
}

/// Newton-Krylov method: inexact Newton with GMRES for the linear systems (Jacobian-vector
/// products from replays of the recorded residual) and a backtracking line search on the norm of
/// the residual.
pub(crate) fn newton_krylov<D: Dimension, S: NonlinearSystem<D>>(
    system: &S,
    x: &mut Array<f64, D>,
    max_iter: usize,
    max_iter_gmres: usize,
    tol: f64,
    solver_log: &mut DFTSolverLog,
    verbosity: Verbosity,
) -> FeosResult<(bool, usize)> {
    let solver = "Newton";

    // the residual of the accepted step is reused, so the (more expensive) recording is only
    // done if another Newton step is required
    let mut res = system.residual(&mut Evaluate, x, solver_log)?;
    for k in 0..max_iter {
        let norm = system.convergence_norm(x, &res);
        if !norm.is_finite() {
            return Err(FeosError::IterationFailed(solver.into()));
        }
        solver_log.add_residual(solver, k, norm, verbosity);
        if norm < tol {
            return Ok((true, k));
        }
        let norm0 = system.norm(&res);

        // linearization
        let mut record = Record::new();
        system.residual(&mut record, x, solver_log)?;

        // inexact Newton: relative tolerance for the linear system, bounded by the absolute one
        // (the replays are timed as a whole by GMRES)
        let rhs = |delta: &Array<f64, D>| {
            let Ok(jvp) = system.residual(&mut record.replay(), delta, &mut DFTSolverLog::new());
            -jvp
        };
        let gmres_tol = (NEWTON_GMRES_RTOL * res.mapv(|r| r * r).sum().sqrt()).max(tol * 1e-2);
        let delta = gmres(rhs, &res, max_iter_gmres, gmres_tol, solver_log)?;

        // backtracking line search on the norm of the residual
        let mut alpha = system.step_length(x, &delta);
        let x0 = x.clone();
        let mut accepted = false;
        for _ in 0..NEWTON_MAX_LINE_SEARCH {
            *x = &x0 + &(alpha * &delta);
            if let Ok(res_new) = system.residual(&mut Evaluate, x, solver_log) {
                let norm = system.norm(&res_new);
                if norm.is_finite() && norm < (1.0 - 1e-4 * alpha) * norm0 {
                    res = res_new;
                    accepted = true;
                    break;
                }
            }
            alpha *= 0.5;
        }
        if !accepted {
            *x = x0;
            return Ok((false, k));
        }
    }
    Ok((false, max_iter))
}

/// Solve the linearized system $-J(x)\delta=b$ with GMRES (Jacobian-vector products from replays
/// of the residual recorded at `x`).
pub(crate) fn solve_linearized<D: Dimension, S: NonlinearSystem<D>>(
    system: &S,
    x: &Array<f64, D>,
    b: &Array<f64, D>,
    max_iter: usize,
    tol: f64,
) -> FeosResult<Array<f64, D>> {
    let solver_log = &mut DFTSolverLog::new();
    let mut record = Record::new();
    system.residual(&mut record, x, solver_log)?;
    let rhs = |delta: &Array<f64, D>| {
        let Ok(jvp) = system.residual(&mut record.replay(), delta, &mut DFTSolverLog::new());
        -jvp
    };
    gmres(rhs, b, max_iter, tol, solver_log)
}

/// GMRES for the linear system $Ax=r_0$, where `rhs` calculates the product $Av$.
pub(crate) fn gmres<D, R>(
    rhs: R,
    r0: &Array<f64, D>,
    max_iter: usize,
    tol: f64,
    solver_log: &mut DFTSolverLog,
) -> FeosResult<Array<f64, D>>
where
    D: Dimension,
    R: Fn(&Array<f64, D>) -> Array<f64, D>,
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
