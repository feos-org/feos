use super::{SolverOptions, State, StateHD, TPSpec, Verbosity};
use crate::equation_of_state::Residual;
use crate::errors::{EosError, EosResult};
use crate::{DensityInitialization, EosUnit};
use nalgebra::{DMatrix, DVector, SVector, SymmetricEigen};
use ndarray::{arr1, Array1};
use num_dual::{
    first_derivative, try_first_derivative, try_jacobian, Dual, Dual3, Dual64, DualNum, DualSVec64,
    DualVec, HyperDual,
};
use num_traits::{One, Zero};
use quantity::si::{SIArray1, SINumber, SIUnit};
use std::convert::TryFrom;
use std::sync::Arc;

const MAX_ITER_CRIT_POINT: usize = 50;
const MAX_ITER_CRIT_POINT_BINARY: usize = 200;
const TOL_CRIT_POINT: f64 = 1e-8;

/// # Critical points
impl<R: Residual> State<R> {
    /// Calculate the pure component critical point of all components.
    pub fn critical_point_pure(
        eos: &Arc<R>,
        initial_temperature: Option<SINumber>,
        options: SolverOptions,
    ) -> EosResult<Vec<Self>>
    where
        SINumber: std::fmt::Display,
    {
        (0..eos.components())
            .map(|i| {
                Self::critical_point(
                    &Arc::new(eos.subset(&[i])),
                    None,
                    initial_temperature,
                    options,
                )
            })
            .collect()
    }

    pub fn critical_point_binary(
        eos: &Arc<R>,
        temperature_or_pressure: SINumber,
        initial_temperature: Option<SINumber>,
        initial_molefracs: Option<[f64; 2]>,
        options: SolverOptions,
    ) -> EosResult<Self>
    where
        SINumber: std::fmt::Display,
    {
        match TPSpec::try_from(temperature_or_pressure)? {
            TPSpec::Temperature(t) => {
                Self::critical_point_binary_t(eos, t, initial_molefracs, options)
            }
            TPSpec::Pressure(p) => Self::critical_point_binary_p(
                eos,
                p,
                initial_temperature,
                initial_molefracs,
                options,
            ),
        }
    }

    /// Calculate the critical point of a system for given moles.
    pub fn critical_point(
        eos: &Arc<R>,
        moles: Option<&SIArray1>,
        initial_temperature: Option<SINumber>,
        options: SolverOptions,
    ) -> EosResult<Self>
    where
        SINumber: std::fmt::Display,
    {
        let moles = eos.validate_moles(moles)?;
        let trial_temperatures = [
            300.0 * SIUnit::reference_temperature(),
            700.0 * SIUnit::reference_temperature(),
            500.0 * SIUnit::reference_temperature(),
        ];
        if let Some(t) = initial_temperature {
            return Self::critical_point_hkm(eos, &moles, t, options);
        }
        for &t in trial_temperatures.iter() {
            let s = Self::critical_point_hkm(eos, &moles, t, options);
            if s.is_ok() {
                return s;
            }
        }
        Err(EosError::NotConverged(String::from("Critical point")))
    }

    fn critical_point_hkm(
        eos: &Arc<R>,
        moles: &SIArray1,
        initial_temperature: SINumber,
        options: SolverOptions,
    ) -> EosResult<Self>
    where
        SINumber: std::fmt::Display,
    {
        let (max_iter, tol, verbosity) = options.unwrap_or(MAX_ITER_CRIT_POINT, TOL_CRIT_POINT);

        let mut t = initial_temperature.to_reduced(SIUnit::reference_temperature())?;
        let max_density = eos
            .max_density(Some(moles))?
            .to_reduced(SIUnit::reference_density())?;
        let mut rho = 0.3 * max_density;
        let n = moles.to_reduced(SIUnit::reference_moles())?;

        log_iter!(
            verbosity,
            " iter |    residual    |   temperature   |       density        "
        );
        log_iter!(verbosity, "{:-<64}", "");
        log_iter!(
            verbosity,
            " {:4} |                | {:13.8} | {:12.8}",
            0,
            t * SIUnit::reference_temperature(),
            rho * SIUnit::reference_density(),
        );

        for i in 1..=max_iter {
            // calculate residuals and derivatives w.r.t. temperature and density
            let res = |x: SVector<DualSVec64<2>, 2>| critical_point_objective(eos, x[0], x[1], &n);
            let (res, jac) = try_jacobian(res, SVector::from([t, rho]))?;

            // calculate Newton step
            let delta = jac.lu().solve(&res);
            let mut delta = delta.ok_or(EosError::IterationFailed("Critical point".into()))?;

            // reduce step if necessary
            if delta[0].abs() > 0.25 * t {
                delta *= 0.25 * t / delta[0].abs()
            }
            if delta[1].abs() > 0.03 * max_density {
                delta *= 0.03 * max_density / delta[1].abs()
            }

            // apply step
            t -= delta[0];
            rho -= delta[1];
            rho = f64::max(rho, 1e-4 * max_density);

            log_iter!(
                verbosity,
                " {:4} | {:14.8e} | {:13.8} | {:12.8}",
                i,
                res.norm(),
                t * SIUnit::reference_temperature(),
                rho * SIUnit::reference_density(),
            );

            // check convergence
            if res.norm() < tol {
                log_result!(
                    verbosity,
                    "Critical point calculation converged in {} step(s)\n",
                    i
                );
                return State::new_nvt(
                    eos,
                    t * SIUnit::reference_temperature(),
                    moles.sum() / (rho * SIUnit::reference_density()),
                    moles,
                );
            }
        }
        Err(EosError::NotConverged(String::from("Critical point")))
    }

    /// Calculate the critical point of a binary system for given temperature.
    fn critical_point_binary_t(
        eos: &Arc<R>,
        temperature: SINumber,
        initial_molefracs: Option<[f64; 2]>,
        options: SolverOptions,
    ) -> EosResult<Self>
    where
        SINumber: std::fmt::Display,
    {
        let (max_iter, tol, verbosity) =
            options.unwrap_or(MAX_ITER_CRIT_POINT_BINARY, TOL_CRIT_POINT);

        let t = temperature.to_reduced(SIUnit::reference_temperature())?;
        let x = SVector::from(initial_molefracs.unwrap_or([0.5, 0.5]));
        let max_density = eos
            .max_density(Some(&(arr1(&x.data.0[0]) * SIUnit::reference_moles())))?
            .to_reduced(SIUnit::reference_density())?;
        let mut rho = x * 0.3 * max_density;

        log_iter!(
            verbosity,
            " iter |    residual    |      density 1       |      density 2       "
        );
        log_iter!(verbosity, "{:-<69}", "");
        log_iter!(
            verbosity,
            " {:4} |                | {:12.8} | {:12.8}",
            0,
            rho[0] * SIUnit::reference_density(),
            rho[1] * SIUnit::reference_density(),
        );

        for i in 1..=max_iter {
            // calculate residuals and derivatives w.r.t. partial densities
            let res = |rho| critical_point_objective_t(eos, t, rho);
            let (res, jac) = try_jacobian(res, rho)?;

            // calculate Newton step
            let delta = jac.lu().solve(&res);
            let mut delta = delta.ok_or(EosError::IterationFailed("Critical point".into()))?;

            // reduce step if necessary
            for i in 0..2 {
                if delta[i].abs() > 0.03 * max_density {
                    delta *= 0.03 * max_density / delta[i].abs()
                }
            }

            // apply step
            rho -= delta;
            rho[0] = f64::max(rho[0], 1e-4 * max_density);
            rho[1] = f64::max(rho[1], 1e-4 * max_density);

            log_iter!(
                verbosity,
                " {:4} | {:14.8e} | {:12.8} | {:12.8}",
                i,
                res.norm(),
                rho[0] * SIUnit::reference_density(),
                rho[1] * SIUnit::reference_density(),
            );

            // check convergence
            if res.norm() < tol {
                log_result!(
                    verbosity,
                    "Critical point calculation converged in {} step(s)\n",
                    i
                );
                return State::new_nvt(
                    eos,
                    t * SIUnit::reference_temperature(),
                    SIUnit::reference_volume(),
                    &(arr1(&rho.data.0[0]) * SIUnit::reference_moles()),
                );
            }
        }
        Err(EosError::NotConverged(String::from("Critical point")))
    }

    /// Calculate the critical point of a binary system for given pressure.
    fn critical_point_binary_p(
        eos: &Arc<R>,
        pressure: SINumber,
        initial_temperature: Option<SINumber>,
        initial_molefracs: Option<[f64; 2]>,
        options: SolverOptions,
    ) -> EosResult<Self>
    where
        SINumber: std::fmt::Display,
    {
        let (max_iter, tol, verbosity) =
            options.unwrap_or(MAX_ITER_CRIT_POINT_BINARY, TOL_CRIT_POINT);

        let p = pressure.to_reduced(SIUnit::reference_pressure())?;
        let mut t = initial_temperature
            .map(|t| t.to_reduced(SIUnit::reference_temperature()))
            .transpose()?
            .unwrap_or(300.0);
        let x = SVector::from(initial_molefracs.unwrap_or([0.5, 0.5]));
        let max_density = eos
            .max_density(Some(&(arr1(&x.data.0[0]) * SIUnit::reference_moles())))?
            .to_reduced(SIUnit::reference_density())?;
        let mut rho = x * 0.3 * max_density;

        log_iter!(
            verbosity,
            " iter |    residual    |   temperature   |      density 1       |      density 2       "
        );
        log_iter!(verbosity, "{:-<87}", "");
        log_iter!(
            verbosity,
            " {:4} |                | {:13.8} | {:12.8} | {:12.8}",
            0,
            t * SIUnit::reference_temperature(),
            rho[0] * SIUnit::reference_density(),
            rho[1] * SIUnit::reference_density(),
        );

        for i in 1..=max_iter {
            // calculate residuals and derivatives w.r.t. temperature and partial densities
            let res = |x: SVector<DualSVec64<3>, 3>| {
                let r = SVector::from([x[1], x[2]]);
                critical_point_objective_p(eos, p, x[0], r)
            };
            let (res, jac) = try_jacobian(res, SVector::from([t, rho[0], rho[1]]))?;

            // calculate Newton step
            let delta = jac.lu().solve(&res);
            let mut delta = delta.ok_or(EosError::IterationFailed("Critical point".into()))?;

            // reduce step if necessary
            if delta[0].abs() > 0.25 * t {
                delta *= 0.25 * t / delta[0].abs()
            }
            if delta[1].abs() > 0.03 * max_density {
                delta *= 0.03 * max_density / delta[1].abs()
            }
            if delta[2].abs() > 0.03 * max_density {
                delta *= 0.03 * max_density / delta[2].abs()
            }

            // apply step
            t -= delta[0];
            rho[0] -= delta[1];
            rho[1] -= delta[2];
            rho[0] = f64::max(rho[0], 1e-4 * max_density);
            rho[1] = f64::max(rho[1], 1e-4 * max_density);

            log_iter!(
                verbosity,
                " {:4} | {:14.8e} | {:13.8} | {:12.8} | {:12.8}",
                i,
                res.norm(),
                t * SIUnit::reference_temperature(),
                rho[0] * SIUnit::reference_density(),
                rho[1] * SIUnit::reference_density(),
            );

            // check convergence
            if res.norm() < tol {
                log_result!(
                    verbosity,
                    "Critical point calculation converged in {} step(s)\n",
                    i
                );
                return State::new_nvt(
                    eos,
                    t * SIUnit::reference_temperature(),
                    SIUnit::reference_volume(),
                    &(arr1(&rho.data.0[0]) * SIUnit::reference_moles()),
                );
            }
        }
        Err(EosError::NotConverged(String::from("Critical point")))
    }

    pub fn spinodal(
        eos: &Arc<R>,
        temperature: SINumber,
        moles: Option<&SIArray1>,
        options: SolverOptions,
    ) -> EosResult<[Self; 2]>
    where
        SINumber: std::fmt::Display,
    {
        let critical_point = Self::critical_point(eos, moles, None, options)?;
        let moles = eos.validate_moles(moles)?;
        let spinodal_vapor = Self::calculate_spinodal(
            eos,
            temperature,
            &moles,
            DensityInitialization::Vapor,
            options,
        )?;
        let rho = 2.0 * critical_point.density - spinodal_vapor.density;
        let spinodal_liquid = Self::calculate_spinodal(
            eos,
            temperature,
            &moles,
            DensityInitialization::InitialDensity(rho),
            options,
        )?;
        Ok([spinodal_vapor, spinodal_liquid])
    }

    fn calculate_spinodal(
        eos: &Arc<R>,
        temperature: SINumber,
        moles: &SIArray1,
        density_initialization: DensityInitialization,
        options: SolverOptions,
    ) -> EosResult<Self>
    where
        SINumber: std::fmt::Display,
    {
        let (max_iter, tol, verbosity) = options.unwrap_or(MAX_ITER_CRIT_POINT, TOL_CRIT_POINT);

        let max_density = eos
            .max_density(Some(moles))?
            .to_reduced(SIUnit::reference_density())?;
        let t = temperature.to_reduced(SIUnit::reference_temperature())?;
        let mut rho = match density_initialization {
            DensityInitialization::Vapor => 1e-5 * max_density,
            DensityInitialization::Liquid => max_density,
            DensityInitialization::InitialDensity(rho) => {
                rho.to_reduced(SIUnit::reference_density())?
            }
            DensityInitialization::None => unreachable!(),
        };
        let n = moles.to_reduced(SIUnit::reference_moles())?;

        log_iter!(verbosity, " iter |    residual    |       density        ");
        log_iter!(verbosity, "{:-<46}", "");
        log_iter!(
            verbosity,
            " {:4} |                | {:12.8}",
            0,
            rho * SIUnit::reference_density(),
        );

        for i in 1..=max_iter {
            // calculate residuals and derivative w.r.t. density
            let (f, df) =
                try_first_derivative(|rho| spinodal_objective(eos, t.into(), rho, &n), rho)?;

            // calculate Newton step
            let mut delta = f / df;

            // reduce step if necessary
            if delta.abs() > 0.03 * max_density {
                delta *= 0.03 * max_density / delta.abs()
            }

            // apply step
            rho -= delta;
            rho = f64::max(rho, 1e-4 * max_density);

            log_iter!(
                verbosity,
                " {:4} | {:14.8e} | {:12.8}",
                i,
                f.abs(),
                rho * SIUnit::reference_density(),
            );

            // check convergence
            if f.abs() < tol {
                log_result!(
                    verbosity,
                    "Spinodal calculation converged in {} step(s)\n",
                    i
                );
                return State::new_nvt(
                    eos,
                    temperature,
                    moles.sum() / (rho * SIUnit::reference_density()),
                    moles,
                );
            }
        }
        Err(EosError::SuperCritical)
    }
}

fn critical_point_objective<R: Residual>(
    eos: &Arc<R>,
    temperature: DualSVec64<2>,
    density: DualSVec64<2>,
    moles: &Array1<f64>,
) -> EosResult<SVector<DualSVec64<2>, 2>> {
    // calculate second partial derivatives w.r.t. moles
    let t = HyperDual::from_re(temperature);
    let v = HyperDual::from_re(density.recip() * moles.sum());
    let qij = DMatrix::from_fn(eos.components(), eos.components(), |i, j| {
        let mut m = moles.mapv(HyperDual::from);
        m[i].eps1 = DualSVec64::one();
        m[j].eps2 = DualSVec64::one();
        let state = StateHD::new(t, v, m);
        // (eos.evaluate_residual(&state).eps1eps2 + eos.ideal_gas().evaluate(&state).eps1eps2)
        eos.evaluate_residual(&state).eps1eps2 * (moles[i] * moles[j]).sqrt()
    });

    // calculate smallest eigenvalue and corresponding eigenvector of q
    let (eval, evec) = smallest_ev(qij);

    // evaluate third partial derivative w.r.t. s
    let moles_hd = Array1::from_shape_fn(eos.components(), |i| {
        Dual3::new(
            DualSVec64::from(moles[i]),
            evec[i] * moles[i].sqrt(),
            DualSVec64::zero(),
            DualSVec64::zero(),
        )
    });
    let state_s = StateHD::new(
        Dual3::from_re(temperature),
        Dual3::from_re(density.recip() * moles.sum()),
        moles_hd,
    );
    // let res = eos.evaluate_residual(&state_s) + eos.ideal_gas().evaluate(&state_s);
    let res = eos.evaluate_residual(&state_s);
    Ok(SVector::from([eval, res.v3]))
}

fn critical_point_objective_t<R: Residual>(
    eos: &Arc<R>,
    temperature: f64,
    density: SVector<DualSVec64<2>, 2>,
) -> EosResult<SVector<DualSVec64<2>, 2>> {
    // calculate second partial derivatives w.r.t. moles
    let t = HyperDual::from(temperature);
    let v = HyperDual::from(1.0);
    let qij = DMatrix::from_fn(eos.components(), eos.components(), |i, j| {
        let mut m = density.map(HyperDual::from_re);
        m[i].eps1 = DualSVec64::one();
        m[j].eps2 = DualSVec64::one();
        let state = StateHD::new(t, v, arr1(&[m[0], m[1]]));
        // (eos.evaluate_residual(&state).eps1eps2 + eos.ideal_gas().evaluate(&state).eps1eps2)
        eos.evaluate_residual(&state).eps1eps2 * (density[i] * density[j]).sqrt()
    });

    // calculate smallest eigenvalue and corresponding eigenvector of q
    let (eval, evec) = smallest_ev(qij);

    // evaluate third partial derivative w.r.t. s
    let moles_hd = Array1::from_shape_fn(eos.components(), |i| {
        Dual3::new(
            density[i],
            evec[i] * density[i].sqrt(),
            DualSVec64::zero(),
            DualSVec64::zero(),
        )
    });
    let state_s = StateHD::new(Dual3::from(temperature), Dual3::from(1.0), moles_hd);
    let res = eos.evaluate_residual(&state_s); // + eos.ideal_gas().evaluate(&state_s);
    Ok(SVector::from([eval, res.v3]))
}

fn critical_point_objective_p<R: Residual>(
    eos: &Arc<R>,
    pressure: f64,
    temperature: DualSVec64<3>,
    density: SVector<DualSVec64<3>, 2>,
) -> EosResult<SVector<DualSVec64<3>, 3>> {
    // calculate second partial derivatives w.r.t. moles
    let t = HyperDual::from_re(temperature);
    let v = HyperDual::from(1.0);
    let qij = DMatrix::from_fn(eos.components(), eos.components(), |i, j| {
        let mut m = density.map(HyperDual::from_re);
        m[i].eps1 = DualSVec64::one();
        m[j].eps2 = DualSVec64::one();
        let state = StateHD::new(t, v, arr1(&[m[0], m[1]]));
        // (eos.evaluate_residual(&state).eps1eps2 + eos.ideal_gas().evaluate(&state).eps1eps2)
        eos.evaluate_residual(&state).eps1eps2 * (density[i] * density[j]).sqrt()
    });

    // calculate smallest eigenvalue and corresponding eigenvector of q
    let (eval, evec) = smallest_ev(qij);

    // evaluate third partial derivative w.r.t. s
    let moles_hd = Array1::from_shape_fn(eos.components(), |i| {
        Dual3::new(
            density[i],
            evec[i] * density[i].sqrt(),
            DualSVec64::zero(),
            DualSVec64::zero(),
        )
    });
    let state_s = StateHD::new(Dual3::from_re(temperature), Dual3::from(1.0), moles_hd);
    let res = eos.evaluate_residual(&state_s); // + eos.ideal_gas().evaluate(&state_s);

    // calculate pressure
    let a = |v| {
        let m = arr1(&[Dual::from_re(density[0]), Dual::from_re(density[1])]);
        let state_p = StateHD::new(Dual::from_re(temperature), v, m);
        eos.evaluate_residual(&state_p) // + eos.ideal_gas().evaluate(&state_p)
    };
    let (_, p) = first_derivative(a, DualVec::one());

    Ok(SVector::from([eval, res.v3, p * temperature + pressure]))
}

fn spinodal_objective<R: Residual>(
    eos: &Arc<R>,
    temperature: Dual64,
    density: Dual64,
    moles: &Array1<f64>,
) -> EosResult<Dual64> {
    // calculate second partial derivatives w.r.t. moles
    let t = HyperDual::from_re(temperature);
    let v = HyperDual::from_re(density.recip() * moles.sum());
    let qij = DMatrix::from_fn(eos.components(), eos.components(), |i, j| {
        let mut m = moles.mapv(HyperDual::from);
        m[i].eps1 = Dual64::one();
        m[j].eps2 = Dual64::one();
        let state = StateHD::new(t, v, m);
        // (eos.evaluate_residual(&state).eps1eps2 + eos.ideal_gas().evaluate(&state).eps1eps2)
        eos.evaluate_residual(&state).eps1eps2 * (moles[i] * moles[j]).sqrt()
    });

    // calculate smallest eigenvalue of q
    let (eval, _) = smallest_ev_scalar(qij);

    Ok(eval)
}

fn smallest_ev<const N: usize>(
    m: DMatrix<DualSVec64<N>>,
) -> (DualSVec64<N>, DVector<DualSVec64<N>>) {
    let eig = SymmetricEigen::new(m);
    let (e, ev) = eig
        .eigenvalues
        .iter()
        .zip(eig.eigenvectors.column_iter())
        .reduce(|e1, e2| if e1.0 < e2.0 { e1 } else { e2 })
        .unwrap();
    (*e, ev.into())
}

fn smallest_ev_scalar(m: DMatrix<Dual64>) -> (Dual64, DVector<Dual64>) {
    let eig = SymmetricEigen::new(m);
    let (e, ev) = eig
        .eigenvalues
        .iter()
        .zip(eig.eigenvectors.column_iter())
        .reduce(|e1, e2| if e1.0 < e2.0 { e1 } else { e2 })
        .unwrap();
    (*e, ev.into())
}
