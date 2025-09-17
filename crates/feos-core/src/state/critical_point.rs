use super::{DensityInitialization, State};
use crate::equation_of_state::Residual;
use crate::errors::{FeosError, FeosResult};
use crate::{ReferenceSystem, SolverOptions, Subset, TemperatureOrPressure, Verbosity};
use nalgebra::allocator::Allocator;
use nalgebra::{DVector, DefaultAllocator, OMatrix, OVector, SVector, U1, U2, U3};
use num_dual::linalg::smallest_ev;
use num_dual::{
    Dual3, DualNum, DualSVec, DualSVec64, DualStruct, Gradients, first_derivative,
    implicit_derivative_binary, implicit_derivative_vec, jacobian, partial, partial2,
    third_derivative,
};
use quantity::{Density, Pressure, Temperature};

const MAX_ITER_CRIT_POINT: usize = 50;
const MAX_ITER_CRIT_POINT_BINARY: usize = 200;
const TOL_CRIT_POINT: f64 = 1e-8;

/// # Critical points
impl<R: Residual + Subset> State<R> {
    /// Calculate the pure component critical point of all components.
    pub fn critical_point_pure(
        eos: &R,
        initial_temperature: Option<Temperature>,
        options: SolverOptions,
    ) -> FeosResult<Vec<Self>> {
        (0..eos.components())
            .map(|i| {
                let pure_eos = eos.subset(&[i]);
                let cp = State::critical_point(&pure_eos, None, initial_temperature, options)?;
                let mut molefracs = DVector::zeros(eos.components());
                molefracs[i] = 1.0;
                State::new_intensive(eos, cp.temperature, cp.density, &molefracs)
            })
            .collect()
    }
}

impl<E: Residual<N, D>, N: Gradients, D: DualNum<f64> + Copy> State<E, N, D>
where
    DefaultAllocator: Allocator<N> + Allocator<N, N> + Allocator<U1, N>,
{
    pub fn critical_point_binary<TP: TemperatureOrPressure<D>>(
        eos: &E,
        temperature_or_pressure: TP,
        initial_temperature: Option<Temperature>,
        initial_molefracs: Option<[f64; 2]>,
        options: SolverOptions,
    ) -> FeosResult<Self> {
        let eos_re = eos.re();
        let n = N::from_usize(2);
        let initial_molefracs = initial_molefracs.unwrap_or([0.5; 2]);
        let initial_molefracs = OVector::from_fn_generic(n, U1, |i, _| initial_molefracs[i]);
        if let Some(t) = temperature_or_pressure.temperature() {
            let [rho0, rho1] =
                critical_point_binary_t(&eos_re, t.re(), initial_molefracs, options)?;
            let rho = implicit_derivative_binary(
                |rho0, rho1, &temperature| {
                    let rho = [rho0, rho1];
                    let density = rho0 + rho1;
                    let molefracs = OVector::from_fn_generic(n, U1, |i, _| rho[i] / density);
                    criticality_conditions(&eos.lift(), temperature, density, &molefracs)
                },
                rho0,
                rho1,
                &t.into_reduced(),
            );
            let density = rho[0] + rho[1];
            let molefracs = OVector::from_fn_generic(n, U1, |i, _| rho[i] / density);
            Self::new_intensive(eos, t, Density::from_reduced(density), &molefracs)
        } else if let Some(p) = temperature_or_pressure.pressure() {
            let x = critical_point_binary_p(
                &eos_re,
                p.re(),
                initial_temperature,
                initial_molefracs,
                options,
            )?;
            let trho = implicit_derivative_vec::<_, _, _, _, U3>(
                |x, &p| {
                    let t = x[0];
                    let rho = [x[1], x[2]];
                    criticality_conditions_p(&eos.lift(), p, t, rho)
                },
                SVector::from(x),
                &p.into_reduced(),
            );
            let density = trho[1] + trho[2];
            let molefracs = OVector::from_fn_generic(n, U1, |i, _| trho[i + 1] / density);
            let t = Temperature::from_reduced(trho[0]);
            Self::new_intensive(eos, t, Density::from_reduced(density), &molefracs)
        } else {
            unreachable!()
        }
    }

    /// Calculate the critical point of a system for given moles.
    pub fn critical_point(
        eos: &E,
        molefracs: Option<&OVector<D, N>>,
        initial_temperature: Option<Temperature>,
        options: SolverOptions,
    ) -> FeosResult<Self> {
        let eos_re = eos.re();
        let molefracs = molefracs.map_or_else(E::pure_molefracs, |x| x.clone());
        let x = &molefracs.map(|x| x.re());
        let trial_temperatures = [300.0, 700.0, 500.0];
        let mut t_rho = None;
        if let Some(t) = initial_temperature {
            t_rho = Some(critical_point_hkm(&eos_re, x, t.into_reduced(), options)?);
        }
        for &t in trial_temperatures.iter() {
            if t_rho.is_some() {
                break;
            }
            t_rho = critical_point_hkm(&eos_re, x, t, options).ok();
        }
        let Some(t_rho) = t_rho else {
            return Err(FeosError::NotConverged(String::from("Critical point")));
        };

        // implicit derivative
        let [temperature, density] = implicit_derivative_binary(
            |t, rho, x| criticality_conditions(&eos.lift(), t, rho, x),
            t_rho[0],
            t_rho[1],
            &molefracs,
        );
        Self::new_intensive(
            eos,
            Temperature::from_reduced(temperature),
            Density::from_reduced(density),
            &molefracs,
        )
    }
}
fn critical_point_hkm<E: Residual<N>, N: Gradients>(
    eos: &E,
    molefracs: &OVector<f64, N>,
    initial_temperature: f64,
    options: SolverOptions,
) -> FeosResult<[f64; 2]>
where
    DefaultAllocator: Allocator<N> + Allocator<N, N> + Allocator<U1, N>,
{
    let (max_iter, tol, verbosity) = options.unwrap_or(MAX_ITER_CRIT_POINT, TOL_CRIT_POINT);

    let mut t = initial_temperature;
    let max_density = eos.compute_max_density(molefracs);
    let mut rho = 0.3 * max_density;

    log_iter!(
        verbosity,
        " iter |    residual    |   temperature   |       density        "
    );
    log_iter!(verbosity, "{:-<64}", "");
    log_iter!(
        verbosity,
        " {:4} |                | {:13.8} | {:12.8}",
        0,
        Temperature::from_reduced(t),
        Density::from_reduced(rho),
    );

    for i in 1..=max_iter {
        // calculate residuals and derivatives w.r.t. temperature and density
        let (res, jac) = jacobian::<_, _, _, U2, U2, _>(
            |x: SVector<DualSVec64<2>, 2>| {
                SVector::from(criticality_conditions(
                    &eos.lift(),
                    x[0],
                    x[1],
                    &molefracs.map(DualSVec::from_re),
                ))
            },
            &SVector::from([t, rho]),
        );

        // calculate Newton step
        let delta = jac.lu().solve::<U2, U1, _>(&res);
        let mut delta = delta.ok_or(FeosError::IterationFailed("Critical point".into()))?;

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
            Temperature::from_reduced(t),
            Density::from_reduced(rho),
        );

        // check convergence
        if res.norm() < tol {
            log_result!(
                verbosity,
                "Critical point calculation converged in {} step(s)\n",
                i
            );
            return Ok([t, rho]);
        }
    }
    Err(FeosError::NotConverged(String::from("Critical point")))
}

/// Calculate the critical point of a binary system for given temperature.
fn critical_point_binary_t<E: Residual<N>, N: Gradients>(
    eos: &E,
    temperature: Temperature,
    initial_molefracs: OVector<f64, N>,
    options: SolverOptions,
) -> FeosResult<[f64; 2]>
where
    DefaultAllocator: Allocator<N> + Allocator<N, N> + Allocator<U1, N>,
{
    let (max_iter, tol, verbosity) = options.unwrap_or(MAX_ITER_CRIT_POINT_BINARY, TOL_CRIT_POINT);

    let t = temperature.to_reduced();
    let n = N::from_usize(2);
    let max_density = eos.compute_max_density(&initial_molefracs);
    let mut rho = SVector::from([initial_molefracs[0], initial_molefracs[1]]) * 0.3 * max_density;

    log_iter!(
        verbosity,
        " iter |    residual    |      density 1       |      density 2       "
    );
    log_iter!(verbosity, "{:-<69}", "");
    log_iter!(
        verbosity,
        " {:4} |                | {:12.8} | {:12.8}",
        0,
        Density::from_reduced(rho[0]),
        Density::from_reduced(rho[1]),
    );

    for i in 1..=max_iter {
        // calculate residuals and derivatives w.r.t. partial densities
        let (res, jac) = jacobian::<_, _, _, U2, U2, _>(
            |rho: SVector<DualSVec64<2>, 2>| {
                let density = rho.sum();
                let x = rho / density;
                let molefracs = OVector::from_fn_generic(n, U1, |i, _| x[i]);
                let t = DualSVec::from_re(t);
                SVector::from(criticality_conditions(&eos.lift(), t, density, &molefracs))
            },
            &rho,
        );

        // calculate Newton step
        let delta = jac.lu().solve(&res);
        let mut delta = delta.ok_or(FeosError::IterationFailed("Critical point".into()))?;

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
            Density::from_reduced(rho[0]),
            Density::from_reduced(rho[1]),
        );

        // check convergence
        if res.norm() < tol {
            log_result!(
                verbosity,
                "Critical point calculation converged in {} step(s)\n",
                i
            );
            return Ok(rho.data.0[0]);
        }
    }
    Err(FeosError::NotConverged(String::from("Critical point")))
}

/// Calculate the critical point of a binary system for given pressure.
fn critical_point_binary_p<E: Residual<N>, N: Gradients>(
    eos: &E,
    pressure: Pressure,
    initial_temperature: Option<Temperature>,
    initial_molefracs: OVector<f64, N>,
    options: SolverOptions,
) -> FeosResult<[f64; 3]>
where
    DefaultAllocator: Allocator<N> + Allocator<N, N> + Allocator<U1, N>,
{
    let (max_iter, tol, verbosity) = options.unwrap_or(MAX_ITER_CRIT_POINT_BINARY, TOL_CRIT_POINT);

    let p = pressure.to_reduced();
    let mut t = initial_temperature.map(|t| t.to_reduced()).unwrap_or(300.0);
    let max_density = eos.compute_max_density(&initial_molefracs);
    let mut rho = SVector::from([initial_molefracs[0], initial_molefracs[1]]) * 0.3 * max_density;

    log_iter!(
        verbosity,
        " iter |    residual    |   temperature   |      density 1       |      density 2       "
    );
    log_iter!(verbosity, "{:-<87}", "");
    log_iter!(
        verbosity,
        " {:4} |                | {:13.8} | {:12.8} | {:12.8}",
        0,
        Temperature::from_reduced(t),
        Density::from_reduced(rho[0]),
        Density::from_reduced(rho[1]),
    );

    for i in 1..=max_iter {
        // calculate residuals and derivatives w.r.t. temperature and partial densities
        let res = |x: SVector<DualSVec64<3>, 3>| {
            let t = x[0];
            let partial_density = [x[1], x[2]];
            let p = DualSVec::from_re(p);
            criticality_conditions_p(&eos.lift(), p, t, partial_density)
        };
        let (res, jac) = jacobian::<_, _, _, U3, U3, _>(res, &SVector::from([t, rho[0], rho[1]]));

        // calculate Newton step
        let delta = jac.lu().solve(&res);
        let mut delta = delta.ok_or(FeosError::IterationFailed("Critical point".into()))?;

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
            Temperature::from_reduced(t),
            Density::from_reduced(rho[0]),
            Density::from_reduced(rho[1]),
        );

        // check convergence
        if res.norm() < tol {
            log_result!(
                verbosity,
                "Critical point calculation converged in {} step(s)\n",
                i
            );
            return Ok([t, rho[0], rho[1]]);
        }
    }
    Err(FeosError::NotConverged(String::from("Critical point")))
}

impl<E: Residual<N>, N: Gradients> State<E, N>
where
    DefaultAllocator: Allocator<N> + Allocator<N, N> + Allocator<U1, N>,
{
    pub fn spinodal(
        eos: &E,
        temperature: Temperature,
        molefracs: Option<&OVector<f64, N>>,
        options: SolverOptions,
    ) -> FeosResult<[Self; 2]> {
        let critical_point = Self::critical_point(eos, molefracs, None, options)?;
        let molefracs = molefracs.map_or_else(E::pure_molefracs, |x| x.clone());
        let spinodal_vapor = Self::calculate_spinodal(
            eos,
            temperature,
            &molefracs,
            DensityInitialization::Vapor,
            options,
        )?;
        let rho = 2.0 * critical_point.density - spinodal_vapor.density;
        let spinodal_liquid = Self::calculate_spinodal(
            eos,
            temperature,
            &molefracs,
            DensityInitialization::InitialDensity(rho),
            options,
        )?;
        Ok([spinodal_vapor, spinodal_liquid])
    }

    fn calculate_spinodal(
        eos: &E,
        temperature: Temperature,
        molefracs: &OVector<f64, N>,
        density_initialization: DensityInitialization,
        options: SolverOptions,
    ) -> FeosResult<Self> {
        let (max_iter, tol, verbosity) = options.unwrap_or(MAX_ITER_CRIT_POINT, TOL_CRIT_POINT);

        let max_density = eos.compute_max_density(molefracs);
        let t = temperature.to_reduced();
        let mut rho = match density_initialization {
            DensityInitialization::Vapor => 1e-5 * max_density,
            DensityInitialization::Liquid => max_density,
            DensityInitialization::InitialDensity(rho) => rho.to_reduced(),
        };

        log_iter!(verbosity, " iter |    residual    |       density        ");
        log_iter!(verbosity, "{:-<46}", "");
        log_iter!(
            verbosity,
            " {:4} |                | {:12.8}",
            0,
            Density::from_reduced(rho),
        );

        for i in 1..=max_iter {
            // calculate residuals and derivative w.r.t. density
            let (f, df) = first_derivative(
                partial(
                    |rho, x| stability_condition(&eos.lift(), t.into(), rho, x),
                    molefracs,
                ),
                rho,
            );

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
                Density::from_reduced(rho),
            );

            // check convergence
            if f.abs() < tol {
                log_result!(
                    verbosity,
                    "Spinodal calculation converged in {} step(s)\n",
                    i
                );
                return Self::new_intensive(
                    eos,
                    temperature,
                    Density::from_reduced(rho),
                    molefracs,
                );
            }
        }
        Err(FeosError::SuperCritical)
    }
}

fn criticality_conditions<E: Residual<N, D>, N: Gradients, D: DualNum<f64> + Copy>(
    eos: &E,
    temperature: D,
    density: D,
    molefracs: &OVector<D, N>,
) -> [D; 2]
where
    DefaultAllocator: Allocator<N> + Allocator<U1, N> + Allocator<N, N>,
{
    // calculate M
    let molar_volume = density.recip();
    let sqrt_z = molefracs.map(|z| z.sqrt());
    let z_mix = &sqrt_z * sqrt_z.transpose();
    let m = dmu_dn(eos, temperature, molar_volume, molefracs);
    let (r, c) = m.shape_generic();
    let m = m.component_mul(&z_mix) + OMatrix::identity_generic(r, c);

    // calculate smallest eigenvalue and corresponding eigenvector
    let (l, u) = smallest_ev(m);

    let (_, _, _, c2) = third_derivative(
        |s| {
            let n = molefracs.map(Dual3::from_re);
            let n = n + sqrt_z.component_mul(&u).map(Dual3::from_re) * s;
            let t = Dual3::from_re(temperature);
            let v = Dual3::from_re(molar_volume);
            let ig = n.dot(&n.map(|n| (n / v).ln() - 1.0));
            eos.lift().residual_helmholtz_energy(t, v, &n) / t + ig
        },
        D::from(0.0),
    );

    [l, c2]
}

fn criticality_conditions_p<E: Residual<N, D>, N: Gradients, D: DualNum<f64> + Copy>(
    eos: &E,
    pressure: D,
    temperature: D,
    partial_density: [D; 2],
) -> SVector<D, 3>
where
    DefaultAllocator: Allocator<N> + Allocator<U1, N> + Allocator<N, N>,
{
    let density = partial_density[0] + partial_density[1];
    let n = N::from_usize(2);
    let molefracs: OVector<D, N> =
        OVector::from_fn_generic(n, U1, |i, _| partial_density[i] / density);

    let [c1, c2] = criticality_conditions(eos, temperature, density, &molefracs);

    // calculate pressure
    let a = partial2(
        |v, &t, x| eos.lift().residual_molar_helmholtz_energy(t, v, x),
        &temperature,
        &molefracs,
    );
    let (_, da) = first_derivative(a, D::one());
    let p_calc = -da + density * temperature;
    SVector::from([c1, c2, -p_calc + pressure])
}

fn stability_condition<E: Residual<N, D>, N: Gradients, D: DualNum<f64> + Copy>(
    eos: &E,
    temperature: D,
    density: D,
    molefracs: &OVector<D, N>,
) -> D
where
    DefaultAllocator: Allocator<N> + Allocator<U1, N> + Allocator<N, N>,
{
    // calculate M
    let molar_volume = density.recip();
    let sqrt_z = molefracs.map(|z| z.sqrt());
    let z_mix = &sqrt_z * sqrt_z.transpose();
    let m = dmu_dn(eos, temperature, molar_volume, molefracs);
    let (r, c) = m.shape_generic();
    let m = m.component_mul(&z_mix) + OMatrix::identity_generic(r, c);

    // calculate smallest eigenvalue and corresponding eigenvector
    let (l, _) = smallest_ev(m);

    l
}

fn dmu_dn<E: Residual<N, D>, N: Gradients, D: DualNum<f64> + Copy>(
    eos: &E,
    temperature: D,
    molar_volume: D,
    molefracs: &OVector<D, N>,
) -> OMatrix<D, N, N>
where
    DefaultAllocator: Allocator<N> + Allocator<N, N>,
{
    let (_, _, h) = N::hessian(
        |n, &(t, v)| eos.lift().residual_helmholtz_energy(t, v, &n) / t,
        molefracs,
        &(temperature, molar_volume),
    );
    h
}
