use super::PhaseEquilibrium;
use crate::errors::FeosResult;
use crate::state::State;
use crate::{Composition, FeosError, ReferenceSystem, SolverOptions, Total, Verbosity};
use nalgebra::{SVector, U2, vector};
use num_dual::linalg::LU;
use num_dual::{
    Dual, Dual64, DualNum, DualStruct, first_derivative, hessian, implicit_derivative_sp, partial,
};
use quantity::{Density, MolarEnergy, MolarEntropy, Pressure, Quantity, SIUnit, Temperature};

const MAX_ITER_PX: usize = 20;
const TOL_PX: f64 = 1e-11;

impl<E: Total<U2, D>, D: DualNum<f64> + Copy> PhaseEquilibrium<E, 2, U2, D> {
    /// Perform a ph-flash calculation. An initial temperature is required
    /// and the system needs to be in the two-phase region at that initial
    /// temperature.
    ///
    /// based on Michelsen's work [State function based flash specifications](https://doi.org/10.1016/S0378-3812(99)00092-8)
    pub fn ph_flash<X: Composition<D, U2>>(
        eos: &E,
        pressure: Pressure<D>,
        molar_enthalpy: MolarEnergy<D>,
        feed: X,
        initial_temperature: Temperature,
        options: SolverOptions,
    ) -> FeosResult<Self> {
        PhaseEquilibrium::px_flash(
            eos,
            pressure,
            molar_enthalpy,
            feed,
            initial_temperature,
            options,
        )
    }

    /// Perform a ps-flash calculation. An initial temperature is required
    /// and the system needs to be in the two-phase region at that initial
    /// temperature.
    ///
    /// based on Michelsen's work [State function based flash specifications](https://doi.org/10.1016/S0378-3812(99)00092-8)
    pub fn ps_flash<X: Composition<D, U2>>(
        eos: &E,
        pressure: Pressure<D>,
        molar_entropy: MolarEntropy<D>,
        feed: X,
        initial_temperature: Temperature,
        options: SolverOptions,
    ) -> FeosResult<Self> {
        PhaseEquilibrium::px_flash(
            eos,
            pressure,
            molar_entropy,
            feed,
            initial_temperature,
            options,
        )
    }

    // Generic implementation of ph and ps flashes.
    fn px_flash<X: Composition<D, U2>, U: PXFlash>(
        eos: &E,
        pressure: Pressure<D>,
        specification: Quantity<D, U>,
        feed: X,
        initial_temperature: Temperature,
        options: SolverOptions,
    ) -> FeosResult<Self>
    where
        Quantity<D, U>: ReferenceSystem<Inner = D>,
        Quantity<Dual64, U>: ReferenceSystem<Inner = Dual64>,
    {
        let (max_iter, tol, verbosity) = options.unwrap_or(MAX_ITER_PX, TOL_PX);
        let (molefracs, total_moles) = feed.into_molefracs(eos)?;

        // initialize with a tp flash
        let eos_f64 = eos.re_total();
        let vle = PhaseEquilibrium::tp_flash(
            &eos_f64,
            initial_temperature,
            pressure.re(),
            molefracs[0].re(),
            None,
            Default::default(),
            None,
        )?;

        // extract specifications
        let p = pressure.into_reduced().re();
        let s = specification.into_reduced().re();
        let [[z, _]] = molefracs.data.0;
        let specs = [p, s, z.re()];

        // extract variables
        let t = initial_temperature.into_reduced();
        let [[y, _]] = vle.vapor().molefracs.data.0;
        let v_v = vle.vapor().molar_volume.into_reduced();
        let [[x, _]] = vle.liquid().molefracs.data.0;
        let v_l = vle.liquid().molar_volume.into_reduced();
        let mut vars = SVector::from([t, v_l, v_v, x, y]);
        let mut old_res = None;

        log_iter!(
            verbosity,
            " iter |  method  | temperature |    residual    |  phase I mole fractions  |  phase II mole fractions  "
        );
        log_iter!(verbosity, "{:-<102}", "");
        log_iter!(
            verbosity,
            " {:4} |          | {:.5} |                | {:10.8?} | {:10.8?}",
            0,
            Temperature::from_reduced(t),
            [y, 1.0 - y],
            [x, 1.0 - x],
        );

        // iterate
        for k in 0..max_iter {
            let (grad, new_vars) = U::newton_step(&eos_f64, &vars, &specs)?;
            let new_res = grad.norm();
            let (method, res) = if let Some(r) = old_res
                && r < new_res
            {
                // if the residual is not reduced, reject the step and do a tp-flash instead
                vars = U::tp_step(&eos_f64, &vars, &specs)?;
                ("Tp-flash", None)
            } else {
                vars = new_vars;
                ("Newton", Some(new_res))
            };

            log_iter!(
                verbosity,
                " {:4} | {:^8} | {:.5} | {} | {:10.8?} | {:10.8?}",
                k + 1,
                method,
                Temperature::from_reduced(vars[0]),
                res.map_or(String::from("              "), |r| format!("{r:14.8e}")),
                [vars[4], 1.0 - vars[4]],
                [vars[3], 1.0 - vars[3]],
            );

            if let Some(res) = res
                && res < tol
            {
                log_result!(
                    verbosity,
                    "px flash: calculation converged in {} step(s)\n",
                    k + 1
                );

                // implicit differentiation
                let specs = [pressure.into_reduced(), specification.into_reduced(), z];
                let [[t, v_l, v_v, x, y]] = implicit_derivative_sp(
                    |variables, specifications| {
                        U::potential_function(&eos.lift_total(), variables, specifications)
                    },
                    vars,
                    &specs,
                )
                .data
                .0;

                // store results in PhaseEquilibrium
                let liquid = State::new(
                    eos,
                    Temperature::from_reduced(t),
                    Density::from_reduced(v_l.recip()),
                    x,
                )?;
                let vapor = State::new(
                    eos,
                    Temperature::from_reduced(t),
                    Density::from_reduced(v_v.recip()),
                    y,
                )?;
                let beta = (-x + z) / (y - x);
                return Ok(PhaseEquilibrium::with_vapor_phase_fraction(
                    vapor,
                    liquid,
                    beta,
                    total_moles,
                ));
            }
            old_res = res;
        }
        Err(FeosError::NotConverged("px flash".to_owned()))
    }
}

trait PXFlash: Sized + Copy {
    // potential function for which the flash solution is a saddle point.
    fn potential_function<E: Total<U2, D>, D: DualNum<f64> + Copy>(
        eos: &E,
        variables: SVector<D, 5>,
        args: &[D; 3],
    ) -> D;

    fn evaluate_property<E: Total<U2, D>, D: DualNum<f64> + Copy>(
        vle: &PhaseEquilibrium<E, 2, U2, D>,
    ) -> Quantity<D, Self>;

    // the potential function for a tp-flash specification (Q = A + V*p_spec)
    fn tp_potential_function<E: Total<U2, D>, D: DualNum<f64> + Copy>(
        eos: &E,
        variables: SVector<D, 4>,
        &[t, p, z]: &[D; 3],
    ) -> D {
        let [[v_l, v_v, x, y]] = variables.data.0;
        let beta = (z - x) / (y - x);
        let potential = |x: D, v, t| {
            let molefracs = vector![x, -x + 1.0];
            let a_res = eos.residual_helmholtz_energy(t, v, &molefracs);
            let a_ig = eos.ideal_gas_molar_helmholtz_energy(t, v, &molefracs);
            a_res + a_ig + v * p
        };
        potential(y, v_v, t) * beta + potential(x, v_l, t) * (-beta + 1.0)
    }

    // An undamped Newton step for the gradients of the potential function.
    // Because the ps and ph flashes are saddle points rather then extrema,
    // the value of the potential can not be used as convergence criterion.
    fn newton_step<E: Total<U2, D>, D: DualNum<f64> + Copy>(
        eos: &E,
        variables: &SVector<D, 5>,
        specifications: &[D; 3],
    ) -> FeosResult<(SVector<D, 5>, SVector<D, 5>)> {
        let (_, grad, hess) = hessian(
            partial(
                |variables, specifications| {
                    Self::potential_function(&eos.lift_total(), variables, specifications)
                },
                specifications,
            ),
            variables,
        );
        let dx = LU::new(hess)?.solve(&grad);
        Ok((grad, variables - dx))
    }

    // A much slower but more robust step that calculates the implicit
    // derivative of the temperature only (which is well behaved
    // according to Michelsen) and then calculates all other variables
    // from a tp-flash.
    fn tp_step<E: Total<U2, f64>>(
        eos: &E,
        variables: &SVector<f64, 5>,
        specifications: &[f64; 3],
    ) -> FeosResult<SVector<f64, 5>>
    where
        Quantity<Dual64, Self>: ReferenceSystem<Inner = Dual64>,
    {
        let &[p, hs_spec, z] = specifications;
        let [[mut t, v_l, v_v, x, y]] = variables.data.0;
        let (hs, dhs) = first_derivative(
            partial(
                |t: Dual<_, _>, args| {
                    let &[p, z] = args;
                    let args = [t, p, z];

                    // implicit differentiation of the tp stationarity condition
                    // to obtain the derivative of the other variables w.r.t. t
                    let [[v_l, v_v, x, y]] = implicit_derivative_sp(
                        |variables, args| {
                            Self::tp_potential_function(
                                &eos.lift_total().lift_total(),
                                variables,
                                args,
                            )
                        },
                        SVector::from([v_l, v_v, x, y]),
                        &args,
                    )
                    .data
                    .0;
                    // Evaluation of the enthalpy/entropy including the derivatives.
                    let liquid = State::new(
                        &eos.lift_total(),
                        Temperature::from_reduced(t),
                        Density::from_reduced(v_l.recip()),
                        x,
                    )
                    .unwrap();
                    let vapor = State::new(
                        &eos.lift_total(),
                        Temperature::from_reduced(t),
                        Density::from_reduced(v_v.recip()),
                        y,
                    )
                    .unwrap();
                    let beta = (-x + z) / (y - x);
                    Self::evaluate_property(&PhaseEquilibrium::with_vapor_phase_fraction(
                        vapor, liquid, beta, None,
                    ))
                    .into_reduced()
                },
                &[p, z],
            ),
            t,
        );

        // Newton step for the temperature
        t -= (hs - hs_spec) / dhs;

        // tp-flash for all other variables
        let vle = PhaseEquilibrium::tp_flash(
            eos,
            Temperature::from_reduced(t),
            Pressure::from_reduced(p),
            z,
            None,
            Default::default(),
            None,
        )?;
        let [[y, _]] = vle.vapor().molefracs.data.0;
        let v_v = vle.vapor().molar_volume.into_reduced();
        let [[x, _]] = vle.liquid().molefracs.data.0;
        let v_l = vle.liquid().molar_volume.into_reduced();
        Ok(SVector::from([t, v_l, v_v, x, y]))
    }
}

impl PXFlash for SIUnit<-2, 2, 1, 0, 0, -1, 0> {
    // the potential function for a ph-flash specification (Q = (A + V*p_spec - H_spec) / T)
    fn potential_function<E: Total<U2, D>, D: DualNum<f64> + Copy>(
        eos: &E,
        variables: SVector<D, 5>,
        &[p, h, z]: &[D; 3],
    ) -> D {
        let [[t, v_l, v_v, x, y]] = variables.data.0;
        let beta = (z - x) / (y - x);
        let potential = |x: D, v, t| {
            let molefracs = vector![x, -x + 1.0];
            let a_res = eos.residual_helmholtz_energy(t, v, &molefracs);
            let a_ig = eos.ideal_gas_molar_helmholtz_energy(t, v, &molefracs);
            (a_res + a_ig + v * p - h) / t
        };
        potential(y, v_v, t) * beta + potential(x, v_l, t) * (-beta + 1.0)
    }

    fn evaluate_property<E: Total<U2, D>, D: DualNum<f64> + Copy>(
        vle: &PhaseEquilibrium<E, 2, U2, D>,
    ) -> Quantity<D, Self> {
        vle.molar_enthalpy()
    }
}

impl PXFlash for SIUnit<-2, 2, 1, 0, -1, -1, 0> {
    // the potential function for a ps-flash specification (Q = A + T*S_spec + V*p_spec)
    fn potential_function<E: Total<U2, D>, D: DualNum<f64> + Copy>(
        eos: &E,
        variables: SVector<D, 5>,
        &[p, s, z]: &[D; 3],
    ) -> D {
        let [[t, v_l, v_v, x, y]] = variables.data.0;
        let beta = (z - x) / (y - x);
        let potential = |x: D, v, t| {
            let molefracs = vector![x, -x + 1.0];
            let a_res = eos.residual_helmholtz_energy(t, v, &molefracs);
            let a_ig = eos.ideal_gas_molar_helmholtz_energy(t, v, &molefracs);
            a_res + a_ig + t * s + v * p
        };
        potential(y, v_v, t) * beta + potential(x, v_l, t) * (-beta + 1.0)
    }

    fn evaluate_property<E: Total<U2, D>, D: DualNum<f64> + Copy>(
        vle: &PhaseEquilibrium<E, 2, U2, D>,
    ) -> Quantity<D, Self> {
        vle.molar_entropy()
    }
}
