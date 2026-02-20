#![expect(clippy::toplevel_ref_arg)]
use super::PhaseEquilibrium;
use crate::errors::FeosResult;
use crate::state::State;
use crate::{Composition, FeosError, ReferenceSystem, SolverOptions, Total, Verbosity};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, DimAdd, OVector, U1, U2, U3, stack, vector};
use num_dual::linalg::LU;
use num_dual::{
    Dual, Dual64, DualNum, DualStruct, Gradients, first_derivative, implicit_derivative_sp, partial,
};
use quantity::{Density, MolarEnergy, MolarEntropy, Pressure, Quantity, SIUnit, Temperature};

const MAX_ITER_PX: usize = 20;
const TOL_PX: f64 = 1e-11;

type PXVars<N> = <N as DimAdd<U3>>::Output;
type TPVars<N> = <N as DimAdd<U2>>::Output;

impl<E: Total<N, D>, N: Gradients + DimAdd<U2> + DimAdd<U3>, D: DualNum<f64> + Copy>
    PhaseEquilibrium<E, 2, N, D>
where
    DefaultAllocator: Allocator<N>
        + Allocator<N, N>
        + Allocator<PXVars<N>>
        + Allocator<U1, PXVars<N>>
        + Allocator<PXVars<N>, PXVars<N>>
        + Allocator<TPVars<N>>
        + Allocator<U1, TPVars<N>>
        + Allocator<TPVars<N>, TPVars<N>>,
    PXVars<N>: Gradients,
    TPVars<N>: Gradients,
{
    /// Perform a ph-flash calculation. An initial temperature is required
    /// and the system needs to be in the two-phase region at that initial
    /// temperature.
    ///
    /// based on Michelsen's work [State function based flash specifications](https://doi.org/10.1016/S0378-3812(99)00092-8)
    pub fn ph_flash<X: Composition<D, N>>(
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
    pub fn ps_flash<X: Composition<D, N>>(
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
    fn px_flash<X: Composition<D, N>, U: PXFlash>(
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
            molefracs.map(|x| x.re()),
            None,
            Default::default(),
            None,
        )?;

        // extract specifications
        let p = pressure.into_reduced().re();
        let hs = specification.into_reduced().re();
        let z = molefracs.map(|x| x.re());
        let specs = (p, hs, z.clone());

        // extract variables
        let t = initial_temperature.into_reduced();
        let beta = vle.vapor_phase_fraction();
        let rho_v = vle.vapor().density.into_reduced();
        let rho_l = vle.liquid().partial_density().into_reduced();
        let mut vars = stack![rho_l; vector![t, beta, rho_v]];
        let mut old_res = None;

        log_iter!(
            verbosity,
            " iter |  method  | temperature |    residual    |  phase I mole fractions  |  phase II mole fractions  "
        );
        log_iter!(verbosity, "{:-<102}", "");
        log_iter!(
            verbosity,
            " {:4} |          | {:9.5} |                | {:10.8?} | {:10.8?}",
            0,
            Temperature::from_reduced(t),
            (&rho_l / rho_l.sum() + (&z - &rho_l / rho_l.sum()) / beta).as_slice(),
            (&rho_l / rho_l.sum()).as_slice(),
        );

        // iterate
        for k in 0..max_iter {
            // always try a Newton step first
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

            if let Verbosity::Iter = verbosity {
                let (t, _, _, _, x, y) = unpack_variables(&z, &vars);
                log_iter!(
                    verbosity,
                    " {:4} | {:^8} | {:9.5} | {} | {:10.8?} | {:10.8?}",
                    k + 1,
                    method,
                    Temperature::from_reduced(t),
                    res.map_or(String::from("              "), |r| format!("{r:14.8e}")),
                    y.as_slice(),
                    x.as_slice(),
                );
            }

            if let Some(res) = res
                && res < tol
            {
                log_result!(
                    verbosity,
                    "px flash: calculation converged in {} step(s)\n",
                    k + 1
                );

                // implicit differentiation
                let specs = (
                    pressure.into_reduced(),
                    specification.into_reduced(),
                    molefracs.clone(),
                );
                let vars = implicit_derivative_sp(
                    |variables, specifications| {
                        U::state_function(&eos.lift_total(), variables, specifications)
                    },
                    vars,
                    &specs,
                );
                let (t, beta, rho_l, rho_v, x, y) = unpack_variables(&molefracs, &vars);

                // store results in PhaseEquilibrium
                let liquid = State::new(
                    eos,
                    Temperature::from_reduced(t),
                    Density::from_reduced(rho_l),
                    x,
                )?;
                let vapor = State::new(
                    eos,
                    Temperature::from_reduced(t),
                    Density::from_reduced(rho_v),
                    y,
                )?;
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

fn unpack_variables<D: DualNum<f64> + Copy, N: Dim + DimAdd<U3>>(
    molefracs: &OVector<D, N>,
    variables: &OVector<D, PXVars<N>>,
) -> (D, D, D, D, OVector<D, N>, OVector<D, N>)
where
    DefaultAllocator: Allocator<N> + Allocator<PXVars<N>>,
{
    let n = molefracs.len();
    let rho_i_l = variables.rows_generic(0, N::from_usize(n)).clone_owned();
    let [[t, beta, rho_v]] = variables.rows_generic(n, U3).clone_owned().data.0;
    let rho_l = rho_i_l.sum();
    let x = rho_i_l / rho_l;
    let y = &x + (molefracs - &x) / beta;
    (t, beta, rho_l, rho_v, x, y)
}

fn unpack_tp_variables<D: DualNum<f64> + Copy, N: Dim + DimAdd<U2>>(
    molefracs: &OVector<D, N>,
    variables: &OVector<D, TPVars<N>>,
) -> (D, D, D, OVector<D, N>, OVector<D, N>)
where
    DefaultAllocator: Allocator<N> + Allocator<TPVars<N>>,
{
    let n = molefracs.len();
    let rho_i_l = variables.rows_generic(0, N::from_usize(n)).clone_owned();
    let [[beta, rho_v]] = variables.rows_generic(n, U2).clone_owned().data.0;
    let rho_l = rho_i_l.sum();
    let x = rho_i_l / rho_l;
    let y = &x + (molefracs - &x) / beta;
    (beta, rho_l, rho_v, x, y)
}

trait PXFlash: Sized + Copy {
    // potential function for which the flash solution is a saddle point.
    fn state_function<E: Total<N, D>, N: Dim + DimAdd<U3>, D: DualNum<f64> + Copy>(
        eos: &E,
        variables: OVector<D, PXVars<N>>,
        args: &(D, D, OVector<D, N>),
    ) -> D
    where
        DefaultAllocator: Allocator<N> + Allocator<PXVars<N>>;

    fn evaluate_property<E: Total<N, D>, N: Gradients, D: DualNum<f64> + Copy>(
        vle: &PhaseEquilibrium<E, 2, N, D>,
    ) -> Quantity<D, Self>
    where
        DefaultAllocator: Allocator<N>;

    // the potential function for a tp-flash specification (Q = A + V*p_spec)
    fn tp_state_function<E: Total<N, D>, N: Dim + DimAdd<U2>, D: DualNum<f64> + Copy>(
        eos: &E,
        variables: OVector<D, TPVars<N>>,
        &(t, p, ref z): &(D, D, OVector<D, N>),
    ) -> D
    where
        DefaultAllocator: Allocator<N> + Allocator<TPVars<N>>,
    {
        let (beta, rho_l, rho_v, x, y) = unpack_tp_variables(z, &variables);
        let potential = |molefracs, rho: D, t| {
            let v = rho.recip();
            let a_res = eos.residual_helmholtz_energy(t, v, &molefracs);
            let a_ig = eos.ideal_gas_molar_helmholtz_energy(t, v, &molefracs);
            a_res + a_ig + v * p
        };
        potential(y, rho_v, t) * beta + potential(x, rho_l, t) * (-beta + 1.0)
    }

    // An undamped Newton step for the gradients of the potential function.
    // Because the ps and ph flashes are saddle points rather then extrema,
    // the value of the potential can not be used as convergence criterion.
    #[expect(clippy::type_complexity)]
    fn newton_step<E: Total<N, D>, N: Dim + DimAdd<U3>, D: DualNum<f64> + Copy>(
        eos: &E,
        variables: &OVector<D, PXVars<N>>,
        specifications: &(D, D, OVector<D, N>),
    ) -> FeosResult<(OVector<D, PXVars<N>>, OVector<D, PXVars<N>>)>
    where
        DefaultAllocator: Allocator<N> + Allocator<PXVars<N>> + Allocator<PXVars<N>, PXVars<N>>,
        PXVars<N>: Gradients,
    {
        let (_, grad, hess) = PXVars::<N>::hessian(
            |variables, specifications| {
                Self::state_function(&eos.lift_total(), variables, specifications)
            },
            variables,
            specifications,
        );
        let dx = LU::<D, f64, PXVars<N>>::new(hess)?.solve(&grad);
        Ok((grad, variables - &dx))
    }

    // A much slower but more robust step that calculates the implicit
    // derivative of the temperature only (which is well behaved
    // according to Michelsen) and then calculates all other variables
    // from a tp-flash.
    fn tp_step<E: Total<N, f64>, N: Gradients + DimAdd<U2> + DimAdd<U3>>(
        eos: &E,
        variables: &OVector<f64, PXVars<N>>,
        &(p, hs_spec, ref z): &(f64, f64, OVector<f64, N>),
    ) -> FeosResult<OVector<f64, PXVars<N>>>
    where
        Quantity<Dual64, Self>: ReferenceSystem<Inner = Dual64>,
        DefaultAllocator: Allocator<N>
            + Allocator<N, N>
            + Allocator<PXVars<N>>
            + Allocator<U1, TPVars<N>>
            + Allocator<TPVars<N>>
            + Allocator<TPVars<N>, TPVars<N>>,
        TPVars<N>: Gradients,
    {
        let (mut t, beta, rho_l, rho_v, x, y) = unpack_variables(z, variables);
        let rho_i_l = rho_l * x;
        let (hs, dhs) = first_derivative(
            partial(
                |t: Dual<_, _>, args: &(_, OVector<_, _>)| {
                    let &(p, ref z) = args;
                    let args = (t, p, z.clone_owned());

                    // implicit differentiation of the tp stationarity condition
                    // to obtain the derivative of the other variables w.r.t. t
                    let tp_vars = implicit_derivative_sp(
                        |variables, args| {
                            Self::tp_state_function(&eos.lift_total().lift_total(), variables, args)
                        },
                        stack![rho_i_l; vector![beta, rho_v]],
                        &args,
                    );
                    let (beta, rho_l, rho_v, x, y) = unpack_tp_variables(z, &tp_vars);

                    // Evaluation of the enthalpy/entropy including the derivatives.
                    let liquid = State::new(
                        &eos.lift_total(),
                        Temperature::from_reduced(t),
                        Density::from_reduced(rho_l),
                        x,
                    )
                    .unwrap();
                    let vapor = State::new(
                        &eos.lift_total(),
                        Temperature::from_reduced(t),
                        Density::from_reduced(rho_v),
                        y,
                    )
                    .unwrap();
                    Self::evaluate_property(&PhaseEquilibrium::with_vapor_phase_fraction(
                        vapor, liquid, beta, None,
                    ))
                    .into_reduced()
                },
                &(p, z.clone_owned()),
            ),
            t,
        );

        // Newton step for the temperature
        t -= (hs - hs_spec) / dhs;

        // pack variables into PhaseEquilibrium for initial values
        let liquid = State::new_density(
            eos,
            Temperature::from_reduced(t),
            Density::from_reduced(rho_i_l),
        )?;
        let vapor = State::new(
            eos,
            Temperature::from_reduced(t),
            Density::from_reduced(rho_v),
            y,
        )?;
        let vle = PhaseEquilibrium::with_vapor_phase_fraction(vapor, liquid, beta, None);

        // tp-flash for all other variables
        let vle = PhaseEquilibrium::tp_flash(
            eos,
            Temperature::from_reduced(t),
            Pressure::from_reduced(p),
            z,
            Some(&vle),
            Default::default(),
            None,
        )?;
        let beta = vle.vapor_phase_fraction();
        let rho_v = vle.vapor().density.into_reduced();
        let rho_l = vle.liquid().partial_density().into_reduced();
        Ok(stack![rho_l; vector![t, beta, rho_v]])
    }
}

impl PXFlash for SIUnit<-2, 2, 1, 0, 0, -1, 0> {
    // the potential function for a ph-flash specification (Q = (A + V*p_spec - H_spec) / T)
    fn state_function<E: Total<N, D>, N: Dim + DimAdd<U3>, D: DualNum<f64> + Copy>(
        eos: &E,
        variables: OVector<D, PXVars<N>>,
        &(p, h, ref z): &(D, D, OVector<D, N>),
    ) -> D
    where
        DefaultAllocator: Allocator<N> + Allocator<PXVars<N>>,
    {
        let (t, beta, rho_l, rho_v, x, y) = unpack_variables(z, &variables);
        let potential = |molefracs, rho: D, t| {
            let v = rho.recip();
            let a_res = eos.residual_helmholtz_energy(t, v, &molefracs);
            let a_ig = eos.ideal_gas_molar_helmholtz_energy(t, v, &molefracs);
            (a_res + a_ig + v * p - h) / t
        };
        potential(y, rho_v, t) * beta + potential(x, rho_l, t) * (-beta + 1.0)
    }

    fn evaluate_property<E: Total<N, D>, N: Gradients, D: DualNum<f64> + Copy>(
        vle: &PhaseEquilibrium<E, 2, N, D>,
    ) -> Quantity<D, Self>
    where
        DefaultAllocator: Allocator<N>,
    {
        vle.molar_enthalpy()
    }
}

impl PXFlash for SIUnit<-2, 2, 1, 0, -1, -1, 0> {
    // the potential function for a ps-flash specification (Q = A + T*S_spec + V*p_spec)
    fn state_function<E: Total<N, D>, N: Dim + DimAdd<U3>, D: DualNum<f64> + Copy>(
        eos: &E,
        variables: OVector<D, PXVars<N>>,
        &(p, s, ref z): &(D, D, OVector<D, N>),
    ) -> D
    where
        DefaultAllocator: Allocator<N> + Allocator<PXVars<N>>,
    {
        let (t, beta, rho_l, rho_v, x, y) = unpack_variables(z, &variables);
        let potential = |molefracs, rho: D, t| {
            let v = rho.recip();
            let a_res = eos.residual_helmholtz_energy(t, v, &molefracs);
            let a_ig = eos.ideal_gas_molar_helmholtz_energy(t, v, &molefracs);
            // Division by t.re() is done to ensure that the state function has the same
            // units (and in conclusion same order of magnitude) as the ph state function.
            // This allows using the same toelrances for both methods.
            (a_res + a_ig + t * s + v * p) / t.re()
        };
        potential(y, rho_v, t) * beta + potential(x, rho_l, t) * (-beta + 1.0)
    }

    fn evaluate_property<E: Total<N, D>, N: Gradients, D: DualNum<f64> + Copy>(
        vle: &PhaseEquilibrium<E, 2, N, D>,
    ) -> Quantity<D, Self>
    where
        DefaultAllocator: Allocator<N>,
    {
        vle.molar_entropy()
    }
}
