use super::{PhaseEquilibrium, SolverOptions, Verbosity};
use crate::errors::{EosError, EosResult};
use crate::state::{
    Contributions,
    DensityInitialization::{InitialDensity, Liquid, Vapor},
    State, StateBuilder, TPSpec,
};
use crate::{equation_of_state::EquationOfState, EosUnit};
use ndarray::*;
use num_dual::linalg::{norm, LU};
use quantity::{QuantityArray1, QuantityScalar};
use std::convert::TryFrom;
use std::rc::Rc;

const MAX_ITER_INNER: usize = 5;
const TOL_INNER: f64 = 1e-9;
const MAX_ITER_OUTER: usize = 400;
const TOL_OUTER: f64 = 1e-10;

const MAX_TSTEP: f64 = 20.0;
const MAX_LNPSTEP: f64 = 0.1;
const PROMISING_F: f64 = 1.0;
const NEWTON_TOL: f64 = 1e-3;

impl<U: EosUnit> TPSpec<U> {
    pub(super) fn temperature_pressure(
        &self,
        tp_init: QuantityScalar<U>,
    ) -> (Self, QuantityScalar<U>, QuantityScalar<U>) {
        match self {
            Self::Temperature(t) => (Self::Pressure(tp_init), *t, tp_init),
            Self::Pressure(p) => (Self::Temperature(tp_init), tp_init, *p),
        }
    }

    fn identifier(&self) -> &str {
        match self {
            Self::Temperature(_) => "temperature",
            Self::Pressure(_) => "pressure",
        }
    }
}

impl<U: EosUnit> std::fmt::Display for TPSpec<U>
where
    QuantityScalar<U>: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Temperature(t) => {
                write!(f, " ")?;
                t.fmt(f)?;
                write!(f, " ")
            }
            Self::Pressure(p) => p.fmt(f),
        }
    }
}

/// # Bubble and dew point calculations
impl<U: EosUnit, E: EquationOfState> PhaseEquilibrium<U, E, 2> {
    /// Calculate a phase equilibrium for a given temperature
    /// or pressure and composition of the liquid phase.
    pub fn bubble_point(
        eos: &Rc<E>,
        temperature_or_pressure: QuantityScalar<U>,
        liquid_molefracs: &Array1<f64>,
        tp_init: Option<QuantityScalar<U>>,
        vapor_molefracs: Option<&Array1<f64>>,
        options: (SolverOptions, SolverOptions),
    ) -> EosResult<Self>
    where
        QuantityScalar<U>: std::fmt::Display,
    {
        Self::bubble_dew_point(
            eos,
            TPSpec::try_from(temperature_or_pressure)?,
            tp_init,
            liquid_molefracs,
            vapor_molefracs,
            true,
            options,
        )
    }

    /// Calculate a phase equilibrium for a given temperature
    /// or pressure and composition of the vapor phase.
    pub fn dew_point(
        eos: &Rc<E>,
        temperature_or_pressure: QuantityScalar<U>,
        vapor_molefracs: &Array1<f64>,
        tp_init: Option<QuantityScalar<U>>,
        liquid_molefracs: Option<&Array1<f64>>,
        options: (SolverOptions, SolverOptions),
    ) -> EosResult<Self>
    where
        QuantityScalar<U>: std::fmt::Display,
    {
        Self::bubble_dew_point(
            eos,
            TPSpec::try_from(temperature_or_pressure)?,
            tp_init,
            vapor_molefracs,
            liquid_molefracs,
            false,
            options,
        )
    }

    pub(super) fn bubble_dew_point(
        eos: &Rc<E>,
        tp_spec: TPSpec<U>,
        tp_init: Option<QuantityScalar<U>>,
        molefracs_spec: &Array1<f64>,
        molefracs_init: Option<&Array1<f64>>,
        bubble: bool,
        options: (SolverOptions, SolverOptions),
    ) -> EosResult<Self>
    where
        QuantityScalar<U>: std::fmt::Display,
    {
        match tp_spec {
            TPSpec::Temperature(t) => {
                // First use given initial pressure if applicable
                let mut vle = tp_init
                    .map(|p| {
                        Self::iterate_bubble_dew(
                            eos,
                            tp_spec,
                            p,
                            molefracs_spec,
                            molefracs_init,
                            bubble,
                            options,
                        )
                    })
                    .and_then(Result::ok);

                // Next try to initialize with an ideal gas assumption
                vle = vle.or_else(|| {
                    let (p, x) =
                        Self::starting_pressure_ideal_gas(eos, t, molefracs_spec, bubble).ok()?;
                    Self::iterate_bubble_dew(
                        eos,
                        tp_spec,
                        p,
                        molefracs_spec,
                        Some(&x),
                        bubble,
                        options,
                    )
                    .ok()
                });

                // Finally use the spinodal to initialize the calculation
                vle.map_or_else(
                    || {
                        Self::iterate_bubble_dew(
                            eos,
                            tp_spec,
                            Self::starting_pressure_spinodal(eos, t, molefracs_spec)?,
                            molefracs_spec,
                            molefracs_init,
                            bubble,
                            options,
                        )
                    },
                    Ok,
                )
            }
            TPSpec::Pressure(_) => {
                let solve = |temperature| {
                    Self::iterate_bubble_dew(
                        eos,
                        tp_spec,
                        temperature,
                        molefracs_spec,
                        molefracs_init,
                        bubble,
                        options,
                    )
                };
                // First use given initial temperature if applicable
                tp_init.map(solve).unwrap()
            }
        }
    }

    fn iterate_bubble_dew(
        eos: &Rc<E>,
        tp_spec: TPSpec<U>,
        tp_init: QuantityScalar<U>,
        molefracs_spec: &Array1<f64>,
        molefracs_init: Option<&Array1<f64>>,
        bubble: bool,
        options: (SolverOptions, SolverOptions),
    ) -> EosResult<Self>
    where
        QuantityScalar<U>: std::fmt::Display,
    {
        let (var, t, p) = tp_spec.temperature_pressure(tp_init);
        let [state1, state2] = if bubble {
            starting_x2_bubble(eos, t, p, molefracs_spec, molefracs_init)
        } else {
            starting_x2_dew(eos, t, p, molefracs_spec, molefracs_init)
        }?;
        bubble_dew(tp_spec, var, state1, state2, options)
    }

    fn starting_pressure_ideal_gas(
        eos: &Rc<E>,
        temperature: QuantityScalar<U>,
        molefracs_spec: &Array1<f64>,
        bubble: bool,
    ) -> EosResult<(QuantityScalar<U>, Array1<f64>)>
    where
        QuantityScalar<U>: std::fmt::Display,
    {
        if bubble {
            Self::starting_pressure_ideal_gas_bubble(eos, temperature, molefracs_spec)
        } else {
            Self::starting_pressure_ideal_gas_dew(eos, temperature, molefracs_spec)
        }
    }

    pub(super) fn starting_pressure_ideal_gas_bubble(
        eos: &Rc<E>,
        temperature: QuantityScalar<U>,
        liquid_molefracs: &Array1<f64>,
    ) -> EosResult<(QuantityScalar<U>, Array1<f64>)> {
        let m = liquid_molefracs * U::reference_moles();
        let density = 0.75 * eos.max_density(Some(&m))?;
        let liquid = State::new_nvt(eos, temperature, m.sum() / density, &m)?;
        let v_l = liquid.molar_volume(Contributions::Total);
        let p_l = liquid.pressure(Contributions::Total);
        let mu_l = liquid.chemical_potential(Contributions::ResidualNvt);
        let p_i = (temperature * density * U::gas_constant() * liquid_molefracs)
            * (mu_l - p_l * v_l)
                .to_reduced(U::gas_constant() * temperature)?
                .mapv(f64::exp);
        let y = p_i.to_reduced(p_i.sum())?;
        Ok((p_i.sum(), y))
    }

    fn starting_pressure_ideal_gas_dew(
        eos: &Rc<E>,
        temperature: QuantityScalar<U>,
        vapor_molefracs: &Array1<f64>,
    ) -> EosResult<(QuantityScalar<U>, Array1<f64>)>
    where
        QuantityScalar<U>: std::fmt::Display,
    {
        let mut p: Option<QuantityScalar<U>> = None;

        let mut x = vapor_molefracs.clone();
        for _ in 0..5 {
            let m = x * U::reference_moles();
            let density = 0.75 * eos.max_density(Some(&m))?;
            let liquid = State::new_nvt(eos, temperature, m.sum() / density, &m)?;
            let v_l = liquid.molar_volume(Contributions::Total);
            let p_l = liquid.pressure(Contributions::Total);
            let mu_l = liquid.chemical_potential(Contributions::ResidualNvt);
            let k = vapor_molefracs
                / (mu_l - p_l * v_l)
                    .to_reduced(U::gas_constant() * temperature)?
                    .mapv(f64::exp);
            let p_new = U::gas_constant() * temperature * density / k.sum();
            x = &k / k.sum();
            println!("{p_new} {x}");
            if let Some(p_old) = p {
                if (p_new - p_old).to_reduced(p_old)?.abs() < 1e-5 {
                    p = Some(p_new);
                    break;
                }
            }
            p = Some(p_new);
        }
        Ok((p.unwrap(), x))
    }

    pub(super) fn starting_pressure_spinodal(
        eos: &Rc<E>,
        temperature: QuantityScalar<U>,
        molefracs: &Array1<f64>,
    ) -> EosResult<QuantityScalar<U>>
    where
        QuantityScalar<U>: std::fmt::Display,
    {
        let moles = molefracs * U::reference_moles();
        let [sp_v, sp_l] = State::spinodal(eos, temperature, Some(&moles), Default::default())?;
        let pv = sp_v.pressure(Contributions::Total);
        let pl = sp_l.pressure(Contributions::Total);
        Ok(0.5 * ((0.0 * U::reference_pressure()).max(pl)? + pv))
    }
}

fn starting_x2_bubble<U: EosUnit, E: EquationOfState>(
    eos: &Rc<E>,
    temperature: QuantityScalar<U>,
    pressure: QuantityScalar<U>,
    liquid_molefracs: &Array1<f64>,
    vapor_molefracs: Option<&Array1<f64>>,
) -> EosResult<[State<U, E>; 2]> {
    let liquid_state = State::new_npt(
        eos,
        temperature,
        pressure,
        &(liquid_molefracs.clone() * U::reference_moles()),
        Liquid,
    )?;
    let xv = match vapor_molefracs {
        Some(xv) => xv.clone(),
        None => liquid_state.ln_phi().mapv(f64::exp) * liquid_molefracs,
    };
    let vapor_state = State::new_npt(
        eos,
        temperature,
        pressure,
        &(xv * U::reference_moles()),
        Vapor,
    )?;
    Ok([liquid_state, vapor_state])
}

fn starting_x2_dew<U: EosUnit, E: EquationOfState>(
    eos: &Rc<E>,
    temperature: QuantityScalar<U>,
    pressure: QuantityScalar<U>,
    vapor_molefracs: &Array1<f64>,
    liquid_molefracs: Option<&Array1<f64>>,
) -> EosResult<[State<U, E>; 2]> {
    let vapor_state = State::new_npt(
        eos,
        temperature,
        pressure,
        &(vapor_molefracs.clone() * U::reference_moles()),
        Vapor,
    )?;
    let xl = match liquid_molefracs {
        Some(xl) => xl.clone(),
        None => {
            let xl = vapor_state.ln_phi().mapv(f64::exp) * vapor_molefracs;
            let liquid_state = State::new_npt(
                eos,
                temperature,
                pressure,
                &(xl * U::reference_moles()),
                Liquid,
            )?;
            (vapor_state.ln_phi() - liquid_state.ln_phi()).mapv(f64::exp) * vapor_molefracs
        }
    };
    let liquid_state = State::new_npt(
        eos,
        temperature,
        pressure,
        &(xl * U::reference_moles()),
        Liquid,
    )?;
    Ok([vapor_state, liquid_state])
}

fn bubble_dew<U: EosUnit, E: EquationOfState>(
    tp_spec: TPSpec<U>,
    mut var_tp: TPSpec<U>,
    mut state1: State<U, E>,
    mut state2: State<U, E>,
    options: (SolverOptions, SolverOptions),
) -> EosResult<PhaseEquilibrium<U, E, 2>>
where
    QuantityScalar<U>: std::fmt::Display,
{
    let (options_inner, options_outer) = options;

    // initialize variables
    let mut err_out = 1.0;
    let mut k_out = 0;

    // If the starting values are insufficient find better ones
    if !promising_values(&state1, &state2) {
        log_iter!(options_outer.verbosity, "Trivial solution encountered!");
        return Err(EosError::TrivialSolution);
        // find_starting_values(&mut var_tp, &mut state1, bubble)?;
    }

    log_iter!(
        options_outer.verbosity,
        "res outer loop | res inner loop | {:^16} | molefracs second phase",
        var_tp.identifier()
    );
    log_iter!(options_outer.verbosity, "{:-<85}", "");
    log_iter!(
        options_outer.verbosity,
        "{:14} | {:14} | {:12.8} | {:.8}",
        "",
        "",
        var_tp,
        state2.molefracs
    );

    // Outer loop for finding x2
    for ko in 0..options_outer.max_iter.unwrap_or(MAX_ITER_OUTER) {
        // Iso-Fugacity equation
        err_out = if err_out > NEWTON_TOL {
            // Inner loop for finding T or p
            for _ in 0..options_inner.max_iter.unwrap_or(MAX_ITER_INNER) {
                // Newton step
                if adjust_t_p(
                    &mut var_tp,
                    &mut state1,
                    &mut state2,
                    options_inner.verbosity,
                )? < options_inner.tol.unwrap_or(TOL_INNER)
                {
                    break;
                }
            }
            adjust_x2(&state1, &mut state2, options_outer.verbosity)
        } else {
            newton_step(
                tp_spec,
                &mut var_tp,
                &mut state1,
                &mut state2,
                options_outer.verbosity,
            )
        }?;

        // if a trivial solution is encountered, reinitialize T or p
        if PhaseEquilibrium::is_trivial_solution(&state1, &state2) {
            log_iter!(options_outer.verbosity, "Trivial solution encountered!");
            return Err(EosError::TrivialSolution);
            // find_starting_values(iterate_t, bubble, &mut itervars)?;
        }

        if err_out < options_outer.tol.unwrap_or(TOL_OUTER) {
            k_out = ko + 1;
            break;
        }
    }

    if err_out < options_outer.tol.unwrap_or(TOL_OUTER) {
        log_result!(
            options_outer.verbosity,
            "Bubble/dew point: calculation converged in {} step(s)\n",
            k_out
        );
        Ok(PhaseEquilibrium::from_states(state1, state2))
    } else {
        // not converged, return EosError
        Err(EosError::NotConverged(String::from("bubble-dew-iteration")))
    }
}

fn adjust_t_p<U: EosUnit, E: EquationOfState>(
    var: &mut TPSpec<U>,
    state1: &mut State<U, E>,
    state2: &mut State<U, E>,
    verbosity: Verbosity,
) -> EosResult<f64>
where
    QuantityScalar<U>: std::fmt::Display,
{
    // calculate K = phi_1/phi_2 = x_2/x_1
    let ln_phi_1 = state1.ln_phi();
    let ln_phi_2 = state2.ln_phi();
    let k = (&ln_phi_1 - &ln_phi_2).mapv(f64::exp);

    // calculate residual
    let f = (&state1.molefracs * &k).sum() - 1.0;

    match var {
        TPSpec::Temperature(t) => {
            // Derivative w.r.t. temperature
            let ln_phi_1_dt = state1.dln_phi_dt();
            let ln_phi_2_dt = state2.dln_phi_dt();
            let df = ((ln_phi_1_dt - ln_phi_2_dt) * &state1.molefracs * &k).sum();
            let mut tstep = -f / df;

            // catch too big t-steps
            if tstep < -MAX_TSTEP * U::reference_temperature() {
                tstep = -MAX_TSTEP * U::reference_temperature();
            } else if tstep > MAX_TSTEP * U::reference_temperature() {
                tstep = MAX_TSTEP * U::reference_temperature();
            }

            // Update t
            *t += tstep;
        }
        TPSpec::Pressure(p) => {
            // Derivative w.r.t. ln(pressure)
            let ln_phi_1_dp = state1.dln_phi_dp();
            let ln_phi_2_dp = state2.dln_phi_dp();
            let df = ((ln_phi_1_dp - ln_phi_2_dp) * *p * &state1.molefracs * &k)
                .sum()
                .into_value()?;
            let mut lnpstep = -f / df;

            // catch too big p-steps
            if lnpstep < -MAX_LNPSTEP {
                lnpstep = -MAX_LNPSTEP;
            } else if lnpstep > MAX_LNPSTEP {
                lnpstep = MAX_LNPSTEP;
            }

            // Update p
            *p = *p * lnpstep.exp();
        }
    };

    // update states with new temperature/pressure
    adjust_states(&*var, state1, state2, None)?;

    // log
    log_iter!(
        verbosity,
        "{:14} | {:<14.8e} | {:12.8} | {:.8}",
        "",
        f.abs(),
        var,
        state2.molefracs
    );

    Ok(f.abs())
}

fn adjust_states<U: EosUnit, E: EquationOfState>(
    var: &TPSpec<U>,
    state1: &mut State<U, E>,
    state2: &mut State<U, E>,
    moles_state2: Option<&QuantityArray1<U>>,
) -> EosResult<()> {
    let (temperature, pressure) = match var {
        TPSpec::Pressure(p) => (state1.temperature, *p),
        TPSpec::Temperature(t) => (*t, state1.pressure(Contributions::Total)),
    };
    *state1 = State::new_npt(
        &state1.eos,
        temperature,
        pressure,
        &state1.moles,
        InitialDensity(state1.density),
    )?;
    *state2 = State::new_npt(
        &state2.eos,
        temperature,
        pressure,
        moles_state2.unwrap_or(&state2.moles),
        InitialDensity(state2.density),
    )?;
    Ok(())
}

fn adjust_x2<U: EosUnit, E: EquationOfState>(
    state1: &State<U, E>,
    state2: &mut State<U, E>,
    verbosity: Verbosity,
) -> EosResult<f64> {
    let x1 = &state1.molefracs;
    let ln_phi_1 = state1.ln_phi();
    let ln_phi_2 = state2.ln_phi();
    let k = (ln_phi_1 - ln_phi_2).mapv(f64::exp);
    let err_out = (&k * x1 / &state2.molefracs - 1.0).mapv(f64::abs).sum();
    let x2 = (x1 * &k) / (&k * x1).sum();
    log_iter!(verbosity, "{:<14.8e} | {:14} | {:16} |", err_out, "", "");
    *state2 = State::new_npt(
        &state2.eos,
        state2.temperature,
        state2.pressure(Contributions::Total),
        &(x2 * U::reference_moles()),
        InitialDensity(state2.density),
    )?;
    Ok(err_out)
}

fn newton_step<U: EosUnit, E: EquationOfState>(
    tp_spec: TPSpec<U>,
    var: &mut TPSpec<U>,
    state1: &mut State<U, E>,
    state2: &mut State<U, E>,
    verbosity: Verbosity,
) -> EosResult<f64>
where
    QuantityScalar<U>: std::fmt::Display,
{
    match tp_spec {
        TPSpec::Temperature(_) => newton_step_t(var, state1, state2, verbosity),
        TPSpec::Pressure(p) => newton_step_p(p, var, state1, state2, verbosity),
    }
}

fn newton_step_t<U: EosUnit, E: EquationOfState>(
    pressure: &mut TPSpec<U>,
    state1: &mut State<U, E>,
    state2: &mut State<U, E>,
    verbosity: Verbosity,
) -> EosResult<f64>
where
    QuantityScalar<U>: std::fmt::Display,
{
    let dmu_drho_1 = (state1.dmu_dni(Contributions::Total) * state1.volume)
        .to_reduced(U::reference_molar_energy() / U::reference_density())?
        .dot(&state1.molefracs);
    let dmu_drho_2 = (state2.dmu_dni(Contributions::Total) * state2.volume)
        .to_reduced(U::reference_molar_energy() / U::reference_density())?;
    let dp_drho_1 = (state1.dp_dni(Contributions::Total) * state1.volume)
        .to_reduced(U::reference_pressure() / U::reference_density())?
        .dot(&state1.molefracs);
    let dp_drho_2 = (state2.dp_dni(Contributions::Total) * state2.volume)
        .to_reduced(U::reference_pressure() / U::reference_density())?;
    let mu_1 = state1
        .chemical_potential(Contributions::Total)
        .to_reduced(U::reference_molar_energy())?;
    let mu_2 = state2
        .chemical_potential(Contributions::Total)
        .to_reduced(U::reference_molar_energy())?;
    let p_1 = state1
        .pressure(Contributions::Total)
        .to_reduced(U::reference_pressure())?;
    let p_2 = state2
        .pressure(Contributions::Total)
        .to_reduced(U::reference_pressure())?;

    // calculate residual
    let res = concatenate![Axis(0), mu_1 - &mu_2, arr1(&[p_1 - p_2])];
    let error = norm(&res);

    // calculate Jacobian
    let jacobian = concatenate![
        Axis(1),
        concatenate![Axis(0), -dmu_drho_2, -dp_drho_2.insert_axis(Axis(0))],
        concatenate![
            Axis(0),
            dmu_drho_1.insert_axis(Axis(1)),
            arr2(&[[dp_drho_1]])
        ]
    ];

    // calculate Newton step
    let dx = LU::new(jacobian)?.solve(&res);

    // apply Newton step
    let rho_l1 = state1.density - dx[dx.len() - 1] * U::reference_density();
    let rho_l2 =
        &state2.partial_density - &(dx.slice(s![0..-1]).to_owned() * U::reference_density());

    // update states
    *state1 = StateBuilder::new(&state1.eos)
        .temperature(state1.temperature)
        .density(rho_l1)
        .molefracs(&state1.molefracs)
        .build()?;
    *state2 = StateBuilder::new(&state2.eos)
        .temperature(state2.temperature)
        .partial_density(&rho_l2)
        .build()?;
    *pressure = TPSpec::Pressure(state1.pressure(Contributions::Total));
    log_iter!(
        verbosity,
        "{:<14.8e} | {:14} | {:12.8} | {:.8} NEWTON",
        error,
        "",
        pressure,
        state2.molefracs
    );
    Ok(error)
}

fn newton_step_p<U: EosUnit, E: EquationOfState>(
    pressure: QuantityScalar<U>,
    temperature: &mut TPSpec<U>,
    state1: &mut State<U, E>,
    state2: &mut State<U, E>,
    verbosity: Verbosity,
) -> EosResult<f64>
where
    QuantityScalar<U>: std::fmt::Display,
{
    let dmu_drho_1 = (state1.dmu_dni(Contributions::Total) * state1.volume)
        .to_reduced(U::reference_molar_energy() / U::reference_density())?
        .dot(&state1.molefracs);
    let dmu_drho_2 = (state2.dmu_dni(Contributions::Total) * state2.volume)
        .to_reduced(U::reference_molar_energy() / U::reference_density())?;
    let dmu_dt_1 = state1
        .dmu_dt(Contributions::Total)
        .to_reduced(U::reference_molar_energy() / U::reference_temperature())?;
    let dmu_dt_2 = state2
        .dmu_dt(Contributions::Total)
        .to_reduced(U::reference_molar_energy() / U::reference_temperature())?;
    let dp_drho_1 = (state1.dp_dni(Contributions::Total) * state1.volume)
        .to_reduced(U::reference_pressure() / U::reference_density())?
        .dot(&state1.molefracs);
    let dp_dt_1 = state1
        .dp_dt(Contributions::Total)
        .to_reduced(U::reference_pressure() / U::reference_temperature())?;
    let dp_dt_2 = state2
        .dp_dt(Contributions::Total)
        .to_reduced(U::reference_pressure() / U::reference_temperature())?;
    let dp_drho_2 = (state2.dp_dni(Contributions::Total) * state2.volume)
        .to_reduced(U::reference_pressure() / U::reference_density())?;
    let mu_1 = state1
        .chemical_potential(Contributions::Total)
        .to_reduced(U::reference_molar_energy())?;
    let mu_2 = state2
        .chemical_potential(Contributions::Total)
        .to_reduced(U::reference_molar_energy())?;
    let p_1 = state1
        .pressure(Contributions::Total)
        .to_reduced(U::reference_pressure())?;
    let p_2 = state2
        .pressure(Contributions::Total)
        .to_reduced(U::reference_pressure())?;
    let p = pressure.to_reduced(U::reference_pressure())?;

    // calculate residual
    let res = concatenate![Axis(0), mu_1 - &mu_2, arr1(&[p_1 - p]), arr1(&[p_2 - p])];
    let error = norm(&res);

    // calculate Jacobian
    let jacobian = concatenate![
        Axis(1),
        concatenate![
            Axis(0),
            -dmu_drho_2,
            Array2::zeros((1, res.len() - 2)),
            dp_drho_2.insert_axis(Axis(0))
        ],
        concatenate![
            Axis(0),
            dmu_drho_1.insert_axis(Axis(1)),
            arr2(&[[dp_drho_1], [0.0]])
        ],
        concatenate![
            Axis(0),
            (dmu_dt_1 - dmu_dt_2).insert_axis(Axis(1)),
            arr2(&[[dp_dt_1], [dp_dt_2]])
        ]
    ];

    // calculate Newton step
    let dx = LU::new(jacobian)?.solve(&res);

    // apply Newton step
    let rho_l1 = state1.density - dx[dx.len() - 2] * U::reference_density();
    let rho_l2 =
        &state2.partial_density - &(dx.slice(s![0..-2]).to_owned() * U::reference_density());
    let t = state1.temperature - dx[dx.len() - 1] * U::reference_temperature();

    // update states
    *state1 = StateBuilder::new(&state1.eos)
        .temperature(t)
        .density(rho_l1)
        .molefracs(&state1.molefracs)
        .build()?;
    *state2 = StateBuilder::new(&state2.eos)
        .temperature(t)
        .partial_density(&rho_l2)
        .build()?;
    *temperature = TPSpec::Temperature(t);
    log_iter!(
        verbosity,
        "{:<14.8e} | {:14} | {:12.8} | {:.8} NEWTON",
        error,
        "",
        temperature,
        state2.molefracs
    );
    Ok(error)
}

fn promising_values<U: EosUnit, E: EquationOfState>(
    state1: &State<U, E>,
    state2: &State<U, E>,
) -> bool {
    if PhaseEquilibrium::is_trivial_solution(state1, state2) {
        return false;
    }

    let ln_phi_1 = state1.ln_phi();
    let ln_phi_2 = state2.ln_phi();
    ((&state1.molefracs * &(ln_phi_1 - ln_phi_2).mapv(f64::exp)).sum() - 1.0).abs() < PROMISING_F
}
