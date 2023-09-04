use super::PhaseEquilibrium;
use crate::equation_of_state::Residual;
use crate::errors::{EosError, EosResult};
use crate::si::{Density, Dimensionless, Moles, Pressure, Quantity, SIUnit, Temperature, RGAS};
use crate::state::{
    Contributions,
    DensityInitialization::{InitialDensity, Liquid, Vapor},
    State, StateBuilder,
};
use crate::{SolverOptions, Verbosity};
use ndarray::*;
use num_dual::linalg::{norm, LU};
use std::fmt;
use std::sync::Arc;
use typenum::{N1, N2, P1, Z0};

const MAX_ITER_INNER: usize = 5;
const TOL_INNER: f64 = 1e-9;
const MAX_ITER_OUTER: usize = 400;
const TOL_OUTER: f64 = 1e-10;

const MAX_TSTEP: f64 = 20.0;
const MAX_LNPSTEP: f64 = 0.1;
const NEWTON_TOL: f64 = 1e-3;

pub trait TemperatureOrPressure: Copy {
    type Other: fmt::Display + Copy;

    const IDENTIFIER: &'static str;

    fn temperature_pressure(&self, tp_init: Self::Other) -> (Temperature, Pressure);

    fn from_state<E: Residual>(state: &State<E>) -> Self::Other;

    fn linspace(
        &self,
        start: Self::Other,
        end: Self::Other,
        n: usize,
    ) -> (Temperature<Array1<f64>>, Pressure<Array1<f64>>);

    fn bubble_dew_point<E: Residual>(
        eos: &Arc<E>,
        tp_spec: Self,
        tp_init: Option<Self::Other>,
        molefracs_spec: &Array1<f64>,
        molefracs_init: Option<&Array1<f64>>,
        bubble: bool,
        options: (SolverOptions, SolverOptions),
    ) -> EosResult<PhaseEquilibrium<E, 2>>;

    fn adjust_t_p<E: Residual>(
        tp_spec: Self,
        var: &mut Self::Other,
        state1: &mut State<E>,
        state2: &mut State<E>,
        verbosity: Verbosity,
    ) -> EosResult<f64>;

    fn newton_step<E: Residual>(
        tp_spec: Self,
        var: &mut Self::Other,
        state1: &mut State<E>,
        state2: &mut State<E>,
        verbosity: Verbosity,
    ) -> EosResult<f64>;
}

impl TemperatureOrPressure for Temperature {
    type Other = Pressure;
    const IDENTIFIER: &'static str = "temperature";

    fn temperature_pressure(&self, tp_init: Self::Other) -> (Temperature, Pressure) {
        (*self, tp_init)
    }

    fn from_state<E: Residual>(state: &State<E>) -> Self::Other {
        state.pressure(Contributions::Total)
    }

    fn linspace(
        &self,
        start: Pressure,
        end: Pressure,
        n: usize,
    ) -> (Temperature<Array1<f64>>, Pressure<Array1<f64>>) {
        (
            Temperature::linspace(*self, *self, n),
            Pressure::linspace(start, end, n),
        )
    }

    fn bubble_dew_point<E: Residual>(
        eos: &Arc<E>,
        temperature: Self,
        p_init: Option<Pressure>,
        molefracs_spec: &Array1<f64>,
        molefracs_init: Option<&Array1<f64>>,
        bubble: bool,
        options: (SolverOptions, SolverOptions),
    ) -> EosResult<PhaseEquilibrium<E, 2>> {
        // First use given initial pressure if applicable
        if let Some(p) = p_init {
            return PhaseEquilibrium::iterate_bubble_dew(
                eos,
                temperature,
                p,
                molefracs_spec,
                molefracs_init,
                bubble,
                options,
            );
        }

        // Next try to initialize with an ideal gas assumption
        let vle =
            PhaseEquilibrium::starting_pressure_ideal_gas(eos, temperature, molefracs_spec, bubble)
                .and_then(|(p, x)| {
                    PhaseEquilibrium::iterate_bubble_dew(
                        eos,
                        temperature,
                        p,
                        molefracs_spec,
                        molefracs_init.or(Some(&x)),
                        bubble,
                        options,
                    )
                });

        // Finally use the spinodal to initialize the calculation
        vle.or_else(|_| {
            PhaseEquilibrium::iterate_bubble_dew(
                eos,
                temperature,
                PhaseEquilibrium::starting_pressure_spinodal(eos, temperature, molefracs_spec)?,
                molefracs_spec,
                molefracs_init,
                bubble,
                options,
            )
        })
    }

    fn adjust_t_p<E: Residual>(
        temperature: Temperature,
        pressure: &mut Pressure,
        state1: &mut State<E>,
        state2: &mut State<E>,
        verbosity: Verbosity,
    ) -> EosResult<f64> {
        // calculate K = phi_1/phi_2 = x_2/x_1
        let ln_phi_1 = state1.ln_phi();
        let ln_phi_2 = state2.ln_phi();
        let k = (&ln_phi_1 - &ln_phi_2).mapv(f64::exp);

        // calculate residual
        let f = (&state1.molefracs * &k).sum() - 1.0;

        // Derivative w.r.t. ln(pressure)
        let ln_phi_1_dp = state1.dln_phi_dp();
        let ln_phi_2_dp = state2.dln_phi_dp();
        let df =
            (((ln_phi_1_dp - ln_phi_2_dp) * *pressure).into_value() * &state1.molefracs * &k).sum();
        let mut lnpstep = -f / df;

        // catch too big p-steps
        lnpstep = lnpstep.clamp(-MAX_LNPSTEP, MAX_LNPSTEP);

        // Update p
        *pressure *= lnpstep.exp();

        // update states with new temperature/pressure
        adjust_states(temperature, *pressure, state1, state2, None)?;

        // log
        log_iter!(
            verbosity,
            "{:14} | {:<14.8e} | {:12.8} | {:.8}",
            "",
            f.abs(),
            pressure,
            state2.molefracs
        );

        Ok(f.abs())
    }

    fn newton_step<E: Residual>(
        _: Temperature,
        pressure: &mut Pressure,
        state1: &mut State<E>,
        state2: &mut State<E>,
        verbosity: Verbosity,
    ) -> EosResult<f64> {
        let dmu_drho_1 = (state1.dmu_dni(Contributions::Total) * state1.volume)
            .to_reduced()
            .dot(&state1.molefracs);
        let dmu_drho_2 = (state2.dmu_dni(Contributions::Total) * state2.volume).to_reduced();
        let dp_drho_1 = (state1.dp_dni(Contributions::Total) * state1.volume)
            .to_reduced()
            .dot(&state1.molefracs);
        let dp_drho_2 = (state2.dp_dni(Contributions::Total) * state2.volume).to_reduced();
        let mu_1_res = state1.residual_chemical_potential().to_reduced();
        let mu_2_res = state2.residual_chemical_potential().to_reduced();
        let p_1 = state1.pressure(Contributions::Total).to_reduced();
        let p_2 = state2.pressure(Contributions::Total).to_reduced();

        // calculate residual
        let dmu_ig = (RGAS * state1.temperature).to_reduced()
            * (&state1.partial_density / &state2.partial_density)
                .into_value()
                .mapv(f64::ln);
        let res = concatenate![Axis(0), mu_1_res - mu_2_res + dmu_ig, arr1(&[p_1 - p_2])];
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
        let rho_l1 = state1.density - Density::from_reduced(dx[dx.len() - 1]);
        let rho_l2 =
            state2.partial_density.clone() - Density::from_reduced(dx.slice(s![0..-1]).to_owned());

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
        *pressure = state1.pressure(Contributions::Total);
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
}

// For some inexplicable reason this does not compile if the `Pressure` type is
// used instead of the explicit unit. Maybe the type is too complicated for the
// compiler?
impl TemperatureOrPressure for Quantity<f64, SIUnit<N2, N1, P1, Z0, Z0, Z0, Z0>> {
    type Other = Temperature;
    const IDENTIFIER: &'static str = "pressure";

    fn temperature_pressure(&self, tp_init: Self::Other) -> (Temperature, Pressure) {
        (tp_init, *self)
    }

    fn from_state<E: Residual>(state: &State<E>) -> Self::Other {
        state.temperature
    }

    fn linspace(
        &self,
        start: Temperature,
        end: Temperature,
        n: usize,
    ) -> (Temperature<Array1<f64>>, Pressure<Array1<f64>>) {
        (
            Temperature::linspace(start, end, n),
            Pressure::linspace(*self, *self, n),
        )
    }

    fn bubble_dew_point<E: Residual>(
        eos: &Arc<E>,
        pressure: Self,
        t_init: Option<Temperature>,
        molefracs_spec: &Array1<f64>,
        molefracs_init: Option<&Array1<f64>>,
        bubble: bool,
        options: (SolverOptions, SolverOptions),
    ) -> EosResult<PhaseEquilibrium<E, 2>> {
        let temperature = t_init.expect("An initial temperature is required for the calculation of bubble/dew points at given pressure!");
        PhaseEquilibrium::iterate_bubble_dew(
            eos,
            pressure,
            temperature,
            molefracs_spec,
            molefracs_init,
            bubble,
            options,
        )
    }

    fn adjust_t_p<E: Residual>(
        pressure: Pressure,
        temperature: &mut Temperature,
        state1: &mut State<E>,
        state2: &mut State<E>,
        verbosity: Verbosity,
    ) -> EosResult<f64> {
        // calculate K = phi_1/phi_2 = x_2/x_1
        let ln_phi_1 = state1.ln_phi();
        let ln_phi_2 = state2.ln_phi();
        let k = (&ln_phi_1 - &ln_phi_2).mapv(f64::exp);

        // calculate residual
        let f = (&state1.molefracs * &k).sum() - 1.0;

        // Derivative w.r.t. temperature
        let ln_phi_1_dt = state1.dln_phi_dt();
        let ln_phi_2_dt = state2.dln_phi_dt();
        let df = ((ln_phi_1_dt - ln_phi_2_dt) * Dimensionless::from(&state1.molefracs * &k)).sum();
        let mut tstep = -f / df;

        // catch too big t-steps
        if tstep < -Temperature::from_reduced(MAX_TSTEP) {
            tstep = -Temperature::from_reduced(MAX_TSTEP);
        } else if tstep > Temperature::from_reduced(MAX_TSTEP) {
            tstep = Temperature::from_reduced(MAX_TSTEP);
        }

        // Update t
        *temperature += tstep;

        // update states with new temperature
        adjust_states(*temperature, pressure, state1, state2, None)?;

        // log
        log_iter!(
            verbosity,
            "{:14} | {:<14.8e} | {:12.8} | {:.8}",
            "",
            f.abs(),
            temperature,
            state2.molefracs
        );

        Ok(f.abs())
    }

    fn newton_step<E: Residual>(
        pressure: Pressure,
        temperature: &mut Temperature,
        state1: &mut State<E>,
        state2: &mut State<E>,
        verbosity: Verbosity,
    ) -> EosResult<f64> {
        let dmu_drho_1 = (state1.dmu_dni(Contributions::Total) * state1.volume)
            .to_reduced()
            .dot(&state1.molefracs);
        let dmu_drho_2 = (state2.dmu_dni(Contributions::Total) * state2.volume).to_reduced();
        let dmu_res_dt_1 = state1.dmu_res_dt().to_reduced();
        let dmu_res_dt_2 = state2.dmu_res_dt().to_reduced();
        let dp_drho_1 = (state1.dp_dni(Contributions::Total) * state1.volume)
            .to_reduced()
            .dot(&state1.molefracs);
        let dp_dt_1 = state1.dp_dt(Contributions::Total).to_reduced();
        let dp_dt_2 = state2.dp_dt(Contributions::Total).to_reduced();
        let dp_drho_2 = (state2.dp_dni(Contributions::Total) * state2.volume).to_reduced();
        let mu_1_res = state1.residual_chemical_potential().to_reduced();
        let mu_2_res = state2.residual_chemical_potential().to_reduced();
        let p_1 = state1.pressure(Contributions::Total).to_reduced();
        let p_2 = state2.pressure(Contributions::Total).to_reduced();
        let p = pressure.to_reduced();

        // calculate residual
        let delta_dmu_ig_dt = (&state1.partial_density / &state2.partial_density)
            .into_value()
            .mapv(f64::ln);
        let delta_mu_ig = (RGAS * state1.temperature).to_reduced() * &delta_dmu_ig_dt;
        let res = concatenate![
            Axis(0),
            mu_1_res - mu_2_res + delta_mu_ig,
            arr1(&[p_1 - p]),
            arr1(&[p_2 - p])
        ];
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
                (dmu_res_dt_1 - dmu_res_dt_2 + delta_dmu_ig_dt).insert_axis(Axis(1)),
                arr2(&[[dp_dt_1], [dp_dt_2]])
            ]
        ];

        // calculate Newton step
        let dx = LU::new(jacobian)?.solve(&res);

        // apply Newton step
        let rho_l1 = state1.density - Density::from_reduced(dx[dx.len() - 2]);
        let rho_l2 =
            state2.partial_density.clone() - Density::from_reduced(dx.slice(s![0..-2]).to_owned());
        let t = state1.temperature - Temperature::from_reduced(dx[dx.len() - 1]);

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
        *temperature = t;
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
}

/// # Bubble and dew point calculations
impl<E: Residual> PhaseEquilibrium<E, 2> {
    /// Calculate a phase equilibrium for a given temperature
    /// or pressure and composition of the liquid phase.
    pub fn bubble_point<TP: TemperatureOrPressure>(
        eos: &Arc<E>,
        temperature_or_pressure: TP,
        liquid_molefracs: &Array1<f64>,
        tp_init: Option<TP::Other>,
        vapor_molefracs: Option<&Array1<f64>>,
        options: (SolverOptions, SolverOptions),
    ) -> EosResult<Self> {
        Self::bubble_dew_point(
            eos,
            temperature_or_pressure,
            tp_init,
            liquid_molefracs,
            vapor_molefracs,
            true,
            options,
        )
    }

    /// Calculate a phase equilibrium for a given temperature
    /// or pressure and composition of the vapor phase.
    pub fn dew_point<TP: TemperatureOrPressure>(
        eos: &Arc<E>,
        temperature_or_pressure: TP,
        vapor_molefracs: &Array1<f64>,
        tp_init: Option<TP::Other>,
        liquid_molefracs: Option<&Array1<f64>>,
        options: (SolverOptions, SolverOptions),
    ) -> EosResult<Self> {
        Self::bubble_dew_point(
            eos,
            temperature_or_pressure,
            tp_init,
            vapor_molefracs,
            liquid_molefracs,
            false,
            options,
        )
    }

    pub(super) fn bubble_dew_point<TP: TemperatureOrPressure>(
        eos: &Arc<E>,
        tp_spec: TP,
        tp_init: Option<TP::Other>,
        molefracs_spec: &Array1<f64>,
        molefracs_init: Option<&Array1<f64>>,
        bubble: bool,
        options: (SolverOptions, SolverOptions),
    ) -> EosResult<Self> {
        TP::bubble_dew_point(
            eos,
            tp_spec,
            tp_init,
            molefracs_spec,
            molefracs_init,
            bubble,
            options,
        )
    }

    fn iterate_bubble_dew<TP: TemperatureOrPressure>(
        eos: &Arc<E>,
        tp_spec: TP,
        tp_init: TP::Other,
        molefracs_spec: &Array1<f64>,
        molefracs_init: Option<&Array1<f64>>,
        bubble: bool,
        options: (SolverOptions, SolverOptions),
    ) -> EosResult<Self> {
        let (t, p) = tp_spec.temperature_pressure(tp_init);
        let [state1, state2] = if bubble {
            starting_x2_bubble(eos, t, p, molefracs_spec, molefracs_init)
        } else {
            starting_x2_dew(eos, t, p, molefracs_spec, molefracs_init)
        }?;
        bubble_dew(tp_spec, tp_init, state1, state2, bubble, options)
    }

    fn starting_pressure_ideal_gas(
        eos: &Arc<E>,
        temperature: Temperature,
        molefracs_spec: &Array1<f64>,
        bubble: bool,
    ) -> EosResult<(Pressure, Array1<f64>)> {
        if bubble {
            Self::starting_pressure_ideal_gas_bubble(eos, temperature, molefracs_spec)
        } else {
            Self::starting_pressure_ideal_gas_dew(eos, temperature, molefracs_spec)
        }
    }

    pub(super) fn starting_pressure_ideal_gas_bubble(
        eos: &Arc<E>,
        temperature: Temperature,
        liquid_molefracs: &Array1<f64>,
    ) -> EosResult<(Pressure, Array1<f64>)> {
        let m = Moles::from_reduced(liquid_molefracs.to_owned());
        let density = 0.75 * eos.max_density(Some(&m))?;
        let liquid = State::new_nvt(eos, temperature, m.sum() / density, &m)?;
        let v_l = liquid.partial_molar_volume();
        let p_l = liquid.pressure(Contributions::Total);
        let mu_l = liquid.residual_chemical_potential();
        let p_i = (liquid_molefracs * temperature * density * RGAS)
            * Dimensionless::from(
                ((mu_l - p_l * v_l) / (RGAS * temperature))
                    .into_value()
                    .mapv(f64::exp),
            );
        let p = p_i.sum();
        let y = (p_i / p).into_value();
        Ok((p, y))
    }

    fn starting_pressure_ideal_gas_dew(
        eos: &Arc<E>,
        temperature: Temperature,
        vapor_molefracs: &Array1<f64>,
    ) -> EosResult<(Pressure, Array1<f64>)> {
        let mut p: Option<Pressure> = None;

        let mut x = vapor_molefracs.clone();
        for _ in 0..5 {
            let m = Moles::from_reduced(x);
            let density = 0.75 * eos.max_density(Some(&m))?;
            let liquid = State::new_nvt(eos, temperature, m.sum() / density, &m)?;
            let v_l = liquid.partial_molar_volume();
            let p_l = liquid.pressure(Contributions::Total);
            let mu_l = liquid.residual_chemical_potential();
            let k = vapor_molefracs
                / ((mu_l - p_l * v_l) / (RGAS * temperature))
                    .into_value()
                    .mapv(f64::exp);
            let p_new = RGAS * temperature * density / k.sum();
            x = &k / k.sum();
            if let Some(p_old) = p {
                if ((p_new - p_old) / p_old).into_value().abs() < 1e-5 {
                    p = Some(p_new);
                    break;
                }
            }
            p = Some(p_new);
        }
        Ok((p.unwrap(), x))
    }

    pub(super) fn starting_pressure_spinodal(
        eos: &Arc<E>,
        temperature: Temperature,
        molefracs: &Array1<f64>,
    ) -> EosResult<Pressure> {
        let moles = Moles::from_reduced(molefracs.clone());
        let [sp_v, sp_l] = State::spinodal(eos, temperature, Some(&moles), Default::default())?;
        let pv = sp_v.pressure(Contributions::Total);
        let pl = sp_l.pressure(Contributions::Total);
        Ok(0.5 * (Pressure::from_reduced(0.0).max(pl) + pv))
    }
}

fn starting_x2_bubble<E: Residual>(
    eos: &Arc<E>,
    temperature: Temperature,
    pressure: Pressure,
    liquid_molefracs: &Array1<f64>,
    vapor_molefracs: Option<&Array1<f64>>,
) -> EosResult<[State<E>; 2]> {
    let liquid_state = State::new_npt(
        eos,
        temperature,
        pressure,
        &Moles::from_reduced(liquid_molefracs.clone()),
        Liquid,
    )?;
    let xv = match vapor_molefracs {
        Some(xv) => xv.clone(),
        None => liquid_state.ln_phi().mapv(f64::exp) * liquid_molefracs,
    };
    let vapor_state = State::new_npt(eos, temperature, pressure, &Moles::from_reduced(xv), Vapor)?;
    Ok([liquid_state, vapor_state])
}

fn starting_x2_dew<E: Residual>(
    eos: &Arc<E>,
    temperature: Temperature,
    pressure: Pressure,
    vapor_molefracs: &Array1<f64>,
    liquid_molefracs: Option<&Array1<f64>>,
) -> EosResult<[State<E>; 2]> {
    let vapor_state = State::new_npt(
        eos,
        temperature,
        pressure,
        &Moles::from_reduced(vapor_molefracs.clone()),
        Vapor,
    )?;
    let xl = match liquid_molefracs {
        Some(xl) => xl.clone(),
        None => {
            let xl = vapor_state.ln_phi().mapv(f64::exp) * vapor_molefracs;
            let liquid_state =
                State::new_npt(eos, temperature, pressure, &Moles::from_reduced(xl), Liquid)?;
            (vapor_state.ln_phi() - liquid_state.ln_phi()).mapv(f64::exp) * vapor_molefracs
        }
    };
    let liquid_state =
        State::new_npt(eos, temperature, pressure, &Moles::from_reduced(xl), Liquid)?;
    Ok([vapor_state, liquid_state])
}

fn bubble_dew<E: Residual, TP: TemperatureOrPressure>(
    tp_spec: TP,
    mut var_tp: TP::Other,
    mut state1: State<E>,
    mut state2: State<E>,
    bubble: bool,
    options: (SolverOptions, SolverOptions),
) -> EosResult<PhaseEquilibrium<E, 2>> {
    let (options_inner, options_outer) = options;

    // initialize variables
    let mut err_out = 1.0;
    let mut k_out = 0;

    if PhaseEquilibrium::is_trivial_solution(&state1, &state2) {
        log_iter!(options_outer.verbosity, "Trivial solution encountered!");
        return Err(EosError::TrivialSolution);
    }

    log_iter!(
        options_outer.verbosity,
        "res outer loop | res inner loop | {:^16} | molefracs second phase",
        TP::IDENTIFIER
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
                if TP::adjust_t_p(
                    tp_spec,
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
            TP::newton_step(
                tp_spec,
                &mut var_tp,
                &mut state1,
                &mut state2,
                options_outer.verbosity,
            )
        }?;

        if PhaseEquilibrium::is_trivial_solution(&state1, &state2) {
            log_iter!(options_outer.verbosity, "Trivial solution encountered!");
            return Err(EosError::TrivialSolution);
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
        if bubble {
            Ok(PhaseEquilibrium([state2, state1]))
        } else {
            Ok(PhaseEquilibrium([state1, state2]))
        }
    } else {
        // not converged, return EosError
        Err(EosError::NotConverged(String::from("bubble-dew-iteration")))
    }
}

fn adjust_states<E: Residual>(
    temperature: Temperature,
    pressure: Pressure,
    state1: &mut State<E>,
    state2: &mut State<E>,
    moles_state2: Option<&Moles<Array1<f64>>>,
) -> EosResult<()> {
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

fn adjust_x2<E: Residual>(
    state1: &State<E>,
    state2: &mut State<E>,
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
        &Moles::from_reduced(x2),
        InitialDensity(state2.density),
    )?;
    Ok(err_out)
}
