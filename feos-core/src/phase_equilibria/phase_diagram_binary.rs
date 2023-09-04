use super::bubble_dew::TemperatureOrPressure;
use super::{PhaseDiagram, PhaseEquilibrium};
use crate::equation_of_state::Residual;
use crate::errors::{EosError, EosResult};
use crate::si::{Density, Moles, Pressure, Temperature, RGAS};
use crate::state::{Contributions, DensityInitialization, State, StateBuilder, TPSpec};
use crate::SolverOptions;
use ndarray::{arr1, arr2, concatenate, s, Array1, Array2, Axis};
use num_dual::linalg::{norm, LU};
use std::sync::Arc;

const DEFAULT_POINTS: usize = 51;

impl<E: Residual> PhaseDiagram<E, 2> {
    /// Create a new binary phase diagram exhibiting a
    /// vapor/liquid equilibrium.
    ///
    /// If a heteroazeotrope occurs and the composition of the liquid
    /// phases are known, they can be passed as `x_lle` to avoid
    /// the calculation of unstable branches.
    pub fn binary_vle<TP: TemperatureOrPressure>(
        eos: &Arc<E>,
        temperature_or_pressure: TP,
        npoints: Option<usize>,
        x_lle: Option<(f64, f64)>,
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> EosResult<Self>
    where
        TPSpec: From<TP>,
    {
        let npoints = npoints.unwrap_or(DEFAULT_POINTS);

        // calculate boiling temperature/vapor pressure of pure components
        let vle_sat = PhaseEquilibrium::vle_pure_comps(eos, temperature_or_pressure);
        let vle_sat = [vle_sat[1].clone(), vle_sat[0].clone()];

        // Only calculate up to specified compositions
        if let Some(x_lle) = x_lle {
            let (states1, states2) = Self::calculate_vlle(
                eos,
                temperature_or_pressure,
                npoints,
                x_lle,
                vle_sat,
                bubble_dew_options,
            )?;

            let states = states1
                .into_iter()
                .chain(states2.into_iter().rev())
                .collect();
            return Ok(Self { states });
        }

        // use dew point when calculating a supercritical tx diagram
        let bubble = match temperature_or_pressure.into() {
            TPSpec::Temperature(_) => true,
            TPSpec::Pressure(_) => false,
        };

        // look for supercritical components
        let (x_lim, vle_lim, bubble) = match vle_sat {
            [None, None] => return Err(EosError::SuperCritical),
            [Some(vle2), None] => {
                let cp = State::critical_point_binary(
                    eos,
                    temperature_or_pressure,
                    None,
                    None,
                    SolverOptions::default(),
                )?;
                let cp_vle = PhaseEquilibrium::from_states(cp.clone(), cp.clone());
                ([0.0, cp.molefracs[0]], (vle2, cp_vle), bubble)
            }
            [None, Some(vle1)] => {
                let cp = State::critical_point_binary(
                    eos,
                    temperature_or_pressure,
                    None,
                    None,
                    SolverOptions::default(),
                )?;
                let cp_vle = PhaseEquilibrium::from_states(cp.clone(), cp.clone());
                ([1.0, cp.molefracs[0]], (vle1, cp_vle), bubble)
            }
            [Some(vle2), Some(vle1)] => ([0.0, 1.0], (vle2, vle1), true),
        };

        let mut states = iterate_vle(
            eos,
            temperature_or_pressure,
            &x_lim,
            vle_lim.0,
            Some(vle_lim.1),
            npoints,
            bubble,
            bubble_dew_options,
        );
        if !bubble {
            states = states.into_iter().rev().collect();
        }
        Ok(Self { states })
    }

    #[allow(clippy::type_complexity)]
    fn calculate_vlle<TP: TemperatureOrPressure>(
        eos: &Arc<E>,
        tp: TP,
        npoints: usize,
        x_lle: (f64, f64),
        vle_sat: [Option<PhaseEquilibrium<E, 2>>; 2],
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> EosResult<(Vec<PhaseEquilibrium<E, 2>>, Vec<PhaseEquilibrium<E, 2>>)> {
        match vle_sat {
            [Some(vle2), Some(vle1)] => {
                let states1 = iterate_vle(
                    eos,
                    tp,
                    &[0.0, x_lle.0],
                    vle2,
                    None,
                    npoints / 2,
                    true,
                    bubble_dew_options,
                );
                let states2 = iterate_vle(
                    eos,
                    tp,
                    &[1.0, x_lle.1],
                    vle1,
                    None,
                    npoints - npoints / 2,
                    true,
                    bubble_dew_options,
                );
                Ok((states1, states2))
            }
            _ => Err(EosError::SuperCritical),
        }
    }

    /// Create a new phase diagram using Tp flash calculations.
    ///
    /// The usual use case for this function is the calculation of
    /// liquid-liquid phase diagrams, but it can be used for vapor-
    /// liquid diagrams as well, as long as the feed composition is
    /// in a two phase region.
    pub fn lle<TP: TemperatureOrPressure>(
        eos: &Arc<E>,
        temperature_or_pressure: TP,
        feed: &Moles<Array1<f64>>,
        min_tp: TP::Other,
        max_tp: TP::Other,
        npoints: Option<usize>,
    ) -> EosResult<Self> {
        let npoints = npoints.unwrap_or(DEFAULT_POINTS);
        let mut states = Vec::with_capacity(npoints);

        let (t_vec, p_vec) = temperature_or_pressure.linspace(min_tp, max_tp, npoints);
        let mut vle = None;
        for i in 0..npoints {
            let (t, p) = (t_vec.get(i), p_vec.get(i));
            vle = PhaseEquilibrium::tp_flash(
                eos,
                t,
                p,
                feed,
                vle.as_ref(),
                SolverOptions::default(),
                None,
            )
            .ok();
            if let Some(vle) = vle.as_ref() {
                states.push(vle.clone());
            }
        }
        Ok(Self { states })
    }
}

fn iterate_vle<E: Residual, TP: TemperatureOrPressure>(
    eos: &Arc<E>,
    tp: TP,
    x_lim: &[f64],
    vle_0: PhaseEquilibrium<E, 2>,
    vle_1: Option<PhaseEquilibrium<E, 2>>,
    npoints: usize,
    bubble: bool,
    bubble_dew_options: (SolverOptions, SolverOptions),
) -> Vec<PhaseEquilibrium<E, 2>> {
    let mut vle_vec = Vec::with_capacity(npoints);

    let x = Array1::linspace(x_lim[0], x_lim[1], npoints);
    let x = if vle_1.is_some() {
        x.slice(s![1..-1])
    } else {
        x.slice(s![1..])
    };

    let tp_0 = Some(TP::from_state(vle_0.vapor()));
    let mut tp_old = tp_0;
    let mut y_old = None;
    vle_vec.push(vle_0);
    for xi in x {
        let vle = PhaseEquilibrium::bubble_dew_point(
            eos,
            tp,
            tp_old,
            &arr1(&[*xi, 1.0 - xi]),
            y_old.as_ref(),
            bubble,
            bubble_dew_options,
        );

        if let Ok(vle) = vle {
            y_old = Some(if bubble {
                vle.vapor().molefracs.clone()
            } else {
                vle.liquid().molefracs.clone()
            });
            tp_old = Some(TP::from_state(vle.vapor()));
            vle_vec.push(vle.clone());
        } else {
            y_old = None;
            tp_old = tp_0;
        }
    }
    if let Some(vle_1) = vle_1 {
        vle_vec.push(vle_1);
    }

    vle_vec
}

/// Phase diagram (Txy or pxy) for a system with heteroazeotropic phase behavior.
pub struct PhaseDiagramHetero<E> {
    pub vle1: PhaseDiagram<E, 2>,
    pub vle2: PhaseDiagram<E, 2>,
    pub lle: Option<PhaseDiagram<E, 2>>,
}

impl<E: Residual> PhaseDiagram<E, 2> {
    /// Create a new binary phase diagram exhibiting a
    /// vapor/liquid/liquid equilibrium.
    ///
    /// The `x_lle` parameter is used as initial values for the calculation
    /// of the heteroazeotrope.
    pub fn binary_vlle<TP: TemperatureOrPressure>(
        eos: &Arc<E>,
        temperature_or_pressure: TP,
        x_lle: (f64, f64),
        tp_lim_lle: Option<TP::Other>,
        tp_init_vlle: Option<TP::Other>,
        npoints_vle: Option<usize>,
        npoints_lle: Option<usize>,
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> EosResult<PhaseDiagramHetero<E>>
    where
        TPSpec: From<TP>,
    {
        let npoints_vle = npoints_vle.unwrap_or(DEFAULT_POINTS);

        // calculate pure components
        let vle_sat = PhaseEquilibrium::vle_pure_comps(eos, temperature_or_pressure);
        let vle_sat = [vle_sat[1].clone(), vle_sat[0].clone()];

        // calculate heteroazeotrope
        let vlle = PhaseEquilibrium::heteroazeotrope(
            eos,
            temperature_or_pressure,
            x_lle,
            tp_init_vlle,
            SolverOptions::default(),
            bubble_dew_options,
        )?;
        let x_hetero = (vlle.liquid1().molefracs[0], vlle.liquid2().molefracs[0]);

        // calculate vapor liquid equilibria
        let (dia1, dia2) = PhaseDiagram::calculate_vlle(
            eos,
            temperature_or_pressure,
            npoints_vle,
            x_hetero,
            vle_sat,
            bubble_dew_options,
        )?;

        // calculate liquid liquid equilibrium
        let lle = tp_lim_lle
            .map(|tp_lim| {
                let tp_hetero = TP::from_state(vlle.vapor());
                let x_feed = 0.5 * (x_hetero.0 + x_hetero.1);
                let feed = Moles::from_reduced(arr1(&[x_feed, 1.0 - x_feed]));
                PhaseDiagram::lle(
                    eos,
                    temperature_or_pressure,
                    &feed,
                    tp_lim,
                    tp_hetero,
                    npoints_lle,
                )
            })
            .transpose()?;

        Ok(PhaseDiagramHetero {
            vle1: PhaseDiagram::new(dia1),
            vle2: PhaseDiagram::new(dia2),
            lle,
        })
    }
}

impl<E> PhaseDiagramHetero<E> {
    pub fn vle(&self) -> PhaseDiagram<E, 2> {
        PhaseDiagram::new(
            self.vle1
                .states
                .iter()
                .chain(self.vle2.states.iter().rev())
                .cloned()
                .collect(),
        )
    }
}

const MAX_ITER_HETERO: usize = 50;
const TOL_HETERO: f64 = 1e-8;

/// # Heteroazeotropes
impl<E: Residual> PhaseEquilibrium<E, 3> {
    /// Calculate a heteroazeotrope (three phase equilbrium) for a binary
    /// system and given temperature or pressure.
    pub fn heteroazeotrope<TP: TemperatureOrPressure>(
        eos: &Arc<E>,
        temperature_or_pressure: TP,
        x_init: (f64, f64),
        tp_init: Option<TP::Other>,
        options: SolverOptions,
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> EosResult<Self>
    where
        TPSpec: From<TP>,
    {
        let tp_init = tp_init.map(|tp_init| temperature_or_pressure.temperature_pressure(tp_init));
        match TPSpec::from(temperature_or_pressure) {
            TPSpec::Temperature(t) => PhaseEquilibrium::heteroazeotrope_t(
                eos,
                t,
                x_init,
                tp_init.map(|tp| tp.1),
                options,
                bubble_dew_options,
            ),
            TPSpec::Pressure(p) => PhaseEquilibrium::heteroazeotrope_p(
                eos,
                p,
                x_init,
                tp_init.map(|tp| tp.0),
                options,
                bubble_dew_options,
            ),
        }
    }

    /// Calculate a heteroazeotrope (three phase equilbrium) for a binary
    /// system and given temperature.
    fn heteroazeotrope_t(
        eos: &Arc<E>,
        temperature: Temperature,
        x_init: (f64, f64),
        p_init: Option<Pressure>,
        options: SolverOptions,
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> EosResult<Self> {
        // calculate initial values using bubble point
        let x1 = arr1(&[x_init.0, 1.0 - x_init.0]);
        let x2 = arr1(&[x_init.1, 1.0 - x_init.1]);
        let vle1 = PhaseEquilibrium::bubble_point(
            eos,
            temperature,
            &x1,
            p_init,
            None,
            bubble_dew_options,
        )?;
        let vle2 = PhaseEquilibrium::bubble_point(
            eos,
            temperature,
            &x2,
            p_init,
            None,
            bubble_dew_options,
        )?;
        let mut l1 = vle1.liquid().clone();
        let mut l2 = vle2.liquid().clone();
        let p0 = (vle1.vapor().pressure(Contributions::Total)
            + vle2.vapor().pressure(Contributions::Total))
            * 0.5;
        let nv0 = (&vle1.vapor().moles + &vle2.vapor().moles) * 0.5;
        let mut v = State::new_npt(eos, temperature, p0, &nv0, DensityInitialization::Vapor)?;

        for _ in 0..options.max_iter.unwrap_or(MAX_ITER_HETERO) {
            // calculate properties
            let dmu_drho_l1 = (l1.dmu_dni(Contributions::Total) * l1.volume).to_reduced();
            let dmu_drho_l2 = (l2.dmu_dni(Contributions::Total) * l2.volume).to_reduced();
            let dmu_drho_v = (v.dmu_dni(Contributions::Total) * v.volume).to_reduced();
            let dp_drho_l1 = (l1.dp_dni(Contributions::Total) * l1.volume).to_reduced();
            let dp_drho_l2 = (l2.dp_dni(Contributions::Total) * l2.volume).to_reduced();
            let dp_drho_v = (v.dp_dni(Contributions::Total) * v.volume).to_reduced();
            let mu_l1_res = l1.residual_chemical_potential().to_reduced();
            let mu_l2_res = l2.residual_chemical_potential().to_reduced();
            let mu_v_res = v.residual_chemical_potential().to_reduced();
            let p_l1 = l1.pressure(Contributions::Total).to_reduced();
            let p_l2 = l2.pressure(Contributions::Total).to_reduced();
            let p_v = v.pressure(Contributions::Total).to_reduced();

            // calculate residual
            let delta_l1v_mu_ig = (RGAS * v.temperature).to_reduced()
                * (&l1.partial_density / &v.partial_density)
                    .into_value()
                    .mapv(f64::ln);
            let delta_l2v_mu_ig = (RGAS * v.temperature).to_reduced()
                * (&l2.partial_density / &v.partial_density)
                    .into_value()
                    .mapv(f64::ln);
            let res = concatenate![
                Axis(0),
                mu_l1_res - &mu_v_res + delta_l1v_mu_ig,
                mu_l2_res - &mu_v_res + delta_l2v_mu_ig,
                arr1(&[p_l1 - p_v]),
                arr1(&[p_l2 - p_v])
            ];

            // check for convergence
            if norm(&res) < options.tol.unwrap_or(TOL_HETERO) {
                return Ok(Self([v, l1, l2]));
            }

            // calculate Jacobian
            let jacobian = concatenate![
                Axis(1),
                concatenate![
                    Axis(0),
                    dmu_drho_l1,
                    Array2::zeros((2, 2)),
                    dp_drho_l1.insert_axis(Axis(0)),
                    Array2::zeros((1, 2))
                ],
                concatenate![
                    Axis(0),
                    Array2::zeros((2, 2)),
                    dmu_drho_l2,
                    Array2::zeros((1, 2)),
                    dp_drho_l2.insert_axis(Axis(0))
                ],
                concatenate![
                    Axis(0),
                    -&dmu_drho_v,
                    -dmu_drho_v,
                    -dp_drho_v.clone().insert_axis(Axis(0)),
                    -dp_drho_v.insert_axis(Axis(0))
                ]
            ];

            // calculate Newton step
            let dx = LU::new(jacobian)?.solve(&res);

            // apply Newton step
            let rho_l1 =
                &l1.partial_density - &Density::from_reduced(dx.slice(s![0..2]).to_owned());
            let rho_l2 =
                &l2.partial_density - &Density::from_reduced(dx.slice(s![2..4]).to_owned());
            let rho_v = &v.partial_density - &Density::from_reduced(dx.slice(s![4..6]).to_owned());

            // check for negative densities
            for i in 0..2 {
                if rho_l1.get(i).is_sign_negative()
                    || rho_l2.get(i).is_sign_negative()
                    || rho_v.get(i).is_sign_negative()
                {
                    return Err(EosError::IterationFailed(String::from(
                        "PhaseEquilibrium::heteroazeotrope_t",
                    )));
                }
            }

            // update states
            l1 = StateBuilder::new(eos)
                .temperature(temperature)
                .partial_density(&rho_l1)
                .build()?;
            l2 = StateBuilder::new(eos)
                .temperature(temperature)
                .partial_density(&rho_l2)
                .build()?;
            v = StateBuilder::new(eos)
                .temperature(temperature)
                .partial_density(&rho_v)
                .build()?;
        }
        Err(EosError::NotConverged(String::from(
            "PhaseEquilibrium::heteroazeotrope_t",
        )))
    }

    /// Calculate a heteroazeotrope (three phase equilbrium) for a binary
    /// system and given pressure.
    fn heteroazeotrope_p(
        eos: &Arc<E>,
        pressure: Pressure,
        x_init: (f64, f64),
        t_init: Option<Temperature>,
        options: SolverOptions,
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> EosResult<Self> {
        let p = pressure.to_reduced();

        // calculate initial values using bubble point
        let x1 = arr1(&[x_init.0, 1.0 - x_init.0]);
        let x2 = arr1(&[x_init.1, 1.0 - x_init.1]);
        let vle1 =
            PhaseEquilibrium::bubble_point(eos, pressure, &x1, t_init, None, bubble_dew_options)?;
        let vle2 =
            PhaseEquilibrium::bubble_point(eos, pressure, &x2, t_init, None, bubble_dew_options)?;
        let mut l1 = vle1.liquid().clone();
        let mut l2 = vle2.liquid().clone();
        let t0 = (vle1.vapor().temperature + vle2.vapor().temperature) * 0.5;
        let nv0 = (&vle1.vapor().moles + &vle2.vapor().moles) * 0.5;
        let mut v = State::new_npt(eos, t0, pressure, &nv0, DensityInitialization::Vapor)?;

        for _ in 0..options.max_iter.unwrap_or(MAX_ITER_HETERO) {
            // calculate properties
            let dmu_drho_l1 = (l1.dmu_dni(Contributions::Total) * l1.volume).to_reduced();
            let dmu_drho_l2 = (l2.dmu_dni(Contributions::Total) * l2.volume).to_reduced();
            let dmu_drho_v = (v.dmu_dni(Contributions::Total) * v.volume).to_reduced();
            let dmu_res_dt_l1 = (l1.dmu_res_dt()).to_reduced();
            let dmu_res_dt_l2 = (l2.dmu_res_dt()).to_reduced();
            let dmu_res_dt_v = (v.dmu_res_dt()).to_reduced();
            let dp_drho_l1 = (l1.dp_dni(Contributions::Total) * l1.volume).to_reduced();
            let dp_drho_l2 = (l2.dp_dni(Contributions::Total) * l2.volume).to_reduced();
            let dp_drho_v = (v.dp_dni(Contributions::Total) * v.volume).to_reduced();
            let dp_dt_l1 = (l1.dp_dt(Contributions::Total)).to_reduced();
            let dp_dt_l2 = (l2.dp_dt(Contributions::Total)).to_reduced();
            let dp_dt_v = (v.dp_dt(Contributions::Total)).to_reduced();
            let mu_l1_res = l1.residual_chemical_potential().to_reduced();
            let mu_l2_res = l2.residual_chemical_potential().to_reduced();
            let mu_v_res = v.residual_chemical_potential().to_reduced();
            let p_l1 = l1.pressure(Contributions::Total).to_reduced();
            let p_l2 = l2.pressure(Contributions::Total).to_reduced();
            let p_v = v.pressure(Contributions::Total).to_reduced();

            // calculate residual
            let delta_l1v_dmu_ig_dt = (&l1.partial_density / &v.partial_density)
                .into_value()
                .mapv(f64::ln);
            let delta_l2v_dmu_ig_dt = (&l2.partial_density / &v.partial_density)
                .into_value()
                .mapv(f64::ln);
            let delta_l1v_mu_ig = (RGAS * v.temperature).to_reduced() * &delta_l1v_dmu_ig_dt;
            let delta_l2v_mu_ig = (RGAS * v.temperature).to_reduced() * &delta_l2v_dmu_ig_dt;
            let res = concatenate![
                Axis(0),
                mu_l1_res - &mu_v_res + delta_l1v_mu_ig,
                mu_l2_res - &mu_v_res + delta_l2v_mu_ig,
                arr1(&[p_l1 - p]),
                arr1(&[p_l2 - p]),
                arr1(&[p_v - p])
            ];

            // check for convergence
            if norm(&res) < options.tol.unwrap_or(TOL_HETERO) {
                return Ok(Self([v, l1, l2]));
            }

            // calculate Jacobian
            let jacobian = concatenate![
                Axis(1),
                concatenate![
                    Axis(0),
                    dmu_drho_l1,
                    Array2::zeros((2, 2)),
                    dp_drho_l1.insert_axis(Axis(0)),
                    Array2::zeros((1, 2)),
                    Array2::zeros((1, 2))
                ],
                concatenate![
                    Axis(0),
                    Array2::zeros((2, 2)),
                    dmu_drho_l2,
                    Array2::zeros((1, 2)),
                    dp_drho_l2.insert_axis(Axis(0)),
                    Array2::zeros((1, 2))
                ],
                concatenate![
                    Axis(0),
                    -&dmu_drho_v,
                    -dmu_drho_v,
                    Array2::zeros((1, 2)),
                    Array2::zeros((1, 2)),
                    dp_drho_v.insert_axis(Axis(0))
                ],
                concatenate![
                    Axis(0),
                    (dmu_res_dt_l1 - &dmu_res_dt_v + delta_l1v_dmu_ig_dt).insert_axis(Axis(1)),
                    (dmu_res_dt_l2 - &dmu_res_dt_v + delta_l2v_dmu_ig_dt).insert_axis(Axis(1)),
                    arr2(&[[dp_dt_l1]]),
                    arr2(&[[dp_dt_l2]]),
                    arr2(&[[dp_dt_v]])
                ]
            ];

            // calculate Newton step
            let dx = LU::new(jacobian)?.solve(&res);

            // apply Newton step
            let rho_l1 = l1.partial_density - Density::from_reduced(dx.slice(s![0..2]).to_owned());
            let rho_l2 = l2.partial_density - Density::from_reduced(dx.slice(s![2..4]).to_owned());
            let rho_v = v.partial_density - Density::from_reduced(dx.slice(s![4..6]).to_owned());
            let t = v.temperature - Temperature::from_reduced(dx[6]);

            // check for negative densities and temperatures
            for i in 0..2 {
                if rho_l1.get(i).is_sign_negative()
                    || rho_l2.get(i).is_sign_negative()
                    || rho_v.get(i).is_sign_negative()
                    || t.is_sign_negative()
                {
                    return Err(EosError::IterationFailed(String::from(
                        "PhaseEquilibrium::heteroazeotrope_p",
                    )));
                }
            }

            // update states
            l1 = StateBuilder::new(eos)
                .temperature(t)
                .partial_density(&rho_l1)
                .build()?;
            l2 = StateBuilder::new(eos)
                .temperature(t)
                .partial_density(&rho_l2)
                .build()?;
            v = StateBuilder::new(eos)
                .temperature(t)
                .partial_density(&rho_v)
                .build()?;
        }
        Err(EosError::NotConverged(String::from(
            "PhaseEquilibrium::heteroazeotrope_p",
        )))
    }
}
