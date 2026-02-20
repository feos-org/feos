use super::bubble_dew::TemperatureOrPressure;
use super::{PhaseDiagram, PhaseEquilibrium};
use crate::errors::{FeosError, FeosResult};
use crate::state::{Contributions, DensityInitialization::Vapor, State};
use crate::{ReferenceSystem, Residual, SolverOptions, Subset};
use nalgebra::{DVector, dvector, matrix, stack, vector};
use ndarray::{Array1, s};
use num_dual::linalg::LU;
use quantity::{Density, Moles, Pressure, RGAS, Temperature};

const DEFAULT_POINTS: usize = 51;

impl<E: Residual + Subset> PhaseDiagram<E, 2> {
    /// Create a new binary phase diagram exhibiting a
    /// vapor/liquid equilibrium.
    ///
    /// If a heteroazeotrope occurs and the composition of the liquid
    /// phases are known, they can be passed as `x_lle` to avoid
    /// the calculation of unstable branches.
    pub fn binary_vle<TP: TemperatureOrPressure>(
        eos: &E,
        temperature_or_pressure: TP,
        npoints: Option<usize>,
        x_lle: Option<(f64, f64)>,
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> FeosResult<Self> {
        let npoints = npoints.unwrap_or(DEFAULT_POINTS);

        // calculate boiling temperature/vapor pressure of pure components
        let vle_sat = PhaseEquilibrium::vle_pure_comps(eos, temperature_or_pressure);
        let vle_sat = [vle_sat[1].clone(), vle_sat[0].clone()];

        // Only calculate up to specified compositions
        if let Some(x_lle) = x_lle {
            let [states1, states2] = Self::calculate_vlle(
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
        let bubble = temperature_or_pressure.temperature().is_some();

        // look for supercritical components
        let (x_lim, vle_lim, bubble) = match vle_sat {
            [None, None] => return Err(FeosError::SuperCritical),
            [Some(vle2), None] => {
                let cp = State::critical_point_binary(
                    eos,
                    temperature_or_pressure,
                    None,
                    None,
                    None,
                    SolverOptions::default(),
                )?;
                let x_max = cp.molefracs[0];
                let cp_vle = PhaseEquilibrium::single_phase(cp);
                ([0.0, x_max], (vle2, cp_vle), bubble)
            }
            [None, Some(vle1)] => {
                let cp = State::critical_point_binary(
                    eos,
                    temperature_or_pressure,
                    None,
                    None,
                    None,
                    SolverOptions::default(),
                )?;
                let x_min = cp.molefracs[0];
                let cp_vle = PhaseEquilibrium::single_phase(cp);
                ([1.0, x_min], (vle1, cp_vle), bubble)
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
        let states = check_for_vlle(temperature_or_pressure, states, npoints, bubble_dew_options);
        Ok(Self { states })
    }

    fn calculate_vlle<TP: TemperatureOrPressure>(
        eos: &E,
        tp: TP,
        npoints: usize,
        x_lle: (f64, f64),
        vle_sat: [Option<PhaseEquilibrium<E, 2>>; 2],
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> FeosResult<[Vec<PhaseEquilibrium<E, 2>>; 2]> {
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
                Ok([states1, states2])
            }
            _ => Err(FeosError::SuperCritical),
        }
    }

    /// Create a new phase diagram using Tp flash calculations.
    ///
    /// The usual use case for this function is the calculation of
    /// liquid-liquid phase diagrams, but it can be used for vapor-
    /// liquid diagrams as well, as long as the feed composition is
    /// in a two phase region.
    pub fn lle<TP: TemperatureOrPressure>(
        eos: &E,
        temperature_or_pressure: TP,
        feed: &Moles<DVector<f64>>,
        min_tp: TP::Other,
        max_tp: TP::Other,
        npoints: Option<usize>,
    ) -> FeosResult<Self> {
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

#[expect(clippy::too_many_arguments)]
fn iterate_vle<E: Residual + Subset, TP: TemperatureOrPressure>(
    eos: &E,
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
            dvector![*xi, 1.0 - xi],
            tp_old,
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

fn check_for_vlle<E: Residual + Subset, TP: TemperatureOrPressure>(
    tp: TP,
    states: Vec<PhaseEquilibrium<E, 2>>,
    npoints: usize,
    bubble_dew_options: (SolverOptions, SolverOptions),
) -> Vec<PhaseEquilibrium<E, 2>> {
    let n = states.len();
    let p: Vec<_> = states
        .iter()
        .map(|s| s.vapor().pressure(Contributions::Total))
        .collect();
    let t: Vec<_> = states.iter().map(|s| s.vapor().temperature).collect();
    let x: Vec<_> = states.iter().map(|s| s.liquid().molefracs[0]).collect();
    let y: Vec<_> = states.iter().map(|s| s.vapor().molefracs[0]).collect();

    // Determine if the dew line intersects with itself
    if let Some(t) = tp.temperature()
        && p[1] > p[0]
        && p[n - 2] > p[n - 1]
    {
        let [mut i, mut j] = [0, n - 1];
        while i != j {
            if p[i] > p[j] {
                j -= 1;
            } else {
                i += 1
            }
            if y[j] < y[i] {
                // intersection found!
                let (xj, yj, pj) = if j == n - 2 {
                    // Use Henry constant of component 2
                    let k_inf = (states[n - 1].liquid().ln_phi() - states[n - 1].vapor().ln_phi())
                        .map(f64::exp)[1];
                    (
                        [1.0, 1.0 - 1.0 / k_inf],
                        [1.0, 0.0],
                        [p[n - 1], p[n - 1] * (2.0 - 1.0 / k_inf)],
                    )
                } else {
                    // or interpolate linearly
                    ([x[j + 1], x[j]], [y[j + 1], y[j]], [p[j + 1], p[j]])
                };
                let (xi, yi, pi) = if i == 1 {
                    // Use Henry constant of component 1
                    let k_inf =
                        (states[0].liquid().ln_phi() - states[0].vapor().ln_phi()).map(f64::exp)[0];
                    (
                        [0.0, 1.0 / k_inf],
                        [0.0, 1.0],
                        [p[0], p[0] * (2.0 - 1.0 / k_inf)],
                    )
                } else {
                    // or interpolate linearly
                    ([x[i - 1], x[i]], [y[i - 1], y[i]], [p[i - 1], p[i]])
                };
                // calculate intersection
                let a = matrix![yi[1] - yi[0], yj[0] - yj[1];
                                (pi[1] - pi[0]).into_reduced(), (pj[0] - pj[1]).into_reduced()];
                let b = vector![yj[0] - yi[0], (pj[0] - pi[0]).into_reduced()];
                let [[r, s]] = LU::new(a).unwrap().solve(&b).data.0;
                let (xi, xj, p) = (
                    xi[0] + r * (xi[1] - xi[0]),
                    xj[0] + s * (xj[1] - xj[0]),
                    pi[0] + r * (pi[1] - pi[0]),
                );
                let Ok(vlle) = PhaseEquilibrium::heteroazeotrope(
                    &states[0].liquid().eos,
                    t,
                    (xi, xj),
                    Some(p),
                    Default::default(),
                    bubble_dew_options,
                ) else {
                    return states;
                };
                let x_hetero = (vlle.liquid1().molefracs[0], vlle.liquid2().molefracs[0]);
                return PhaseDiagram::binary_vle(
                    &states[0].liquid().eos,
                    tp,
                    Some(npoints),
                    Some(x_hetero),
                    bubble_dew_options,
                )
                .map_or(states, |dia| dia.states);
            }
        }
    } else if let Some(p) = tp.pressure()
        && t[1] < t[0]
        && t[n - 2] < t[n - 1]
    {
        let [mut i, mut j] = [0, n - 1];
        while i != j {
            if t[i] < t[j] {
                j -= 1;
            } else {
                i += 1
            }
            if y[j] < y[i] {
                // intersection found!
                let (xj, yj, tj) = if j == n - 2 {
                    // Use Henry constant of component 2
                    let vle = &states[n - 1];
                    let k_inf = (vle.liquid().ln_phi() - vle.vapor().ln_phi()).map(f64::exp)[1];
                    let dh = vle.vapor().residual_molar_enthalpy()
                        - vle.liquid().residual_molar_enthalpy();
                    let dv = 1.0 / vle.vapor().density - 1.0 / vle.liquid().density;
                    let pdv_dh = (p * dv).convert_into(dh);
                    (
                        [1.0, 1.0 - 1.0 / k_inf],
                        [1.0, 0.0],
                        [t[n - 1], t[n - 1] * (1.0 - (k_inf - 1.0) / k_inf * pdv_dh)],
                    )
                } else {
                    // or interpolate linearly
                    ([x[j + 1], x[j]], [y[j + 1], y[j]], [t[j + 1], t[j]])
                };
                let (xi, yi, ti) = if i == 1 {
                    // Use Henry constant of component 1
                    let vle = &states[0];
                    let k_inf = (vle.liquid().ln_phi() - vle.vapor().ln_phi()).map(f64::exp)[0];
                    let dh = vle.vapor().residual_molar_enthalpy()
                        - vle.liquid().residual_molar_enthalpy();
                    let dv = 1.0 / vle.vapor().density - 1.0 / vle.liquid().density;
                    let pdv_dh = (p * dv).convert_into(dh);
                    (
                        [0.0, 1.0 / k_inf],
                        [0.0, 1.0],
                        [t[0], t[0] * (1.0 - (k_inf - 1.0) / k_inf * pdv_dh)],
                    )
                } else {
                    // or interpolate linearly
                    ([x[i - 1], x[i]], [y[i - 1], y[i]], [t[i - 1], t[i]])
                };
                // calculate intersection
                let a = matrix![yi[1] - yi[0], yj[0] - yj[1];
                                (ti[1] - ti[0]).into_reduced(), (tj[0] - tj[1]).into_reduced()];
                let b = vector![yj[0] - yi[0], (tj[0] - ti[0]).into_reduced()];
                let [[r, s]] = LU::new(a).unwrap().solve(&b).data.0;
                let (xi, xj, t) = (
                    xi[0] + r * (xi[1] - xi[0]),
                    xj[0] + s * (xj[1] - xj[0]),
                    ti[0] + r * (ti[1] - ti[0]),
                );
                let Ok(vlle) = PhaseEquilibrium::heteroazeotrope(
                    &states[0].liquid().eos,
                    p,
                    (xi, xj),
                    Some(t),
                    Default::default(),
                    bubble_dew_options,
                ) else {
                    return states;
                };
                let x_hetero = (vlle.liquid1().molefracs[0], vlle.liquid2().molefracs[0]);
                return PhaseDiagram::binary_vle(
                    &states[0].liquid().eos,
                    tp,
                    Some(npoints),
                    Some(x_hetero),
                    bubble_dew_options,
                )
                .map_or(states, |dia| dia.states);
            }
        }
    }
    states
}

/// Phase diagram (Txy or pxy) for a system with heteroazeotropic phase behavior.
pub struct PhaseDiagramHetero<E> {
    pub vle1: PhaseDiagram<E, 2>,
    pub vle2: PhaseDiagram<E, 2>,
    pub lle: Option<PhaseDiagram<E, 2>>,
}

impl<E: Residual + Subset> PhaseDiagram<E, 2> {
    /// Create a new binary phase diagram exhibiting a
    /// vapor/liquid/liquid equilibrium.
    ///
    /// The `x_lle` parameter is used as initial values for the calculation
    /// of the heteroazeotrope.
    #[expect(clippy::too_many_arguments)]
    pub fn binary_vlle<TP: TemperatureOrPressure>(
        eos: &E,
        temperature_or_pressure: TP,
        x_lle: (f64, f64),
        tp_lim_lle: Option<TP::Other>,
        tp_init_vlle: Option<TP::Other>,
        npoints_vle: Option<usize>,
        npoints_lle: Option<usize>,
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> FeosResult<PhaseDiagramHetero<E>> {
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
        let [dia1, dia2] = PhaseDiagram::calculate_vlle(
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
                let feed = Moles::from_reduced(dvector![x_feed, 1.0 - x_feed]);
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

impl<E: Clone> PhaseDiagramHetero<E> {
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
        eos: &E,
        temperature_or_pressure: TP,
        x_init: (f64, f64),
        tp_init: Option<TP::Other>,
        options: SolverOptions,
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> FeosResult<Self> {
        let (temperature, pressure, iterate_p) =
            temperature_or_pressure.temperature_pressure(tp_init);
        if iterate_p {
            PhaseEquilibrium::heteroazeotrope_t(
                eos,
                temperature.unwrap(),
                x_init,
                pressure,
                options,
                bubble_dew_options,
            )
        } else {
            PhaseEquilibrium::heteroazeotrope_p(
                eos,
                pressure.unwrap(),
                x_init,
                temperature,
                options,
                bubble_dew_options,
            )
        }
    }

    /// Calculate a heteroazeotrope (three phase equilbrium) for a binary
    /// system and given temperature.
    #[expect(clippy::toplevel_ref_arg)]
    fn heteroazeotrope_t(
        eos: &E,
        temperature: Temperature,
        x_init: (f64, f64),
        p_init: Option<Pressure>,
        options: SolverOptions,
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> FeosResult<Self> {
        // calculate initial values using bubble point
        let x1 = dvector![x_init.0, 1.0 - x_init.0];
        let x2 = dvector![x_init.1, 1.0 - x_init.1];
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
        let y0 = (&vle1.vapor().molefracs + &vle2.vapor().molefracs) * 0.5;
        let mut v = State::new_npt(eos, temperature, p0, y0, Some(Vapor))?;

        for _ in 0..options.max_iter.unwrap_or(MAX_ITER_HETERO) {
            // calculate properties
            let dmu_drho_l1 = (l1.n_dmu_dni(Contributions::Total) * l1.molar_volume).to_reduced();
            let dmu_drho_l2 = (l2.n_dmu_dni(Contributions::Total) * l2.molar_volume).to_reduced();
            let dmu_drho_v = (v.n_dmu_dni(Contributions::Total) * v.molar_volume).to_reduced();
            let dp_drho_l1 = (l1.n_dp_dni(Contributions::Total) * l1.molar_volume)
                .to_reduced()
                .transpose();
            let dp_drho_l2 = (l2.n_dp_dni(Contributions::Total) * l2.molar_volume)
                .to_reduced()
                .transpose();
            let dp_drho_v = (v.n_dp_dni(Contributions::Total) * v.molar_volume)
                .to_reduced()
                .transpose();
            let mu_l1_res = l1.residual_chemical_potential().to_reduced();
            let mu_l2_res = l2.residual_chemical_potential().to_reduced();
            let mu_v_res = v.residual_chemical_potential().to_reduced();
            let p_l1 = l1.pressure(Contributions::Total).to_reduced();
            let p_l2 = l2.pressure(Contributions::Total).to_reduced();
            let p_v = v.pressure(Contributions::Total).to_reduced();

            // calculate residual
            let delta_l1v_mu_ig = (RGAS * v.temperature).to_reduced()
                * (l1
                    .partial_density()
                    .to_reduced()
                    .component_div(&v.partial_density().to_reduced()))
                .map(f64::ln);
            let delta_l2v_mu_ig = (RGAS * v.temperature).to_reduced()
                * (l2
                    .partial_density()
                    .to_reduced()
                    .component_div(&v.partial_density().to_reduced()))
                .map(f64::ln);
            let res = stack![
                mu_l1_res - &mu_v_res + delta_l1v_mu_ig;
                mu_l2_res - &mu_v_res + delta_l2v_mu_ig;
                vector![p_l1 - p_v];
                vector![p_l2 - p_v]
            ];

            // check for convergence
            if res.norm() < options.tol.unwrap_or(TOL_HETERO) {
                return Ok(Self::new(v, l1, l2));
            }

            // calculate Jacobian
            let jacobian = stack![
                dmu_drho_l1, 0          , -&dmu_drho_v;
                0          , dmu_drho_l2, -dmu_drho_v;
                dp_drho_l1 , 0          , -&dp_drho_v;
                0          , dp_drho_l2 , -dp_drho_v
            ];

            // calculate Newton step
            let dx = LU::new(jacobian)?.solve(&res);

            // apply Newton step
            let rho_l1 =
                &l1.partial_density() - &Density::from_reduced(dx.rows_range(0..2).into_owned());
            let rho_l2 =
                &l2.partial_density() - &Density::from_reduced(dx.rows_range(2..4).into_owned());
            let rho_v =
                &v.partial_density() - &Density::from_reduced(dx.rows_range(4..6).into_owned());

            // check for negative densities
            for i in 0..2 {
                if rho_l1.get(i).is_sign_negative()
                    || rho_l2.get(i).is_sign_negative()
                    || rho_v.get(i).is_sign_negative()
                {
                    return Err(FeosError::IterationFailed(String::from(
                        "PhaseEquilibrium::heteroazeotrope_t",
                    )));
                }
            }

            // update states
            l1 = State::new_density(eos, temperature, rho_l1)?;
            l2 = State::new_density(eos, temperature, rho_l2)?;
            v = State::new_density(eos, temperature, rho_v)?;
        }
        Err(FeosError::NotConverged(String::from(
            "PhaseEquilibrium::heteroazeotrope_t",
        )))
    }

    /// Calculate a heteroazeotrope (three phase equilbrium) for a binary
    /// system and given pressure.
    #[expect(clippy::toplevel_ref_arg)]
    fn heteroazeotrope_p(
        eos: &E,
        pressure: Pressure,
        x_init: (f64, f64),
        t_init: Option<Temperature>,
        options: SolverOptions,
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> FeosResult<Self> {
        let p = pressure.to_reduced();

        // calculate initial values using bubble point
        let x1 = dvector![x_init.0, 1.0 - x_init.0];
        let x2 = dvector![x_init.1, 1.0 - x_init.1];
        let vle1 =
            PhaseEquilibrium::bubble_point(eos, pressure, &x1, t_init, None, bubble_dew_options)?;
        let vle2 =
            PhaseEquilibrium::bubble_point(eos, pressure, &x2, t_init, None, bubble_dew_options)?;
        let mut l1 = vle1.liquid().clone();
        let mut l2 = vle2.liquid().clone();
        let t0 = (vle1.vapor().temperature + vle2.vapor().temperature) * 0.5;
        let y0 = (&vle1.vapor().molefracs + &vle2.vapor().molefracs) * 0.5;
        let mut v = State::new_npt(eos, t0, pressure, y0, Some(Vapor))?;

        for _ in 0..options.max_iter.unwrap_or(MAX_ITER_HETERO) {
            // calculate properties
            let dmu_drho_l1 = (l1.n_dmu_dni(Contributions::Total) * l1.molar_volume).to_reduced();
            let dmu_drho_l2 = (l2.n_dmu_dni(Contributions::Total) * l2.molar_volume).to_reduced();
            let dmu_drho_v = (v.n_dmu_dni(Contributions::Total) * v.molar_volume).to_reduced();
            let dmu_res_dt_l1 = (l1.dmu_res_dt()).to_reduced();
            let dmu_res_dt_l2 = (l2.dmu_res_dt()).to_reduced();
            let dmu_res_dt_v = (v.dmu_res_dt()).to_reduced();
            let dp_drho_l1 = (l1.n_dp_dni(Contributions::Total) * l1.molar_volume)
                .to_reduced()
                .transpose();
            let dp_drho_l2 = (l2.n_dp_dni(Contributions::Total) * l2.molar_volume)
                .to_reduced()
                .transpose();
            let dp_drho_v = (v.n_dp_dni(Contributions::Total) * v.molar_volume)
                .to_reduced()
                .transpose();
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
            let delta_l1v_dmu_ig_dt = l1
                .partial_density()
                .to_reduced()
                .component_div(&v.partial_density().to_reduced())
                .map(f64::ln);
            let delta_l2v_dmu_ig_dt = l2
                .partial_density()
                .to_reduced()
                .component_div(&v.partial_density().to_reduced())
                .map(f64::ln);
            let delta_l1v_mu_ig = (RGAS * v.temperature).to_reduced() * &delta_l1v_dmu_ig_dt;
            let delta_l2v_mu_ig = (RGAS * v.temperature).to_reduced() * &delta_l2v_dmu_ig_dt;
            let res = stack![
                mu_l1_res - &mu_v_res + delta_l1v_mu_ig;
                mu_l2_res - &mu_v_res + delta_l2v_mu_ig;
                vector![p_l1 - p];
                vector![p_l2 - p];
                vector![p_v - p]
            ];

            // check for convergence
            if res.norm() < options.tol.unwrap_or(TOL_HETERO) {
                return Ok(Self::new(v, l1, l2));
            }

            let jacobian = stack![
                dmu_drho_l1, 0, -&dmu_drho_v, dmu_res_dt_l1 - &dmu_res_dt_v + delta_l1v_dmu_ig_dt;
                0, dmu_drho_l2, -dmu_drho_v, dmu_res_dt_l2 - &dmu_res_dt_v + delta_l2v_dmu_ig_dt;
                dp_drho_l1, 0, 0, vector![dp_dt_l1];
                0, dp_drho_l2, 0, vector![dp_dt_l2];
                0, 0, dp_drho_v, vector![dp_dt_v]
            ];

            // calculate Newton step
            let dx = LU::new(jacobian)?.solve(&res);

            // apply Newton step
            let rho_l1 =
                l1.partial_density() - Density::from_reduced(dx.rows_range(0..2).into_owned());
            let rho_l2 =
                l2.partial_density() - Density::from_reduced(dx.rows_range(2..4).into_owned());
            let rho_v =
                v.partial_density() - Density::from_reduced(dx.rows_range(4..6).into_owned());
            let t = v.temperature - Temperature::from_reduced(dx[6]);

            // check for negative densities and temperatures
            for i in 0..2 {
                if rho_l1.get(i).is_sign_negative()
                    || rho_l2.get(i).is_sign_negative()
                    || rho_v.get(i).is_sign_negative()
                    || t.is_sign_negative()
                {
                    return Err(FeosError::IterationFailed(String::from(
                        "PhaseEquilibrium::heteroazeotrope_p",
                    )));
                }
            }

            // update states
            l1 = State::new_density(eos, t, rho_l1)?;
            l2 = State::new_density(eos, t, rho_l2)?;
            v = State::new_density(eos, t, rho_v)?;
        }
        Err(FeosError::NotConverged(String::from(
            "PhaseEquilibrium::heteroazeotrope_p",
        )))
    }
}
