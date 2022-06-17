use super::{PhaseDiagram, PhaseEquilibrium, SolverOptions};
use crate::equation_of_state::EquationOfState;
use crate::errors::{EosError, EosResult};
use crate::state::{Contributions, DensityInitialization, State, StateBuilder, TPSpec};
use crate::EosUnit;
use ndarray::{arr1, arr2, concatenate, s, Array1, Array2, Axis};
use num_dual::linalg::{norm, LU};
use quantity::{QuantityArray1, QuantityScalar};
use std::convert::{TryFrom, TryInto};
use std::rc::Rc;

const DEFAULT_POINTS: usize = 51;

impl<U: EosUnit, E: EquationOfState> PhaseDiagram<U, E> {
    /// Create a new binary phase diagram exhibiting a
    /// vapor/liquid equilibrium.
    ///
    /// If a heteroazeotrope occurs and the composition of the liquid
    /// phases are known, they can be passed as `x_lle` to avoid
    /// the calculation of unstable branches.
    pub fn binary_vle(
        eos: &Rc<E>,
        temperature_or_pressure: QuantityScalar<U>,
        npoints: Option<usize>,
        x_lle: Option<(f64, f64)>,
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> EosResult<Self>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let npoints = npoints.unwrap_or(DEFAULT_POINTS);
        let tp = temperature_or_pressure.try_into()?;

        // calculate boiling temperature/vapor pressure of pure components
        let vle_sat = PhaseEquilibrium::vle_pure_comps(eos, temperature_or_pressure);
        let vle_sat = [vle_sat[1].clone(), vle_sat[0].clone()];

        // Only calculate up to specified compositions
        if let Some(x_lle) = x_lle {
            let (states1, states2) =
                Self::calculate_vlle(eos, tp, npoints, x_lle, vle_sat, bubble_dew_options)?;

            let states = states1
                .into_iter()
                .chain(states2.into_iter().rev())
                .collect();
            return Ok(Self { states });
        }

        // use dew point when calculating a supercritical tx diagram
        let bubble = match tp {
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
            tp,
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
    fn calculate_vlle(
        eos: &Rc<E>,
        tp: TPSpec<U>,
        npoints: usize,
        x_lle: (f64, f64),
        vle_sat: [Option<PhaseEquilibrium<U, E, 2>>; 2],
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> EosResult<(
        Vec<PhaseEquilibrium<U, E, 2>>,
        Vec<PhaseEquilibrium<U, E, 2>>,
    )>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
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
    pub fn lle(
        eos: &Rc<E>,
        temperature_or_pressure: QuantityScalar<U>,
        feed: &QuantityArray1<U>,
        min_tp: QuantityScalar<U>,
        max_tp: QuantityScalar<U>,
        npoints: Option<usize>,
    ) -> EosResult<Self>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let npoints = npoints.unwrap_or(DEFAULT_POINTS);
        let mut states = Vec::with_capacity(npoints);
        let tp: TPSpec<U> = temperature_or_pressure.try_into()?;

        let tp_vec = QuantityArray1::linspace(min_tp, max_tp, npoints)?;
        let mut vle = None;
        for i in 0..npoints {
            let (_, t, p) = tp.temperature_pressure(tp_vec.get(i));
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
            if let Some(vle) = &vle {
                states.push(vle.clone());
            }
        }
        Ok(Self { states })
    }
}

fn iterate_vle<U: EosUnit, E: EquationOfState>(
    eos: &Rc<E>,
    tp: TPSpec<U>,
    x_lim: &[f64],
    vle_0: PhaseEquilibrium<U, E, 2>,
    vle_1: Option<PhaseEquilibrium<U, E, 2>>,
    npoints: usize,
    bubble: bool,
    bubble_dew_options: (SolverOptions, SolverOptions),
) -> Vec<PhaseEquilibrium<U, E, 2>>
where
    QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
{
    let mut vle_vec = Vec::with_capacity(npoints);

    let x = Array1::linspace(x_lim[0], x_lim[1], npoints);
    let x = if vle_1.is_some() {
        x.slice(s![1..-1])
    } else {
        x.slice(s![1..])
    };

    let mut tp_old = Some(vle_0.vapor().tp(tp));
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
            tp_old = Some(match tp {
                TPSpec::Temperature(_) => vle.vapor().pressure(Contributions::Total),
                TPSpec::Pressure(_) => vle.vapor().temperature,
            });
            vle_vec.push(vle.clone());
        } else {
            y_old = None;
            tp_old = None;
        }
    }
    if let Some(vle_1) = vle_1 {
        vle_vec.push(vle_1);
    }

    vle_vec
}

impl<U: EosUnit, E: EquationOfState> State<U, E> {
    fn tp(&self, tp: TPSpec<U>) -> QuantityScalar<U> {
        match tp {
            TPSpec::Temperature(_) => self.pressure(Contributions::Total),
            TPSpec::Pressure(_) => self.temperature,
        }
    }
}

/// Phase diagram (Txy or pxy) for a system with heteroazeotropic phase behavior.
pub struct PhaseDiagramHetero<U, E> {
    pub vle1: PhaseDiagram<U, E>,
    pub vle2: PhaseDiagram<U, E>,
    pub lle: Option<PhaseDiagram<U, E>>,
}

impl<U: EosUnit, E: EquationOfState> PhaseDiagram<U, E> {
    /// Create a new binary phase diagram exhibiting a
    /// vapor/liquid/liquid equilibrium.
    ///
    /// The `x_lle` parameter is used as initial values for the calculation
    /// of the heteroazeotrope.
    pub fn binary_vlle(
        eos: &Rc<E>,
        temperature_or_pressure: QuantityScalar<U>,
        x_lle: (f64, f64),
        tp_lim_lle: Option<QuantityScalar<U>>,
        npoints_vle: Option<usize>,
        npoints_lle: Option<usize>,
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> EosResult<PhaseDiagramHetero<U, E>>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let npoints_vle = npoints_vle.unwrap_or(DEFAULT_POINTS);
        let tp = temperature_or_pressure.try_into()?;

        // calculate pure components
        let vle_sat = PhaseEquilibrium::vle_pure_comps(eos, temperature_or_pressure);
        let vle_sat = [vle_sat[1].clone(), vle_sat[0].clone()];

        // calculate heteroazeotrope
        let vlle = match tp {
            TPSpec::Temperature(t) => PhaseEquilibrium::heteroazeotrope_t(
                eos,
                t,
                x_lle,
                SolverOptions::default(),
                bubble_dew_options,
            ),
            TPSpec::Pressure(p) => PhaseEquilibrium::heteroazeotrope_p(
                eos,
                p,
                x_lle,
                SolverOptions::default(),
                bubble_dew_options,
            ),
        }?;
        let x_hetero = (vlle.liquid1().molefracs[0], vlle.liquid2().molefracs[0]);

        // calculate vapor liquid equilibria
        let (dia1, dia2) = PhaseDiagram::calculate_vlle(
            eos,
            tp,
            npoints_vle,
            x_hetero,
            vle_sat,
            bubble_dew_options,
        )?;

        // calculate liquid liquid equilibrium
        let lle = tp_lim_lle
            .map(|tp_lim| {
                let tp_hetero = match tp {
                    TPSpec::Pressure(_) => vlle.vapor().temperature,
                    TPSpec::Temperature(_) => vlle.vapor().pressure(Contributions::Total),
                };
                let x_feed = 0.5 * (x_hetero.0 + x_hetero.1);
                let feed = arr1(&[x_feed, 1.0 - x_feed]) * U::reference_moles();
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
            vle1: PhaseDiagram { states: dia1 },
            vle2: PhaseDiagram { states: dia2 },
            lle,
        })
    }
}

impl<U: Clone, E> PhaseDiagramHetero<U, E> {
    pub fn vle(&self) -> PhaseDiagram<U, E> {
        PhaseDiagram {
            states: self
                .vle1
                .states
                .iter()
                .chain(self.vle2.states.iter().rev())
                .cloned()
                .collect(),
        }
    }
}

const MAX_ITER_HETERO: usize = 50;
const TOL_HETERO: f64 = 1e-8;

/// # Heteroazeotropes
impl<U: EosUnit, E: EquationOfState> PhaseEquilibrium<U, E, 3>
where
    QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
{
    /// Calculate a heteroazeotrope (three phase equilbrium) for a binary
    /// system and given pressure.
    pub fn heteroazeotrope(
        eos: &Rc<E>,
        temperature_or_pressure: QuantityScalar<U>,
        x_init: (f64, f64),
        options: SolverOptions,
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> EosResult<Self> {
        match TPSpec::try_from(temperature_or_pressure)? {
            TPSpec::Temperature(t) => {
                Self::heteroazeotrope_t(eos, t, x_init, options, bubble_dew_options)
            }
            TPSpec::Pressure(p) => {
                Self::heteroazeotrope_p(eos, p, x_init, options, bubble_dew_options)
            }
        }
    }

    /// Calculate a heteroazeotrope (three phase equilbrium) for a binary
    /// system and given temperature.
    fn heteroazeotrope_t(
        eos: &Rc<E>,
        temperature: QuantityScalar<U>,
        x_init: (f64, f64),
        options: SolverOptions,
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> EosResult<Self> {
        // calculate initial values using bubble point
        let x1 = arr1(&[x_init.0, 1.0 - x_init.0]);
        let x2 = arr1(&[x_init.1, 1.0 - x_init.1]);
        let vle1 =
            PhaseEquilibrium::bubble_point(eos, temperature, &x1, None, None, bubble_dew_options)?;
        let vle2 =
            PhaseEquilibrium::bubble_point(eos, temperature, &x2, None, None, bubble_dew_options)?;
        let mut l1 = vle1.liquid().clone();
        let mut l2 = vle2.liquid().clone();
        let p0 = (vle1.vapor().pressure(Contributions::Total)
            + vle2.vapor().pressure(Contributions::Total))
            * 0.5;
        let nv0 = (&vle1.vapor().moles + &vle2.vapor().moles) * 0.5;
        let mut v = State::new_npt(eos, temperature, p0, &nv0, DensityInitialization::Vapor)?;

        for _ in 0..options.max_iter.unwrap_or(MAX_ITER_HETERO) {
            // calculate properties
            let dmu_drho_l1 = (l1.dmu_dni(Contributions::Total) * l1.volume)
                .to_reduced(U::reference_molar_energy() / U::reference_density())?;
            let dmu_drho_l2 = (l2.dmu_dni(Contributions::Total) * l2.volume)
                .to_reduced(U::reference_molar_energy() / U::reference_density())?;
            let dmu_drho_v = (v.dmu_dni(Contributions::Total) * v.volume)
                .to_reduced(U::reference_molar_energy() / U::reference_density())?;
            let dp_drho_l1 = (l1.dp_dni(Contributions::Total) * l1.volume)
                .to_reduced(U::reference_pressure() / U::reference_density())?;
            let dp_drho_l2 = (l2.dp_dni(Contributions::Total) * l2.volume)
                .to_reduced(U::reference_pressure() / U::reference_density())?;
            let dp_drho_v = (v.dp_dni(Contributions::Total) * v.volume)
                .to_reduced(U::reference_pressure() / U::reference_density())?;
            let mu_l1 = l1
                .chemical_potential(Contributions::Total)
                .to_reduced(U::reference_molar_energy())?;
            let mu_l2 = l2
                .chemical_potential(Contributions::Total)
                .to_reduced(U::reference_molar_energy())?;
            let mu_v = v
                .chemical_potential(Contributions::Total)
                .to_reduced(U::reference_molar_energy())?;
            let p_l1 = l1
                .pressure(Contributions::Total)
                .to_reduced(U::reference_pressure())?;
            let p_l2 = l2
                .pressure(Contributions::Total)
                .to_reduced(U::reference_pressure())?;
            let p_v = v
                .pressure(Contributions::Total)
                .to_reduced(U::reference_pressure())?;

            // calculate residual
            let res = concatenate![
                Axis(0),
                mu_l1 - &mu_v,
                mu_l2 - &mu_v,
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
                &l1.partial_density - &(dx.slice(s![0..2]).to_owned() * U::reference_density());
            let rho_l2 =
                &l2.partial_density - &(dx.slice(s![2..4]).to_owned() * U::reference_density());
            let rho_v =
                &v.partial_density - &(dx.slice(s![4..6]).to_owned() * U::reference_density());

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
        eos: &Rc<E>,
        pressure: QuantityScalar<U>,
        x_init: (f64, f64),
        options: SolverOptions,
        bubble_dew_options: (SolverOptions, SolverOptions),
    ) -> EosResult<Self> {
        let p = pressure.to_reduced(U::reference_pressure())?;

        // calculate initial values using bubble point
        let x1 = arr1(&[x_init.0, 1.0 - x_init.0]);
        let x2 = arr1(&[x_init.1, 1.0 - x_init.1]);
        let vle1 =
            PhaseEquilibrium::bubble_point(eos, pressure, &x1, None, None, bubble_dew_options)?;
        let vle2 =
            PhaseEquilibrium::bubble_point(eos, pressure, &x2, None, None, bubble_dew_options)?;
        let mut l1 = vle1.liquid().clone();
        let mut l2 = vle2.liquid().clone();
        let t0 = (vle1.vapor().temperature + vle2.vapor().temperature) * 0.5;
        let nv0 = (&vle1.vapor().moles + &vle2.vapor().moles) * 0.5;
        let mut v = State::new_npt(eos, t0, pressure, &nv0, DensityInitialization::Vapor)?;

        for _ in 0..options.max_iter.unwrap_or(MAX_ITER_HETERO) {
            // calculate properties
            let dmu_drho_l1 = (l1.dmu_dni(Contributions::Total) * l1.volume)
                .to_reduced(U::reference_molar_energy() / U::reference_density())?;
            let dmu_drho_l2 = (l2.dmu_dni(Contributions::Total) * l2.volume)
                .to_reduced(U::reference_molar_energy() / U::reference_density())?;
            let dmu_drho_v = (v.dmu_dni(Contributions::Total) * v.volume)
                .to_reduced(U::reference_molar_energy() / U::reference_density())?;
            let dmu_dt_l1 = (l1.dmu_dt(Contributions::Total))
                .to_reduced(U::reference_molar_energy() / U::reference_temperature())?;
            let dmu_dt_l2 = (l2.dmu_dt(Contributions::Total))
                .to_reduced(U::reference_molar_energy() / U::reference_temperature())?;
            let dmu_dt_v = (v.dmu_dt(Contributions::Total))
                .to_reduced(U::reference_molar_energy() / U::reference_temperature())?;
            let dp_drho_l1 = (l1.dp_dni(Contributions::Total) * l1.volume)
                .to_reduced(U::reference_pressure() / U::reference_density())?;
            let dp_drho_l2 = (l2.dp_dni(Contributions::Total) * l2.volume)
                .to_reduced(U::reference_pressure() / U::reference_density())?;
            let dp_drho_v = (v.dp_dni(Contributions::Total) * v.volume)
                .to_reduced(U::reference_pressure() / U::reference_density())?;
            let dp_dt_l1 = (l1.dp_dt(Contributions::Total))
                .to_reduced(U::reference_pressure() / U::reference_temperature())?;
            let dp_dt_l2 = (l2.dp_dt(Contributions::Total))
                .to_reduced(U::reference_pressure() / U::reference_temperature())?;
            let dp_dt_v = (v.dp_dt(Contributions::Total))
                .to_reduced(U::reference_pressure() / U::reference_temperature())?;
            let mu_l1 = l1
                .chemical_potential(Contributions::Total)
                .to_reduced(U::reference_molar_energy())?;
            let mu_l2 = l2
                .chemical_potential(Contributions::Total)
                .to_reduced(U::reference_molar_energy())?;
            let mu_v = v
                .chemical_potential(Contributions::Total)
                .to_reduced(U::reference_molar_energy())?;
            let p_l1 = l1
                .pressure(Contributions::Total)
                .to_reduced(U::reference_pressure())?;
            let p_l2 = l2
                .pressure(Contributions::Total)
                .to_reduced(U::reference_pressure())?;
            let p_v = v
                .pressure(Contributions::Total)
                .to_reduced(U::reference_pressure())?;

            // calculate residual
            let res = concatenate![
                Axis(0),
                mu_l1 - &mu_v,
                mu_l2 - &mu_v,
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
                    (dmu_dt_l1 - &dmu_dt_v).insert_axis(Axis(1)),
                    (dmu_dt_l2 - &dmu_dt_v).insert_axis(Axis(1)),
                    arr2(&[[dp_dt_l1]]),
                    arr2(&[[dp_dt_l2]]),
                    arr2(&[[dp_dt_v]])
                ]
            ];

            // calculate Newton step
            let dx = LU::new(jacobian)?.solve(&res);

            // apply Newton step
            let rho_l1 =
                &l1.partial_density - &(dx.slice(s![0..2]).to_owned() * U::reference_density());
            let rho_l2 =
                &l2.partial_density - &(dx.slice(s![2..4]).to_owned() * U::reference_density());
            let rho_v =
                &v.partial_density - &(dx.slice(s![4..6]).to_owned() * U::reference_density());
            let t = v.temperature - dx[6] * U::reference_temperature();

            // check for negative densities and temperatures
            for i in 0..2 {
                if rho_l1.get(i).is_sign_negative()
                    || rho_l2.get(i).is_sign_negative()
                    || rho_v.get(i).is_sign_negative()
                    || t.is_sign_negative()
                {
                    return Err(EosError::IterationFailed(String::from(
                        "PhaseEquilibrium::heteroazeotrope_t",
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
            "PhaseEquilibrium::heteroazeotrope_t",
        )))
    }
}
