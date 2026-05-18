use crate::ad::Gradient;
use crate::ad::parameter_optimization::dataset::DatasetRecord;
use crate::ad::{ParametersAD, vectorize, vectorize_ad};
use crate::{Composition, Contributions, FeosResult, PhaseEquilibrium, ReferenceSystem, Residual};
use nalgebra::{SVector, U2};
use ndarray::{Array1, Array2, ArrayView2};
use quantity::{KELVIN, PASCAL, Pressure, Temperature};
use serde::{Deserialize, Serialize};

/// The pressure column doubles as the initial guess passed to the VLE solver.
#[derive(Deserialize, Serialize)]
pub struct DewPointRecord {
    pub temperature_k: f64,
    pub vapor_molefrac_1: f64,
    pub dew_pressure_pa: f64,
}

impl DatasetRecord for DewPointRecord {
    const N_INPUTS: usize = 3;

    fn input(&self, column: usize) -> f64 {
        match column {
            0 => self.temperature_k,
            1 => self.vapor_molefrac_1,
            2 => self.dew_pressure_pa,
            _ => unreachable!("invalid dew point input column"),
        }
    }

    fn target(&self) -> f64 {
        self.dew_pressure_pa
    }
}

pub fn dew_point_pressure_ad<
    E: Residual<U2, Gradient<P>>,
    const P: usize,
    X: Composition<f64, U2>,
>(
    eos: &E,
    temperature: Temperature,
    pressure: Option<Pressure>,
    vapor_molefracs: X,
) -> FeosResult<Pressure<Gradient<P>>> {
    let eos_f64 = eos.re();
    let (vapor_molefracs, _) = vapor_molefracs.into_molefracs(&eos_f64)?;
    let vle = PhaseEquilibrium::dew_point(
        &eos_f64,
        temperature,
        vapor_molefracs,
        pressure,
        None,
        Default::default(),
    )?;

    let v_l = 1.0 / vle.liquid().density.to_reduced();
    let v_v = 1.0 / vle.vapor().density.to_reduced();
    let x = &vle.liquid().molefracs;
    let x: SVector<_, 2> = SVector::from_fn(|i, _| x[i]);
    let t = temperature.into_reduced();
    let (a_l, a_v, v_l, v_v) = {
        let t = Gradient::from(t);
        let v_l = Gradient::from(v_l);
        let v_v = Gradient::from(v_v);
        let x = x.map(Gradient::from);
        let y = vapor_molefracs.map(Gradient::from);

        let a_l = eos.residual_helmholtz_energy(t, v_l, &x);
        let (p_v, mu_res_v, dp_v, dmu_v) = eos.dmu_dv(t, v_v, &y);
        let vi_v = dmu_v / dp_v;
        let v_v = vi_v.dot(&x);
        let a_v = (mu_res_v - vi_v * p_v).dot(&x);
        (a_l, a_v, v_l, v_v)
    };
    let rho_l = vle.liquid().partial_density().to_reduced();
    let rho_l = [rho_l[0], rho_l[1]];
    let rho_v = vle.vapor().partial_density().to_reduced();
    let rho_v = [rho_v[0], rho_v[1]];
    let p = -(a_l - a_v
        + t * (x[0] * (rho_l[0] / rho_v[0]).ln() + x[1] * (rho_l[1] / rho_v[1]).ln() - 1.0))
        / (v_l - v_v);
    Ok(Pressure::from_reduced(p))
}

pub fn dew_point_pressure<E: Residual>(
    eos: &E,
    temperature: Temperature,
    pressure_init: Option<Pressure>,
    vapor_molefrac_1: f64,
) -> FeosResult<Pressure> {
    let vle = PhaseEquilibrium::dew_point(
        eos,
        temperature,
        vapor_molefrac_1,
        pressure_init,
        None,
        Default::default(),
    )?;
    Ok(vle.vapor().pressure(Contributions::Total))
}

pub fn dew_point_pressure_parallel<E: Residual + Sync>(
    eos: &E,
    input: ArrayView2<f64>,
) -> (Array1<f64>, Array1<bool>) {
    vectorize(eos, input, |eos, inp| {
        dew_point_pressure(eos, inp[0] * KELVIN, Some(inp[2] * PASCAL), inp[1])
            .map(|p| p.convert_into(PASCAL))
    })
}

pub fn dew_point_pressure_parallel_ad<T: ParametersAD<2>, const P: usize>(
    parameter_names: [String; P],
    parameters: ArrayView2<f64>,
    input: ArrayView2<f64>,
) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
    vectorize_ad::<_, T, 2, P>(parameter_names, parameters, input, |eos, inp| {
        dew_point_pressure_ad(eos, inp[0] * KELVIN, Some(inp[2] * PASCAL), inp[1])
            .map(|p| p.convert_into(PASCAL))
    })
}
