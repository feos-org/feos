use crate::DensityInitialization::Liquid;
use crate::ad::Gradient;
use crate::ad::dataset::DatasetRecord;
use crate::ad::{ParametersAD, vectorize, vectorize_ad};
use crate::density_iteration::density_iteration;
use crate::{FeosResult, ReferenceSystem, Residual, State};
use nalgebra::U1;
use ndarray::{Array1, Array2, ArrayView2};
use num_dual::DualStruct;
use quantity::{Density, KELVIN, KILO, METER, MOL, Moles, PASCAL, Pressure, Temperature};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct LiquidDensityRecord {
    pub temperature_k: f64,
    pub pressure_pa: f64,
    pub liquid_density_kmol_m3: f64,
}

impl DatasetRecord for LiquidDensityRecord {
    const N_INPUTS: usize = 2;

    fn input(&self, column: usize) -> f64 {
        match column {
            0 => self.temperature_k,
            1 => self.pressure_pa,
            _ => unreachable!("invalid liquid density input column"),
        }
    }

    fn target(&self) -> f64 {
        self.liquid_density_kmol_m3
    }
}

pub fn liquid_density_ad<E: Residual<U1, Gradient<P>>, const P: usize>(
    eos: &E,
    temperature: Temperature,
    pressure: Pressure,
) -> FeosResult<Density<Gradient<P>>> {
    let x = E::pure_molefracs();
    let t = Temperature::from_inner(&temperature);
    let p = Pressure::from_inner(&pressure);
    density_iteration(eos, t, p, &x, Some(Liquid))
}

pub fn liquid_density<E: Residual>(
    eos: &E,
    temperature: Temperature,
    pressure: Pressure,
) -> FeosResult<Density> {
    let state = State::new_npt(
        eos,
        temperature,
        pressure,
        &Moles::from_reduced(nalgebra::DVector::from_element(eos.components(), 1.0)),
        Some(Liquid),
    )?;
    Ok(state.density)
}

pub fn liquid_density_parallel<E: Residual + Sync>(
    eos: &E,
    input: ArrayView2<f64>,
) -> (Array1<f64>, Array1<bool>) {
    vectorize(eos, input, |eos, inp| {
        liquid_density(eos, inp[0] * KELVIN, inp[1] * PASCAL)
            .map(|d| d.convert_into(KILO * MOL / (METER * METER * METER)))
    })
}

pub fn liquid_density_parallel_ad<T: ParametersAD<1>, const P: usize>(
    parameter_names: [String; P],
    parameters: ArrayView2<f64>,
    input: ArrayView2<f64>,
) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
    vectorize_ad::<_, T, 1, P>(parameter_names, parameters, input, |eos, inp| {
        liquid_density_ad(eos, inp[0] * KELVIN, inp[1] * PASCAL)
            .map(|d| d.convert_into(KILO * MOL / (METER * METER * METER)))
    })
}
