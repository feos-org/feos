use crate::ad::Gradient;
use crate::ad::parameter_optimization::dataset::DatasetRecord;
use crate::ad::{ParametersAD, vectorize, vectorize_ad};
use crate::{FeosResult, PhaseEquilibrium, Residual};
use nalgebra::U1;
use ndarray::{Array1, Array2, ArrayView2};
use num_dual::DualStruct;
use quantity::{Density, KELVIN, KILO, METER, MOL, Pressure, Temperature};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct EquilibriumLiquidDensityRecord {
    pub temperature_k: f64,
    pub liquid_density_kmol_m3: f64,
}

impl DatasetRecord for EquilibriumLiquidDensityRecord {
    const N_INPUTS: usize = 1;

    fn input(&self, _column: usize) -> f64 {
        self.temperature_k
    }

    fn target(&self) -> f64 {
        self.liquid_density_kmol_m3
    }
}

pub fn equilibrium_liquid_density_ad<E: Residual<U1, Gradient<P>>, const P: usize>(
    eos: &E,
    temperature: Temperature,
) -> FeosResult<(Pressure<Gradient<P>>, Density<Gradient<P>>)> {
    let t = Temperature::from_inner(&temperature);
    PhaseEquilibrium::pure_t(eos, t, None, Default::default()).map(|(p, [_, rho])| (p, rho))
}

pub fn equilibrium_liquid_density<E: Residual>(
    eos: &E,
    temperature: Temperature,
) -> FeosResult<Density> {
    let (_, [_, rho]) = PhaseEquilibrium::pure_t(eos, temperature, None, Default::default())?;
    Ok(rho)
}

pub fn equilibrium_liquid_density_parallel<E: Residual + Sync>(
    eos: &E,
    input: ArrayView2<f64>,
) -> (Array1<f64>, Array1<bool>) {
    vectorize(eos, input, |eos, inp| {
        equilibrium_liquid_density(eos, inp[0] * KELVIN)
            .map(|d| d.convert_into(KILO * MOL / (METER * METER * METER)))
    })
}

pub fn equilibrium_liquid_density_parallel_ad<T: ParametersAD<1>, const P: usize>(
    parameter_names: [String; P],
    parameters: ArrayView2<f64>,
    input: ArrayView2<f64>,
) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
    vectorize_ad::<_, T, 1, P>(parameter_names, parameters, input, |eos, inp| {
        equilibrium_liquid_density_ad(eos, inp[0] * KELVIN)
            .map(|(_, d)| d.convert_into(KILO * MOL / (METER * METER * METER)))
    })
}
