use crate::DensityInitialization::Liquid;
use crate::ad::Gradient;
use crate::ad::{ParametersAD, vectorize, vectorize_ad};
use crate::density_iteration::density_iteration;
use crate::{FeosResult, ReferenceSystem, Residual, State};
use nalgebra::U1;
use ndarray::{Array1, Array2, ArrayView2};
use num_dual::DualStruct;
use quantity::{JOULE, KELVIN, MOL, MolarEntropy, Moles, PASCAL, Pressure, Temperature};

/// Residual isobaric molar heat capacity of the liquid phase at the given
/// temperature and pressure.
pub fn residual_isobaric_heat_capacity_ad<E: Residual<U1, Gradient<P>>, const P: usize>(
    eos: &E,
    temperature: Temperature,
    pressure: Pressure,
) -> FeosResult<MolarEntropy<Gradient<P>>> {
    let x = E::pure_molefracs();
    let t = Temperature::from_inner(&temperature);
    let p = Pressure::from_inner(&pressure);
    let density = density_iteration(eos, t, p, &x, Some(Liquid))?;
    let state = State::new_pure(eos, t, density)?;
    Ok(state.residual_molar_isobaric_heat_capacity())
}

pub fn residual_isobaric_heat_capacity<E: Residual>(
    eos: &E,
    temperature: Temperature,
    pressure: Pressure,
) -> FeosResult<MolarEntropy> {
    let state = State::new_npt(
        eos,
        temperature,
        pressure,
        &Moles::from_reduced(nalgebra::DVector::from_element(eos.components(), 1.0)),
        Some(Liquid),
    )?;
    Ok(state.residual_molar_isobaric_heat_capacity())
}

pub fn residual_isobaric_heat_capacity_parallel<E: Residual + Sync>(
    eos: &E,
    input: ArrayView2<f64>,
) -> (Array1<f64>, Array1<bool>) {
    vectorize(eos, input, |eos, inp| {
        residual_isobaric_heat_capacity(eos, inp[0] * KELVIN, inp[1] * PASCAL)
            .map(|cp| cp.convert_into(JOULE / (MOL * KELVIN)))
    })
}

pub fn residual_isobaric_heat_capacity_parallel_ad<T: ParametersAD<1>, const P: usize>(
    parameter_names: [String; P],
    parameters: ArrayView2<f64>,
    input: ArrayView2<f64>,
) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
    vectorize_ad::<_, T, 1, P>(parameter_names, parameters, input, |eos, inp| {
        residual_isobaric_heat_capacity_ad(eos, inp[0] * KELVIN, inp[1] * PASCAL)
            .map(|cp| cp.convert_into(JOULE / (MOL * KELVIN)))
    })
}
