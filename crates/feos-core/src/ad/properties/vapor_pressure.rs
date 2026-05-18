use crate::ad::Gradient;
use crate::ad::{ParametersAD, vectorize, vectorize_ad};
use crate::{FeosResult, PhaseEquilibrium, ReferenceSystem, Residual};
use nalgebra::U1;
use ndarray::{Array1, Array2, ArrayView2};
use quantity::{KELVIN, PASCAL, Pressure, Temperature};

pub fn vapor_pressure_ad<E: Residual<U1, Gradient<P>>, const P: usize>(
    eos: &E,
    temperature: Temperature,
) -> FeosResult<Pressure<Gradient<P>>> {
    let eos_f64 = eos.re();
    let (_, [vapor_density, liquid_density]) =
        PhaseEquilibrium::pure_t(&eos_f64, temperature, None, Default::default())?;

    // implicit differentiation is implemented here instead of just calling pure_t with dual
    // numbers, because for the first derivative, we can avoid calculating density derivatives.
    let v1 = 1.0 / liquid_density.to_reduced();
    let v2 = 1.0 / vapor_density.to_reduced();
    let t = temperature.into_reduced();
    let (a1, a2) = {
        let t = Gradient::from(t);
        let v1 = Gradient::from(v1);
        let v2 = Gradient::from(v2);
        let x = E::pure_molefracs();

        let a1 = eos.residual_helmholtz_energy(t, v1, &x);
        let a2 = eos.residual_helmholtz_energy(t, v2, &x);
        (a1, a2)
    };

    let p = -(a1 - a2 + t * (v2 / v1).ln()) / (v1 - v2);
    Ok(Pressure::from_reduced(p))
}

/// Non-AD vapor pressure for a single-component model.
pub fn vapor_pressure<E: Residual>(eos: &E, temperature: Temperature) -> FeosResult<Pressure> {
    let (p, _) = PhaseEquilibrium::pure_t(eos, temperature, None, Default::default())?;
    Ok(p)
}

/// Non-AD batched evaluation over input rows. Single shared model.
pub fn vapor_pressure_parallel<E: Residual + Sync>(
    eos: &E,
    input: ArrayView2<f64>,
) -> (Array1<f64>, Array1<bool>) {
    vectorize(eos, input, |eos, inp| {
        vapor_pressure(eos, inp[0] * KELVIN).map(|p| p.convert_into(PASCAL))
    })
}

pub fn vapor_pressure_parallel_ad<T: ParametersAD<1>, const P: usize>(
    parameter_names: [String; P],
    parameters: ArrayView2<f64>,
    input: ArrayView2<f64>,
) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
    vectorize_ad::<_, T, 1, P>(parameter_names, parameters, input, |eos, inp| {
        vapor_pressure_ad(eos, inp[0] * KELVIN).map(|p| p.convert_into(PASCAL))
    })
}
