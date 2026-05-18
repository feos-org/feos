use crate::ad::Gradient;
use crate::ad::{ParametersAD, vectorize, vectorize_ad};
use crate::{FeosResult, PhaseEquilibrium, ReferenceSystem, Residual};
use nalgebra::U1;
use ndarray::{Array1, Array2, ArrayView2};
use num_dual::{DualNum, first_derivative, partial2};
use quantity::{KELVIN, PASCAL, Pressure, Temperature};

pub fn boiling_temperature_ad<E: Residual<U1, Gradient<P>>, const P: usize>(
    eos: &E,
    pressure: Pressure,
) -> FeosResult<Temperature<Gradient<P>>> {
    let eos_f64 = eos.re();
    let (temperature, [vapor_density, liquid_density]) =
        PhaseEquilibrium::pure_p(&eos_f64, pressure, None, Default::default())?;

    let t = temperature.into_reduced();
    let v1 = 1.0 / liquid_density.to_reduced();
    let v2 = 1.0 / vapor_density.to_reduced();
    let p = pressure.into_reduced();
    let t = Gradient::from(t);
    let t = t + {
        let v1 = Gradient::from(v1);
        let v2 = Gradient::from(v2);
        let p = Gradient::from(p);
        let x = E::pure_molefracs();

        let residual_entropy = |v| {
            let (a, s) = first_derivative(
                partial2(
                    |t, &v, x| eos.lift().residual_helmholtz_energy(t, v, x),
                    &v,
                    &x,
                ),
                t,
            );
            (a, -s)
        };
        let (a1, s1) = residual_entropy(v1);
        let (a2, s2) = residual_entropy(v2);

        let ln_rho = (v1 / v2).ln();
        (p * (v2 - v1) + (a2 - a1 + t * ln_rho)) / (s2 - s1 - ln_rho)
    };
    Ok(Temperature::from_reduced(t))
}

pub fn boiling_temperature<E: Residual>(eos: &E, pressure: Pressure) -> FeosResult<Temperature> {
    let (t, _) = PhaseEquilibrium::pure_p(eos, pressure, None, Default::default())?;
    Ok(t)
}

pub fn boiling_temperature_parallel<E: Residual + Sync>(
    eos: &E,
    input: ArrayView2<f64>,
) -> (Array1<f64>, Array1<bool>) {
    vectorize(eos, input, |eos, inp| {
        boiling_temperature(eos, inp[0] * PASCAL).map(|t| t.convert_into(KELVIN))
    })
}

pub fn boiling_temperature_parallel_ad<T: ParametersAD<1>, const P: usize>(
    parameter_names: [String; P],
    parameters: ArrayView2<f64>,
    input: ArrayView2<f64>,
) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
    vectorize_ad::<_, T, 1, P>(parameter_names, parameters, input, |eos, inp| {
        boiling_temperature_ad(eos, inp[0] * PASCAL).map(|t| t.convert_into(KELVIN))
    })
}
