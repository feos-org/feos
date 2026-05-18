use crate::ad::Gradient;
use crate::ad::parameter_optimization::dataset::DatasetRecord;
use crate::ad::{ParametersAD, vectorize, vectorize_ad};
use crate::{FeosResult, PhaseEquilibrium, ReferenceSystem, Residual};
use nalgebra::U1;
use ndarray::{Array1, Array2, ArrayView2};
use num_dual::{DualNum, DualStruct, first_derivative, partial2};
use quantity::{JOULE, KELVIN, MOL, MolarEnergy, Temperature};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct EnthalpyOfVaporizationRecord {
    pub temperature_k: f64,
    pub dh_vap_j_mol: f64,
}

impl DatasetRecord for EnthalpyOfVaporizationRecord {
    const N_INPUTS: usize = 1;

    fn input(&self, _column: usize) -> f64 {
        self.temperature_k
    }

    fn target(&self) -> f64 {
        self.dh_vap_j_mol
    }
}

pub fn enthalpy_of_vaporization_ad<E: Residual<U1, Gradient<P>>, const P: usize>(
    eos: &E,
    temperature: Temperature,
) -> FeosResult<MolarEnergy<Gradient<P>>> {
    let t = Temperature::from_inner(&temperature);
    let (_, [vapor_density, liquid_density]) =
        PhaseEquilibrium::pure_t(eos, t, None, Default::default())?;

    let v1 = liquid_density.into_reduced().recip();
    let v2 = vapor_density.into_reduced().recip();
    let x = E::pure_molefracs();
    let t = t.into_reduced();
    let residual_entropy = |v| {
        let (_a, s) = first_derivative(
            partial2(
                |t, &v, x| eos.lift().residual_helmholtz_energy(t, v, x),
                &v,
                &x,
            ),
            t,
        );
        -s
    };

    let s1 = residual_entropy(v1);
    let s2 = residual_entropy(v2);

    let dh = t * ((v2 / v1).ln() + s2 - s1);
    Ok(MolarEnergy::from_reduced(dh))
}

pub fn enthalpy_of_vaporization<E: Residual>(
    eos: &E,
    temperature: Temperature,
) -> FeosResult<MolarEnergy> {
    let vle = PhaseEquilibrium::pure(eos, temperature, None, Default::default())?;
    let h_v = vle.vapor().residual_molar_enthalpy();
    let h_l = vle.liquid().residual_molar_enthalpy();
    Ok(h_v - h_l)
}

pub fn enthalpy_of_vaporization_parallel<E: Residual + Sync>(
    eos: &E,
    input: ArrayView2<f64>,
) -> (Array1<f64>, Array1<bool>) {
    vectorize(eos, input, |eos, inp| {
        enthalpy_of_vaporization(eos, inp[0] * KELVIN).map(|dh| dh.convert_into(JOULE / MOL))
    })
}

pub fn enthalpy_of_vaporization_parallel_ad<T: ParametersAD<1>, const P: usize>(
    parameter_names: [String; P],
    parameters: ArrayView2<f64>,
    input: ArrayView2<f64>,
) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
    vectorize_ad::<_, T, 1, P>(parameter_names, parameters, input, |eos, inp| {
        enthalpy_of_vaporization_ad(eos, inp[0] * KELVIN).map(|dh| dh.convert_into(JOULE / MOL))
    })
}
