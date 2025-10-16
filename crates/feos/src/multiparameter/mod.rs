use std::f64::consts::E;

use feos_core::parameter::Parameters;
use feos_core::{EquationOfState, IdealGas, Molarweight, ResidualDyn, StateHD, Subset};
use nalgebra::DVector;
use num_dual::DualNum;
use quantity::MolarWeight;
use serde::Deserialize;

mod ideal_gas_function;
mod residual_function;
use ideal_gas_function::{IdealGasFunction, IdealGasFunctionJson};
use residual_function::{ResidualFunction, ResidualFunctionJson};

// record

#[derive(Clone, Deserialize)]
pub struct MultiParameterRecord {
    tc: f64,
    rhoc: f64,
    residual: Vec<ResidualFunctionJson>,
    ideal_gas: Vec<IdealGasFunctionJson>,
}

pub type MultiParameterParameters = Parameters<MultiParameterRecord, (), ()>;

// structs

#[derive(Clone)]
pub struct MultiParameter {
    tc: f64,
    rhoc: f64,
    terms: Vec<ResidualFunction>,
    molar_weight: MolarWeight<DVector<f64>>,
}

#[derive(Clone)]
pub struct MultiParameterIdealGas {
    tc: f64,
    rhoc: f64,
    terms: Vec<IdealGasFunction>,
}

pub type MultiParameterEquationOfState =
    EquationOfState<Vec<MultiParameterIdealGas>, MultiParameter>;

impl MultiParameter {
    pub fn new(mut parameters: MultiParameterParameters) -> MultiParameterEquationOfState {
        if parameters.pure.len() != 1 {
            panic!("Multiparameter equations of state are only implemented for pure components!");
        }
        let record = parameters.pure.pop().unwrap().model_record;
        let terms = record.residual.into_iter().flatten().collect();

        let residual = Self {
            tc: record.tc,
            rhoc: record.rhoc,
            terms,
            molar_weight: parameters.molar_weight,
        };

        let terms = record.ideal_gas.into_iter().flatten().collect();
        let ideal_gas = MultiParameterIdealGas {
            tc: record.tc,
            rhoc: record.rhoc,
            terms,
        };

        EquationOfState::new(vec![ideal_gas], residual)
    }
}

// eos trait implementations

impl ResidualDyn for MultiParameter {
    fn components(&self) -> usize {
        1
    }

    fn compute_max_density<D: DualNum<f64> + Copy>(&self, _: &DVector<D>) -> D {
        // Not sure what value works well here. This one is based on rho_c = 0.31*rho_max.
        D::from(6.02214076e-7 * self.rhoc / 0.31)
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        let rho = state.partial_density.sum();
        let delta = rho / (6.02214076e-7 * self.rhoc);
        let tau = state.temperature.recip() * self.tc;
        vec![(
            "Multiparameter",
            self.terms
                .iter()
                .map(|r| r.evaluate(delta, tau) * rho)
                .sum(),
        )]
    }
}

impl Molarweight for MultiParameter {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.molar_weight.clone()
    }
}

impl Subset for MultiParameter {
    fn subset(&self, _: &[usize]) -> Self {
        self.clone()
    }
}

impl IdealGas for MultiParameterIdealGas {
    fn ln_lambda3<D2: DualNum<f64, Inner = f64> + Copy>(&self, temperature: D2) -> D2 {
        let tau = temperature.recip() * self.tc;
        // bit of a hack to convert from phi^0 into ln Lambda^3
        let delta = D2::from(E / (6.02214076e-7 * self.rhoc));
        self.terms.iter().map(|r| r.evaluate(delta, tau)).sum()
    }

    fn ideal_gas_model(&self) -> &'static str {
        "Ideal Gas (Multiparameter)"
    }
}

#[cfg(test)]
mod test {
    use approx::{assert_relative_eq, assert_relative_ne};
    use feos_core::parameter::IdentifierOption;
    use feos_core::{SolverOptions, State, Total};
    use nalgebra::{Dyn, SVector, U2, dvector};
    use num_dual::{Dual2Vec, hessian};
    use quantity::{GRAM, KELVIN, KILO, KILOGRAM, METER, MOL, RGAS};
    use typenum::P3;

    use super::*;

    fn water() -> MultiParameterEquationOfState {
        let parameters = Parameters::from_json(
            vec!["Water"],
            "../../parameters/multiparameter/coolprop.json",
            None,
            IdentifierOption::Name,
        )
        .unwrap();
        MultiParameter::new(parameters)
    }

    #[test]
    fn test_phi_r_1() {
        let t = 500.;
        let rho = 838.025;
        let eos = water();
        let tau = eos.tc / t;
        let delta = rho / (eos.rhoc * eos.molar_weight.get(0).convert_into(KILO * GRAM / MOL));
        let (phi, dphi, d2phi) = hessian(
            |x| {
                let [delta, tau] = x.data.0[0];
                eos.terms
                    .iter()
                    .map(|f| f.evaluate(delta, tau))
                    .sum::<Dual2Vec<f64, f64, U2>>()
            },
            &SVector::from([delta, tau]),
        );
        println!("{}\n{}\n{}", phi, dphi, d2phi);
        assert_eq!(format!("{phi:.8}"), "-3.42693206");
        assert_eq!(format!("{:.8}", dphi[0]), "-0.36436665");
        assert_eq!(format!("{:.8}", dphi[1]), "-5.81403435");
        assert_eq!(format!("{:.8}", d2phi[(0, 0)]), "0.85606370");
        assert_eq!(format!("{:.8}", d2phi[(0, 1)]), "-1.12176915");
        assert_eq!(format!("{:.8}", d2phi[(1, 1)]), "-2.23440737");
    }

    #[test]
    fn test_phi_r_2() {
        let t = 647.;
        let rho = 358.;
        let eos = water();
        let tau = eos.tc / t;
        let delta = rho / (eos.rhoc * eos.molar_weight.get(0).convert_into(KILO * GRAM / MOL));
        let (phi, dphi, d2phi) = hessian(
            |x| {
                let [delta, tau] = x.data.0[0];
                eos.terms
                    .iter()
                    .map(|f| f.evaluate(delta, tau))
                    .sum::<Dual2Vec<f64, f64, U2>>()
            },
            &SVector::from([delta, tau]),
        );
        println!("{}\n{}\n{}", phi, dphi, d2phi);
        assert_eq!(format!("{phi:.8}"), "-1.21202657");
        assert_eq!(format!("{:.8}", dphi[0]), "-0.71401202");
        assert_eq!(format!("{:.8}", dphi[1]), "-3.21722501");
        assert_eq!(format!("{:.8}", d2phi[(0, 0)]), "0.47573070");
        assert_eq!(format!("{:.8}", d2phi[(0, 1)]), "-1.33214720");
        assert_eq!(format!("{:.8}", d2phi[(1, 1)]), "-9.96029507");
    }

    #[test]
    fn test_phi_o_1() {
        let t = 500.;
        let rho = 838.025;
        let eos = water();
        let tau = eos.tc / t;
        let delta = rho / (eos.rhoc * eos.molar_weight.get(0).convert_into(KILO * GRAM / MOL));
        let (phi, dphi, d2phi) = hessian(
            |x| {
                let [delta, tau] = x.data.0[0];
                eos.ideal_gas[0]
                    .terms
                    .iter()
                    .map(|r| r.evaluate(delta, tau))
                    .sum::<Dual2Vec<f64, f64, U2>>()
            },
            &SVector::from([delta, tau]),
        );
        println!("{}\n{}\n{}", phi, dphi, d2phi);
        assert_eq!(format!("{phi:.7}"), "2.0479773");
        assert_eq!(format!("{:.8}", dphi[0]), "0.38423675");
        assert_eq!(format!("{:.8}", dphi[1]), "9.04611106");
        assert_eq!(format!("{:.8}", d2phi[(0, 0)]), "-0.14763788");
        assert_eq!(format!("{:.8}", d2phi[(0, 1)]), "-0.00000000");
        assert_eq!(format!("{:.8}", d2phi[(1, 1)]), "-1.93249185");
    }

    #[test]
    fn test_phi_o_2() {
        let t = 647.;
        let rho = 358.;
        let eos = water();
        let tau = eos.tc / t;
        let delta = rho / (eos.rhoc * eos.molar_weight.get(0).convert_into(KILO * GRAM / MOL));
        let (phi, dphi, d2phi) = hessian(
            |x| {
                let [delta, tau] = x.data.0[0];
                eos.ideal_gas[0]
                    .terms
                    .iter()
                    .map(|r| r.evaluate(delta, tau))
                    .sum::<Dual2Vec<f64, f64, U2>>()
            },
            &SVector::from([delta, tau]),
        );
        println!("{}\n{}\n{}", phi, dphi, d2phi);
        assert_eq!(format!("{phi:.8}"), "-1.56319605");
        assert_eq!(format!("{:.8}", dphi[0]), "0.89944134");
        assert_eq!(format!("{:.8}", dphi[1]), "9.80343918");
        assert_eq!(format!("{:.8}", d2phi[(0, 0)]), "-0.80899473");
        assert_eq!(format!("{:.8}", d2phi[(0, 1)]), "-0.00000000");
        assert_eq!(format!("{:.8}", d2phi[(1, 1)]), "-3.43316334");
    }

    #[test]
    fn test_ideal_gas_hack() {
        let t = 647. * KELVIN;
        let rho = 358. * KILOGRAM / METER.powi::<P3>();
        let eos = &water();
        let mw = eos.molar_weight.get(0);
        let moles = dvector![1.8] * MOL;
        let a_feos = eos.ideal_gas_helmholtz_energy(t, moles.sum() * mw / rho, &moles);
        let phi_feos = (a_feos / RGAS / moles.sum() / t).into_value();
        println!("A:          {a_feos}");
        println!("phi(feos):  {phi_feos}");
        let delta = (rho / (eos.rhoc * MOL / METER.powi::<P3>() * mw)).into_value();
        let tau = (eos.tc * KELVIN / t).into_value();
        let phi = eos.ideal_gas[0]
            .terms
            .iter()
            .map(|r| r.evaluate(delta, tau))
            .sum::<f64>();
        println!("phi(IAPWS): {phi}");
        assert_relative_eq!(phi_feos, phi, max_relative = 1e-15)
    }

    #[test]
    fn test_critical_point() {
        let eos = &water();
        let options = SolverOptions {
            verbosity: feos_core::Verbosity::Iter,
            ..Default::default()
        };
        let cp: State<_, Dyn, f64> =
            State::critical_point(&eos, None, Some(647. * KELVIN), None, options).unwrap();
        println!("{cp}");
        assert_relative_eq!(cp.temperature, eos.tc * KELVIN, max_relative = 1e-13);
        let cp: State<_, Dyn, f64> =
            State::critical_point(&eos, None, None, None, Default::default()).unwrap();
        println!("{cp}");
        assert_relative_ne!(cp.temperature, eos.tc * KELVIN, max_relative = 1e-13);
        let cp: State<_, Dyn, f64> =
            State::critical_point(&eos, None, Some(700.0 * KELVIN), None, Default::default())
                .unwrap();
        println!("{cp}");
        assert_relative_eq!(cp.temperature, eos.tc * KELVIN, max_relative = 1e-13)
    }
}
