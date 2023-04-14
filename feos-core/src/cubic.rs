//! Implementation of the Peng-Robinson equation of state.
//!
//! This module acts as a reference on how a simple equation
//! of state - with a single contribution to the Helmholtz energy - can be implemented.
//! The implementation closely follows the form of the equations given in
//! [this wikipedia article](https://en.wikipedia.org/wiki/Cubic_equations_of_state#Peng%E2%80%93Robinson_equation_of_state).
use crate::equation_of_state::{
    EquationOfState, HelmholtzEnergy, HelmholtzEnergyDual, IdealGasContribution,
};
use crate::joback::{Joback, JobackRecord};
use crate::parameter::{Identifier, Parameter, ParameterError, PureRecord};
use crate::si::{GRAM, MOL};
use crate::state::StateHD;
use crate::MolarWeight;
use ndarray::{Array1, Array2};
use num_dual::DualNum;
use quantity::si::SIArray1;
use serde::{Deserialize, Serialize};
use std::f64::consts::SQRT_2;
use std::fmt;
use std::sync::Arc;

const KB_A3: f64 = 13806490.0;

/// Peng-Robinson parameters for a single substance.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct PengRobinsonRecord {
    /// critical temperature in Kelvin
    tc: f64,
    /// critical pressure in Pascal
    pc: f64,
    /// acentric factor
    acentric_factor: f64,
}

impl PengRobinsonRecord {
    /// Create a new pure substance record for the Peng-Robinson equation of state.
    pub fn new(tc: f64, pc: f64, acentric_factor: f64) -> Self {
        Self {
            tc,
            pc,
            acentric_factor,
        }
    }
}

impl std::fmt::Display for PengRobinsonRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PengRobinsonRecord(tc={} K", self.tc)?;
        write!(f, ", pc={} Pa", self.pc)?;
        write!(f, ", acentric factor={}", self.acentric_factor)
    }
}

/// Peng-Robinson parameters for one ore more substances.
pub struct PengRobinsonParameters {
    /// Critical temperature in Kelvin
    tc: Array1<f64>,
    a: Array1<f64>,
    b: Array1<f64>,
    /// Binary interaction parameter
    k_ij: Array2<f64>,
    kappa: Array1<f64>,
    /// Molar weight in units of g/mol
    molarweight: Array1<f64>,
    /// List of pure component records
    pure_records: Vec<PureRecord<PengRobinsonRecord, JobackRecord>>,
    /// List of ideal gas Joback records
    joback_records: Option<Vec<JobackRecord>>,
}

impl std::fmt::Display for PengRobinsonParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.pure_records
            .iter()
            .try_for_each(|pr| writeln!(f, "{}", pr))?;
        writeln!(f, "\nk_ij:\n{}", self.k_ij)
    }
}

impl PengRobinsonParameters {
    /// Build a simple parameter set without binary interaction parameters.
    pub fn new_simple(
        tc: &[f64],
        pc: &[f64],
        acentric_factor: &[f64],
        molarweight: &[f64],
    ) -> Result<Self, crate::parameter::ParameterError> {
        if [pc.len(), acentric_factor.len(), molarweight.len()]
            .iter()
            .any(|&l| l != tc.len())
        {
            return Err(ParameterError::IncompatibleParameters(String::from(
                "each component has to have parameters.",
            )));
        }
        let records = (0..tc.len())
            .map(|i| {
                let record = PengRobinsonRecord {
                    tc: tc[i],
                    pc: pc[i],
                    acentric_factor: acentric_factor[i],
                };
                let id = Identifier::default();
                PureRecord::new(id, molarweight[i], record, None)
            })
            .collect();
        Ok(PengRobinsonParameters::from_records(
            records,
            Array2::zeros([pc.len(); 2]),
        ))
    }
}

impl Parameter for PengRobinsonParameters {
    type Pure = PengRobinsonRecord;
    type IdealGas = JobackRecord;
    type Binary = f64;

    /// Creates parameters from pure component records.
    fn from_records(
        pure_records: Vec<PureRecord<Self::Pure, Self::IdealGas>>,
        binary_records: Array2<Self::Binary>,
    ) -> Self {
        let n = pure_records.len();

        let mut tc = Array1::zeros(n);
        let mut a = Array1::zeros(n);
        let mut b = Array1::zeros(n);
        let mut molarweight = Array1::zeros(n);
        let mut kappa = Array1::zeros(n);

        for (i, record) in pure_records.iter().enumerate() {
            molarweight[i] = record.molarweight;
            let r = &record.model_record;
            tc[i] = r.tc;
            a[i] = 0.45724 * r.tc.powi(2) * KB_A3 / r.pc;
            b[i] = 0.07780 * r.tc * KB_A3 / r.pc;
            kappa[i] = 0.37464 + (1.54226 - 0.26992 * r.acentric_factor) * r.acentric_factor;
        }

        let joback_records = pure_records
            .iter()
            .map(|r| r.ideal_gas_record.clone())
            .collect();

        Self {
            tc,
            a,
            b,
            k_ij: binary_records,
            kappa,
            molarweight,
            pure_records,
            joback_records,
        }
    }

    fn records(
        &self,
    ) -> (
        &[PureRecord<PengRobinsonRecord, JobackRecord>],
        &Array2<f64>,
    ) {
        (&self.pure_records, &self.k_ij)
    }
}

struct PengRobinsonContribution {
    parameters: Arc<PengRobinsonParameters>,
}

impl<D: DualNum<f64>> HelmholtzEnergyDual<D> for PengRobinsonContribution {
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        // temperature dependent a parameter
        let p = &self.parameters;
        let x = &state.molefracs;
        let ak = (&p.tc.mapv(|tc| (D::one() - (state.temperature / tc).sqrt())) * &p.kappa + 1.0)
            .mapv(|x| x.powi(2))
            * &p.a;

        // Mixing rules
        let mut ak_mix = D::zero();
        for i in 0..ak.len() {
            for j in 0..ak.len() {
                ak_mix += (ak[i] * ak[j]).sqrt() * (x[i] * x[j] * (1.0 - p.k_ij[(i, j)]));
            }
        }
        let b = (x * &p.b).sum();

        // Helmholtz energy
        let n = state.moles.sum();
        let v = state.volume;
        n * ((v / (v - b * n)).ln()
            - ak_mix / (b * SQRT_2 * 2.0 * state.temperature)
                * ((v + b * n * (1.0 + SQRT_2)) / (v + b * n * (1.0 - SQRT_2))).ln())
    }
}

impl fmt::Display for PengRobinsonContribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Peng Robinson")
    }
}

/// A simple version of the Peng-Robinson equation of state.
pub struct PengRobinson {
    /// Parameters
    parameters: Arc<PengRobinsonParameters>,
    /// Ideal gas contributions to the Helmholtz energy
    ideal_gas: Joback,
    /// Non-ideal contributions to the Helmholtz energy
    contributions: Vec<Box<dyn HelmholtzEnergy>>,
}

impl PengRobinson {
    /// Create a new equation of state from a set of parameters.
    pub fn new(parameters: Arc<PengRobinsonParameters>) -> Self {
        let ideal_gas = parameters.joback_records.as_ref().map_or_else(
            || Joback::default(parameters.tc.len()),
            |j| Joback::new(j.clone()),
        );
        let contributions: Vec<Box<dyn HelmholtzEnergy>> =
            vec![Box::new(PengRobinsonContribution {
                parameters: parameters.clone(),
            })];
        Self {
            parameters,
            ideal_gas,
            contributions,
        }
    }
}

impl EquationOfState for PengRobinson {
    fn components(&self) -> usize {
        self.parameters.b.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::new(Arc::new(self.parameters.subset(component_list)))
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        let b = (moles * &self.parameters.b).sum() / moles.sum();
        0.9 / b
    }

    fn residual(&self) -> &[Box<dyn HelmholtzEnergy>] {
        &self.contributions
    }

    fn ideal_gas(&self) -> &dyn IdealGasContribution {
        &self.ideal_gas
    }
}

impl MolarWeight for PengRobinson {
    fn molar_weight(&self) -> SIArray1 {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase_equilibria::SolverOptions;
    use crate::state::State;
    use crate::Contributions;
    use crate::{EosResult, Verbosity};
    use approx::*;
    use quantity::si::*;
    use std::sync::Arc;

    fn pure_record_vec() -> Vec<PureRecord<PengRobinsonRecord, JobackRecord>> {
        let records = r#"[
            {
                "identifier": {
                    "cas": "74-98-6",
                    "name": "propane",
                    "iupac_name": "propane",
                    "smiles": "CCC",
                    "inchi": "InChI=1/C3H8/c1-3-2/h3H2,1-2H3",
                    "formula": "C3H8"
                },
                "model_record": {
                    "tc": 369.96,
                    "pc": 4250000.0,
                    "acentric_factor": 0.153
                },
                "molarweight": 44.0962
            },
            {
                "identifier": {
                    "cas": "106-97-8",
                    "name": "butane",
                    "iupac_name": "butane",
                    "smiles": "CCCC",
                    "inchi": "InChI=1/C4H10/c1-3-4-2/h3-4H2,1-2H3",
                    "formula": "C4H10"
                },
                "model_record": {
                    "tc": 425.2,
                    "pc": 3800000.0,
                    "acentric_factor": 0.199
                },
                "molarweight": 58.123
            }
        ]"#;
        serde_json::from_str(records).expect("Unable to parse json.")
    }

    #[test]
    fn peng_robinson() -> EosResult<()> {
        let mixture = pure_record_vec();
        let propane = mixture[0].clone();
        let tc = propane.model_record.tc;
        let pc = propane.model_record.pc;
        let parameters = PengRobinsonParameters::from_records(vec![propane], Array2::zeros((1, 1)));
        let pr = Arc::new(PengRobinson::new(Arc::new(parameters)));
        let options = SolverOptions::new().verbosity(Verbosity::Iter);
        let cp = State::critical_point(&pr, None, None, options)?;
        println!("{} {}", cp.temperature, cp.pressure(Contributions::Total));
        assert_relative_eq!(cp.temperature, tc * KELVIN, max_relative = 1e-4);
        assert_relative_eq!(
            cp.pressure(Contributions::Total),
            pc * PASCAL,
            max_relative = 1e-4
        );
        Ok(())
    }
}
