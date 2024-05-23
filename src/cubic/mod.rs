use feos_core::parameter::{Identifier, Parameter, ParameterError, PureRecord};
use feos_core::si::{MolarWeight, GRAM, MOL};
use feos_core::{Components, Residual, StateHD};
use ndarray::{Array1, Array2, ScalarOperand};
use num_dual::DualNum;
use serde::{Deserialize, Serialize};
use std::f64::consts::SQRT_2;
use std::fmt;
use std::sync::Arc;

mod alpha;
mod mixing_rules;

const KB_A3: f64 = 13806490.0;

/// Cubic parameters for a single substance.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct CubicRecord {
    /// critical temperature in Kelvin
    tc: f64,
    /// critical pressure in Pascal
    pc: f64,
    /// acentric factor
    acentric_factor: f64,
}

impl CubicRecord {
    /// Create a new pure substance record for the Cubic equation of state.
    pub fn new(tc: f64, pc: f64, acentric_factor: f64) -> Self {
        Self {
            tc,
            pc,
            acentric_factor,
        }
    }
}

impl std::fmt::Display for CubicRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CubicRecord(tc={} K", self.tc)?;
        write!(f, ", pc={} Pa", self.pc)?;
        write!(f, ", acentric factor={}", self.acentric_factor)
    }
}

/// Cubic parameters for one ore more substances.
pub struct CubicParameters {
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
    pure_records: Vec<PureRecord<CubicRecord>>,
}

impl std::fmt::Display for CubicParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.pure_records
            .iter()
            .try_for_each(|pr| writeln!(f, "{}", pr))?;
        writeln!(f, "\nk_ij:\n{}", self.k_ij)
    }
}

impl CubicParameters {
    /// Build a simple parameter set without binary interaction parameters.
    pub fn new_simple(
        tc: &[f64],
        pc: &[f64],
        acentric_factor: &[f64],
        molarweight: &[f64],
    ) -> Result<Self, ParameterError> {
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
                let record = CubicRecord {
                    tc: tc[i],
                    pc: pc[i],
                    acentric_factor: acentric_factor[i],
                };
                let id = Identifier::default();
                PureRecord::new(id, molarweight[i], record)
            })
            .collect();
        CubicParameters::from_records(records, None)
    }
}

impl Parameter for CubicParameters {
    type Pure = CubicRecord;
    type Binary = f64;

    /// Creates parameters from pure component records.
    fn from_records(
        pure_records: Vec<PureRecord<Self::Pure>>,
        binary_records: Option<Array2<Self::Binary>>,
    ) -> Result<Self, ParameterError> {
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

        let k_ij = binary_records.unwrap_or_else(|| Array2::zeros([n; 2]));

        Ok(Self {
            tc,
            a,
            b,
            k_ij,
            kappa,
            molarweight,
            pure_records,
        })
    }

    fn records(&self) -> (&[PureRecord<CubicRecord>], Option<&Array2<f64>>) {
        (&self.pure_records, Some(&self.k_ij))
    }
}

/// A simple version of the Cubic equation of state.
pub struct Cubic {
    /// Parameters
    parameters: Arc<CubicParameters>,
}

impl Cubic {
    /// Create a new equation of state from a set of parameters.
    pub fn new(parameters: Arc<CubicParameters>) -> Self {
        Self { parameters }
    }
}

impl fmt::Display for Cubic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Peng Robinson")
    }
}

impl Components for Cubic {
    fn components(&self) -> usize {
        self.parameters.b.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::new(Arc::new(self.parameters.subset(component_list)))
    }
}

impl Residual for Cubic {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        let b = (moles * &self.parameters.b).sum() / moles.sum();
        0.9 / b
    }

    fn residual_helmholtz_energy<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> D {
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

    fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)> {
        vec![(
            "Peng Robinson".to_string(),
            self.residual_helmholtz_energy(state),
        )]
    }

    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        &self.parameters.molarweight * (GRAM / MOL)
    }
}
