use feos_core::parameter::{Identifier, Parameter, PureRecord};
use feos_core::{FeosError, FeosResult};
use ndarray::{Array1, Array2};
use num_traits::Zero;
use serde::{Deserialize, Serialize};

/// Cubic parameters for a single substance.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct CubicRecord {
    /// critical temperature in Kelvin
    pub(crate) tc: f64,
    /// critical pressure in Pascal
    pub(crate) pc: f64,
    /// acentric factor
    pub(crate) acentric_factor: f64,
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

/// Cubic binary interaction parameters.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct CubicBinaryRecord {
    /// Binary interaction parameter for a
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub k_ij: f64,
    /// Binary interaction parameter for b
    #[serde(skip_serializing_if = "f64::is_zero")]
    #[serde(default)]
    pub l_ij: f64,
    // /// Binary association parameters
    // #[serde(flatten)]
    // association: Option<BinaryAssociationRecord>,
}

impl CubicBinaryRecord {
    pub fn new(
        k_ij: Option<f64>,
        l_ij: Option<f64>,
        // rc_ab: Option<f64>,
        // epsilon_k_ab: Option<f64>,
    ) -> Self {
        let k_ij = k_ij.unwrap_or_default();
        let l_ij = l_ij.unwrap_or_default();
        // let association = if rc_ab.is_none() && epsilon_k_ab.is_none() {
        //     None
        // } else {
        //     Some(BinaryAssociationRecord::new(rc_ab, epsilon_k_ab, None))
        // };
        Self {
            k_ij,
            l_ij,
            // association,
        }
    }
}

impl From<f64> for CubicBinaryRecord {
    fn from(k_ij: f64) -> Self {
        Self {
            k_ij,
            l_ij: f64::default(),
            // association: None,
        }
    }
}

impl From<CubicBinaryRecord> for f64 {
    fn from(binary_record: CubicBinaryRecord) -> Self {
        binary_record.k_ij
    }
}

impl std::fmt::Display for CubicBinaryRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut tokens = vec![];
        if !self.k_ij.is_zero() {
            tokens.push(format!("k_ij={}", self.k_ij));
        }
        if !self.l_ij.is_zero() {
            tokens.push(format!("l_ij={}", self.l_ij));
        }
        // if let Some(association) = self.association {
        //     if let Some(rc_ab) = association.rc_ab {
        //         tokens.push(format!("rc_ab={}", rc_ab));
        //     }
        //     if let Some(epsilon_k_ab) = association.epsilon_k_ab {
        //         tokens.push(format!("epsilon_k_ab={}", epsilon_k_ab));
        //     }
        // }
        write!(f, "CubicBinaryRecord({})", tokens.join(", "))
    }
}

/// Cubic parameters for one ore more substances.
pub struct CubicParameters {
    /// Critical temperature in Kelvin
    pub(super) tc: Array1<f64>,
    pub(super) pc: Array1<f64>,
    pub(super) acentric_factor: Array1<f64>,
    /// Binary interaction parameter for a
    pub(super) k_ij: Array2<f64>,
    /// Binary interaction parameter for b
    pub(super) l_ij: Array2<f64>,
    /// Molar weight in units of g/mol
    pub(super) molarweight: Array1<f64>,
    /// List of pure component records
    pub(super) pure_records: Vec<PureRecord<CubicRecord>>,
    /// List of binary records
    pub binary_records: Option<Array2<CubicBinaryRecord>>,
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
    ) -> Result<Self, FeosError> {
        if [pc.len(), acentric_factor.len(), molarweight.len()]
            .iter()
            .any(|&l| l != tc.len())
        {
            return Err(FeosError::IncompatibleParameters(String::from(
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
    type Binary = CubicBinaryRecord;

    /// Creates parameters from pure component records.
    fn from_records(
        pure_records: Vec<PureRecord<Self::Pure>>,
        binary_records: Option<Array2<Self::Binary>>,
    ) -> FeosResult<Self> {
        let n = pure_records.len();

        let mut tc = Array1::zeros(n);
        let mut pc = Array1::zeros(n);
        let mut acentric_factor = Array1::zeros(n);
        let mut molarweight = Array1::zeros(n);

        for (i, record) in pure_records.iter().enumerate() {
            molarweight[i] = record.molarweight;
            let r = &record.model_record;
            tc[i] = r.tc;
            pc[i] = r.pc;
            acentric_factor[i] = r.acentric_factor;
        }

        let br = binary_records.as_ref();
        let k_ij = br.map_or_else(|| Array2::zeros([n; 2]), |br| br.mapv(|br| br.k_ij));
        let l_ij = br.map_or_else(|| Array2::zeros([n; 2]), |br| br.mapv(|br| br.l_ij));

        Ok(Self {
            tc,
            pc,
            acentric_factor,
            k_ij,
            l_ij,
            molarweight,
            pure_records,
            binary_records,
        })
    }

    fn records(
        &self,
    ) -> (
        &[PureRecord<CubicRecord>],
        Option<&Array2<CubicBinaryRecord>>,
    ) {
        (&self.pure_records, self.binary_records.as_ref())
    }
}
