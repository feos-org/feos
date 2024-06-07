use feos_core::parameter::Parameter;
use feos_core::si::{MolarWeight, GRAM, MOL};
use feos_core::{Components, EosResult, Residual, StateHD};
use mixing_rules::{MixingRule, MixingRuleFunction, MixtureParameters, Quadratic};
use ndarray::{Array1, ScalarOperand, Zip};
use num_dual::DualNum;
use std::f64::consts::SQRT_2;
use std::fmt;
use std::sync::Arc;

mod alpha;
use alpha::{Alpha, AlphaFunction, PengRobinson1976, RedlichKwong1972};
mod mixing_rules;
mod parameters;
use parameters::CubicParameters;
// mod volume_translation;

#[cfg(feature = "python")]
pub(crate) mod python;

const KB_A3: f64 = 13806490.0;

#[derive(Debug, Clone, Copy)]
pub struct Delta {
    d1: f64,
    d2: f64,
    d12: f64,
}

impl From<(f64, f64)> for Delta {
    fn from(value: (f64, f64)) -> Self {
        Delta {
            d1: value.0,
            d2: value.1,
            d12: value.0 - value.1,
        }
    }
}

impl Delta {
    // Calculate universal critical constants from universal cubic parameters.
    //
    // See https://doi.org/10.1016/j.fluid.2012.05.008
    fn critical_constants(&self) -> (f64, f64) {
        let (r1, r2) = (-self.d1, -self.d2);
        let eta_c = 1.0
            / (((1.0 - r1) * (1.0 - r2).powi(2)).cbrt()
                + ((1.0 - r2) * (1.0 - r1).powi(2)).cbrt()
                + 1.0);
        let omega_a = (1.0 - eta_c * r1) * (1.0 - eta_c * r2) / (1.0 - eta_c)
            * (2.0 - eta_c * (r1 + r2))
            / (3.0 - eta_c * (1.0 + r1 + r2)).powi(2);
        let omega_b = eta_c / (3.0 - eta_c * (1.0 + r1 + r2));
        (omega_a, omega_b)
    }
}

/// Parameters processed using model constants and substance critial data.
#[derive(Debug)]
pub struct CriticalParameters {
    ac: Array1<f64>,
    bc: Array1<f64>,
    omega_a: f64,
    omega_b: f64,
}

impl CriticalParameters {
    fn new(p: &Arc<CubicParameters>, delta: &Delta) -> Self {
        let (omega_a, omega_b) = delta.critical_constants();
        let ac = omega_a * &p.tc.mapv(|tc| tc.powi(2)) * KB_A3 / &p.pc;
        let bc = omega_b * &p.tc * KB_A3 / &p.pc;
        Self {
            ac,
            bc,
            omega_a,
            omega_b,
        }
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        let n = component_list.len();
        let mut ac = Array1::zeros(n);
        let mut bc = Array1::zeros(n);
        Zip::from(&mut ac)
            .and(&mut bc)
            .and(component_list)
            .for_each(|a, b, &i| {
                *a = self.ac[i];
                *b = self.bc[i];
            });
        Self {
            ac,
            bc,
            omega_a: self.omega_a,
            omega_b: self.omega_b,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CubicOptions {
    pub(crate) alpha: Alpha,
    pub(crate) mixing: MixingRule,
    pub(crate) delta: Delta,
}

/// A generic cubic equation of state.
pub struct Cubic {
    /// Parameters
    pub parameters: Arc<CubicParameters>,
    pub options: CubicOptions,
    /// processed parameters using model and substance critical data
    pub critical_parameters: CriticalParameters,
}

impl Cubic {
    /// Generic cubic equation of state with adjustable universal constants.
    pub fn new(parameters: Arc<CubicParameters>, options: CubicOptions) -> EosResult<Self> {
        let p = CriticalParameters::new(&parameters, &options.delta);
        options.alpha.validate(&parameters)?;
        Ok(Self {
            parameters,
            options,
            critical_parameters: p,
        })
    }

    /// Peng Robinson equation of state.
    ///
    /// Universal constants:
    /// - $\delta_1 = 1 + \sqrt{2}$
    /// - $\delta_2 = 1 - \sqrt{2}$
    ///
    /// If no options are supplied, the following is used:
    /// - alpha function: Peng Robinson (1976)
    /// - mixing rules: quadratic mixing
    pub fn peng_robinson(
        parameters: Arc<CubicParameters>,
        alpha: Option<Alpha>,
        mixing: Option<MixingRule>,
    ) -> EosResult<Self> {
        let delta: Delta = (1.0 + SQRT_2, 1.0 - SQRT_2).into();
        let p = CriticalParameters::new(&parameters, &delta);
        let options = CubicOptions {
            alpha: alpha.unwrap_or(PengRobinson1976.into()),
            mixing: mixing.unwrap_or(Quadratic.into()),
            delta,
        };
        options.alpha.validate(&parameters)?;
        Ok(Self {
            parameters,
            options,
            critical_parameters: p,
        })
    }

    /// Create equation of state of (Suave) Redlich Kwong.
    ///
    /// Universal constants:
    /// - $\delta_1 = 1$
    /// - $\delta_2 = 0$
    ///
    /// If no options are supplied, the following is used:
    /// - alpha function: Soave (1972)
    /// - mixing rules: quadratic mixing
    pub fn redlich_kwong(
        parameters: Arc<CubicParameters>,
        alpha: Option<Alpha>,
        mixing: Option<MixingRule>,
    ) -> EosResult<Self> {
        let delta: Delta = (1.0, 0.0).into();
        let p = CriticalParameters::new(&parameters, &delta);
        let options = CubicOptions {
            alpha: alpha.unwrap_or(RedlichKwong1972.into()),
            mixing: mixing.unwrap_or(Quadratic.into()),
            delta,
        };
        options.alpha.validate(&parameters)?;
        Ok(Self {
            parameters,
            options,
            critical_parameters: p,
        })
    }
}

impl fmt::Display for Cubic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cubic")
    }
}

impl Components for Cubic {
    fn components(&self) -> usize {
        self.parameters.tc.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self {
            parameters: Arc::new(self.parameters.subset(component_list)),
            options: self.options.clone(),
            critical_parameters: self.critical_parameters.subset(component_list),
        }
    }
}

impl Residual for Cubic {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        let b = (moles * &self.critical_parameters.bc).sum() / moles.sum();
        0.9 / b
    }

    fn residual_helmholtz_energy<D: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        state: &StateHD<D>,
    ) -> D {
        let MixtureParameters { a, b, c: _ } = self.options.mixing.apply(self, state);
        let n = state.moles.sum();
        let v = state.volume;
        let bn = b * n;
        n * ((v / (v - bn)).ln()
            - a / (b * self.options.delta.d12 * state.temperature)
                * ((v + bn * self.options.delta.d1) / (v + bn * self.options.delta.d2)).ln())
    }

    fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)> {
        vec![("cubic".to_string(), self.residual_helmholtz_energy(state))]
    }

    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        &self.parameters.molarweight * (GRAM / MOL)
    }
}

#[cfg(test)]
mod tests {
    use feos_core::{
        cubic::{PengRobinson, PengRobinsonParameters, PengRobinsonRecord},
        parameter::{Identifier, PureRecord},
    };
    use ndarray::arr1;
    use parameters::CubicRecord;

    use super::*;

    #[test]
    fn a_res() {
        let propane = PureRecord::new(
            Identifier::new(None, Some("propane"), None, None, None, None),
            44.0962,
            CubicRecord::new(369.96, 4250000.0, 0.153),
        );
        let parameters = Arc::new(CubicParameters::new_pure(propane).unwrap());
        let eos = Cubic::peng_robinson(parameters, None, None).unwrap();
        dbg!(&eos.critical_parameters);
        let state = StateHD::new(300.0, 1e5, arr1(&[5.0]));

        let propane = PureRecord::new(
            Identifier::new(None, Some("propane"), None, None, None, None),
            44.0962,
            PengRobinsonRecord::new(369.96, 4250000.0, 0.153),
        );
        let parameters = PengRobinsonParameters::new_pure(propane).unwrap();
        let pr = Arc::new(PengRobinson::new(Arc::new(parameters)));

        assert_eq!(
            pr.residual_helmholtz_energy(&state),
            eos.residual_helmholtz_energy(&state)
        )
    }
}
