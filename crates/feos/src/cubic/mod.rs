use alpha::{Alpha, AlphaFunction, PengRobinson1976, RedlichKwong1972};
use feos_core::FeosResult;
use feos_core::parameter::Parameter;
use feos_core::{Components, Residual};
use feos_core::{Molarweight, StateHD};
use mixing_rules::{MixingRule, MixingRuleFunction, MixtureParameters, Quadratic};
use ndarray::{Array1, ScalarOperand, Zip};
use num_dual::DualNum;
use parameters::CubicParameters;
use quantity::{GRAM, MOL, MolarWeight};
use std::f64::consts::SQRT_2;
use std::fmt;
use std::sync::Arc;

mod alpha;
mod mixing_rules;
mod parameters;

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

impl CubicOptions {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self {
            alpha: self.alpha.subset(component_list),
            mixing: self.mixing.clone(),
            delta: self.delta.clone(),
        }
    }
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
    pub fn new(parameters: Arc<CubicParameters>, options: CubicOptions) -> FeosResult<Self> {
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
    ) -> FeosResult<Self> {
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
    ) -> FeosResult<Self> {
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
            options: self.options.subset(component_list),
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
}

impl Molarweight for Cubic {
    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        &self.parameters.molarweight * (GRAM / MOL)
    }
}

#[cfg(test)]
mod tests {
    // general import
    use super::{
        Cubic, PengRobinson1976,
        alpha::Alpha,
        mixing_rules::{MixingRule, Quadratic},
        parameters::{CubicParameters, CubicRecord},
    };
    use feos_core::{
        Contributions::{IdealGas, Total},
        StateBuilder, StateHD,
        cubic::{PengRobinson, PengRobinsonParameters, PengRobinsonRecord},
        parameter::{Identifier, Parameter, PureRecord},
    };
    use ndarray::arr1;
    use quantity::*;
    use std::sync::Arc;
    use typenum::P3;

    use super::*;

    #[test]
    fn a_res() {
        println!("PengRobinson Propane Residual Helmholtz Free Energy Test");

        let pc = 4.21e6; // Pa
        let tc = 369.83; // K
        let mw = 44.1; // g/mol
        let omega = 0.153; // dimensionless

        // Create the newly implemented PR record
        let propane_implemented = PureRecord::new(
            Identifier::new(None, Some("propane"), None, None, None, None),
            mw,
            CubicRecord::new(tc, pc, omega),
        );

        let parameters_implemented =
            Arc::new(CubicParameters::new_pure(propane_implemented).unwrap());

        let eos_implemented = Cubic::peng_robinson(
            parameters_implemented,
            Some(Alpha::PengRobinson1976(PengRobinson1976)),
            Some(MixingRule::Quadratic(Quadratic)),
        )
        .unwrap();

        dbg!(&eos_implemented.critical_parameters);

        // Create the original PR record for comparison from feos-core.
        let propane_compare = PureRecord::new(
            Identifier::new(None, Some("propane"), None, None, None, None),
            mw,
            PengRobinsonRecord::new(tc, pc, omega),
        );

        let parameters_compare = PengRobinsonParameters::new_pure(propane_compare).unwrap();
        let eos_compare = Arc::new(PengRobinson::new(Arc::new(parameters_compare)));

        // Test state. Residual Helmholtz energy of both eos will be used to test.
        let state = StateHD::new(300.0, 1.0e5, arr1(&[5.0]));

        dbg!(&state);

        assert_eq!(
            eos_implemented.residual_helmholtz_energy(&state),
            eos_compare.residual_helmholtz_energy(&state)
        )
    }

    #[test]
    fn pressure() {
        let pc = 4.21e6; // Pa
        let tc = 369.83; // K
        let mw = 44.1; // g/mol
        let omega = 0.153; // dimensionless

        // Create the newly implemented PR record
        let propane_implemented = PureRecord::new(
            Identifier::new(None, Some("propane"), None, None, None, None),
            mw,
            CubicRecord::new(tc, pc, omega),
        );

        let parameters_implemented =
            Arc::new(CubicParameters::new_pure(propane_implemented).unwrap());

        let eos_implemented = Arc::new(
            Cubic::peng_robinson(
                parameters_implemented,
                Some(Alpha::PengRobinson1976(PengRobinson1976)),
                Some(MixingRule::Quadratic(Quadratic)),
            )
            .unwrap(),
        );

        // Create the original PR record for comparison from feos-core.
        let propane_compare = PureRecord::new(
            Identifier::new(None, Some("propane"), None, None, None, None),
            mw,
            PengRobinsonRecord::new(tc, pc, omega),
        );

        let parameters_compare = PengRobinsonParameters::new_pure(propane_compare).unwrap();
        let eos_compare = Arc::new(PengRobinson::new(Arc::new(parameters_compare)));

        // Build the test state
        // Set volume and moles to define the state
        let temp = 300.0 * KELVIN;
        let vol = 8.7e-5 * METER.powi::<P3>();
        let mol = arr1(&[1.0]) * MOL;

        // Build the state with implemented pr eos
        let state_implemented = StateBuilder::new(&eos_implemented)
            .temperature(temp)
            .volume(vol)
            .moles(&mol)
            .build()
            .unwrap();

        // Build the state with compare eos
        let state_compare = StateBuilder::new(&eos_compare)
            .temperature(temp)
            .volume(vol)
            .moles(&mol)
            .build()
            .unwrap();

        // Pressure
        println!(
            "Implemented Total pressure {}",
            state_compare.pressure(Total)
        );

        println!("Compare Total pressure {}", state_compare.pressure(Total));

        // // other properties:
        // // feos-core::state -> residual_properties.rs work fine to get from the state
        // let a_r = state_implemented.residual_helmholtz_energy();
        // // feos-core::state --> properties.rs complain that IdeaGas was not implemented correctly
        // let a = state_implemented.helmholtz_energy();

        assert_eq!(
            state_implemented.pressure(Total),
            state_compare.pressure(Total)
        );
    }
}
