use ndarray::Array1;
use num_dual::DualNum;
use quantity::si::{SIArray1, SINumber};
use std::{fmt::Display, sync::Arc};

pub mod debroglie;
pub mod helmholtz_energy;
pub mod ideal_gas;
pub mod residual;
use crate::{EosResult, StateHD};

pub use self::debroglie::{DeBroglieWavelength, DeBroglieWavelengthDual};
pub use helmholtz_energy::{HelmholtzEnergy, HelmholtzEnergyDual};
pub use ideal_gas::IdealGas;
pub use residual::{EntropyScaling, Residual};

/// Molar weight of all components.
///
/// The trait is required to be able to calculate (mass)
/// specific properties.
pub trait MolarWeight {
    fn molar_weight(&self) -> SIArray1;
}

#[derive(Clone)]
pub struct EquationOfState<I, R> {
    pub ideal_gas: Arc<I>,
    pub residual: Arc<R>,
    components: usize,
}

impl<I: IdealGas, R: Residual> Display for EquationOfState<I, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {}",
            self.ideal_gas.to_string(),
            self.residual.to_string()
        )
    }
}

impl<I: IdealGas, R: Residual> EquationOfState<I, R> {
    pub fn new(ideal_gas: Arc<I>, residual: Arc<R>) -> Self {
        // assert_eq!(residual.components(), ideal_gas.components());
        let components = residual.components();
        Self {
            ideal_gas,
            residual,
            components,
        }
    }
}

impl<I: IdealGas, R: Residual> IdealGas for EquationOfState<I, R> {
    fn evaluate_ideal_gas<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> D
    where
        dyn DeBroglieWavelength: DeBroglieWavelengthDual<D>,
    {
        self.ideal_gas.evaluate_ideal_gas(state)
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::new(
            Arc::new(self.ideal_gas.subset(component_list)),
            Arc::new(self.residual.subset(component_list)),
        )
    }

    fn ideal_gas_model(&self) -> &Box<dyn DeBroglieWavelength> {
        self.ideal_gas.ideal_gas_model()
    }
}

impl<I: IdealGas, R: Residual> Residual for EquationOfState<I, R> {
    fn components(&self) -> usize {
        self.residual.components()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::new(
            Arc::new(self.ideal_gas.subset(component_list)),
            Arc::new(self.residual.subset(component_list)),
        )
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.residual.compute_max_density(moles)
    }

    fn contributions(&self) -> &[Box<dyn HelmholtzEnergy>] {
        self.residual.contributions()
    }
}

impl<I: IdealGas, R: Residual + MolarWeight> MolarWeight for EquationOfState<I, R> {
    fn molar_weight(&self) -> SIArray1 {
        self.residual.molar_weight()
    }
}

impl<I: IdealGas, R: Residual + EntropyScaling> EntropyScaling for EquationOfState<I, R> {
    fn viscosity_reference(
        &self,
        temperature: SINumber,
        volume: SINumber,
        moles: &SIArray1,
    ) -> EosResult<SINumber> {
        self.residual
            .viscosity_reference(temperature, volume, moles)
    }
    fn viscosity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        self.residual.viscosity_correlation(s_res, x)
    }
    fn diffusion_reference(
        &self,
        temperature: SINumber,
        volume: SINumber,
        moles: &SIArray1,
    ) -> EosResult<SINumber> {
        self.residual
            .diffusion_reference(temperature, volume, moles)
    }
    fn diffusion_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        self.residual.diffusion_correlation(s_res, x)
    }
    fn thermal_conductivity_reference(
        &self,
        temperature: SINumber,
        volume: SINumber,
        moles: &SIArray1,
    ) -> EosResult<SINumber> {
        self.residual
            .thermal_conductivity_reference(temperature, volume, moles)
    }
    fn thermal_conductivity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        self.residual.thermal_conductivity_correlation(s_res, x)
    }
}
