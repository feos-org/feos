use crate::EosResult;
use ndarray::Array1;
use quantity::si::{SIArray1, SINumber};
use std::sync::Arc;

mod helmholtz_energy;
mod ideal_gas;
mod residual;

pub use helmholtz_energy::{HelmholtzEnergy, HelmholtzEnergyDual};
pub use ideal_gas::{DeBroglieWavelength, DeBroglieWavelengthDual, IdealGas};
pub use residual::{EntropyScaling, Residual};

/// Molar weight of all components.
///
/// The trait is required to be able to calculate (mass)
/// specific properties.
pub trait MolarWeight {
    fn molar_weight(&self) -> SIArray1;
}

pub trait Components {
    /// Return the number of components of the model.
    fn components(&self) -> usize;

    /// Return a model consisting of the components
    /// contained in component_list.
    fn subset(&self, component_list: &[usize]) -> Self;
}

#[derive(Clone)]
pub struct EquationOfState<I, R> {
    pub ideal_gas: Arc<I>,
    pub residual: Arc<R>,
}

impl<I, R> EquationOfState<I, R> {
    pub fn new(ideal_gas: Arc<I>, residual: Arc<R>) -> Self {
        Self {
            ideal_gas,
            residual,
        }
    }
}

impl<I: Components, R: Components> Components for EquationOfState<I, R> {
    fn components(&self) -> usize {
        assert_eq!(self.residual.components(), self.ideal_gas.components());
        self.residual.components()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::new(
            Arc::new(self.ideal_gas.subset(component_list)),
            Arc::new(self.residual.subset(component_list)),
        )
    }
}

impl<I: IdealGas, R: Residual> IdealGas for EquationOfState<I, R> {
    fn ideal_gas_model(&self) -> &dyn DeBroglieWavelength {
        self.ideal_gas.ideal_gas_model()
    }
}

impl<I: IdealGas, R: Residual> Residual for EquationOfState<I, R> {
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
