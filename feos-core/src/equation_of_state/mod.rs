use crate::{EosError, EosResult, EosUnit, StateHD};
use ndarray::{Array, Array1};
use num_dual::{Dual3, Dual3_64, Dual64, DualNum, HyperDual, HyperDual64};
use num_traits::{One, Zero};
use quantity::si::{SIArray1, SINumber, SIUnit, MOL};
use std::sync::Arc;

pub use ideal_gas::IdealGas;
pub use residual::Residual;
pub mod debroglie;
pub mod helmholtz_energy;
pub mod ideal_gas;
pub mod residual;
pub use helmholtz_energy::{HelmholtzEnergy, HelmholtzEnergyDual};
pub use self::debroglie::{DeBroglieWavelength, DeBroglieWavelengthDual};

/// Molar weight of all components.
///
/// The trait is required to be able to calculate (mass)
/// specific properties.
pub trait MolarWeight {
    fn molar_weight(&self) -> SIArray1;
}

// #[derive(Clone)]
// pub struct Model<I, R> {
//     pub ideal_gas: Arc<I>,
//     pub residual: Arc<R>,
//     components: usize,
// }

// impl<I: IdealGas, R: Residual> Model<I, R> {
//     pub fn new(ideal_gas: Arc<I>, residual: Arc<R>) -> Self {
//         assert_eq!(residual.components(), ideal_gas.components());
//         let components = residual.components();
//         Self {
//             ideal_gas,
//             residual,
//             components,
//         }
//     }

//     pub fn components(&self) -> usize {
//         self.components
//     }

//     pub fn subset(&self, component_list: &[usize]) -> Self {
//         Self::new(
//             Arc::new(self.ideal_gas.subset(component_list)),
//             Arc::new(self.residual.subset(component_list)),
//         )
//     }

//     /// Check if the provided optional mole number is consistent with the
//     /// equation of state.
//     ///
//     /// In general, the number of elements in `moles` needs to match the number
//     /// of components of the equation of state. For a pure component, however,
//     /// no moles need to be provided. In that case, it is set to the constant
//     /// reference value.
//     pub fn validate_moles(&self, moles: Option<&SIArray1>) -> EosResult<SIArray1> {
//         let l = moles.map_or(1, |m| m.len());
//         if self.components() == l {
//             match moles {
//                 Some(m) => Ok(m.to_owned()),
//                 None => Ok(Array::ones(1) * SIUnit::reference_moles()),
//             }
//         } else {
//             Err(EosError::IncompatibleComponents(self.components(), l))
//         }
//     }

//     /// Calculate the maximum density.
//     ///
//     /// This value is used as an estimate for a liquid phase for phase
//     /// equilibria and other iterations. It is not explicitly meant to
//     /// be a mathematical limit for the density (if those exist in the
//     /// equation of state anyways).
//     pub fn max_density(&self, moles: Option<&SIArray1>) -> EosResult<SINumber> {
//         let mr = self
//             .residual
//             .validate_moles(moles)?
//             .to_reduced(SIUnit::reference_moles())?;
//         Ok(self.residual.compute_max_density(&mr) * SIUnit::reference_density())
//     }

//     pub fn evaluate_residual<D: DualNum<f64>>(&self, state: &StateHD<D>) -> D
//     where
//         dyn HelmholtzEnergy: HelmholtzEnergyDual<D>,
//     {
//         self.residual.helmholtz_energy(state)
//     }

//     pub fn evaluate_ideal_gas<D: DualNum<f64>>(&self, state: &StateHD<D>) -> D
//     where
//         dyn DeBroglieWavelength: DeBroglieWavelengthDual<D>,
//     {
//         self.ideal_gas.evaluate_ideal_gas(state)
//     }

//     /// Calculate the second virial coefficient $B(T)$
//     pub fn second_virial_coefficient(
//         &self,
//         temperature: SINumber,
//         moles: Option<&SIArray1>,
//     ) -> EosResult<SINumber> {
//         let mr = self.validate_moles(moles)?;
//         let x = mr.to_reduced(mr.sum())?;
//         let mut rho = HyperDual64::zero();
//         rho.eps1[0] = 1.0;
//         rho.eps2[0] = 1.0;
//         let t = HyperDual64::from(temperature.to_reduced(SIUnit::reference_temperature())?);
//         let s = StateHD::new_virial(t, rho, x);
//         Ok(self.evaluate_residual(&s).eps1eps2[(0, 0)] * 0.5 / SIUnit::reference_density())
//     }

//     /// Calculate the third virial coefficient $C(T)$
//     pub fn third_virial_coefficient(
//         &self,
//         temperature: SINumber,
//         moles: Option<&SIArray1>,
//     ) -> EosResult<SINumber> {
//         let mr = self.validate_moles(moles)?;
//         let x = mr.to_reduced(mr.sum())?;
//         let rho = Dual3_64::zero().derive();
//         let t = Dual3_64::from(temperature.to_reduced(SIUnit::reference_temperature())?);
//         let s = StateHD::new_virial(t, rho, x);
//         Ok(self.evaluate_residual(&s).v3 / 3.0 / SIUnit::reference_density().powi(2))
//     }

//     /// Calculate the temperature derivative of the second virial coefficient $B'(T)$
//     pub fn second_virial_coefficient_temperature_derivative(
//         &self,
//         temperature: SINumber,
//         moles: Option<&SIArray1>,
//     ) -> EosResult<SINumber> {
//         let mr = self.validate_moles(moles)?;
//         let x = mr.to_reduced(mr.sum())?;
//         let mut rho = HyperDual::zero();
//         rho.eps1[0] = Dual64::one();
//         rho.eps2[0] = Dual64::one();
//         let t = HyperDual::from_re(
//             Dual64::from(temperature.to_reduced(SIUnit::reference_temperature())?).derive(),
//         );
//         let s = StateHD::new_virial(t, rho, x);
//         Ok(self.evaluate_residual(&s).eps1eps2[(0, 0)].eps[0] * 0.5
//             / (SIUnit::reference_density() * SIUnit::reference_temperature()))
//     }

//     /// Calculate the temperature derivative of the third virial coefficient $C'(T)$
//     pub fn third_virial_coefficient_temperature_derivative(
//         &self,
//         temperature: SINumber,
//         moles: Option<&SIArray1>,
//     ) -> EosResult<SINumber> {
//         let mr = self.validate_moles(moles)?;
//         let x = mr.to_reduced(mr.sum())?;
//         let rho = Dual3::zero().derive();
//         let t = Dual3::from_re(
//             Dual64::from(temperature.to_reduced(SIUnit::reference_temperature())?).derive(),
//         );
//         let s = StateHD::new_virial(t, rho, x);
//         Ok(self.evaluate_residual(&s).v3.eps[0]
//             / 3.0
//             / (SIUnit::reference_density().powi(2) * SIUnit::reference_temperature()))
//     }
// }

// impl<I: IdealGas, R: Residual + MolarWeight> Model<I, R> {
//     pub fn molar_weight(&self) -> Array1<f64> {
//         self.residual.molar_weight().to_reduced(MOL).unwrap()
//     }
// }
