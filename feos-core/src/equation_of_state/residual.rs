use super::{Components, HelmholtzEnergy, HelmholtzEnergyDual};
use crate::StateHD;
use crate::{EosError, EosResult, EosUnit};
use ndarray::prelude::*;
use num_dual::*;
use num_traits::{One, Zero};
use quantity::si::{SIArray1, SINumber, SIUnit};

/// A general equation of state.
pub trait Residual: Components + Send + Sync {
    /// Return the maximum density in Angstrom^-3.
    ///
    /// This value is used as an estimate for a liquid phase for phase
    /// equilibria and other iterations. It is not explicitly meant to
    /// be a mathematical limit for the density (if those exist in the
    /// equation of state anyways).
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64;

    /// Return a slice of the individual contributions (excluding the ideal gas)
    /// of the equation of state.
    fn contributions(&self) -> &[Box<dyn HelmholtzEnergy>];

    /// Evaluate the residual reduced Helmholtz energy $\beta A^\mathrm{res}$.
    fn evaluate_residual<D: DualNum<f64>>(&self, state: &StateHD<D>) -> D
    where
        dyn HelmholtzEnergy: HelmholtzEnergyDual<D>,
    {
        self.contributions()
            .iter()
            .map(|c| c.helmholtz_energy(state))
            .sum()
    }

    /// Evaluate the reduced Helmholtz energy of each individual contribution
    /// and return them together with a string representation of the contribution.
    fn evaluate_residual_contributions<D: DualNum<f64>>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)>
    where
        dyn HelmholtzEnergy: HelmholtzEnergyDual<D>,
    {
        self.contributions()
            .iter()
            .map(|c| (c.to_string(), c.helmholtz_energy(state)))
            .collect()
    }

    /// Check if the provided optional mole number is consistent with the
    /// equation of state.
    ///
    /// In general, the number of elements in `moles` needs to match the number
    /// of components of the equation of state. For a pure component, however,
    /// no moles need to be provided. In that case, it is set to the constant
    /// reference value.
    fn validate_moles(&self, moles: Option<&SIArray1>) -> EosResult<SIArray1> {
        let l = moles.map_or(1, |m| m.len());
        if self.components() == l {
            match moles {
                Some(m) => Ok(m.to_owned()),
                None => Ok(Array::ones(1) * SIUnit::reference_moles()),
            }
        } else {
            Err(EosError::IncompatibleComponents(self.components(), l))
        }
    }

    /// Calculate the maximum density.
    ///
    /// This value is used as an estimate for a liquid phase for phase
    /// equilibria and other iterations. It is not explicitly meant to
    /// be a mathematical limit for the density (if those exist in the
    /// equation of state anyways).
    fn max_density(&self, moles: Option<&SIArray1>) -> EosResult<SINumber> {
        let mr = self
            .validate_moles(moles)?
            .to_reduced(SIUnit::reference_moles())?;
        Ok(self.compute_max_density(&mr) * SIUnit::reference_density())
    }

    /// Calculate the second virial coefficient $B(T)$
    fn second_virial_coefficient(
        &self,
        temperature: SINumber,
        moles: Option<&SIArray1>,
    ) -> EosResult<SINumber> {
        let mr = self.validate_moles(moles)?;
        let x = mr.to_reduced(mr.sum())?;
        let mut rho = HyperDual64::zero();
        rho.eps1 = 1.0;
        rho.eps2 = 1.0;
        let t = HyperDual64::from(temperature.to_reduced(SIUnit::reference_temperature())?);
        let s = StateHD::new_virial(t, rho, x);
        Ok(self.evaluate_residual(&s).eps1eps2 * 0.5 / SIUnit::reference_density())
    }

    /// Calculate the third virial coefficient $C(T)$
    fn third_virial_coefficient(
        &self,
        temperature: SINumber,
        moles: Option<&SIArray1>,
    ) -> EosResult<SINumber> {
        let mr = self.validate_moles(moles)?;
        let x = mr.to_reduced(mr.sum())?;
        let rho = Dual3_64::zero().derivative();
        let t = Dual3_64::from(temperature.to_reduced(SIUnit::reference_temperature())?);
        let s = StateHD::new_virial(t, rho, x);
        Ok(self.evaluate_residual(&s).v3 / 3.0 / SIUnit::reference_density().powi(2))
    }

    /// Calculate the temperature derivative of the second virial coefficient $B'(T)$
    fn second_virial_coefficient_temperature_derivative(
        &self,
        temperature: SINumber,
        moles: Option<&SIArray1>,
    ) -> EosResult<SINumber> {
        let mr = self.validate_moles(moles)?;
        let x = mr.to_reduced(mr.sum())?;
        let mut rho = HyperDual::zero();
        rho.eps1 = Dual64::one();
        rho.eps2 = Dual64::one();
        let t = HyperDual::from_re(
            Dual64::from(temperature.to_reduced(SIUnit::reference_temperature())?).derivative(),
        );
        let s = StateHD::new_virial(t, rho, x);
        Ok(self.evaluate_residual(&s).eps1eps2.eps * 0.5
            / (SIUnit::reference_density() * SIUnit::reference_temperature()))
    }

    /// Calculate the temperature derivative of the third virial coefficient $C'(T)$
    fn third_virial_coefficient_temperature_derivative(
        &self,
        temperature: SINumber,
        moles: Option<&SIArray1>,
    ) -> EosResult<SINumber> {
        let mr = self.validate_moles(moles)?;
        let x = mr.to_reduced(mr.sum())?;
        let rho = Dual3::zero().derivative();
        let t = Dual3::from_re(
            Dual64::from(temperature.to_reduced(SIUnit::reference_temperature())?).derivative(),
        );
        let s = StateHD::new_virial(t, rho, x);
        Ok(self.evaluate_residual(&s).v3.eps
            / 3.0
            / (SIUnit::reference_density().powi(2) * SIUnit::reference_temperature()))
    }
}

/// Reference values and residual entropy correlations for entropy scaling.
pub trait EntropyScaling {
    fn viscosity_reference(
        &self,
        temperature: SINumber,
        volume: SINumber,
        moles: &SIArray1,
    ) -> EosResult<SINumber>;
    fn viscosity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64>;
    fn diffusion_reference(
        &self,
        temperature: SINumber,
        volume: SINumber,
        moles: &SIArray1,
    ) -> EosResult<SINumber>;
    fn diffusion_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64>;
    fn thermal_conductivity_reference(
        &self,
        temperature: SINumber,
        volume: SINumber,
        moles: &SIArray1,
    ) -> EosResult<SINumber>;
    fn thermal_conductivity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64>;
}
