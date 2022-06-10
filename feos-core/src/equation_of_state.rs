use crate::errors::{EosError, EosResult};
use crate::state::StateHD;
use crate::EosUnit;
use ndarray::prelude::*;
use num_dual::{Dual, Dual3, Dual3_64, Dual64, DualNum, DualVec64, HyperDual, HyperDual64};
use num_traits::{One, Zero};
use quantity::{QuantityArray1, QuantityScalar};
use std::fmt;

/// Individual Helmholtz energy contribution that can
/// be evaluated using generalized (hyper) dual numbers.
///
/// This trait needs to be implemented generically or for
/// the specific types in the supertraits of [HelmholtzEnergy]
/// so that the implementor can be used as a Helmholtz energy
/// contribution in the equation of state.
pub trait HelmholtzEnergyDual<D: DualNum<f64>> {
    /// The Helmholtz energy contribution $\beta A$ of a given state in reduced units.
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D;
}

/// Object safe version of the [HelmholtzEnergyDual] trait.
///
/// The trait is implemented automatically for every struct that implements
/// the supertraits.
pub trait HelmholtzEnergy:
    HelmholtzEnergyDual<f64>
    + HelmholtzEnergyDual<Dual64>
    + HelmholtzEnergyDual<Dual<DualVec64<3>, f64>>
    + HelmholtzEnergyDual<HyperDual64>
    + HelmholtzEnergyDual<Dual3_64>
    + HelmholtzEnergyDual<HyperDual<Dual64, f64>>
    + HelmholtzEnergyDual<HyperDual<DualVec64<2>, f64>>
    + HelmholtzEnergyDual<HyperDual<DualVec64<3>, f64>>
    + HelmholtzEnergyDual<Dual3<Dual64, f64>>
    + HelmholtzEnergyDual<Dual3<DualVec64<2>, f64>>
    + HelmholtzEnergyDual<Dual3<DualVec64<3>, f64>>
    + fmt::Display
{
}

impl<T> HelmholtzEnergy for T where
    T: HelmholtzEnergyDual<f64>
        + HelmholtzEnergyDual<Dual64>
        + HelmholtzEnergyDual<Dual<DualVec64<3>, f64>>
        + HelmholtzEnergyDual<HyperDual64>
        + HelmholtzEnergyDual<Dual3_64>
        + HelmholtzEnergyDual<HyperDual<Dual64, f64>>
        + HelmholtzEnergyDual<HyperDual<DualVec64<2>, f64>>
        + HelmholtzEnergyDual<HyperDual<DualVec64<3>, f64>>
        + HelmholtzEnergyDual<Dual3<Dual64, f64>>
        + HelmholtzEnergyDual<Dual3<DualVec64<2>, f64>>
        + HelmholtzEnergyDual<Dual3<DualVec64<3>, f64>>
        + fmt::Display
{
}

/// Ideal gas Helmholtz energy contribution that can
/// be evaluated using generalized (hyper) dual numbers.
///
/// This trait needs to be implemented generically or for
/// the specific types in the supertraits of [IdealGasContribution]
/// so that the implementor can be used as an ideal gas
/// contribution in the equation of state.
pub trait IdealGasContributionDual<D: DualNum<f64>> {
    /// The thermal de Broglie wavelength of each component in the form $\ln\left(\frac{\Lambda^3}{\AA^3}\right)$
    fn de_broglie_wavelength(&self, temperature: D, components: usize) -> Array1<D>;

    /// Evaluate the ideal gas contribution for a given state.
    ///
    /// In some cases it could be advantageous to overwrite this
    /// implementation instead of implementing the de Broglie
    /// wavelength.
    fn evaluate(&self, state: &StateHD<D>) -> D {
        let lambda = self.de_broglie_wavelength(state.temperature, state.moles.len());
        ((lambda
            + state.partial_density.mapv(|x| {
                if x.re() == 0.0 {
                    D::from(0.0)
                } else {
                    x.ln() - 1.0
                }
            }))
            * &state.moles)
            .sum()
    }
}

/// Object safe version of the [IdealGasContributionDual] trait.
///
/// The trait is implemented automatically for every struct that implements
/// the supertraits.
pub trait IdealGasContribution:
    IdealGasContributionDual<f64>
    + IdealGasContributionDual<Dual64>
    + IdealGasContributionDual<Dual<DualVec64<3>, f64>>
    + IdealGasContributionDual<HyperDual64>
    + IdealGasContributionDual<Dual3_64>
    + IdealGasContributionDual<HyperDual<Dual64, f64>>
    + IdealGasContributionDual<HyperDual<DualVec64<2>, f64>>
    + IdealGasContributionDual<HyperDual<DualVec64<3>, f64>>
    + IdealGasContributionDual<Dual3<Dual64, f64>>
    + IdealGasContributionDual<Dual3<DualVec64<2>, f64>>
    + IdealGasContributionDual<Dual3<DualVec64<3>, f64>>
    + fmt::Display
{
}

impl<T> IdealGasContribution for T where
    T: IdealGasContributionDual<f64>
        + IdealGasContributionDual<Dual64>
        + IdealGasContributionDual<Dual<DualVec64<3>, f64>>
        + IdealGasContributionDual<HyperDual64>
        + IdealGasContributionDual<Dual3_64>
        + IdealGasContributionDual<HyperDual<Dual64, f64>>
        + IdealGasContributionDual<HyperDual<DualVec64<2>, f64>>
        + IdealGasContributionDual<HyperDual<DualVec64<3>, f64>>
        + IdealGasContributionDual<Dual3<Dual64, f64>>
        + IdealGasContributionDual<Dual3<DualVec64<2>, f64>>
        + IdealGasContributionDual<Dual3<DualVec64<3>, f64>>
        + fmt::Display
{
}

struct DefaultIdealGasContribution;
impl<D: DualNum<f64>> IdealGasContributionDual<D> for DefaultIdealGasContribution {
    fn de_broglie_wavelength(&self, _: D, components: usize) -> Array1<D> {
        Array1::zeros(components)
    }
}

impl fmt::Display for DefaultIdealGasContribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ideal gas (default)")
    }
}

/// Molar weight of all components.
///
/// The trait is required to be able to calculate (mass)
/// specific properties.
pub trait MolarWeight<U: EosUnit> {
    fn molar_weight(&self) -> QuantityArray1<U>;
}

/// A general equation of state.
pub trait EquationOfState {
    /// Return the number of components of the equation of state.
    fn components(&self) -> usize;

    /// Return an equation of state consisting of the components
    /// contained in component_list.
    fn subset(&self, component_list: &[usize]) -> Self;

    /// Return the maximum density in Angstrom^-3.
    ///
    /// This value is used as an estimate for a liquid phase for phase
    /// equilibria and other iterations. It is not explicitly meant to
    /// be a mathematical limit for the density (if those exist in the
    /// equation of state anyways).
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64;

    /// Return a slice of the individual contributions (excluding the ideal gas)
    /// of the equation of state.
    fn residual(&self) -> &[Box<dyn HelmholtzEnergy>];

    /// Evaluate the residual reduced Helmholtz energy $\beta A^\mathrm{res}$.
    fn evaluate_residual<D: DualNum<f64>>(&self, state: &StateHD<D>) -> D
    where
        dyn HelmholtzEnergy: HelmholtzEnergyDual<D>,
    {
        self.residual()
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
        self.residual()
            .iter()
            .map(|c| (c.to_string(), c.helmholtz_energy(state)))
            .collect()
    }

    /// Return the ideal gas contribution.
    ///
    /// Per default this function returns an ideal gas contribution
    /// in which the de Broglie wavelength is 1 for every component.
    /// Therefore, the correct ideal gas pressure is obtained even
    /// with no explicit ideal gas term. If a more detailed model is
    /// required (e.g. for the calculation of enthalpies) this function
    /// has to be overwritten.
    fn ideal_gas(&self) -> &dyn IdealGasContribution {
        &DefaultIdealGasContribution
    }

    /// Check if the provided optional mole number is consistent with the
    /// equation of state.
    ///
    /// In general, the number of elements in `moles` needs to match the number
    /// of components of the equation of state. For a pure component, however,
    /// no moles need to be provided. In that case, it is set to the constant
    /// reference value.
    fn validate_moles<U: EosUnit>(
        &self,
        moles: Option<&QuantityArray1<U>>,
    ) -> EosResult<QuantityArray1<U>> {
        let l = moles.map_or(1, |m| m.len());
        if self.components() == l {
            match moles {
                Some(m) => Ok(m.to_owned()),
                None => Ok(Array::ones(1) * U::reference_moles()),
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
    fn max_density<U: EosUnit>(
        &self,
        moles: Option<&QuantityArray1<U>>,
    ) -> EosResult<QuantityScalar<U>> {
        let mr = self
            .validate_moles(moles)?
            .to_reduced(U::reference_moles())?;
        Ok(self.compute_max_density(&mr) * U::reference_density())
    }

    /// Calculate the second virial coefficient $B(T)$
    fn second_virial_coefficient<U: EosUnit>(
        &self,
        temperature: QuantityScalar<U>,
        moles: Option<&QuantityArray1<U>>,
    ) -> EosResult<QuantityScalar<U>> {
        let mr = self.validate_moles(moles)?;
        let x = mr.to_reduced(mr.sum())?;
        let mut rho = HyperDual64::zero();
        rho.eps1[0] = 1.0;
        rho.eps2[0] = 1.0;
        let t = HyperDual64::from(temperature.to_reduced(U::reference_temperature())?);
        let s = StateHD::new_virial(t, rho, x);
        Ok(self.evaluate_residual(&s).eps1eps2[(0, 0)] * 0.5 / U::reference_density())
    }

    /// Calculate the third virial coefficient $C(T)$
    fn third_virial_coefficient<U: EosUnit>(
        &self,
        temperature: QuantityScalar<U>,
        moles: Option<&QuantityArray1<U>>,
    ) -> EosResult<QuantityScalar<U>> {
        let mr = self.validate_moles(moles)?;
        let x = mr.to_reduced(mr.sum())?;
        let rho = Dual3_64::zero().derive();
        let t = Dual3_64::from(temperature.to_reduced(U::reference_temperature())?);
        let s = StateHD::new_virial(t, rho, x);
        Ok(self.evaluate_residual(&s).v3 / 3.0 / U::reference_density().powi(2))
    }

    /// Calculate the temperature derivative of the second virial coefficient $B'(T)$
    fn second_virial_coefficient_temperature_derivative<U: EosUnit>(
        &self,
        temperature: QuantityScalar<U>,
        moles: Option<&QuantityArray1<U>>,
    ) -> EosResult<QuantityScalar<U>> {
        let mr = self.validate_moles(moles)?;
        let x = mr.to_reduced(mr.sum())?;
        let mut rho = HyperDual::zero();
        rho.eps1[0] = Dual64::one();
        rho.eps2[0] = Dual64::one();
        let t = HyperDual::from_re(
            Dual64::from(temperature.to_reduced(U::reference_temperature())?).derive(),
        );
        let s = StateHD::new_virial(t, rho, x);
        Ok(self.evaluate_residual(&s).eps1eps2[(0, 0)].eps[0] * 0.5
            / (U::reference_density() * U::reference_temperature()))
    }

    /// Calculate the temperature derivative of the third virial coefficient $C'(T)$
    fn third_virial_coefficient_temperature_derivative<U: EosUnit>(
        &self,
        temperature: QuantityScalar<U>,
        moles: Option<&QuantityArray1<U>>,
    ) -> EosResult<QuantityScalar<U>> {
        let mr = self.validate_moles(moles)?;
        let x = mr.to_reduced(mr.sum())?;
        let rho = Dual3::zero().derive();
        let t = Dual3::from_re(
            Dual64::from(temperature.to_reduced(U::reference_temperature())?).derive(),
        );
        let s = StateHD::new_virial(t, rho, x);
        Ok(self.evaluate_residual(&s).v3.eps[0]
            / 3.0
            / (U::reference_density().powi(2) * U::reference_temperature()))
    }
}

/// Reference values and residual entropy correlations for entropy scaling.
pub trait EntropyScaling<U: EosUnit> {
    fn viscosity_reference(
        &self,
        temperature: QuantityScalar<U>,
        volume: QuantityScalar<U>,
        moles: &QuantityArray1<U>,
    ) -> EosResult<QuantityScalar<U>>;
    fn viscosity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64>;
    fn diffusion_reference(
        &self,
        temperature: QuantityScalar<U>,
        volume: QuantityScalar<U>,
        moles: &QuantityArray1<U>,
    ) -> EosResult<QuantityScalar<U>>;
    fn diffusion_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64>;
    fn thermal_conductivity_reference(
        &self,
        temperature: QuantityScalar<U>,
        volume: QuantityScalar<U>,
        moles: &QuantityArray1<U>,
    ) -> EosResult<QuantityScalar<U>>;
    fn thermal_conductivity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64>;
}
