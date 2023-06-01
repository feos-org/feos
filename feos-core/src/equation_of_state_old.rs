use crate::errors::{EosError, EosResult};
use crate::state::StateHD;
use crate::EosUnit;
use ndarray::prelude::*;
use num_dual::{
    first_derivative, second_derivative, third_derivative, Dual, Dual2, Dual2_64, Dual3, Dual3_64,
    Dual64, DualNum, DualSVec64, HyperDual, HyperDual64,
};
use num_traits::Zero;
use quantity::si::{SIArray1, SINumber, SIUnit};
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
    + HelmholtzEnergyDual<Dual<DualSVec64<3>, f64>>
    + HelmholtzEnergyDual<HyperDual64>
    + HelmholtzEnergyDual<Dual2_64>
    + HelmholtzEnergyDual<Dual3_64>
    + HelmholtzEnergyDual<HyperDual<Dual64, f64>>
    + HelmholtzEnergyDual<HyperDual<DualSVec64<2>, f64>>
    + HelmholtzEnergyDual<HyperDual<DualSVec64<3>, f64>>
    + HelmholtzEnergyDual<Dual2<Dual64, f64>>
    + HelmholtzEnergyDual<Dual3<Dual64, f64>>
    + HelmholtzEnergyDual<Dual3<DualSVec64<2>, f64>>
    + HelmholtzEnergyDual<Dual3<DualSVec64<3>, f64>>
    + fmt::Display
    + Send
    + Sync
{
}

impl<T> HelmholtzEnergy for T where
    T: HelmholtzEnergyDual<f64>
        + HelmholtzEnergyDual<Dual64>
        + HelmholtzEnergyDual<Dual<DualSVec64<3>, f64>>
        + HelmholtzEnergyDual<HyperDual64>
        + HelmholtzEnergyDual<Dual2_64>
        + HelmholtzEnergyDual<Dual3_64>
        + HelmholtzEnergyDual<HyperDual<Dual64, f64>>
        + HelmholtzEnergyDual<HyperDual<DualSVec64<2>, f64>>
        + HelmholtzEnergyDual<HyperDual<DualSVec64<3>, f64>>
        + HelmholtzEnergyDual<Dual2<Dual64, f64>>
        + HelmholtzEnergyDual<Dual3<Dual64, f64>>
        + HelmholtzEnergyDual<Dual3<DualSVec64<2>, f64>>
        + HelmholtzEnergyDual<Dual3<DualSVec64<3>, f64>>
        + fmt::Display
        + Send
        + Sync
{
}

/// Ideal gas Helmholtz energy contribution that can
/// be evaluated using generalized (hyper) dual numbers.
///
/// This trait needs to be implemented generically or for
/// the specific types in the supertraits of [IdealGasContribution]
/// so that the implementor can be used as an ideal gas
/// contribution in the equation of state.
pub trait IdealGasContributionDual<D: DualNum<f64> + Copy> {
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
    + IdealGasContributionDual<Dual<DualSVec64<3>, f64>>
    + IdealGasContributionDual<HyperDual64>
    + IdealGasContributionDual<Dual2_64>
    + IdealGasContributionDual<Dual3_64>
    + IdealGasContributionDual<HyperDual<Dual64, f64>>
    + IdealGasContributionDual<HyperDual<DualSVec64<2>, f64>>
    + IdealGasContributionDual<HyperDual<DualSVec64<3>, f64>>
    + IdealGasContributionDual<Dual2<Dual64, f64>>
    + IdealGasContributionDual<Dual3<Dual64, f64>>
    + IdealGasContributionDual<Dual3<DualSVec64<2>, f64>>
    + IdealGasContributionDual<Dual3<DualSVec64<3>, f64>>
    + fmt::Display
{
}

impl<T> IdealGasContribution for T where
    T: IdealGasContributionDual<f64>
        + IdealGasContributionDual<Dual64>
        + IdealGasContributionDual<Dual<DualSVec64<3>, f64>>
        + IdealGasContributionDual<HyperDual64>
        + IdealGasContributionDual<Dual2_64>
        + IdealGasContributionDual<Dual3_64>
        + IdealGasContributionDual<HyperDual<Dual64, f64>>
        + IdealGasContributionDual<HyperDual<DualSVec64<2>, f64>>
        + IdealGasContributionDual<HyperDual<DualSVec64<3>, f64>>
        + IdealGasContributionDual<Dual2<Dual64, f64>>
        + IdealGasContributionDual<Dual3<Dual64, f64>>
        + IdealGasContributionDual<Dual3<DualSVec64<2>, f64>>
        + IdealGasContributionDual<Dual3<DualSVec64<3>, f64>>
        + fmt::Display
{
}

struct DefaultIdealGasContribution;
impl<D: DualNum<f64> + Copy> IdealGasContributionDual<D> for DefaultIdealGasContribution {
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
pub trait MolarWeight {
    fn molar_weight(&self) -> SIArray1;
}

/// A general equation of state.
pub trait EquationOfState: Send + Sync {
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
    fn evaluate_residual<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> D
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
    fn evaluate_residual_contributions<D: DualNum<f64> + Copy>(
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
        let t = temperature.to_reduced(SIUnit::reference_temperature())?;
        let a_res = |rho| self.evaluate_residual(&StateHD::new_virial(t.into(), rho, x));
        let (_, _, b) = second_derivative(a_res, 0.0);
        Ok(b * 0.5 / SIUnit::reference_density())
    }

    /// Calculate the third virial coefficient $C(T)$
    fn third_virial_coefficient(
        &self,
        temperature: SINumber,
        moles: Option<&SIArray1>,
    ) -> EosResult<SINumber> {
        let mr = self.validate_moles(moles)?;
        let x = mr.to_reduced(mr.sum())?;
        let t = temperature.to_reduced(SIUnit::reference_temperature())?;
        let a_res = |rho| self.evaluate_residual(&StateHD::new_virial(t.into(), rho, x));
        let (_, _, _, c) = third_derivative(a_res, 0.0);
        Ok(c / 3.0 / SIUnit::reference_density().powi(2))
    }

    /// Calculate the temperature derivative of the second virial coefficient $B'(T)$
    fn second_virial_coefficient_temperature_derivative(
        &self,
        temperature: SINumber,
        moles: Option<&SIArray1>,
    ) -> EosResult<SINumber> {
        let mr = self.validate_moles(moles)?;
        let x = mr.to_reduced(mr.sum())?;
        let t = temperature.to_reduced(SIUnit::reference_temperature())?;
        let b = |t| {
            let a_res = |rho: Dual2<Dual64, f64>| {
                self.evaluate_residual(&StateHD::new_virial(Dual2::from_re(t), rho, x))
            };
            let (_, _, b) = second_derivative(a_res, Dual64::zero());
            b
        };
        let (_, b_t) = first_derivative(b, t);
        Ok(b_t * 0.5 / (SIUnit::reference_density() * SIUnit::reference_temperature()))
    }

    /// Calculate the temperature derivative of the third virial coefficient $C'(T)$
    fn third_virial_coefficient_temperature_derivative(
        &self,
        temperature: SINumber,
        moles: Option<&SIArray1>,
    ) -> EosResult<SINumber> {
        let mr = self.validate_moles(moles)?;
        let x = mr.to_reduced(mr.sum())?;
        let t = temperature.to_reduced(SIUnit::reference_temperature())?;
        let c = |t| {
            let a_res =
                |rho| self.evaluate_residual(&StateHD::new_virial(Dual3::from_re(t), rho, x));
            let (_, _, _, c) = third_derivative(a_res, Dual64::zero());
            c
        };
        let (_, c_t) = first_derivative(c, t);
        Ok(c_t / 3.0 / (SIUnit::reference_density().powi(2) * SIUnit::reference_temperature()))
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
