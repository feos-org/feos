use crate::ReferenceSystem;
use crate::state::StateHD;
use nalgebra::{
    Const, DVector, DefaultAllocator, Dim, Dyn, OVector, SVector, allocator::Allocator,
};
use num_dual::DualNum;
use quantity::{Dimensionless, MolarEnergy, MolarVolume, Temperature};
use std::ops::Deref;

mod residual;
pub use residual::{EntropyScaling, Molarweight, NoResidual, Residual, ResidualDyn, Subset};

/// An equation of state consisting of an ideal gas model
/// and a residual Helmholtz energy model.
#[derive(Clone)]
pub struct EquationOfState<I, R> {
    pub ideal_gas: I,
    pub residual: R,
}

impl<I, R> Deref for EquationOfState<I, R> {
    type Target = R;
    fn deref(&self) -> &R {
        &self.residual
    }
}

impl<I, R> EquationOfState<I, R> {
    /// Return a new [EquationOfState] with the given ideal gas
    /// and residual models.
    pub fn new(ideal_gas: I, residual: R) -> Self {
        Self {
            ideal_gas,
            residual,
        }
    }
}

impl<I> EquationOfState<Vec<I>, NoResidual> {
    /// Return a new [EquationOfState] that only consists of
    /// an ideal gas models.
    pub fn ideal_gas(ideal_gas: Vec<I>) -> Self {
        let residual = NoResidual(ideal_gas.len());
        Self {
            ideal_gas,
            residual,
        }
    }
}

impl<I, R: ResidualDyn> ResidualDyn for EquationOfState<Vec<I>, R> {
    fn components(&self) -> usize {
        self.residual.components()
    }

    fn compute_max_density<D: DualNum<f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        self.residual.compute_max_density(molefracs)
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        self.residual
            .reduced_helmholtz_energy_density_contributions(state)
    }
}

impl<I: Clone, R: Subset> Subset for EquationOfState<Vec<I>, R> {
    fn subset(&self, component_list: &[usize]) -> Self {
        let ideal_gas = component_list
            .iter()
            .map(|&i| self.ideal_gas[i].clone())
            .collect();
        EquationOfState {
            ideal_gas,
            residual: self.residual.subset(component_list),
        }
    }
}

impl<I: Clone, R: Residual<Const<N>, D>, D: DualNum<f64> + Copy, const N: usize>
    Residual<Const<N>, D> for EquationOfState<[I; N], R>
{
    fn components(&self) -> usize {
        N
    }

    type Real = EquationOfState<[I; N], R::Real>;
    type Lifted<D2: DualNum<f64, Inner = D> + Copy> = EquationOfState<[I; N], R::Lifted<D2>>;
    fn re(&self) -> Self::Real {
        EquationOfState::new(self.ideal_gas.clone(), self.residual.re())
    }
    fn lift<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        EquationOfState::new(self.ideal_gas.clone(), self.residual.lift())
    }

    fn compute_max_density(&self, molefracs: &SVector<D, N>) -> D {
        self.residual.compute_max_density(molefracs)
    }

    fn reduced_helmholtz_energy_density_contributions(
        &self,
        state: &StateHD<D, Const<N>>,
    ) -> Vec<(&'static str, D)> {
        self.residual
            .reduced_helmholtz_energy_density_contributions(state)
    }

    fn reduced_residual_helmholtz_energy_density(&self, state: &StateHD<D, Const<N>>) -> D {
        self.residual
            .reduced_residual_helmholtz_energy_density(state)
    }
}

/// Ideal gas Helmholtz energy contribution.
pub trait IdealGas {
    /// Implementation of an ideal gas model in terms of the
    /// logarithm of the cubic thermal de Broglie wavelength
    /// in units ln(A³) for each component in the system.
    fn ln_lambda3<D: DualNum<f64> + Copy>(&self, temperature: D) -> D;

    /// The name of the ideal gas model.
    fn ideal_gas_model(&self) -> &'static str;
}

/// Ideal gas Helmholtz energy contribution.
pub trait IdealGasAD<D = f64>: Clone {
    type Real: IdealGasAD;
    type Lifted<D2: DualNum<f64, Inner = D> + Copy>: IdealGasAD<D2>;
    fn re(&self) -> Self::Real;
    fn lift<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2>;

    /// Implementation of an ideal gas model in terms of the
    /// logarithm of the cubic thermal de Broglie wavelength
    /// in units ln(A³) for each component in the system.
    fn ln_lambda3(&self, temperature: D) -> D;

    /// The name of the ideal gas model.
    fn ideal_gas_model(&self) -> &'static str;
}

/// A total Helmholtz energy model consisting of a [Residual] model and an [IdealGas] part.
pub trait Total<N: Dim = Dyn, D: DualNum<f64> + Copy = f64>: Residual<N, D>
where
    DefaultAllocator: Allocator<N>,
{
    type RealTotal: Total<N, f64>;
    type LiftedTotal<D2: DualNum<f64, Inner = D> + Copy>: Total<N, D2>;
    fn re_total(&self) -> Self::RealTotal;
    fn lift_total<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::LiftedTotal<D2>;

    fn ideal_gas_model(&self) -> &'static str;

    fn ln_lambda3(&self, temperature: D) -> OVector<D, N>;

    fn ideal_gas_molar_helmholtz_energy(
        &self,
        temperature: D,
        molar_volume: D,
        molefracs: &OVector<D, N>,
    ) -> D {
        let partial_density = molefracs / molar_volume;
        let mut res = D::from(0.0);
        for (&l, &r) in self
            .ln_lambda3(temperature)
            .iter()
            .zip(partial_density.iter())
        {
            let ln_rho_m1 = if r.re() == 0.0 {
                D::from(0.0)
            } else {
                r.ln() - 1.0
            };
            res += r * (l + ln_rho_m1)
        }
        res * molar_volume * temperature
    }

    fn ideal_gas_helmholtz_energy(
        &self,
        temperature: Temperature<D>,
        volume: MolarVolume<D>,
        moles: &OVector<D, N>,
    ) -> MolarEnergy<D> {
        let total_moles = moles.sum();
        let molefracs = moles / total_moles;
        let molar_volume = volume.into_reduced() / total_moles;
        MolarEnergy::from_reduced(self.ideal_gas_molar_helmholtz_energy(
            temperature.into_reduced(),
            molar_volume,
            &molefracs,
        )) * Dimensionless::new(total_moles)
    }
}

impl<
    I: IdealGas + 'static,
    C: Deref<Target = EquationOfState<Vec<I>, R>> + Clone,
    R: ResidualDyn + 'static,
    D: DualNum<f64> + Copy,
> Total<Dyn, D> for C
{
    type RealTotal = Self;
    type LiftedTotal<D2: DualNum<f64, Inner = D> + Copy> = Self;
    fn re_total(&self) -> Self::RealTotal {
        self.clone()
    }
    fn lift_total<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::LiftedTotal<D2> {
        self.clone()
    }

    fn ideal_gas_model(&self) -> &'static str {
        self.ideal_gas[0].ideal_gas_model()
    }

    fn ln_lambda3(&self, temperature: D) -> DVector<D> {
        DVector::from_vec(
            self.ideal_gas
                .iter()
                .map(|i| i.ln_lambda3(temperature))
                .collect(),
        )
    }
}

impl<I: IdealGasAD<D>, R: Residual<Const<N>, D>, D: DualNum<f64> + Copy, const N: usize>
    Total<Const<N>, D> for EquationOfState<[I; N], R>
{
    type RealTotal = EquationOfState<[I::Real; N], R::Real>;
    type LiftedTotal<D2: DualNum<f64, Inner = D> + Copy> =
        EquationOfState<[I::Lifted<D2>; N], R::Lifted<D2>>;
    fn re_total(&self) -> Self::RealTotal {
        EquationOfState::new(
            self.ideal_gas.each_ref().map(|i| i.re()),
            self.residual.re(),
        )
    }
    fn lift_total<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::LiftedTotal<D2> {
        EquationOfState::new(
            self.ideal_gas.each_ref().map(|i| i.lift()),
            self.residual.lift(),
        )
    }

    fn ideal_gas_model(&self) -> &'static str {
        self.ideal_gas[0].ideal_gas_model()
    }

    fn ln_lambda3(&self, temperature: D) -> SVector<D, N> {
        SVector::from(self.ideal_gas.each_ref().map(|i| i.ln_lambda3(temperature)))
    }
}
