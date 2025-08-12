use std::ops::Deref;

use crate::state::StateHD;
use nalgebra::{
    Const, DVector, DefaultAllocator, Dim, Dyn, OVector, SVector, U1, allocator::Allocator,
};
use num_dual::DualNum;
use quantity::MolarWeight;

mod ideal_gas;
mod residual;

pub use ideal_gas::{IdealGas, IdealGasDyn};
pub use residual::{Molarweight, NoResidual, Residual, ResidualConst, ResidualDyn, Subset};

/// An equation of state consisting of an ideal gas model
/// and a residual Helmholtz energy model.
#[derive(Clone)]
pub struct EquationOfState<I, R> {
    pub ideal_gas: I,
    pub residual: R,
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

impl<I: Clone, R: ResidualDyn> ResidualDyn for EquationOfState<Vec<I>, R> {
    fn components(&self) -> usize {
        self.residual.components()
    }

    fn compute_max_density<D: DualNum<f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        self.residual.compute_max_density(molefracs)
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)> {
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

impl<I: IdealGas<D>, R: ResidualConst<N, D>, D: DualNum<f64> + Copy, const N: usize>
    ResidualConst<N, D> for EquationOfState<[I; N], R>
{
    const NAME: &str = R::NAME;
    type Real = EquationOfState<[I::Real; N], R::Real>;
    type Lifted<D2: DualNum<f64, Inner = D> + Copy> =
        EquationOfState<[I::Lifted<D2>; N], R::Lifted<D2>>;
    fn re(&self) -> Self::Real {
        EquationOfState::new(self.ideal_gas.each_ref().map(I::re), self.residual.re())
    }
    fn lift<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        EquationOfState::new(self.ideal_gas.each_ref().map(I::lift), self.residual.lift())
    }

    fn compute_max_density(&self, molefracs: &SVector<D, N>) -> D {
        self.residual.compute_max_density(molefracs)
    }

    fn reduced_residual_helmholtz_energy_density(&self, state: &StateHD<D, Const<N>>) -> D {
        self.residual
            .reduced_residual_helmholtz_energy_density(state)
    }
}

impl<I, R: Molarweight<N, D>, N: Dim, D: DualNum<f64> + Copy> Molarweight<N, D>
    for EquationOfState<I, R>
where
    DefaultAllocator: Allocator<N>,
{
    fn molar_weight(&self) -> MolarWeight<OVector<D, N>> {
        self.residual.molar_weight()
    }
}

pub trait Total<N: Dim = Dyn, D: DualNum<f64> + Copy = f64>: Residual<N, D>
where
    DefaultAllocator: Allocator<N>,
{
    type IdealGas<'a>: IdealGas<D>
    where
        Self: 'a;

    fn ideal_gas_model(&self) -> &'static str;

    fn ideal_gas<'a>(&'a self) -> impl Iterator<Item = Self::IdealGas<'a>>;

    fn ln_lambda3<D2: DualNum<f64, Inner = D> + Copy>(&self, temperature: D2) -> OVector<D2, N> {
        OVector::from_iterator_generic(
            N::from_usize(self.components()),
            U1,
            self.ideal_gas().map(|i| i.lift().ln_lambda3(temperature)),
        )
    }

    fn ideal_gas_helmholtz_energy<D2: DualNum<f64, Inner = D> + Copy>(
        &self,
        temperature: D2,
        volume: D2,
        moles: &OVector<D2, N>,
    ) -> D2 {
        let partial_density = moles / volume;
        let mut res = D2::from(0.0);
        for (i, &r) in self.ideal_gas().zip(partial_density.iter()) {
            let ln_rho_m1 = if r.re() == 0.0 {
                D2::from(0.0)
            } else {
                r.ln() - 1.0
            };
            res += r * (i.lift().ln_lambda3(temperature) + ln_rho_m1)
        }
        res * volume * temperature
    }

    fn ideal_gas_helmholtz_energy_0(&self, temperature: D, volume: D, moles: &OVector<D, N>) -> D {
        let partial_density = moles / volume;
        let mut res = D::from(0.0);
        for (i, &r) in self.ideal_gas().zip(partial_density.iter()) {
            let ln_rho_m1 = if r.re() == 0.0 {
                D::from(0.0)
            } else {
                r.ln() - 1.0
            };
            res += r * (i.ln_lambda3(temperature) + ln_rho_m1)
        }
        res * volume * temperature
    }
}

impl<
    I: IdealGasDyn + Clone + 'static,
    C: Deref<Target = EquationOfState<Vec<I>, R>> + Clone,
    R: ResidualDyn + 'static,
    D: DualNum<f64> + Copy,
> Total<Dyn, D> for C
{
    type IdealGas<'a>
        = &'a I
    where
        Self: 'a;

    fn ideal_gas_model(&self) -> &'static str {
        self.ideal_gas[0].ideal_gas_model()
    }

    fn ideal_gas<'a>(&'a self) -> impl Iterator<Item = Self::IdealGas<'a>> {
        self.ideal_gas.iter()
    }
}

impl<I: IdealGas<D>, R: ResidualConst<N, D> + 'static, D: DualNum<f64> + Copy, const N: usize>
    Total<Const<N>, D> for EquationOfState<[I; N], R>
{
    type IdealGas<'a>
        = I
    where
        I: 'a;

    fn ideal_gas_model(&self) -> &'static str {
        self.ideal_gas[0].ideal_gas_model()
    }

    fn ideal_gas<'a>(&'a self) -> impl Iterator<Item = Self::IdealGas<'a>> {
        self.ideal_gas.clone().into_iter()
    }
}

// impl<I: IdealGas, R: Residual + EntropyScaling> EntropyScaling for EquationOfState<I, R> {
//     fn viscosity_reference(
//         &self,
//         temperature: Temperature,
//         volume: Volume,
//         moles: &Moles<DVector<f64>>,
//     ) -> FeosResult<Viscosity> {
//         self.residual
//             .viscosity_reference(temperature, volume, moles)
//     }
//     fn viscosity_correlation(&self, s_res: f64, x: &DVector<f64>) -> FeosResult<f64> {
//         self.residual.viscosity_correlation(s_res, x)
//     }
//     fn diffusion_reference(
//         &self,
//         temperature: Temperature,
//         volume: Volume,
//         moles: &Moles<DVector<f64>>,
//     ) -> FeosResult<Diffusivity> {
//         self.residual
//             .diffusion_reference(temperature, volume, moles)
//     }
//     fn diffusion_correlation(&self, s_res: f64, x: &DVector<f64>) -> FeosResult<f64> {
//         self.residual.diffusion_correlation(s_res, x)
//     }
//     fn thermal_conductivity_reference(
//         &self,
//         temperature: Temperature,
//         volume: Volume,
//         moles: &Moles<DVector<f64>>,
//     ) -> FeosResult<ThermalConductivity> {
//         self.residual
//             .thermal_conductivity_reference(temperature, volume, moles)
//     }
//     fn thermal_conductivity_correlation(&self, s_res: f64, x: &DVector<f64>) -> FeosResult<f64> {
//         self.residual.thermal_conductivity_correlation(s_res, x)
//     }
// }
