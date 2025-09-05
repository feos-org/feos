use crate::{FeosError, FeosResult, ReferenceSystem, state::StateHD};
use nalgebra::{
    Const, DVector, DefaultAllocator, Dim, Dyn, OMatrix, OVector, SVector, U1, allocator::Allocator,
};
use num_dual::{DualNum, Gradients, partial, partial2, second_derivative, third_derivative};
use quantity::ad::first_derivative;
use quantity::*;
use std::ops::Deref;
use std::sync::Arc;
use typenum::Quot;

/// Molar weight of all components.
///
/// Enables calculation of (mass) specific properties.
pub trait Molarweight<N: Dim = Dyn, D: DualNum<f64> + Copy = f64>
where
    DefaultAllocator: Allocator<N>,
{
    fn molar_weight(&self) -> MolarWeight<OVector<D, N>>;
}

impl<T: Molarweight<N, D>, N: Dim, D: DualNum<f64> + Copy> Molarweight<N, D> for Arc<T>
where
    DefaultAllocator: Allocator<N>,
{
    fn molar_weight(&self) -> MolarWeight<OVector<D, N>> {
        T::molar_weight(self)
    }
}

pub trait Subset {
    /// Return a model consisting of the components
    /// contained in component_list.
    fn subset(&self, component_list: &[usize]) -> Self;
}

impl<T: Subset> Subset for Arc<T> {
    fn subset(&self, component_list: &[usize]) -> Self {
        Arc::new(T::subset(self, component_list))
    }
}

pub trait ResidualDyn {
    fn components(&self) -> usize;

    /// Return the maximum density in Angstrom^-3.
    ///
    /// This value is used as an estimate for a liquid phase for phase
    /// equilibria and other iterations. It is not explicitly meant to
    /// be a mathematical limit for the density (if those exist in the
    /// equation of state anyways).
    fn compute_max_density<D: DualNum<f64> + Copy>(&self, molefracs: &DVector<D>) -> D;

    /// Evaluate the reduced Helmholtz energy density of each individual contribution
    /// and return them together with a string representation of the contribution.
    fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)>;
}

impl<C: Deref<Target = T> + Clone, T: ResidualDyn, D: DualNum<f64> + Copy> Residual<Dyn, D> for C {
    type Real = Self;
    type Lifted<D2: DualNum<f64, Inner = D> + Copy> = Self;
    fn re(&self) -> Self::Real {
        self.clone()
    }
    fn lift<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        self.clone()
    }
    fn components(&self) -> usize {
        ResidualDyn::components(self.deref())
    }
    fn compute_max_density(&self, molefracs: &DVector<D>) -> D {
        ResidualDyn::compute_max_density(self.deref(), molefracs)
    }
    fn reduced_helmholtz_energy_density_contributions(
        &self,
        state: &StateHD<D, Dyn>,
    ) -> Vec<(String, D)> {
        ResidualDyn::reduced_helmholtz_energy_density_contributions(self.deref(), state)
    }
}

pub trait ResidualConst<const N: usize, D: DualNum<f64> + Copy>: Clone {
    const NAME: &str;
    type Real: ResidualConst<N, f64>;
    type Lifted<D2: DualNum<f64, Inner = D> + Copy>: ResidualConst<N, D2>;
    fn re(&self) -> Self::Real;
    fn lift<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2>;
    fn compute_max_density(&self, molefracs: &SVector<D, N>) -> D;
    fn reduced_residual_helmholtz_energy_density(&self, state: &StateHD<D, Const<N>>) -> D;
}

impl<T: ResidualConst<N, D>, const N: usize, D: DualNum<f64> + Copy> Residual<Const<N>, D> for T {
    fn components(&self) -> usize {
        N
    }

    type Real = T::Real;

    type Lifted<D2: DualNum<f64, Inner = D> + Copy> = T::Lifted<D2>;

    fn re(&self) -> Self::Real {
        T::re(self)
    }

    fn lift<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        T::lift(self)
    }

    fn compute_max_density(&self, molefracs: &SVector<D, N>) -> D {
        T::compute_max_density(self, molefracs)
    }

    fn reduced_helmholtz_energy_density_contributions(
        &self,
        state: &StateHD<D, Const<N>>,
    ) -> Vec<(String, D)> {
        vec![(
            T::NAME.into(),
            T::reduced_residual_helmholtz_energy_density(self, state),
        )]
    }

    fn reduced_residual_helmholtz_energy_density(&self, state: &StateHD<D, Const<N>>) -> D {
        T::reduced_residual_helmholtz_energy_density(self, state)
    }
}

/// A residual Helmholtz energy model.
pub trait Residual<N: Dim = Dyn, D: DualNum<f64> + Copy = f64>: Clone
where
    DefaultAllocator: Allocator<N>,
{
    fn components(&self) -> usize;
    fn pure_molefracs() -> OVector<D, N> {
        OVector::from_element_generic(N::from_usize(1), U1, D::one())
    }

    type Real: Residual<N>;
    type Lifted<D2: DualNum<f64, Inner = D> + Copy>: Residual<N, D2>;
    fn re(&self) -> Self::Real;
    fn lift<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2>;

    /// Return the maximum density in Angstrom^-3.
    ///
    /// This value is used as an estimate for a liquid phase for phase
    /// equilibria and other iterations. It is not explicitly meant to
    /// be a mathematical limit for the density (if those exist in the
    /// equation of state anyways).
    fn compute_max_density(&self, molefracs: &OVector<D, N>) -> D;

    /// Evaluate the reduced Helmholtz energy density of each individual contribution
    /// and return them together with a string representation of the contribution.
    fn reduced_helmholtz_energy_density_contributions(
        &self,
        state: &StateHD<D, N>,
    ) -> Vec<(String, D)>;

    /// Evaluate the residual reduced Helmholtz energy density $\beta f^\mathrm{res}$.
    fn reduced_residual_helmholtz_energy_density(&self, state: &StateHD<D, N>) -> D {
        self.reduced_helmholtz_energy_density_contributions(state)
            .iter()
            .fold(D::zero(), |acc, (_, a)| acc + a)
    }

    /// Evaluate the molar Helmholtz energy of each individual contribution
    /// and return them together with a string representation of the contribution.
    fn molar_helmholtz_energy_contributions(
        &self,
        temperature: D,
        molar_volume: D,
        molefracs: &OVector<D, N>,
    ) -> Vec<(String, D)> {
        let state = StateHD::new(temperature, molar_volume, molefracs);
        self.reduced_helmholtz_energy_density_contributions(&state)
            .into_iter()
            .map(|(n, f)| (n, f * temperature * molar_volume))
            .collect()
    }

    /// Evaluate the residual molar Helmholtz energy $a^\mathrm{res}$.
    fn residual_molar_helmholtz_energy(
        &self,
        temperature: D,
        molar_volume: D,
        molefracs: &OVector<D, N>,
    ) -> D {
        let state = StateHD::new(temperature, molar_volume, molefracs);
        self.reduced_residual_helmholtz_energy_density(&state) * temperature * molar_volume
    }

    /// Evaluate the residual Helmholtz energy $A^\mathrm{res}$.
    fn residual_helmholtz_energy(&self, temperature: D, volume: D, moles: &OVector<D, N>) -> D {
        let state = StateHD::new_density(temperature, &(moles / volume));
        self.reduced_residual_helmholtz_energy_density(&state) * temperature * volume
    }

    /// Evaluate the residual Helmholtz energy $A^\mathrm{res}$.
    fn residual_helmholtz_energy_unit(
        &self,
        temperature: Temperature<D>,
        volume: Volume<D>,
        moles: &Moles<OVector<D, N>>,
    ) -> Energy<D> {
        let temperature = temperature.into_reduced();
        let total_moles = moles.sum();
        let molar_volume = (volume / total_moles).into_reduced();
        let molefracs = moles / total_moles;
        let state = StateHD::new(temperature, molar_volume, &molefracs);
        Pressure::from_reduced(self.reduced_residual_helmholtz_energy_density(&state) * temperature)
            * volume
    }

    /// Check if the provided optional molar concentration is consistent with the
    /// equation of state.
    ///
    /// In general, the number of elements in `molefracs` needs to match the number
    /// of components of the equation of state. For a pure component, however,
    /// no molefracs need to be provided.
    fn validate_molefracs(&self, molefracs: &Option<OVector<D, N>>) -> FeosResult<OVector<D, N>> {
        let l = molefracs.as_ref().map_or(1, |m| m.len());
        if self.components() == l {
            match molefracs {
                Some(m) => Ok(m.clone()),
                None => Ok(OVector::from_element_generic(
                    N::from_usize(1),
                    U1,
                    D::one(),
                )),
            }
        } else {
            Err(FeosError::IncompatibleComponents(self.components(), l))
        }
    }

    /// Calculate the maximum density.
    ///
    /// This value is used as an estimate for a liquid phase for phase
    /// equilibria and other iterations. It is not explicitly meant to
    /// be a mathematical limit for the density (if those exist in the
    /// equation of state anyways).
    fn max_density(&self, molefracs: &Option<OVector<D, N>>) -> FeosResult<Density<D>> {
        let x = self.validate_molefracs(molefracs)?;
        Ok(Density::from_reduced(self.compute_max_density(&x)))
    }

    /// Calculate the second virial coefficient $B(T)$
    fn second_virial_coefficient(
        &self,
        temperature: Temperature<D>,
        molefracs: &Option<OVector<D, N>>,
    ) -> MolarVolume<D> {
        let x = self.validate_molefracs(molefracs).unwrap();
        let (_, _, d2f) = second_derivative(
            partial2(
                |rho, &t, x| {
                    let state = StateHD::new_virial(t, rho, x);
                    self.lift()
                        .reduced_residual_helmholtz_energy_density(&state)
                },
                &temperature.into_reduced(),
                &x,
            ),
            D::from(0.0),
        );

        Quantity::from_reduced(d2f * 0.5)
    }

    /// Calculate the third virial coefficient $C(T)$
    fn third_virial_coefficient(
        &self,
        temperature: Temperature<D>,
        molefracs: &Option<OVector<D, N>>,
    ) -> Quot<MolarVolume<D>, Density<D>> {
        let x = self.validate_molefracs(molefracs).unwrap();
        let (_, _, _, d3f) = third_derivative(
            partial2(
                |rho, &t, x| {
                    let state = StateHD::new_virial(t, rho, x);
                    self.lift()
                        .reduced_residual_helmholtz_energy_density(&state)
                },
                &temperature.into_reduced(),
                &x,
            ),
            D::from(0.0),
        );

        Quantity::from_reduced(d3f / 3.0)
    }

    /// Calculate the temperature derivative of the second virial coefficient $B'(T)$
    fn second_virial_coefficient_temperature_derivative(
        &self,
        temperature: Temperature<D>,
        molefracs: &Option<OVector<D, N>>,
    ) -> Quot<MolarVolume<D>, Temperature<D>> {
        let (_, db_dt) = first_derivative(
            partial(
                |t, x| self.lift().second_virial_coefficient(t, x),
                molefracs,
            ),
            temperature,
        );
        db_dt
    }

    /// Calculate the temperature derivative of the third virial coefficient $C'(T)$
    fn third_virial_coefficient_temperature_derivative(
        &self,
        temperature: Temperature<D>,
        molefracs: &Option<OVector<D, N>>,
    ) -> Quot<Quot<MolarVolume<D>, Density<D>>, Temperature<D>> {
        let (_, dc_dt) = first_derivative(
            partial(|t, x| self.lift().third_virial_coefficient(t, x), molefracs),
            temperature,
        );
        dc_dt
    }

    fn _p_dpdrho(&self, temperature: D, density: D, molefracs: &OVector<D, N>) -> (D, D, D) {
        let molar_volume = density.recip();
        let (a, da, d2a) = second_derivative(
            partial2(
                |molar_volume, &t, x| {
                    self.lift()
                        .residual_molar_helmholtz_energy(t, molar_volume, x)
                },
                &temperature,
                molefracs,
            ),
            molar_volume,
        );
        (
            a * density,
            -da + temperature * density,
            molar_volume * molar_volume * d2a + temperature,
        )
    }

    fn _p_dpdrho_d2pdrho2(
        &self,
        temperature: D,
        density: D,
        molefracs: &OVector<D, N>,
    ) -> (D, D, D) {
        let molar_volume = density.recip();
        let (_, da, d2a, d3a) = third_derivative(
            partial2(
                |molar_volume, &t, x| {
                    self.lift()
                        .residual_molar_helmholtz_energy(t, molar_volume, x)
                },
                &temperature,
                molefracs,
            ),
            molar_volume,
        );
        (
            -da + temperature * density,
            molar_volume * molar_volume * d2a + temperature,
            -molar_volume * molar_volume * molar_volume * (d2a * 2.0 + molar_volume * d3a),
        )
    }

    /// calculates p, mu_res, dp_drho, dmu_drho
    #[expect(clippy::type_complexity)]
    fn dmu_drho(
        &self,
        temperature: D,
        partial_density: &OVector<D, N>,
    ) -> (D, OVector<D, N>, OVector<D, N>, OMatrix<D, N, N>)
    where
        N: Gradients,
        DefaultAllocator: Allocator<N> + Allocator<N, N>,
    {
        let (f_res, mu_res, dmu_res) = N::hessian(
            |rho, &t| {
                let state = StateHD::new_density(t, &rho);
                self.lift()
                    .reduced_residual_helmholtz_energy_density(&state)
                    * t
            },
            partial_density,
            &temperature,
        );
        let p = mu_res.dot(partial_density) - f_res + temperature * partial_density.sum();
        let dmu = dmu_res + OMatrix::from_diagonal(&partial_density.map(|d| temperature / d));
        let dp = &dmu * partial_density;
        (p, mu_res, dp, dmu)
    }

    /// calculates p, mu_res, dp_dv, dmu_dv
    fn dmu_dv(
        &self,
        temperature: D,
        molar_volume: D,
        molefracs: &OVector<D, N>,
    ) -> (D, OVector<D, N>, D, OVector<D, N>)
    where
        N: Gradients,
        DefaultAllocator: Allocator<N> + Allocator<N, N>,
    {
        let (_, mu_res, a_res_v, mu_res_v) = N::partial_hessian(
            |x, v, &t| self.lift().residual_molar_helmholtz_energy(t, v, &x),
            molefracs,
            molar_volume,
            &temperature,
        );
        let p = -a_res_v + temperature / molar_volume;
        let mu_v = mu_res_v.map(|m| m - temperature / molar_volume);
        let p_v = mu_v.dot(molefracs) / molar_volume;
        (p, mu_res, p_v, mu_v)
    }
}

// /// Reference values and residual entropy correlations for entropy scaling.
// pub trait EntropyScaling {
//     fn viscosity_reference(
//         parameters: &Self::Parameters<D>,
//         temperature: Temperature,
//         volume: Volume,
//         moles: &Moles<DVector<f64>>,
//     ) -> FeosResult<Viscosity>;
//     fn viscosity_correlation(&self, s_res: f64, x: &DVector<f64>) -> FeosResult<f64>;
//     fn diffusion_reference(
//         parameters: &Self::Parameters<D>,
//         temperature: Temperature,
//         volume: Volume,
//         moles: &Moles<DVector<f64>>,
//     ) -> FeosResult<Diffusivity>;
//     fn diffusion_correlation(&self, s_res: f64, x: &DVector<f64>) -> FeosResult<f64>;
//     fn thermal_conductivity_reference(
//         parameters: &Self::Parameters<D>,
//         temperature: Temperature,
//         volume: Volume,
//         moles: &Moles<DVector<f64>>,
//     ) -> FeosResult<ThermalConductivity>;
//     fn thermal_conductivity_correlation(&self, s_res: f64, x: &DVector<f64>) -> FeosResult<f64>;
// }

/// Dummy implementation for [EquationOfState](super::EquationOfState)s that only contain an ideal gas contribution.
pub struct NoResidual(pub usize);

impl Subset for NoResidual {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self(component_list.len())
    }
}

impl ResidualDyn for NoResidual {
    fn components(&self) -> usize {
        self.0
    }

    fn compute_max_density<D: DualNum<f64> + Copy>(&self, _: &DVector<D>) -> D {
        D::one()
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy>(
        &self,
        _: &StateHD<D>,
    ) -> Vec<(String, D)> {
        vec![]
    }
}
