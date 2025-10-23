use crate::{FeosError, FeosResult, ReferenceSystem, state::StateHD};
use nalgebra::{DVector, DefaultAllocator, Dim, Dyn, OMatrix, OVector, U1, allocator::Allocator};
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

impl<C: Deref<Target = T>, T: Molarweight<N, D>, N: Dim, D: DualNum<f64> + Copy> Molarweight<N, D>
    for C
where
    DefaultAllocator: Allocator<N>,
{
    fn molar_weight(&self) -> MolarWeight<OVector<D, N>> {
        T::molar_weight(self)
    }
}

/// A model from which models for subsets of its components can be extracted.
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

/// A simple residual Helmholtz energy model for arbitrary many components
/// and no automatic differentiation of model parameters.
///
/// This is a shortcut to implementing `Residual<Dyn, f64>`. To avoid unnecessary
/// cloning, `Residual<Dyn, f64>` is automatically implemented for all pointer
/// types that deref to the struct implementing `ResidualDyn` and are `Clone`
/// (i.e., `Rc<T>`, `Arc<T>`, `&T`, ...).
pub trait ResidualDyn {
    /// Return the number of components in the system.
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
    ) -> Vec<(&'static str, D)>;
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
    ) -> Vec<(&'static str, D)> {
        ResidualDyn::reduced_helmholtz_energy_density_contributions(self.deref(), state)
    }
}

/// A residual Helmholtz energy model.
pub trait Residual<N: Dim = Dyn, D: DualNum<f64> + Copy = f64>: Clone
where
    DefaultAllocator: Allocator<N>,
{
    /// Return the number of components in the system.
    fn components(&self) -> usize;

    /// Return a generic composition vector for a pure component.
    ///
    /// Panics if N is not Dyn(1) or Const<1>.
    fn pure_molefracs() -> OVector<D, N> {
        OVector::from_element_generic(N::from_usize(1), U1, D::one())
    }

    /// The residual model with only the real parts of the model parameters.
    type Real: Residual<N>;

    /// The residual model with the model parameters lifted to a higher dual number.
    type Lifted<D2: DualNum<f64, Inner = D> + Copy>: Residual<N, D2>;

    /// Return the real part of the residual model.
    fn re(&self) -> Self::Real;

    /// Return the lifted residual model.
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
    ) -> Vec<(&'static str, D)>;

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
    ) -> Vec<(&'static str, D)> {
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

    // The following methods are used in phase equilibrium algorithms

    /// calculates a_res, p, dp_drho
    fn p_dpdrho(&self, temperature: D, density: D, molefracs: &OVector<D, N>) -> (D, D, D) {
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

    /// calculates p, dp_drho, d2p_drho2
    fn p_dpdrho_d2pdrho2(
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
        DefaultAllocator: Allocator<N, N>,
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
    {
        let (_, mu_res, a_res_v, mu_res_v) = N::partial_hessian(
            |x, v, &t| self.lift().residual_helmholtz_energy(t, v, &x),
            molefracs,
            molar_volume,
            &temperature,
        );
        let p = -a_res_v + temperature / molar_volume;
        let mu_v = mu_res_v.map(|m| m - temperature / molar_volume);
        let p_v = mu_v.dot(molefracs) / molar_volume;
        (p, mu_res, p_v, mu_v)
    }

    /// calculates dp_dt, dmu_res_dt
    fn dmu_dt(&self, temperature: D, partial_density: &OVector<D, N>) -> (D, OVector<D, N>)
    where
        N: Gradients,
    {
        let (_, _, f_res_t, mu_res_t) = N::partial_hessian(
            |rho, t, _: &()| {
                let state = StateHD::new_density(t, &rho);
                self.lift()
                    .reduced_residual_helmholtz_energy_density(&state)
                    * t
            },
            partial_density,
            temperature,
            &(),
        );
        let p_t = -f_res_t + partial_density.dot(&mu_res_t) + partial_density.sum();
        (p_t, mu_res_t)
    }
}

/// Reference values and residual entropy correlations for entropy scaling.
pub trait EntropyScaling<N: Dim = Dyn, D: DualNum<f64> + Copy = f64>
where
    DefaultAllocator: Allocator<N>,
{
    fn viscosity_reference(
        &self,
        temperature: Temperature<D>,
        volume: Volume<D>,
        moles: &Moles<OVector<D, N>>,
    ) -> Viscosity<D>;
    fn viscosity_correlation(&self, s_res: D, x: &OVector<D, N>) -> D;
    fn diffusion_reference(
        &self,
        temperature: Temperature<D>,
        volume: Volume<D>,
        moles: &Moles<OVector<D, N>>,
    ) -> Diffusivity<D>;
    fn diffusion_correlation(&self, s_res: D, x: &OVector<D, N>) -> D;
    fn thermal_conductivity_reference(
        &self,
        temperature: Temperature<D>,
        volume: Volume<D>,
        moles: &Moles<OVector<D, N>>,
    ) -> ThermalConductivity<D>;
    fn thermal_conductivity_correlation(&self, s_res: D, x: &OVector<D, N>) -> D;
}

impl<C: Deref<Target = T>, T: EntropyScaling<N, D>, N: Dim, D: DualNum<f64> + Copy>
    EntropyScaling<N, D> for C
where
    DefaultAllocator: Allocator<N>,
{
    fn viscosity_reference(
        &self,
        temperature: Temperature<D>,
        volume: Volume<D>,
        moles: &Moles<OVector<D, N>>,
    ) -> Viscosity<D> {
        self.deref().viscosity_reference(temperature, volume, moles)
    }
    fn viscosity_correlation(&self, s_res: D, x: &OVector<D, N>) -> D {
        self.deref().viscosity_correlation(s_res, x)
    }
    fn diffusion_reference(
        &self,
        temperature: Temperature<D>,
        volume: Volume<D>,
        moles: &Moles<OVector<D, N>>,
    ) -> Diffusivity<D> {
        self.deref().diffusion_reference(temperature, volume, moles)
    }
    fn diffusion_correlation(&self, s_res: D, x: &OVector<D, N>) -> D {
        self.deref().diffusion_correlation(s_res, x)
    }
    fn thermal_conductivity_reference(
        &self,
        temperature: Temperature<D>,
        volume: Volume<D>,
        moles: &Moles<OVector<D, N>>,
    ) -> ThermalConductivity<D> {
        self.deref()
            .thermal_conductivity_reference(temperature, volume, moles)
    }
    fn thermal_conductivity_correlation(&self, s_res: D, x: &OVector<D, N>) -> D {
        self.deref().thermal_conductivity_correlation(s_res, x)
    }
}

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
    ) -> Vec<(&'static str, D)> {
        vec![]
    }
}
