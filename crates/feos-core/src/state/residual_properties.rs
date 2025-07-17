use super::{Contributions, Derivative::*, PartialDerivative, State};
use crate::equation_of_state::{Molarweight, Residual};
use crate::errors::FeosResult;
use crate::phase_equilibria::PhaseEquilibrium;
use crate::state::cache::Cache;
use crate::state::critical_point::kronecker;
use crate::{ReferenceSystem, StateGeneric, StateHD};
use nalgebra::SVector;
use ndarray::{Array1, Array2, arr1};
use num_dual::linalg::smallest_ev;
use num_dual::{
    Dual, Dual2, Dual3, Dual64, DualNum, DualSVec, DualSVec64, HyperDual, second_derivative,
    third_derivative,
};
use num_traits::{One, Zero};
use quantity::*;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::{Arc, Mutex};
use typenum::P2;

type DpDn<T> = Quantity<T, <_Pressure as Sub<_Moles>>::Output>;

/// # State properties
impl<E: Residual> State<E> {
    pub(super) fn get_or_compute_derivative_residual(&self, derivative: PartialDerivative) -> f64 {
        let mut cache = self.cache.lock().unwrap();

        match derivative {
            PartialDerivative::Zeroth => {
                let new_state = self.derive0();
                let computation =
                    || self.eos.residual_helmholtz_energy(&new_state) * new_state.temperature;
                cache.get_or_insert_with_f64(computation)
            }
            PartialDerivative::First(v) => {
                let new_state = self.derive1(v);
                let computation =
                    || self.eos.residual_helmholtz_energy(&new_state) * new_state.temperature;
                cache.get_or_insert_with_d64(v, computation)
            }
            PartialDerivative::Second(v) => {
                let new_state = self.derive2(v);
                let computation =
                    || self.eos.residual_helmholtz_energy(&new_state) * new_state.temperature;
                cache.get_or_insert_with_d2_64(v, computation)
            }
            PartialDerivative::SecondMixed(v1, v2) => {
                let new_state = self.derive2_mixed(v1, v2);
                let computation =
                    || self.eos.residual_helmholtz_energy(&new_state) * new_state.temperature;
                cache.get_or_insert_with_hd64(v1, v2, computation)
            }
            PartialDerivative::Third(v) => {
                let new_state = self.derive3(v);
                let computation =
                    || self.eos.residual_helmholtz_energy(&new_state) * new_state.temperature;
                cache.get_or_insert_with_hd364(v, computation)
            }
        }
    }
}

impl<E: HelmholtzEnergyDerivatives<D>, D: DualNum<f64> + Copy>
    StateGeneric<E, D, E::Molefracs, E::Cache>
{
    fn contributions<T: Add<T, Output = T>, U>(
        ideal_gas: Quantity<T, U>,
        residual: Quantity<T, U>,
        contributions: Contributions,
    ) -> Quantity<T, U> {
        match contributions {
            Contributions::IdealGas => ideal_gas,
            Contributions::Total => ideal_gas + residual,
            Contributions::Residual => residual,
        }
    }
}

pub trait HelmholtzEnergyDerivatives<D: DualNum<f64> + Copy>: Clone {
    // cache
    type Cache;
    fn new_cache(&self) -> Self::Cache;

    // AD
    type Real: HelmholtzEnergyDerivatives<f64>;
    fn re(&self) -> Self::Real;

    // generic data types for components
    type Molefracs: Clone
        + Mul<D, Output = Self::Molefracs>
        + Div<D, Output = Self::Molefracs>
        + Sub<Output = Self::Molefracs>
        + Neg<Output = Self::Molefracs>;
    fn pure_molefracs() -> Self::Molefracs;
    fn molefracs_re(
        molefracs: &Self::Molefracs,
    ) -> <Self::Real as HelmholtzEnergyDerivatives<f64>>::Molefracs;
    // fn extract_molefracs(moles: &Moles<Self::Molefracs>) -> (Moles<D>, Self::Molefracs);
    fn iter_molefracs(molefracs: &Self::Molefracs) -> impl Iterator<Item = D>;
    fn compute_max_density(&self, molefracs: &Self::Molefracs) -> D;

    // helmholtz energy and derivatives
    fn _residual_helmholtz_energy(state: &StateGeneric<Self, D, Self::Molefracs, Self::Cache>)
    -> D;
    fn _residual_entropy(state: &StateGeneric<Self, D, Self::Molefracs, Self::Cache>) -> D;
    fn _residual_pressure(state: &StateGeneric<Self, D, Self::Molefracs, Self::Cache>) -> D;
    fn _residual_chemical_potential(
        state: &StateGeneric<Self, D, Self::Molefracs, Self::Cache>,
    ) -> Self::Molefracs;
    fn _dp_res_dv(state: &StateGeneric<Self, D, Self::Molefracs, Self::Cache>) -> D;
    fn _dp_res_dt(state: &StateGeneric<Self, D, Self::Molefracs, Self::Cache>) -> D;
    fn _dp_dn(state: &StateGeneric<Self, D, Self::Molefracs, Self::Cache>) -> Self::Molefracs;
    fn _d2p_res_dv2(state: &StateGeneric<Self, D, Self::Molefracs, Self::Cache>) -> D;
    fn _ds_res_dt(state: &StateGeneric<Self, D, Self::Molefracs, Self::Cache>) -> D;
    fn _d2s_res_dt2(state: &StateGeneric<Self, D, Self::Molefracs, Self::Cache>) -> D;

    // special uncached functions for phase equilibria
    fn _residual_molar_helmholtz_energy<D2: DualNum<f64, Inner = D> + Copy>(
        &self,
        temperature: D2,
        molar_volume: D2,
        molefracs: &Self::Molefracs,
    ) -> D2;
    fn _p_dpdrho(&self, temperature: D, density: D, molefracs: &Self::Molefracs) -> (D, D, D) {
        let molar_volume = density.recip();
        let t = Dual2::from_inner(temperature);
        let (a, da, d2a) = second_derivative(
            |molar_volume| self._residual_molar_helmholtz_energy(t, molar_volume, molefracs),
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
        molefracs: &Self::Molefracs,
    ) -> (D, D, D) {
        let molar_volume = density.recip();
        let t = Dual3::from_inner(temperature);
        let (_, da, d2a, d3a) = third_derivative(
            |molar_volume| self._residual_molar_helmholtz_energy(t, molar_volume, molefracs),
            molar_volume,
        );
        (
            -da + temperature * density,
            molar_volume * molar_volume * d2a + temperature,
            -molar_volume * molar_volume * molar_volume * (d2a * 2.0 + molar_volume * d3a),
        )
    }

    // critical points and spinodals
    fn stability_condition(
        &self,
        temperature: Dual<D, f64>,
        density: Dual<D, f64>,
        molefracs: &Self::Molefracs,
    ) -> Dual<D, f64>;
    fn criticality_conditions(
        &self,
        temperature: DualSVec<D, f64, 2>,
        density: DualSVec<D, f64, 2>,
        molefracs: &Self::Molefracs,
    ) -> SVector<DualSVec<D, f64, 2>, 2>;
}

impl<E: Residual> HelmholtzEnergyDerivatives<f64> for Arc<E> {
    type Cache = Mutex<Cache>;
    fn new_cache(&self) -> Self::Cache {
        Mutex::new(Cache::with_capacity(self.components()))
    }

    type Real = Self;
    fn re(&self) -> Self {
        self.clone()
    }

    type Molefracs = Array1<f64>;
    fn pure_molefracs() -> Array1<f64> {
        arr1(&[1.0])
    }
    fn molefracs_re(molefracs: &Array1<f64>) -> Array1<f64> {
        molefracs.clone()
    }
    fn iter_molefracs(molefracs: &Array1<f64>) -> impl Iterator<Item = f64> {
        molefracs.iter().copied()
    }
    fn compute_max_density(&self, molefracs: &Array1<f64>) -> f64 {
        Residual::compute_max_density(self as &E, molefracs)
    }

    fn _residual_helmholtz_energy(state: &State<E>) -> f64 {
        state.get_or_compute_derivative_residual(PartialDerivative::Zeroth)
    }

    fn _residual_entropy(state: &State<E>) -> f64 {
        -state.get_or_compute_derivative_residual(PartialDerivative::First(DT))
    }

    fn _residual_pressure(state: &State<E>) -> f64 {
        -state.get_or_compute_derivative_residual(PartialDerivative::First(DV))
    }

    fn _residual_chemical_potential(state: &State<E>) -> Array1<f64> {
        Array1::from_shape_fn(state.eos.components(), |i| {
            state.get_or_compute_derivative_residual(PartialDerivative::First(DN(i)))
        })
    }

    fn _dp_res_dv(state: &State<E>) -> f64 {
        -state.get_or_compute_derivative_residual(PartialDerivative::Second(DV))
    }

    fn _dp_res_dt(state: &State<E>) -> f64 {
        -state.get_or_compute_derivative_residual(PartialDerivative::SecondMixed(DV, DT))
    }

    fn _dp_dn(state: &State<E>) -> Array1<f64> {
        Array1::from_shape_fn(state.eos.components(), |i| {
            -state.get_or_compute_derivative_residual(PartialDerivative::SecondMixed(DV, DN(i)))
                + (state.temperature / state.volume * RGAS).into_reduced()
        })
    }

    fn _d2p_res_dv2(state: &State<E>) -> f64 {
        -state.get_or_compute_derivative_residual(PartialDerivative::Third(DV))
    }

    fn _ds_res_dt(state: &State<E>) -> f64 {
        -state.get_or_compute_derivative_residual(PartialDerivative::Second(DT))
    }

    fn _d2s_res_dt2(state: &State<E>) -> f64 {
        -state.get_or_compute_derivative_residual(PartialDerivative::Third(DT))
    }

    fn _residual_molar_helmholtz_energy<D: DualNum<f64> + Copy>(
        &self,
        temperature: D,
        molar_volume: D,
        molefracs: &Array1<f64>,
    ) -> D {
        let x = molefracs.mapv(D::from);
        let state = StateHD::new(temperature, molar_volume, x);
        self.residual_helmholtz_energy(&state) * temperature
    }

    fn stability_condition(
        &self,
        temperature: Dual64,
        density: Dual64,
        molefracs: &Array1<f64>,
    ) -> Dual64 {
        // calculate second partial derivatives w.r.t. moles
        let t = HyperDual::from_re(temperature);
        let v = HyperDual::from_re(density.recip());
        let qij = Array2::from_shape_fn((self.components(), self.components()), |(i, j)| {
            let mut m = molefracs.mapv(HyperDual::from);
            m[i].eps1 = Dual64::one();
            m[j].eps2 = Dual64::one();
            let state = StateHD::new(t, v, m);
            self.residual_helmholtz_energy(&state).eps1eps2 * (molefracs[i] * molefracs[j]).sqrt()
                + kronecker(i, j)
        });

        // calculate smallest eigenvalue of q
        let (eval, _) = smallest_ev(qij);

        eval
    }

    fn criticality_conditions(
        &self,
        temperature: DualSVec64<2>,
        density: DualSVec64<2>,
        molefracs: &Array1<f64>,
    ) -> SVector<DualSVec64<2>, 2> {
        // calculate second partial derivatives w.r.t. moles
        let t = HyperDual::from_re(temperature);
        let v = HyperDual::from_re(density.recip());
        let qij = Array2::from_shape_fn((self.components(), self.components()), |(i, j)| {
            let mut m = molefracs.mapv(HyperDual::from);
            m[i].eps1 = DualSVec64::one();
            m[j].eps2 = DualSVec64::one();
            let state = StateHD::new(t, v, m);
            self.residual_helmholtz_energy(&state).eps1eps2 * (molefracs[i] * molefracs[j]).sqrt()
                + kronecker(i, j)
        });

        // calculate smallest eigenvalue and corresponding eigenvector of q
        let (eval, evec) = smallest_ev(qij);

        // evaluate third partial derivative w.r.t. s
        let molefracs_hd = Array1::from_shape_fn(self.components(), |i| {
            Dual3::new(
                DualSVec64::from(molefracs[i]),
                evec[i] * molefracs[i].sqrt(),
                DualSVec64::zero(),
                DualSVec64::zero(),
            )
        });
        let state_s = StateHD::new(
            Dual3::from_re(temperature),
            Dual3::from_re(density.recip()),
            molefracs_hd,
        );
        let ig = (&state_s.moles * (state_s.partial_density.mapv(|x| x.ln()) - 1.0)).sum();
        let res = self.residual_helmholtz_energy(&state_s);
        SVector::from([eval, (res + ig).v3])
    }
}

impl<E: HelmholtzEnergyDerivatives<D>, D: DualNum<f64> + Copy>
    StateGeneric<E, D, E::Molefracs, E::Cache>
{
    /// Residual Helmholtz energy $A^\text{res}$
    pub fn residual_helmholtz_energy(&self) -> Energy<D> {
        Energy::from_reduced(E::_residual_helmholtz_energy(self))
    }

    /// Residual molar Helmholtz energy $a^\text{res}$
    pub fn residual_molar_helmholtz_energy(&self) -> MolarEnergy<D> {
        self.residual_helmholtz_energy() / self.total_moles
    }
}

impl<E: Residual> State<E> {
    /// Residual Helmholtz energy $A^\text{res}$ evaluated for each contribution of the equation of state.
    pub fn residual_helmholtz_energy_contributions(&self) -> Vec<(String, Energy)> {
        let new_state = self.derive0();
        let residual_contributions = self.eos.residual_helmholtz_energy_contributions(&new_state);
        let mut res = Vec::with_capacity(residual_contributions.len());
        for (s, v) in residual_contributions {
            res.push((s, Energy::from_reduced(v * new_state.temperature)));
        }
        res
    }
}

impl<E: HelmholtzEnergyDerivatives<D>, D: DualNum<f64> + Copy>
    StateGeneric<E, D, E::Molefracs, E::Cache>
{
    /// Residual entropy $S^\text{res}=\left(\frac{\partial A^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn residual_entropy(&self) -> Entropy<D> {
        Entropy::from_reduced(E::_residual_entropy(self))
    }

    /// Residual entropy $s^\text{res}=\left(\frac{\partial a^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn residual_molar_entropy(&self) -> MolarEntropy<D> {
        self.residual_entropy() / self.total_moles
    }

    /// Pressure: $p=-\left(\frac{\partial A}{\partial V}\right)_{T,N_i}$
    pub fn pressure(&self, contributions: Contributions) -> Pressure<D> {
        let ideal_gas = self.density * RGAS * self.temperature;
        let residual = Pressure::from_reduced(E::_residual_pressure(self));
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Residual chemical potential: $\mu_i^\text{res}=\left(\frac{\partial A^\text{res}}{\partial N_i}\right)_{T,V,N_j}$
    pub fn residual_chemical_potential(&self) -> MolarEnergy<E::Molefracs> {
        MolarEnergy::new(
            E::_residual_chemical_potential(self) * D::from(MolarEnergy::<f64>::FACTOR),
        )
    }
}

impl<E: Residual> State<E> {
    /// Chemical potential $\mu_i^\text{res}$ evaluated for each contribution of the equation of state.
    pub fn residual_chemical_potential_contributions(
        &self,
        component: usize,
    ) -> Vec<(String, MolarEnergy)> {
        let new_state = self.derive1(DN(component));
        let contributions = self.eos.residual_helmholtz_energy_contributions(&new_state);
        let mut res = Vec::with_capacity(contributions.len());
        for (s, v) in contributions {
            res.push((
                s,
                MolarEnergy::from_reduced((v * new_state.temperature).eps),
            ));
        }
        res
    }
}

impl<E: HelmholtzEnergyDerivatives<D>, D: DualNum<f64> + Copy>
    StateGeneric<E, D, E::Molefracs, E::Cache>
{
    /// Compressibility factor: $Z=\frac{pV}{NRT}$
    pub fn compressibility(&self, contributions: Contributions) -> D {
        (self.pressure(contributions) / (self.density * self.temperature * RGAS)).into_value()
    }

    // pressure derivatives

    /// Partial derivative of pressure w.r.t. volume: $\left(\frac{\partial p}{\partial V}\right)_{T,N_i}$
    pub fn dp_dv(&self, contributions: Contributions) -> <Pressure<D> as Div<Volume<D>>>::Output {
        let ideal_gas = -self.density * RGAS * self.temperature / self.volume;
        let residual = Quantity::from_reduced(E::_dp_res_dv(self));
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Partial derivative of pressure w.r.t. density: $\left(\frac{\partial p}{\partial \rho}\right)_{T,N_i}$
    pub fn dp_drho(
        &self,
        contributions: Contributions,
    ) -> <Pressure<D> as Div<Density<D>>>::Output {
        -self.volume / self.density * self.dp_dv(contributions)
    }

    /// Partial derivative of pressure w.r.t. temperature: $\left(\frac{\partial p}{\partial T}\right)_{V,N_i}$
    pub fn dp_dt(
        &self,
        contributions: Contributions,
    ) -> <Pressure<D> as Div<Temperature<D>>>::Output {
        let ideal_gas = self.density * RGAS;
        let residual = Quantity::from_reduced(E::_dp_res_dt(self));
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Partial derivative of pressure w.r.t. moles: $\left(\frac{\partial p}{\partial N_i}\right)_{T,V,N_j}$
    pub fn dp_dni(&self) -> DpDn<E::Molefracs> {
        Quantity::new(E::_dp_dn(self) * D::from(DpDn::<f64>::FACTOR))
    }

    /// Second partial derivative of pressure w.r.t. volume: $\left(\frac{\partial^2 p}{\partial V^2}\right)_{T,N_j}$
    pub fn d2p_dv2(
        &self,
        contributions: Contributions,
    ) -> <<Pressure<D> as Div<Volume<D>>>::Output as Div<Volume<D>>>::Output {
        let ideal_gas = self.density * RGAS * self.temperature / (self.volume * self.volume) * 2.0;
        let residual = Quantity::from_reduced(E::_d2p_res_dv2(self));
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Second partial derivative of pressure w.r.t. density: $\left(\frac{\partial^2 p}{\partial \rho^2}\right)_{T,N_j}$
    pub fn d2p_drho2(
        &self,
        contributions: Contributions,
    ) -> <<Pressure<D> as Div<Density<D>>>::Output as Div<Density<D>>>::Output {
        self.volume / (self.density * self.density)
            * (self.volume * self.d2p_dv2(contributions) + self.dp_dv(contributions) * 2.0)
    }

    /// Structure factor: $S(0)=k_BT\left(\frac{\partial\rho}{\partial p}\right)_{T,N_i}$
    pub fn structure_factor(&self) -> D {
        -(self.temperature * self.density * RGAS / (self.volume * self.dp_dv(Contributions::Total)))
            .into_value()
    }

    // This function is designed specifically for use in density iterations
    pub(crate) fn p_dpdrho(&self) -> (Pressure<D>, <Pressure<D> as Div<Density<D>>>::Output) {
        let dp_dv = self.dp_dv(Contributions::Total);
        (
            self.pressure(Contributions::Total),
            (-self.volume * dp_dv / self.density),
        )
    }
}

impl<E: HelmholtzEnergyDerivatives<D>, D: DualNum<f64> + Copy>
    StateGeneric<E, D, E::Molefracs, E::Cache>
{
    /// Partial molar volume: $v_i=\left(\frac{\partial V}{\partial N_i}\right)_{T,p,N_j}$
    pub fn partial_molar_volume(&self) -> MolarVolume<E::Molefracs> {
        -self.dp_dni() / self.dp_dv(Contributions::Total)
    }
}

impl<E: Residual> State<E> {
    /// Partial derivative of chemical potential w.r.t. moles: $\left(\frac{\partial\mu_i}{\partial N_j}\right)_{T,V,N_k}$
    pub fn dmu_dni(
        &self,
        contributions: Contributions,
    ) -> <MolarEnergy<Array2<f64>> as Div<Moles>>::Output {
        let n = self.eos.components();
        Quantity::from_shape_fn((n, n), |(i, j)| {
            let ideal_gas = if i == j {
                RGAS * self.temperature / (self.molefracs[i] * self.total_moles)
            } else {
                Quantity::from_reduced(0.0)
            };
            let residual =
                Quantity::from_reduced(self.get_or_compute_derivative_residual(
                    PartialDerivative::SecondMixed(DN(i), DN(j)),
                ));
            Self::contributions(ideal_gas, residual, contributions)
        })
    }
}

impl<E: Residual> State<E> {
    /// Isothermal compressibility: $\kappa_T=-\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_{T,N_i}$
    pub fn isothermal_compressibility(&self) -> <f64 as Div<Pressure>>::Output {
        -1.0 / (self.dp_dv(Contributions::Total) * self.volume)
    }

    /// Pressure $p$ evaluated for each contribution of the equation of state.
    pub fn pressure_contributions(&self) -> Vec<(String, Pressure)> {
        let new_state = self.derive1(DV);
        let contributions = self.eos.residual_helmholtz_energy_contributions(&new_state);
        let mut res = Vec::with_capacity(contributions.len() + 1);
        res.push(("Ideal gas".into(), self.density * RGAS * self.temperature));
        for (s, v) in contributions {
            res.push((s, Pressure::from_reduced(-(v * new_state.temperature).eps)));
        }
        res
    }
}

impl<E: HelmholtzEnergyDerivatives<D>, D: DualNum<f64> + Copy>
    StateGeneric<E, D, E::Molefracs, E::Cache>
{
    // entropy derivatives

    /// Partial derivative of the residual entropy w.r.t. temperature: $\left(\frac{\partial S^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn ds_res_dt(&self) -> <Entropy<D> as Div<Temperature<D>>>::Output {
        Quantity::from_reduced(E::_ds_res_dt(self))
    }

    /// Second partial derivative of the residual entropy w.r.t. temperature: $\left(\frac{\partial^2S^\text{res}}{\partial T^2}\right)_{V,N_i}$
    pub fn d2s_res_dt2(
        &self,
    ) -> <<Entropy<D> as Div<Temperature<D>>>::Output as Div<Temperature<D>>>::Output {
        Quantity::from_reduced(E::_d2s_res_dt2(self))
    }
}

impl<E: Residual> State<E> {
    /// Partial derivative of chemical potential w.r.t. temperature: $\left(\frac{\partial\mu_i}{\partial T}\right)_{V,N_i}$
    pub fn dmu_res_dt(&self) -> <MolarEnergy<Array1<f64>> as Div<Temperature>>::Output {
        Quantity::from_reduced(Array1::from_shape_fn(self.eos.components(), |i| {
            self.get_or_compute_derivative_residual(PartialDerivative::SecondMixed(DT, DN(i)))
        }))
    }

    /// Logarithm of the fugacity coefficient: $\ln\varphi_i=\beta\mu_i^\mathrm{res}\left(T,p,\lbrace N_i\rbrace\right)$
    pub fn ln_phi(&self) -> Array1<f64> {
        (self.residual_chemical_potential() / (RGAS * self.temperature)).into_value()
            - self.compressibility(Contributions::Total).ln()
    }

    /// Logarithm of the fugacity coefficient of all components treated as pure substance at mixture temperature and pressure.
    pub fn ln_phi_pure_liquid(&self) -> FeosResult<Array1<f64>> {
        let pressure = self.pressure(Contributions::Total);
        (0..self.eos.components())
            .map(|i| {
                let eos = Arc::new(self.eos.subset(&[i]));
                let state = Self::new_npt(
                    &eos,
                    self.temperature,
                    pressure,
                    &Moles::from_reduced(arr1(&[1.0])),
                    crate::DensityInitialization::Liquid,
                )?;
                Ok(state.ln_phi()[0])
            })
            .collect()
    }

    /// Activity coefficient $\ln \gamma_i = \ln \varphi_i(T, p, \mathbf{N}) - \ln \varphi_i^\mathrm{pure}(T, p)$
    pub fn ln_symmetric_activity_coefficient(&self) -> FeosResult<Array1<f64>> {
        match self.eos.components() {
            1 => Ok(arr1(&[0.0])),
            _ => Ok(self.ln_phi() - &self.ln_phi_pure_liquid()?),
        }
    }

    /// Henry's law constant $H_{i,s}=\lim_{x_i\to 0}\frac{y_ip}{x_i}=p_s^\mathrm{sat}\frac{\varphi_i^{\infty,\mathrm{L}}}{\varphi_i^{\infty,\mathrm{V}}}$
    ///
    /// The composition of the (possibly mixed) solvent is determined by the molefracs. All components for which the composition is 0 are treated as solutes.
    pub fn henrys_law_constant(
        eos: &Arc<E>,
        temperature: Temperature,
        molefracs: &Array1<f64>,
    ) -> FeosResult<Pressure<Array1<f64>>> {
        // Calculate the phase equilibrium (bubble point) of the solvent only
        let (solvent_comps, solvent_molefracs): (Vec<_>, Vec<_>) = molefracs
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| (x != 0.0).then_some((i, x)))
            .unzip();
        let solvent_molefracs = Array1::from_vec(solvent_molefracs);
        let solvent = Arc::new(eos.subset(&solvent_comps));
        let vle = if solvent_comps.len() == 1 {
            PhaseEquilibrium::pure(&solvent, temperature, None, Default::default())
        } else {
            PhaseEquilibrium::bubble_point(
                &solvent,
                temperature,
                &solvent_molefracs,
                None,
                None,
                Default::default(),
            )
        }?;

        // Calculate the liquid state including the Henry components
        let liquid = State::new_nvt(
            eos,
            temperature,
            vle.liquid().volume,
            &(molefracs * vle.liquid().total_moles),
        )?;

        // Calculate the vapor state including the Henry components
        let mut molefracs_vapor = molefracs.clone();
        solvent_comps
            .into_iter()
            .zip(&vle.vapor().molefracs)
            .for_each(|(i, &y)| molefracs_vapor[i] = y);
        let vapor = State::new_nvt(
            eos,
            temperature,
            vle.vapor().volume,
            &(molefracs_vapor * vle.vapor().total_moles),
        )?;

        // Determine the Henry's law coefficients and return only those of the Henry components
        let p = vle.vapor().pressure(Contributions::Total);
        let h = (liquid.ln_phi() - vapor.ln_phi()).mapv(f64::exp) * p;
        Ok(h.into_iter()
            .zip(molefracs)
            .filter_map(|(h, &x)| (x == 0.0).then_some(h))
            .collect())
    }

    /// Henry's law constant $H_{i,s}=\lim_{x_i\to 0}\frac{y_ip}{x_i}=p_s^\mathrm{sat}\frac{\varphi_i^{\infty,\mathrm{L}}}{\varphi_i^{\infty,\mathrm{V}}}$ for a binary system
    ///
    /// The solute (i) is the first component and the solvent (s) the second component.
    pub fn henrys_law_constant_binary(
        eos: &Arc<E>,
        temperature: Temperature,
    ) -> FeosResult<Pressure> {
        Ok(Self::henrys_law_constant(eos, temperature, &arr1(&[0.0, 1.0]))?.get(0))
    }

    /// Partial derivative of the logarithm of the fugacity coefficient w.r.t. temperature: $\left(\frac{\partial\ln\varphi_i}{\partial T}\right)_{p,N_i}$
    pub fn dln_phi_dt(&self) -> <f64 as Div<Temperature<Array1<f64>>>>::Output {
        let vi = self.partial_molar_volume();
        (self.dmu_res_dt()
            - self.residual_chemical_potential() / self.temperature
            - vi * self.dp_dt(Contributions::Total))
            / (RGAS * self.temperature)
            + 1.0 / self.temperature
    }

    /// Partial derivative of the logarithm of the fugacity coefficient w.r.t. pressure: $\left(\frac{\partial\ln\varphi_i}{\partial p}\right)_{T,N_i}$
    pub fn dln_phi_dp(&self) -> <f64 as Div<Pressure<Array1<f64>>>>::Output {
        self.partial_molar_volume() / (RGAS * self.temperature)
            - 1.0 / self.pressure(Contributions::Total)
    }

    /// Partial derivative of the logarithm of the fugacity coefficient w.r.t. moles: $\left(\frac{\partial\ln\varphi_i}{\partial N_j}\right)_{T,p,N_k}$
    pub fn dln_phi_dnj(&self) -> <f64 as Div<Moles<Array2<f64>>>>::Output {
        let n = self.eos.components();
        let dmu_dni = self.dmu_dni(Contributions::Residual);
        let dp_dni = self.dp_dni();
        let dp_dv = self.dp_dv(Contributions::Total);
        let dp_dn_2 = Quantity::from_shape_fn((n, n), |(i, j)| dp_dni.get(i) * dp_dni.get(j));
        (dmu_dni + dp_dn_2 / dp_dv) / (RGAS * self.temperature) + 1.0 / self.total_moles
    }

    /// Thermodynamic factor: $\Gamma_{ij}=\delta_{ij}+x_i\left(\frac{\partial\ln\varphi_i}{\partial x_j}\right)_{T,p,\Sigma}$
    pub fn thermodynamic_factor(&self) -> Array2<f64> {
        let dln_phi_dnj = (self.dln_phi_dnj() * Moles::from_reduced(1.0)).into_value();
        let moles = &self.molefracs * self.total_moles.to_reduced();
        let n = self.eos.components() - 1;
        Array2::from_shape_fn((n, n), |(i, j)| {
            moles[i] * (dln_phi_dnj[[i, j]] - dln_phi_dnj[[i, n]]) + if i == j { 1.0 } else { 0.0 }
        })
    }

    /// Residual molar isochoric heat capacity: $c_v^\text{res}=\left(\frac{\partial u^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn residual_molar_isochoric_heat_capacity(&self) -> MolarEntropy {
        self.temperature * self.ds_res_dt() / self.total_moles
    }

    /// Partial derivative of the residual molar isochoric heat capacity w.r.t. temperature: $\left(\frac{\partial c_V^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn dc_v_res_dt(&self) -> <MolarEntropy as Div<Temperature>>::Output {
        (self.temperature * self.d2s_res_dt2() + self.ds_res_dt()) / self.total_moles
    }

    /// Residual molar isobaric heat capacity: $c_p^\text{res}=\left(\frac{\partial h^\text{res}}{\partial T}\right)_{p,N_i}$
    pub fn residual_molar_isobaric_heat_capacity(&self) -> MolarEntropy {
        self.temperature / self.total_moles
            * (self.ds_res_dt()
                - self.dp_dt(Contributions::Total).powi::<P2>() / self.dp_dv(Contributions::Total))
            - RGAS
    }

    /// Residual enthalpy: $H^\text{res}(T,p,\mathbf{n})=A^\text{res}+TS^\text{res}+p^\text{res}V$
    pub fn residual_enthalpy(&self) -> Energy {
        self.temperature * self.residual_entropy()
            + self.residual_helmholtz_energy()
            + self.pressure(Contributions::Residual) * self.volume
    }

    /// Residual molar enthalpy: $h^\text{res}(T,p,\mathbf{n})=a^\text{res}+Ts^\text{res}+p^\text{res}v$
    pub fn residual_molar_enthalpy(&self) -> MolarEnergy {
        self.residual_enthalpy() / self.total_moles
    }

    /// Residual internal energy: $U^\text{res}(T,V,\mathbf{n})=A^\text{res}+TS^\text{res}$
    pub fn residual_internal_energy(&self) -> Energy {
        self.temperature * self.residual_entropy() + self.residual_helmholtz_energy()
    }

    /// Residual molar internal energy: $u^\text{res}(T,V,\mathbf{n})=a^\text{res}+Ts^\text{res}$
    pub fn residual_molar_internal_energy(&self) -> MolarEnergy {
        self.residual_internal_energy() / self.total_moles
    }
}

impl<E: HelmholtzEnergyDerivatives<D>, D: DualNum<f64> + Copy>
    StateGeneric<E, D, E::Molefracs, E::Cache>
{
    /// Residual Gibbs energy: $G^\text{res}(T,p,\mathbf{n})=A^\text{res}+p^\text{res}V-NRT \ln Z$
    pub fn residual_gibbs_energy(&self) -> Energy<D> {
        self.pressure(Contributions::Residual) * self.volume + self.residual_helmholtz_energy()
            - self.total_moles
                * RGAS
                * self.temperature
                * Dimensionless::new(self.compressibility(Contributions::Total).ln())
    }

    /// Residual Gibbs energy: $g^\text{res}(T,p,\mathbf{n})=a^\text{res}+p^\text{res}v-RT \ln Z$
    pub fn residual_molar_gibbs_energy(&self) -> MolarEnergy<D> {
        self.residual_gibbs_energy() / self.total_moles
    }
}

impl<E: Residual + Molarweight> State<E> {
    /// Total molar weight: $MW=\sum_ix_iMW_i$
    pub fn total_molar_weight(&self) -> MolarWeight {
        (self.eos.molar_weight() * Dimensionless::new(&self.molefracs)).sum()
    }

    /// Mass of each component: $m_i=n_iMW_i$
    pub fn mass(&self) -> Mass<Array1<f64>> {
        self.eos.molar_weight() * Dimensionless::new(&self.molefracs) * self.total_moles
    }

    /// Total mass: $m=\sum_im_i=nMW$
    pub fn total_mass(&self) -> Mass {
        self.total_moles * self.total_molar_weight()
    }

    /// Mass density: $\rho^{(m)}=\frac{m}{V}$
    pub fn mass_density(&self) -> MassDensity {
        self.density * self.total_molar_weight()
    }

    /// Mass fractions: $w_i=\frac{m_i}{m}$
    pub fn massfracs(&self) -> Array1<f64> {
        (self.mass() / self.total_mass()).into_value()
    }
}

// /// # Transport properties
// ///
// /// These properties are available for equations of state
// /// that implement the [EntropyScaling] trait.
// impl<E: Residual + EntropyScaling> State<E> {
//     /// Return the viscosity via entropy scaling.
//     pub fn viscosity(&self) -> FeosResult<Viscosity> {
//         let s = self.residual_molar_entropy().to_reduced();
//         Ok(self
//             .eos
//             .viscosity_reference(self.temperature, self.volume, &self.moles)?
//             * self.eos.viscosity_correlation(s, &self.molefracs)?.exp())
//     }

//     /// Return the logarithm of the reduced viscosity.
//     ///
//     /// This term equals the viscosity correlation function
//     /// that is used for entropy scaling.
//     pub fn ln_viscosity_reduced(&self) -> FeosResult<f64> {
//         let s = self.residual_molar_entropy().to_reduced();
//         self.eos.viscosity_correlation(s, &self.molefracs)
//     }

//     /// Return the viscosity reference as used in entropy scaling.
//     pub fn viscosity_reference(&self) -> FeosResult<Viscosity> {
//         self.eos
//             .viscosity_reference(self.temperature, self.volume, &self.moles)
//     }

//     /// Return the diffusion via entropy scaling.
//     pub fn diffusion(&self) -> FeosResult<Diffusivity> {
//         let s = self.residual_molar_entropy().to_reduced();
//         Ok(self
//             .eos
//             .diffusion_reference(self.temperature, self.volume, &self.moles)?
//             * self.eos.diffusion_correlation(s, &self.molefracs)?.exp())
//     }

//     /// Return the logarithm of the reduced diffusion.
//     ///
//     /// This term equals the diffusion correlation function
//     /// that is used for entropy scaling.
//     pub fn ln_diffusion_reduced(&self) -> FeosResult<f64> {
//         let s = self.residual_molar_entropy().to_reduced();
//         self.eos.diffusion_correlation(s, &self.molefracs)
//     }

//     /// Return the diffusion reference as used in entropy scaling.
//     pub fn diffusion_reference(&self) -> FeosResult<Diffusivity> {
//         self.eos
//             .diffusion_reference(self.temperature, self.volume, &self.moles)
//     }

//     /// Return the thermal conductivity via entropy scaling.
//     pub fn thermal_conductivity(&self) -> FeosResult<ThermalConductivity> {
//         let s = self.residual_molar_entropy().to_reduced();
//         Ok(self
//             .eos
//             .thermal_conductivity_reference(self.temperature, self.volume, &self.moles)?
//             * self
//                 .eos
//                 .thermal_conductivity_correlation(s, &self.molefracs)?
//                 .exp())
//     }

//     /// Return the logarithm of the reduced thermal conductivity.
//     ///
//     /// This term equals the thermal conductivity correlation function
//     /// that is used for entropy scaling.
//     pub fn ln_thermal_conductivity_reduced(&self) -> FeosResult<f64> {
//         let s = self.residual_molar_entropy().to_reduced();
//         self.eos
//             .thermal_conductivity_correlation(s, &self.molefracs)
//     }

//     /// Return the thermal conductivity reference as used in entropy scaling.
//     pub fn thermal_conductivity_reference(&self) -> FeosResult<ThermalConductivity> {
//         self.eos
//             .thermal_conductivity_reference(self.temperature, self.volume, &self.moles)
//     }
// }
