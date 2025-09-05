use super::{Contributions, State};
use crate::equation_of_state::{Molarweight, Total};
use crate::{ReferenceSystem, Residual};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, OVector};
use num_dual::{Dual, DualNum, Gradients, partial, partial2};
use quantity::*;
use std::ops::{Div, Neg};

type InvP<T> = Quantity<T, <_Pressure as Neg>::Output>;
type InvT<T> = Quantity<T, <_Temperature as Neg>::Output>;

impl<E: Total<N, D>, N: Gradients, D: DualNum<f64> + Copy> State<E, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    /// Chemical potential: $\mu_i=\left(\frac{\partial A}{\partial N_i}\right)_{T,V,N_j}$
    pub fn chemical_potential(&self, contributions: Contributions) -> MolarEnergy<OVector<D, N>> {
        let residual = || self.residual_chemical_potential();
        let ideal_gas = || {
            quantity::ad::gradient_copy(
                partial2(
                    |n, &t, &v| self.eos.ideal_gas_helmholtz_energy(t, v, &n),
                    &self.temperature,
                    &self.volume,
                ),
                &self.moles,
            )
            .1
        };
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Partial derivative of chemical potential w.r.t. temperature: $\left(\frac{\partial\mu_i}{\partial T}\right)_{V,N_i}$
    pub fn dmu_dt(&self, contributions: Contributions) -> MolarEntropy<OVector<D, N>> {
        let residual = || self.dmu_res_dt();
        let ideal_gas = || {
            quantity::ad::partial_hessian_copy(
                partial(
                    |(n, t), &v| self.eos.ideal_gas_helmholtz_energy(t, v, &n),
                    &self.volume,
                ),
                (&self.moles, self.temperature),
            )
            .3
        };
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Molar isochoric heat capacity: $c_v=\left(\frac{\partial u}{\partial T}\right)_{V,N_i}$
    pub fn molar_isochoric_heat_capacity(&self, contributions: Contributions) -> MolarEntropy<D> {
        self.temperature * self.ds_dt(contributions) / self.total_moles
    }

    /// Partial derivative of the molar isochoric heat capacity w.r.t. temperature: $\left(\frac{\partial c_V}{\partial T}\right)_{V,N_i}$
    pub fn dc_v_dt(
        &self,
        contributions: Contributions,
    ) -> <MolarEntropy<D> as Div<Temperature<D>>>::Output {
        (self.temperature * self.d2s_dt2(contributions) + self.ds_dt(contributions))
            / self.total_moles
    }

    /// Molar isobaric heat capacity: $c_p=\left(\frac{\partial h}{\partial T}\right)_{p,N_i}$
    pub fn molar_isobaric_heat_capacity(&self, contributions: Contributions) -> MolarEntropy<D> {
        match contributions {
            Contributions::Residual => self.residual_molar_isobaric_heat_capacity(),
            _ => {
                self.temperature / self.total_moles
                    * (self.ds_dt(contributions)
                        - (self.dp_dt(contributions) * self.dp_dt(contributions))
                            / self.dp_dv(contributions))
            }
        }
    }

    /// Entropy: $S=-\left(\frac{\partial A}{\partial T}\right)_{V,N_i}$
    pub fn entropy(&self, contributions: Contributions) -> Entropy<D> {
        let residual = || self.residual_entropy();
        let ideal_gas = || {
            -quantity::ad::first_derivative(
                partial2(
                    |t, &v, n| self.eos.ideal_gas_helmholtz_energy(t, v, n),
                    &self.volume,
                    &self.moles,
                ),
                self.temperature,
            )
            .1
        };
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Molar entropy: $s=\frac{S}{N}$
    pub fn molar_entropy(&self, contributions: Contributions) -> MolarEntropy<D> {
        self.entropy(contributions) / self.total_moles
    }

    /// Partial molar entropy: $s_i=\left(\frac{\partial S}{\partial N_i}\right)_{T,p,N_j}$
    pub fn partial_molar_entropy(&self) -> MolarEntropy<OVector<D, N>> {
        let c = Contributions::Total;
        -(self.dmu_dt(c) + self.dp_dni(c) * (self.dp_dt(c) / self.dp_dv(c)))
    }

    /// Partial derivative of the entropy w.r.t. temperature: $\left(\frac{\partial S}{\partial T}\right)_{V,N_i}$
    pub fn ds_dt(
        &self,
        contributions: Contributions,
    ) -> <Entropy<D> as Div<Temperature<D>>>::Output {
        let residual = || self.ds_res_dt();
        let ideal_gas = || {
            -quantity::ad::second_derivative(
                partial2(
                    |t, &v, n| self.eos.ideal_gas_helmholtz_energy(t, v, n),
                    &self.volume,
                    &self.moles,
                ),
                self.temperature,
            )
            .2
        };
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Second partial derivative of the entropy w.r.t. temperature: $\left(\frac{\partial^2 S}{\partial T^2}\right)_{V,N_i}$
    pub fn d2s_dt2(
        &self,
        contributions: Contributions,
    ) -> <<Entropy<D> as Div<Temperature<D>>>::Output as Div<Temperature<D>>>::Output {
        let residual = || self.d2s_res_dt2();
        let ideal_gas = || {
            -quantity::ad::third_derivative(
                partial2(
                    |t, &v, n| self.eos.ideal_gas_helmholtz_energy(t, v, n),
                    &self.volume,
                    &self.moles,
                ),
                self.temperature,
            )
            .3
        };
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Enthalpy: $H=A+TS+pV$
    pub fn enthalpy(&self, contributions: Contributions) -> Energy<D> {
        self.temperature * self.entropy(contributions)
            + self.helmholtz_energy(contributions)
            + self.pressure(contributions) * self.volume
    }

    /// Molar enthalpy: $h=\frac{H}{N}$
    pub fn molar_enthalpy(&self, contributions: Contributions) -> MolarEnergy<D> {
        self.enthalpy(contributions) / self.total_moles
    }

    /// Partial molar enthalpy: $h_i=\left(\frac{\partial H}{\partial N_i}\right)_{T,p,N_j}$
    pub fn partial_molar_enthalpy(&self) -> MolarEnergy<OVector<D, N>> {
        let s = self.partial_molar_entropy();
        let mu = self.chemical_potential(Contributions::Total);
        s * self.temperature + mu
    }

    /// Helmholtz energy: $A$
    pub fn helmholtz_energy(&self, contributions: Contributions) -> Energy<D> {
        let residual = || self.residual_helmholtz_energy();
        let ideal_gas = || {
            quantity::ad::zeroth_derivative(
                partial2(
                    |t, &v, n| self.eos.ideal_gas_helmholtz_energy(t, v, n),
                    &self.volume,
                    &self.moles,
                ),
                self.temperature,
            )
        };
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Molar Helmholtz energy: $a=\frac{A}{N}$
    pub fn molar_helmholtz_energy(&self, contributions: Contributions) -> MolarEnergy<D> {
        self.helmholtz_energy(contributions) / self.total_moles
    }

    /// Internal energy: $U=A+TS$
    pub fn internal_energy(&self, contributions: Contributions) -> Energy<D> {
        self.temperature * self.entropy(contributions) + self.helmholtz_energy(contributions)
    }

    /// Molar internal energy: $u=\frac{U}{N}$
    pub fn molar_internal_energy(&self, contributions: Contributions) -> MolarEnergy<D> {
        self.internal_energy(contributions) / self.total_moles
    }

    /// Gibbs energy: $G=A+pV$
    pub fn gibbs_energy(&self, contributions: Contributions) -> Energy<D> {
        self.pressure(contributions) * self.volume + self.helmholtz_energy(contributions)
    }

    /// Molar Gibbs energy: $g=\frac{G}{N}$
    pub fn molar_gibbs_energy(&self, contributions: Contributions) -> MolarEnergy<D> {
        self.gibbs_energy(contributions) / self.total_moles
    }

    /// Joule Thomson coefficient: $\mu_{JT}=\left(\frac{\partial T}{\partial p}\right)_{H,N_i}$
    pub fn joule_thomson(&self) -> <Temperature<D> as Div<Pressure<D>>>::Output {
        let c = Contributions::Total;
        -(self.volume + self.temperature * self.dp_dt(c) / self.dp_dv(c))
            / (self.total_moles * self.molar_isobaric_heat_capacity(c))
    }

    /// Isentropic compressibility: $\kappa_s=-\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_{S,N_i}$
    pub fn isentropic_compressibility(&self) -> InvP<D> {
        let c = Contributions::Total;
        -self.molar_isochoric_heat_capacity(c)
            / (self.molar_isobaric_heat_capacity(c) * self.dp_dv(c) * self.volume)
    }

    /// Isenthalpic compressibility: $\kappa_H=-\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_{H,N_i}$
    pub fn isenthalpic_compressibility(&self) -> InvP<D> {
        self.isentropic_compressibility() * Dimensionless::new(self.grueneisen_parameter() + 1.0)
    }

    /// Thermal expansivity: $\alpha_p=-\frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_{p,N_i}$
    pub fn thermal_expansivity(&self) -> InvT<D> {
        let c = Contributions::Total;
        -self.dp_dt(c) / self.dp_dv(c) / self.volume
    }

    /// Grueneisen parameter: $\phi=V\left(\frac{\partial p}{\partial U}\right)_{V,n_i}=\frac{v}{c_v}\left(\frac{\partial p}{\partial T}\right)_{v,n_i}=\frac{\rho}{T}\left(\frac{\partial T}{\partial \rho}\right)_{s, n_i}$
    pub fn grueneisen_parameter(&self) -> D {
        let c = Contributions::Total;
        (self.dp_dt(c) / (self.molar_isochoric_heat_capacity(c) * self.density)).into_value()
    }

    /// Chemical potential $\mu_i$ evaluated for each contribution of the equation of state.
    pub fn chemical_potential_contributions(
        &self,
        component: usize,
        contributions: Contributions,
    ) -> Vec<(String, MolarEnergy<D>)> {
        let t = Dual::from_re(self.temperature.into_reduced());
        let v = Dual::from_re(self.density.into_reduced().recip());
        let mut x = self.molefracs.map(Dual::from_re);
        x[component].eps = D::one();
        let mut res = Vec::new();
        if let Contributions::IdealGas | Contributions::Total = contributions {
            res.push((
                self.eos.ideal_gas_model().into(),
                self.eos.ideal_gas_molar_helmholtz_energy(t, v, &x),
            ));
        }
        if let Contributions::Residual | Contributions::Total = contributions {
            res.extend(
                self.eos
                    .lift()
                    .molar_helmholtz_energy_contributions(t, v, &x),
            );
        }
        res.into_iter()
            .map(|(s, v)| (s, MolarEnergy::from_reduced(v.eps)))
            .collect()
    }
}

impl<E: Total<N, D> + Molarweight<N, D>, N: Gradients, D: DualNum<f64> + Copy> State<E, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    /// Specific isochoric heat capacity: $c_v^{(m)}=\frac{C_v}{m}$
    pub fn specific_isochoric_heat_capacity(
        &self,
        contributions: Contributions,
    ) -> SpecificEntropy<D> {
        self.molar_isochoric_heat_capacity(contributions) / self.total_molar_weight()
    }

    /// Specific isobaric heat capacity: $c_p^{(m)}=\frac{C_p}{m}$
    pub fn specific_isobaric_heat_capacity(
        &self,
        contributions: Contributions,
    ) -> SpecificEntropy<D> {
        self.molar_isobaric_heat_capacity(contributions) / self.total_molar_weight()
    }

    /// Specific entropy: $s^{(m)}=\frac{S}{m}$
    pub fn specific_entropy(&self, contributions: Contributions) -> SpecificEntropy<D> {
        self.molar_entropy(contributions) / self.total_molar_weight()
    }

    /// Specific enthalpy: $h^{(m)}=\frac{H}{m}$
    pub fn specific_enthalpy(&self, contributions: Contributions) -> SpecificEnergy<D> {
        self.molar_enthalpy(contributions) / self.total_molar_weight()
    }

    /// Specific Helmholtz energy: $a^{(m)}=\frac{A}{m}$
    pub fn specific_helmholtz_energy(&self, contributions: Contributions) -> SpecificEnergy<D> {
        self.molar_helmholtz_energy(contributions) / self.total_molar_weight()
    }

    /// Specific internal energy: $u^{(m)}=\frac{U}{m}$
    pub fn specific_internal_energy(&self, contributions: Contributions) -> SpecificEnergy<D> {
        self.molar_internal_energy(contributions) / self.total_molar_weight()
    }

    /// Specific Gibbs energy: $g^{(m)}=\frac{G}{m}$
    pub fn specific_gibbs_energy(&self, contributions: Contributions) -> SpecificEnergy<D> {
        self.molar_gibbs_energy(contributions) / self.total_molar_weight()
    }
}

impl<E: Total<N> + Molarweight<N>, N: Gradients> State<E, N>
where
    DefaultAllocator: Allocator<N>,
{
    /// Speed of sound: $c=\sqrt{\left(\frac{\partial p}{\partial\rho^{(m)}}\right)_{S,N_i}}$
    pub fn speed_of_sound(&self) -> Velocity {
        (self.density * self.total_molar_weight() * self.isentropic_compressibility())
            .inv()
            .sqrt()
    }
}
