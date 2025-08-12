use super::{Contributions, PartialDerivative, State, VectorPartialDerivative};
use crate::equation_of_state::{Molarweight, Total};
use crate::state::Derivative;
use crate::{ReferenceSystem, Residual};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, OVector};
use num_dual::{
    Dual, DualNum, Gradients, first_derivative, partial, partial2, second_derivative,
    second_partial_derivative, third_derivative,
};
use quantity::*;
use std::ops::{Div, Neg};

type InvP<T> = Quantity<T, <_Pressure as Neg>::Output>;
type InvT<T> = Quantity<T, <_Temperature as Neg>::Output>;

impl<E: Total<N, D>, N: Gradients, D: DualNum<f64> + Copy> State<E, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    fn get_or_compute_scalar_derivative(
        &self,
        derivative: PartialDerivative,
        contributions: Contributions,
    ) -> D {
        let residual = match contributions {
            Contributions::IdealGas => None,
            _ => Some(self.get_or_compute_scalar_derivative_residual(derivative)),
        };

        let t = self.temperature.into_reduced();
        let v = self.volume.into_reduced();
        let n = &self.molefracs * self.total_moles.into_reduced();
        let ideal_gas = match contributions {
            Contributions::Residual => None,
            _ => Some(match derivative {
                PartialDerivative::Zeroth => self.eos.ideal_gas_helmholtz_energy_0(t, v, &n),
                PartialDerivative::First(Derivative::DV) => {
                    first_derivative(
                        partial2(
                            |v, &t, n| self.eos.ideal_gas_helmholtz_energy(t, v, n),
                            &t,
                            &n,
                        ),
                        v,
                    )
                    .1
                }
                PartialDerivative::First(Derivative::DT) => {
                    first_derivative(
                        partial2(
                            |t, &v, n| self.eos.ideal_gas_helmholtz_energy(t, v, n),
                            &v,
                            &n,
                        ),
                        t,
                    )
                    .1
                }
                PartialDerivative::Second(Derivative::DV) => {
                    second_derivative(
                        partial2(
                            |v, &t, n| self.eos.ideal_gas_helmholtz_energy(t, v, n),
                            &t,
                            &n,
                        ),
                        v,
                    )
                    .2
                }
                PartialDerivative::Second(Derivative::DT) => {
                    second_derivative(
                        partial2(
                            |t, &v, n| self.eos.ideal_gas_helmholtz_energy(t, v, n),
                            &v,
                            &n,
                        ),
                        t,
                    )
                    .2
                }
                PartialDerivative::SecondMixed => {
                    second_partial_derivative(
                        partial(|(t, v), n| self.eos.ideal_gas_helmholtz_energy(t, v, n), &n),
                        (t, v),
                    )
                    .3
                }
                PartialDerivative::Third(Derivative::DV) => {
                    third_derivative(
                        partial2(
                            |v, &t, n| self.eos.ideal_gas_helmholtz_energy(t, v, n),
                            &t,
                            &n,
                        ),
                        v,
                    )
                    .3
                }
                PartialDerivative::Third(Derivative::DT) => {
                    third_derivative(
                        partial2(
                            |t, &v, n| self.eos.ideal_gas_helmholtz_energy(t, v, n),
                            &v,
                            &n,
                        ),
                        t,
                    )
                    .3
                }
            }),
        };

        match (ideal_gas, residual) {
            (Some(i), Some(r)) => i + r,
            (Some(i), None) => i,
            (None, Some(r)) => r,
            (None, None) => unreachable!(),
        }
    }

    fn get_or_compute_vector_derivative(
        &self,
        derivative: VectorPartialDerivative,
        contributions: Contributions,
    ) -> OVector<D, N> {
        let residual = match contributions {
            Contributions::IdealGas => None,
            _ => Some(self.get_or_compute_vector_derivative_residual(derivative)),
        };

        let t = self.temperature.into_reduced();
        let v = self.volume.into_reduced();
        let n = &self.molefracs * self.total_moles.into_reduced();

        let ideal_gas = match contributions {
            Contributions::Residual => None,
            _ => Some(match derivative {
                VectorPartialDerivative::First => {
                    N::gradient(
                        |n, &(t, v)| self.eos.ideal_gas_helmholtz_energy(t, v, &n),
                        &n,
                        &(t, v),
                    )
                    .1
                }
                VectorPartialDerivative::SecondMixed(Derivative::DV) => {
                    N::partial_hessian(
                        |n, v, &t| self.eos.ideal_gas_helmholtz_energy(t, v, &n),
                        &n,
                        v,
                        &t,
                    )
                    .3
                }
                VectorPartialDerivative::SecondMixed(Derivative::DT) => {
                    N::partial_hessian(
                        |n, t, &v| self.eos.ideal_gas_helmholtz_energy(t, v, &n),
                        &n,
                        t,
                        &v,
                    )
                    .3
                }
            }),
        };

        match (ideal_gas, residual) {
            (Some(i), Some(r)) => i + r,
            (Some(i), None) => i,
            (None, Some(r)) => r,
            (None, None) => unreachable!(),
        }
    }

    /// Chemical potential: $\mu_i=\left(\frac{\partial A}{\partial N_i}\right)_{T,V,N_j}$
    pub fn chemical_potential(&self, contributions: Contributions) -> MolarEnergy<OVector<D, N>> {
        MolarEnergy::new(
            self.get_or_compute_vector_derivative(VectorPartialDerivative::First, contributions)
                * D::from(MolarEnergy::<f64>::FACTOR),
        )
    }

    /// Partial derivative of chemical potential w.r.t. temperature: $\left(\frac{\partial\mu_i}{\partial T}\right)_{V,N_i}$
    pub fn dmu_dt(&self, contributions: Contributions) -> MolarEntropy<OVector<D, N>> {
        Quantity::new(
            self.get_or_compute_vector_derivative(
                VectorPartialDerivative::SecondMixed(Derivative::DT),
                contributions,
            ) * D::from(MolarEntropy::<f64>::FACTOR),
        )
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
        println!(
            "{:?} {:?}",
            self.ds_dt(contributions),
            (self.dp_dt(contributions) * self.dp_dt(contributions)) / self.dp_dv(contributions)
        );
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
        Entropy::from_reduced(-self.get_or_compute_scalar_derivative(
            PartialDerivative::First(Derivative::DT),
            contributions,
        ))
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
        Quantity::from_reduced(-self.get_or_compute_scalar_derivative(
            PartialDerivative::Second(super::Derivative::DT),
            contributions,
        ))
    }

    /// Second partial derivative of the entropy w.r.t. temperature: $\left(\frac{\partial^2 S}{\partial T^2}\right)_{V,N_i}$
    pub fn d2s_dt2(
        &self,
        contributions: Contributions,
    ) -> <<Entropy<D> as Div<Temperature<D>>>::Output as Div<Temperature<D>>>::Output {
        Quantity::from_reduced(-self.get_or_compute_scalar_derivative(
            PartialDerivative::Third(super::Derivative::DT),
            contributions,
        ))
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
        Energy::from_reduced(
            self.get_or_compute_scalar_derivative(PartialDerivative::Zeroth, contributions),
        )
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
        let v = Dual::from_re(self.temperature.into_reduced());
        let mut x = self.molefracs.map(Dual::from_re);
        x[component].eps = D::one();
        let mut res = Vec::new();
        if let Contributions::IdealGas | Contributions::Total = contributions {
            res.push((
                self.eos.ideal_gas_model().into(),
                self.eos.ideal_gas_helmholtz_energy(t, v, &x),
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
