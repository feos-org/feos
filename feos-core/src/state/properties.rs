use std::ops::Div;

use super::{Contributions, Derivative::*, PartialDerivative, State};
use crate::equation_of_state::{IdealGas, Residual};
use crate::si::*;
use ndarray::Array1;
use typenum::P2;

impl<E: Residual + IdealGas> State<E> {
    fn get_or_compute_derivative(
        &self,
        derivative: PartialDerivative,
        contributions: Contributions,
    ) -> f64 {
        let residual = match contributions {
            Contributions::IdealGas => None,
            _ => Some(self.get_or_compute_derivative_residual(derivative)),
        };

        let ideal_gas = match contributions {
            Contributions::Residual => None,
            _ => Some(match derivative {
                PartialDerivative::Zeroth => {
                    let new_state = self.derive0();
                    self.eos.evaluate_ideal_gas(&new_state) * new_state.temperature
                }
                PartialDerivative::First(v) => {
                    let new_state = self.derive1(v);
                    (self.eos.evaluate_ideal_gas(&new_state) * new_state.temperature).eps
                }
                PartialDerivative::Second(v) => {
                    let new_state = self.derive2(v);
                    (self.eos.evaluate_ideal_gas(&new_state) * new_state.temperature).v2
                }
                PartialDerivative::SecondMixed(v1, v2) => {
                    let new_state = self.derive2_mixed(v1, v2);
                    (self.eos.evaluate_ideal_gas(&new_state) * new_state.temperature).eps1eps2
                }
                PartialDerivative::Third(v) => {
                    let new_state = self.derive3(v);
                    (self.eos.evaluate_ideal_gas(&new_state) * new_state.temperature).v3
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
    pub fn chemical_potential(&self, contributions: Contributions) -> MolarEnergy<Array1<f64>> {
        Quantity::from_reduced(Array1::from_shape_fn(self.eos.components(), |i| {
            self.get_or_compute_derivative(PartialDerivative::First(DN(i)), contributions)
        }))
    }

    /// Partial derivative of chemical potential w.r.t. temperature: $\left(\frac{\partial\mu_i}{\partial T}\right)_{V,N_i}$
    pub fn dmu_dt(
        &self,
        contributions: Contributions,
    ) -> <MolarEnergy<Array1<f64>> as Div<Temperature>>::Output {
        Quantity::from_reduced(Array1::from_shape_fn(self.eos.components(), |i| {
            self.get_or_compute_derivative(PartialDerivative::SecondMixed(DT, DN(i)), contributions)
        }))
    }

    /// Molar isochoric heat capacity: $c_v=\left(\frac{\partial u}{\partial T}\right)_{V,N_i}$
    pub fn molar_isochoric_heat_capacity(&self, contributions: Contributions) -> MolarEntropy {
        self.temperature * self.ds_dt(contributions) / self.total_moles
    }

    /// Specific isochoric heat capacity: $c_v^{(m)}=\frac{C_v}{m}$
    pub fn specific_isochoric_heat_capacity(
        &self,
        contributions: Contributions,
    ) -> SpecificEntropy {
        self.molar_isochoric_heat_capacity(contributions) / self.total_molar_weight()
    }

    /// Partial derivative of the molar isochoric heat capacity w.r.t. temperature: $\left(\frac{\partial c_V}{\partial T}\right)_{V,N_i}$
    pub fn dc_v_dt(
        &self,
        contributions: Contributions,
    ) -> <MolarEntropy as Div<Temperature>>::Output {
        (self.temperature * self.d2s_dt2(contributions) + self.ds_dt(contributions))
            / self.total_moles
    }

    /// Molar isobaric heat capacity: $c_p=\left(\frac{\partial h}{\partial T}\right)_{p,N_i}$
    pub fn molar_isobaric_heat_capacity(&self, contributions: Contributions) -> MolarEntropy {
        match contributions {
            Contributions::Residual => self.residual_molar_isobaric_heat_capacity(),
            _ => {
                self.temperature / self.total_moles
                    * (self.ds_dt(contributions)
                        - self.dp_dt(contributions).powi::<P2>() / self.dp_dv(contributions))
            }
        }
    }

    /// Specific isobaric heat capacity: $c_p^{(m)}=\frac{C_p}{m}$
    pub fn specific_isobaric_heat_capacity(&self, contributions: Contributions) -> SpecificEntropy {
        self.molar_isobaric_heat_capacity(contributions) / self.total_molar_weight()
    }

    /// Entropy: $S=-\left(\frac{\partial A}{\partial T}\right)_{V,N_i}$
    pub fn entropy(&self, contributions: Contributions) -> Entropy {
        Entropy::from_reduced(
            -self.get_or_compute_derivative(PartialDerivative::First(DT), contributions),
        )
    }

    /// Molar entropy: $s=\frac{S}{N}$
    pub fn molar_entropy(&self, contributions: Contributions) -> MolarEntropy {
        self.entropy(contributions) / self.total_moles
    }

    /// Specific entropy: $s^{(m)}=\frac{S}{m}$
    pub fn specific_entropy(&self, contributions: Contributions) -> SpecificEntropy {
        self.molar_entropy(contributions) / self.total_molar_weight()
    }

    /// Partial molar entropy: $s_i=\left(\frac{\partial S}{\partial N_i}\right)_{T,p,N_j}$
    pub fn partial_molar_entropy(&self) -> MolarEntropy<Array1<f64>> {
        let c = Contributions::Total;
        -(self.dmu_dt(c) + self.dp_dni(c) * (self.dp_dt(c) / self.dp_dv(c)))
    }

    /// Partial derivative of the entropy w.r.t. temperature: $\left(\frac{\partial S}{\partial T}\right)_{V,N_i}$
    pub fn ds_dt(&self, contributions: Contributions) -> <Entropy as Div<Temperature>>::Output {
        Quantity::from_reduced(
            -self.get_or_compute_derivative(PartialDerivative::Second(DT), contributions),
        )
    }

    /// Second partial derivative of the entropy w.r.t. temperature: $\left(\frac{\partial^2 S}{\partial T^2}\right)_{V,N_i}$
    pub fn d2s_dt2(
        &self,
        contributions: Contributions,
    ) -> <<Entropy as Div<Temperature>>::Output as Div<Temperature>>::Output {
        Quantity::from_reduced(
            -self.get_or_compute_derivative(PartialDerivative::Third(DT), contributions),
        )
    }

    /// Enthalpy: $H=A+TS+pV$
    pub fn enthalpy(&self, contributions: Contributions) -> Energy {
        self.temperature * self.entropy(contributions)
            + self.helmholtz_energy(contributions)
            + self.pressure(contributions) * self.volume
    }

    /// Molar enthalpy: $h=\frac{H}{N}$
    pub fn molar_enthalpy(&self, contributions: Contributions) -> MolarEnergy {
        self.enthalpy(contributions) / self.total_moles
    }

    /// Specific enthalpy: $h^{(m)}=\frac{H}{m}$
    pub fn specific_enthalpy(&self, contributions: Contributions) -> SpecificEnergy {
        self.molar_enthalpy(contributions) / self.total_molar_weight()
    }

    /// Partial molar enthalpy: $h_i=\left(\frac{\partial H}{\partial N_i}\right)_{T,p,N_j}$
    pub fn partial_molar_enthalpy(&self) -> MolarEnergy<Array1<f64>> {
        let s = self.partial_molar_entropy();
        let mu = self.chemical_potential(Contributions::Total);
        s * self.temperature + mu
    }

    /// Helmholtz energy: $A$
    pub fn helmholtz_energy(&self, contributions: Contributions) -> Energy {
        Energy::from_reduced(
            self.get_or_compute_derivative(PartialDerivative::Zeroth, contributions),
        )
    }

    /// Molar Helmholtz energy: $a=\frac{A}{N}$
    pub fn molar_helmholtz_energy(&self, contributions: Contributions) -> MolarEnergy {
        self.helmholtz_energy(contributions) / self.total_moles
    }

    /// Specific Helmholtz energy: $a^{(m)}=\frac{A}{m}$
    pub fn specific_helmholtz_energy(&self, contributions: Contributions) -> SpecificEnergy {
        self.molar_helmholtz_energy(contributions) / self.total_molar_weight()
    }

    /// Internal energy: $U=A+TS$
    pub fn internal_energy(&self, contributions: Contributions) -> Energy {
        self.temperature * self.entropy(contributions) + self.helmholtz_energy(contributions)
    }

    /// Molar internal energy: $u=\frac{U}{N}$
    pub fn molar_internal_energy(&self, contributions: Contributions) -> MolarEnergy {
        self.internal_energy(contributions) / self.total_moles
    }

    /// Specific internal energy: $u^{(m)}=\frac{U}{m}$
    pub fn specific_internal_energy(&self, contributions: Contributions) -> SpecificEnergy {
        self.molar_internal_energy(contributions) / self.total_molar_weight()
    }

    /// Gibbs energy: $G=A+pV$
    pub fn gibbs_energy(&self, contributions: Contributions) -> Energy {
        self.pressure(contributions) * self.volume + self.helmholtz_energy(contributions)
    }

    /// Molar Gibbs energy: $g=\frac{G}{N}$
    pub fn molar_gibbs_energy(&self, contributions: Contributions) -> MolarEnergy {
        self.gibbs_energy(contributions) / self.total_moles
    }

    /// Specific Gibbs energy: $g^{(m)}=\frac{G}{m}$
    pub fn specific_gibbs_energy(&self, contributions: Contributions) -> SpecificEnergy {
        self.molar_gibbs_energy(contributions) / self.total_molar_weight()
    }

    /// Joule Thomson coefficient: $\mu_{JT}=\left(\frac{\partial T}{\partial p}\right)_{H,N_i}$
    pub fn joule_thomson(&self) -> <Temperature as Div<Pressure>>::Output {
        let c = Contributions::Total;
        -(self.volume + self.temperature * self.dp_dt(c) / self.dp_dv(c))
            / (self.total_moles * self.molar_isobaric_heat_capacity(c))
    }

    /// Isentropic compressibility: $\kappa_s=-\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_{S,N_i}$
    pub fn isentropic_compressibility(&self) -> <f64 as Div<Pressure>>::Output {
        let c = Contributions::Total;
        -self.molar_isochoric_heat_capacity(c)
            / (self.molar_isobaric_heat_capacity(c) * self.dp_dv(c) * self.volume)
    }

    /// Isenthalpic compressibility: $\kappa_H=-\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_{H,N_i}$
    pub fn isenthalpic_compressibility(&self) -> <f64 as Div<Pressure>>::Output {
        self.isentropic_compressibility() * (1.0 + self.grueneisen_parameter())
    }

    /// Thermal expansivity: $\alpha_p=-\frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_{p,N_i}$
    pub fn thermal_expansivity(&self) -> <f64 as Div<Temperature>>::Output {
        let c = Contributions::Total;
        -self.dp_dt(c) / self.dp_dv(c) / self.volume
    }

    /// Grueneisen parameter: $\phi=V\left(\frac{\partial p}{\partial U}\right)_{V,n_i}=\frac{v}{c_v}\left(\frac{\partial p}{\partial T}\right)_{v,n_i}=\frac{\rho}{T}\left(\frac{\partial T}{\partial \rho}\right)_{s, n_i}$
    pub fn grueneisen_parameter(&self) -> f64 {
        let c = Contributions::Total;
        (self.volume / (self.total_moles * self.molar_isochoric_heat_capacity(c)) * self.dp_dt(c))
            .into_value()
    }

    /// Chemical potential $\mu_i$ evaluated for each contribution of the equation of state.
    pub fn chemical_potential_contributions(&self, component: usize) -> Vec<(String, MolarEnergy)> {
        let new_state = self.derive1(DN(component));
        let contributions = self.eos.evaluate_residual_contributions(&new_state);
        let mut res = Vec::with_capacity(contributions.len() + 1);
        res.push((
            self.eos.ideal_gas_model().to_string(),
            MolarEnergy::from_reduced(
                (self.eos.evaluate_ideal_gas(&new_state) * new_state.temperature).eps,
            ),
        ));
        for (s, v) in contributions {
            res.push((
                s,
                MolarEnergy::from_reduced((v * new_state.temperature).eps),
            ));
        }
        res
    }

    /// Speed of sound: $c=\sqrt{\left(\frac{\partial p}{\partial\rho^{(m)}}\right)_{S,N_i}}$
    pub fn speed_of_sound(&self) -> Velocity {
        (1.0 / (self.density * self.total_molar_weight() * self.isentropic_compressibility()))
            .sqrt()
    }
}
