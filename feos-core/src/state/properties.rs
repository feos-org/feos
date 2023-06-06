use super::{Contributions, Derivative::*, PartialDerivative, State};
use crate::equation_of_state::{IdealGas, MolarWeight, Residual};
use crate::EosUnit;
use ndarray::Array1;
use quantity::si::*;
use std::ops::{Add, Sub};

#[derive(Clone, Copy)]
pub(crate) enum Evaluate {
    IdealGas,
    Residual,
    Total,
}

impl<E: Residual + IdealGas> State<E> {
    fn get_or_compute_derivative(
        &self,
        derivative: PartialDerivative,
        evaluate: Evaluate,
    ) -> SINumber {
        let residual = match evaluate {
            Evaluate::IdealGas => None,
            _ => Some(self.get_or_compute_derivative_residual(derivative)),
        };

        let ideal_gas = match evaluate {
            Evaluate::Residual => None,
            _ => Some(match derivative {
                PartialDerivative::Zeroth => {
                    let new_state = self.derive0();
                    self.eos.evaluate_ideal_gas(&new_state)
                        * SIUnit::reference_energy()
                        * new_state.temperature
                }
                PartialDerivative::First(v) => {
                    let new_state = self.derive1(v);
                    (self.eos.evaluate_ideal_gas(&new_state) * new_state.temperature).eps
                        * SIUnit::reference_energy()
                        / v.reference()
                }
                PartialDerivative::Second(v) => {
                    let new_state = self.derive2(v);
                    (self.eos.evaluate_ideal_gas(&new_state) * new_state.temperature).v2
                        * SIUnit::reference_energy()
                        / (v.reference() * v.reference())
                }
                PartialDerivative::SecondMixed(v1, v2) => {
                    let new_state = self.derive2_mixed(v1, v2);
                    (self.eos.evaluate_ideal_gas(&new_state) * new_state.temperature).eps1eps2
                        * SIUnit::reference_energy()
                        / (v1.reference() * v2.reference())
                }
                PartialDerivative::Third(v) => {
                    let new_state = self.derive3(v);
                    (self.eos.evaluate_ideal_gas(&new_state) * new_state.temperature).v3
                        * SIUnit::reference_energy()
                        / (v.reference() * v.reference() * v.reference())
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

    fn evaluate_property<R, F>(&self, f: F, contributions: Contributions, additive: bool) -> R
    where
        R: Add<Output = R> + Sub<Output = R>,
        F: Fn(&Self, Evaluate) -> R,
    {
        match contributions {
            Contributions::IdealGas => f(self, Evaluate::IdealGas),
            Contributions::Total => f(self, Evaluate::Total),
            Contributions::Residual => {
                if additive {
                    f(self, Evaluate::Residual)
                } else {
                    f(self, Evaluate::Total) - f(self, Evaluate::IdealGas)
                }
            }
        }
    }

    fn helmholtz_energy_(&self, evaluate: Evaluate) -> SINumber {
        self.get_or_compute_derivative(PartialDerivative::Zeroth, evaluate)
    }

    fn pressure_(&self, evaluate: Evaluate) -> SINumber {
        -self.get_or_compute_derivative(PartialDerivative::First(DV), evaluate)
    }

    fn entropy_(&self, evaluate: Evaluate) -> SINumber {
        -self.get_or_compute_derivative(PartialDerivative::First(DT), evaluate)
    }

    fn chemical_potential_(&self, evaluate: Evaluate) -> SIArray1 {
        SIArray::from_shape_fn(self.eos.components(), |i| {
            self.get_or_compute_derivative(PartialDerivative::First(DN(i)), evaluate)
        })
    }

    fn dp_dv_(&self, evaluate: Evaluate) -> SINumber {
        -self.get_or_compute_derivative(PartialDerivative::Second(DV), evaluate)
    }

    fn dp_dt_(&self, evaluate: Evaluate) -> SINumber {
        -self.get_or_compute_derivative(PartialDerivative::SecondMixed(DV, DT), evaluate)
    }

    fn dp_dni_(&self, evaluate: Evaluate) -> SIArray1 {
        SIArray::from_shape_fn(self.eos.components(), |i| {
            -self.get_or_compute_derivative(PartialDerivative::SecondMixed(DV, DN(i)), evaluate)
        })
    }

    fn dmu_dt_(&self, evaluate: Evaluate) -> SIArray1 {
        SIArray::from_shape_fn(self.eos.components(), |i| {
            self.get_or_compute_derivative(PartialDerivative::SecondMixed(DT, DN(i)), evaluate)
        })
    }

    fn ds_dt_(&self, evaluate: Evaluate) -> SINumber {
        -self.get_or_compute_derivative(PartialDerivative::Second(DT), evaluate)
    }

    fn d2s_dt2_(&self, evaluate: Evaluate) -> SINumber {
        -self.get_or_compute_derivative(PartialDerivative::Third(DT), evaluate)
    }

    /// Chemical potential: $\mu_i=\left(\frac{\partial A}{\partial N_i}\right)_{T,V,N_j}$
    pub fn chemical_potential(&self, contributions: Contributions) -> SIArray1 {
        self.evaluate_property(Self::chemical_potential_, contributions, true)
    }

    /// Partial derivative of chemical potential w.r.t. temperature: $\left(\frac{\partial\mu_i}{\partial T}\right)_{V,N_i}$
    pub fn dmu_dt(&self, contributions: Contributions) -> SIArray1 {
        self.evaluate_property(Self::dmu_dt_, contributions, true)
    }

    /// Molar isochoric heat capacity: $c_v=\left(\frac{\partial u}{\partial T}\right)_{V,N_i}$
    pub fn c_v(&self, contributions: Contributions) -> SINumber {
        let func =
            |s: &Self, evaluate: Evaluate| s.temperature * s.ds_dt_(evaluate) / s.total_moles;
        self.evaluate_property(func, contributions, true)
    }

    /// Partial derivative of the molar isochoric heat capacity w.r.t. temperature: $\left(\frac{\partial c_V}{\partial T}\right)_{V,N_i}$
    pub fn dc_v_dt(&self, contributions: Contributions) -> SINumber {
        let func = |s: &Self, evaluate: Evaluate| {
            (s.temperature * s.d2s_dt2_(evaluate) + s.ds_dt_(evaluate)) / s.total_moles
        };
        self.evaluate_property(func, contributions, true)
    }

    /// Molar isobaric heat capacity: $c_p=\left(\frac{\partial h}{\partial T}\right)_{p,N_i}$
    pub fn c_p(&self, contributions: Contributions) -> SINumber {
        let func = |s: &Self, evaluate: Evaluate| {
            s.temperature / s.total_moles
                * (s.ds_dt_(evaluate) - s.dp_dt_(evaluate).powi(2) / s.dp_dv_(evaluate))
        };
        self.evaluate_property(func, contributions, false)
    }

    /// Entropy: $S=-\left(\frac{\partial A}{\partial T}\right)_{V,N_i}$
    pub fn entropy(&self, contributions: Contributions) -> SINumber {
        self.evaluate_property(Self::entropy_, contributions, true)
    }

    /// Partial derivative of the entropy w.r.t. temperature: $\left(\frac{\partial S}{\partial T}\right)_{V,N_i}$
    pub fn ds_dt(&self, contributions: Contributions) -> SINumber {
        self.evaluate_property(Self::ds_dt_, contributions, true)
    }

    /// molar entropy: $s=\frac{S}{N}$
    pub fn molar_entropy(&self, contributions: Contributions) -> SINumber {
        self.entropy(contributions) / self.total_moles
    }

    /// Enthalpy: $H=A+TS+pV$
    pub fn enthalpy(&self, contributions: Contributions) -> SINumber {
        let func = |s: &Self, evaluate: Evaluate| {
            s.temperature * s.entropy_(evaluate)
                + s.helmholtz_energy_(evaluate)
                + s.pressure_(evaluate) * s.volume
        };
        self.evaluate_property(func, contributions, true)
    }

    /// molar enthalpy: $h=\frac{H}{N}$
    pub fn molar_enthalpy(&self, contributions: Contributions) -> SINumber {
        self.enthalpy(contributions) / self.total_moles
    }

    /// Helmholtz energy: $A$
    pub fn helmholtz_energy(&self, contributions: Contributions) -> SINumber {
        self.evaluate_property(Self::helmholtz_energy_, contributions, true)
    }

    /// molar Helmholtz energy: $a=\frac{A}{N}$
    pub fn molar_helmholtz_energy(&self, contributions: Contributions) -> SINumber {
        self.helmholtz_energy(contributions) / self.total_moles
    }

    /// Internal energy: $U=A+TS$
    pub fn internal_energy(&self, contributions: Contributions) -> SINumber {
        let func = |s: &Self, evaluate: Evaluate| {
            s.temperature * s.entropy_(evaluate) + s.helmholtz_energy_(evaluate)
        };
        self.evaluate_property(func, contributions, true)
    }

    /// Molar internal energy: $u=\frac{U}{N}$
    pub fn molar_internal_energy(&self, contributions: Contributions) -> SINumber {
        self.internal_energy(contributions) / self.total_moles
    }

    /// Gibbs energy: $G=A+pV$
    pub fn gibbs_energy(&self, contributions: Contributions) -> SINumber {
        let func = |s: &Self, evaluate: Evaluate| {
            s.pressure_(evaluate) * s.volume + s.helmholtz_energy_(evaluate)
        };
        self.evaluate_property(func, contributions, true)
    }

    /// Molar Gibbs energy: $g=\frac{G}{N}$
    pub fn molar_gibbs_energy(&self, contributions: Contributions) -> SINumber {
        self.gibbs_energy(contributions) / self.total_moles
    }

    /// Partial molar entropy: $s_i=\left(\frac{\partial S}{\partial N_i}\right)_{T,p,N_j}$
    pub fn partial_molar_entropy(&self, contributions: Contributions) -> SIArray1 {
        let func = |s: &Self, evaluate: Evaluate| {
            -(s.dmu_dt_(evaluate) + s.dp_dni_(evaluate) * (s.dp_dt_(evaluate) / s.dp_dv_(evaluate)))
        };
        self.evaluate_property(func, contributions, false)
    }

    /// Partial molar enthalpy: $h_i=\left(\frac{\partial H}{\partial N_i}\right)_{T,p,N_j}$
    pub fn partial_molar_enthalpy(&self, contributions: Contributions) -> SIArray1 {
        let s = self.partial_molar_entropy(contributions);
        let mu = self.chemical_potential(contributions);
        s * self.temperature + mu
    }

    /// Joule Thomson coefficient: $\mu_{JT}=\left(\frac{\partial T}{\partial p}\right)_{H,N_i}$
    pub fn joule_thomson(&self) -> SINumber {
        let c = Contributions::Total;
        -(self.volume + self.temperature * self.dp_dt(c) / self.dp_dv(c))
            / (self.total_moles * self.c_p(c))
    }

    /// Isentropic compressibility: $\kappa_s=-\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_{S,N_i}$
    pub fn isentropic_compressibility(&self) -> SINumber {
        let c = Contributions::Total;
        -self.c_v(c) / (self.c_p(c) * self.dp_dv(c) * self.volume)
    }

    /// Helmholtz energy $A$ evaluated for each contribution of the equation of state.
    pub fn helmholtz_energy_contributions(&self) -> Vec<(String, SINumber)> {
        let new_state = self.derive0();
        let contributions = self.eos.evaluate_residual_contributions(&new_state);
        let mut res = Vec::with_capacity(contributions.len() + 1);
        res.push((
            self.eos.ideal_gas_model().to_string(),
            self.eos.evaluate_ideal_gas(&new_state)
                * new_state.temperature
                * SIUnit::reference_energy(),
        ));
        for (s, v) in contributions {
            res.push((s, v * new_state.temperature * SIUnit::reference_energy()));
        }
        res
    }

    /// Chemical potential $\mu_i$ evaluated for each contribution of the equation of state.
    pub fn chemical_potential_contributions(&self, component: usize) -> Vec<(String, SINumber)> {
        let new_state = self.derive1(DN(component));
        let contributions = self.eos.evaluate_residual_contributions(&new_state);
        let mut res = Vec::with_capacity(contributions.len() + 1);
        res.push((
            self.eos.ideal_gas_model().to_string(),
            (self.eos.evaluate_ideal_gas(&new_state) * new_state.temperature).eps
                * SIUnit::reference_molar_energy(),
        ));
        for (s, v) in contributions {
            res.push((
                s,
                (v * new_state.temperature).eps * SIUnit::reference_molar_energy(),
            ));
        }
        res
    }
}

/// # Mass specific state properties
///
/// These properties are available for equations of state
/// that implement the [MolarWeight] trait.
impl<E: Residual + MolarWeight> State<E> {
    /// Total molar weight: $MW=\sum_ix_iMW_i$
    pub fn total_molar_weight(&self) -> SINumber {
        (self.eos.molar_weight() * &self.molefracs).sum()
    }

    /// Mass of each component: $m_i=n_iMW_i$
    pub fn mass(&self) -> SIArray1 {
        self.moles.clone() * self.eos.molar_weight()
    }

    /// Total mass: $m=\sum_im_i=nMW$
    pub fn total_mass(&self) -> SINumber {
        self.total_moles * self.total_molar_weight()
    }

    /// Mass density: $\rho^{(m)}=\frac{m}{V}$
    pub fn mass_density(&self) -> SINumber {
        self.density * self.total_molar_weight()
    }

    /// Mass fractions: $w_i=\frac{m_i}{m}$
    pub fn massfracs(&self) -> Array1<f64> {
        self.mass().to_reduced(self.total_mass()).unwrap()
    }
}

impl<E: Residual + IdealGas + MolarWeight> State<E> {
    /// Specific entropy: $s^{(m)}=\frac{S}{m}$
    pub fn specific_entropy(&self, contributions: Contributions) -> SINumber {
        self.molar_entropy(contributions) / self.total_molar_weight()
    }

    /// Specific enthalpy: $h^{(m)}=\frac{H}{m}$
    pub fn specific_enthalpy(&self, contributions: Contributions) -> SINumber {
        self.molar_enthalpy(contributions) / self.total_molar_weight()
    }

    /// Specific Helmholtz energy: $a^{(m)}=\frac{A}{m}$
    pub fn specific_helmholtz_energy(&self, contributions: Contributions) -> SINumber {
        self.molar_helmholtz_energy(contributions) / self.total_molar_weight()
    }

    /// Specific internal energy: $u^{(m)}=\frac{U}{m}$
    pub fn specific_internal_energy(&self, contributions: Contributions) -> SINumber {
        self.molar_internal_energy(contributions) / self.total_molar_weight()
    }

    /// Specific Gibbs energy: $g^{(m)}=\frac{G}{m}$
    pub fn specific_gibbs_energy(&self, contributions: Contributions) -> SINumber {
        self.molar_gibbs_energy(contributions) / self.total_molar_weight()
    }

    /// Speed of sound: $c=\sqrt{\left(\frac{\partial p}{\partial\rho^{(m)}}\right)_{S,N_i}}$
    pub fn speed_of_sound(&self) -> SINumber {
        (1.0 / (self.density * self.total_molar_weight() * self.isentropic_compressibility()))
            .sqrt()
            .unwrap()
    }
}
