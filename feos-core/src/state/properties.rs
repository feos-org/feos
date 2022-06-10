use super::{Derivative::*, PartialDerivative, State};
use crate::equation_of_state::{EntropyScaling, EquationOfState, MolarWeight};
use crate::errors::EosResult;
use crate::EosUnit;
use ndarray::{arr1, Array1, Array2};
use num_dual::DualNum;
use quantity::{QuantityArray, QuantityArray1, QuantityArray2, QuantityScalar};
use std::iter::FromIterator;
use std::ops::{Add, Deref, Sub};
use std::rc::Rc;

#[derive(Clone, Copy)]
pub(crate) enum Evaluate {
    IdealGas,
    Residual,
    Total,
    IdealGasDelta,
}

/// Possible contributions that can be computed.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub enum Contributions {
    /// Only compute the ideal gas contribution
    IdealGas,
    /// Only compute the difference between the total and the ideal gas contribution
    ResidualNvt,
    /// Compute the differnce between the total and the ideal gas contribution for a (N,p,T) reference state
    ResidualNpt,
    /// Compute ideal gas and residual contributions
    Total,
}

/// # State properties
impl<U: EosUnit, E: EquationOfState> State<U, E> {
    fn get_or_compute_derivative(
        &self,
        derivative: PartialDerivative,
        evaluate: Evaluate,
    ) -> QuantityScalar<U> {
        if let Evaluate::IdealGasDelta = evaluate {
            return match derivative {
                PartialDerivative::Zeroth => {
                    let new_state = self.derive0();
                    -(new_state.moles.sum() * new_state.temperature * new_state.volume.ln())
                        * U::reference_energy()
                }
                PartialDerivative::First(v) => {
                    let new_state = self.derive1(v);
                    -(new_state.moles.sum() * new_state.temperature * new_state.volume.ln()).eps[0]
                        * (U::reference_energy() / v.reference())
                }
                PartialDerivative::Second(v1, v2) => {
                    let new_state = self.derive2(v1, v2);
                    -(new_state.moles.sum() * new_state.temperature * new_state.volume.ln())
                        .eps1eps2[(0, 0)]
                        * (U::reference_energy() / (v1.reference() * v2.reference()))
                }
                PartialDerivative::Third(v) => {
                    let new_state = self.derive3(v);
                    -(new_state.moles.sum() * new_state.temperature * new_state.volume.ln()).v3
                        * (U::reference_energy() / (v.reference() * v.reference() * v.reference()))
                }
            };
        }

        let mut cache = self.cache.borrow_mut();

        let residual = match evaluate {
            Evaluate::IdealGas => None,
            _ => Some(match derivative {
                PartialDerivative::Zeroth => {
                    let new_state = self.derive0();
                    let computation =
                        || self.eos.evaluate_residual(&new_state) * new_state.temperature;
                    cache.get_or_insert_with_f64(&computation) * U::reference_energy()
                }
                PartialDerivative::First(v) => {
                    let new_state = self.derive1(v);
                    let computation =
                        || self.eos.evaluate_residual(&new_state) * new_state.temperature;
                    cache.get_or_insert_with_d64(v, &computation) * U::reference_energy()
                        / v.reference()
                }
                PartialDerivative::Second(v1, v2) => {
                    let new_state = self.derive2(v1, v2);
                    let computation =
                        || self.eos.evaluate_residual(&new_state) * new_state.temperature;
                    cache.get_or_insert_with_hd64(v1, v2, &computation) * U::reference_energy()
                        / (v1.reference() * v2.reference())
                }
                PartialDerivative::Third(v) => {
                    let new_state = self.derive3(v);
                    let computation =
                        || self.eos.evaluate_residual(&new_state) * new_state.temperature;
                    cache.get_or_insert_with_hd364(v, &computation) * U::reference_energy()
                        / (v.reference() * v.reference() * v.reference())
                }
            }),
        };

        let ideal_gas = match evaluate {
            Evaluate::Residual => None,
            _ => Some(match derivative {
                PartialDerivative::Zeroth => {
                    let new_state = self.derive0();
                    self.eos.ideal_gas().evaluate(&new_state)
                        * U::reference_energy()
                        * new_state.temperature
                }
                PartialDerivative::First(v) => {
                    let new_state = self.derive1(v);
                    (self.eos.ideal_gas().evaluate(&new_state) * new_state.temperature).eps[0]
                        * U::reference_energy()
                        / v.reference()
                }
                PartialDerivative::Second(v1, v2) => {
                    let new_state = self.derive2(v1, v2);
                    (self.eos.ideal_gas().evaluate(&new_state) * new_state.temperature).eps1eps2
                        [(0, 0)]
                        * U::reference_energy()
                        / (v1.reference() * v2.reference())
                }
                PartialDerivative::Third(v) => {
                    let new_state = self.derive3(v);
                    (self.eos.ideal_gas().evaluate(&new_state) * new_state.temperature).v3
                        * U::reference_energy()
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
            Contributions::ResidualNvt => {
                if additive {
                    f(self, Evaluate::Residual)
                } else {
                    f(self, Evaluate::Total) - f(self, Evaluate::IdealGas)
                }
            }
            Contributions::ResidualNpt => {
                let p = self.pressure_(Evaluate::Total);
                let state_p = Self::new_nvt(
                    &self.eos,
                    self.temperature,
                    self.total_moles * U::gas_constant() * self.temperature / p,
                    &self.moles,
                )
                .unwrap();
                if additive {
                    f(self, Evaluate::Residual) + f(self, Evaluate::IdealGasDelta)
                        - f(&state_p, Evaluate::IdealGasDelta)
                } else {
                    f(self, Evaluate::Total) - f(&state_p, Evaluate::IdealGas)
                }
            }
        }
    }

    fn helmholtz_energy_(&self, evaluate: Evaluate) -> QuantityScalar<U> {
        self.get_or_compute_derivative(PartialDerivative::Zeroth, evaluate)
    }

    fn pressure_(&self, evaluate: Evaluate) -> QuantityScalar<U> {
        -self.get_or_compute_derivative(PartialDerivative::First(DV), evaluate)
    }

    fn entropy_(&self, evaluate: Evaluate) -> QuantityScalar<U> {
        -self.get_or_compute_derivative(PartialDerivative::First(DT), evaluate)
    }

    fn chemical_potential_(&self, evaluate: Evaluate) -> QuantityArray1<U> {
        QuantityArray::from_shape_fn(self.eos.components(), |i| {
            self.get_or_compute_derivative(PartialDerivative::First(DN(i)), evaluate)
        })
    }

    fn dp_dv_(&self, evaluate: Evaluate) -> QuantityScalar<U> {
        -self.get_or_compute_derivative(PartialDerivative::Second(DV, DV), evaluate)
    }

    fn dp_dt_(&self, evaluate: Evaluate) -> QuantityScalar<U> {
        -self.get_or_compute_derivative(PartialDerivative::Second(DV, DT), evaluate)
    }

    fn dp_dni_(&self, evaluate: Evaluate) -> QuantityArray1<U> {
        QuantityArray::from_shape_fn(self.eos.components(), |i| {
            -self.get_or_compute_derivative(PartialDerivative::Second(DV, DN(i)), evaluate)
        })
    }

    fn d2p_dv2_(&self, evaluate: Evaluate) -> QuantityScalar<U> {
        -self.get_or_compute_derivative(PartialDerivative::Third(DV), evaluate)
    }

    fn dmu_dt_(&self, evaluate: Evaluate) -> QuantityArray1<U> {
        QuantityArray::from_shape_fn(self.eos.components(), |i| {
            self.get_or_compute_derivative(PartialDerivative::Second(DT, DN(i)), evaluate)
        })
    }

    fn dmu_dni_(&self, evaluate: Evaluate) -> QuantityArray2<U> {
        let n = self.eos.components();
        QuantityArray::from_shape_fn((n, n), |(i, j)| {
            self.get_or_compute_derivative(PartialDerivative::Second(DN(i), DN(j)), evaluate)
        })
    }

    fn ds_dt_(&self, evaluate: Evaluate) -> QuantityScalar<U> {
        -self.get_or_compute_derivative(PartialDerivative::Second(DT, DT), evaluate)
    }

    fn d2s_dt2_(&self, evaluate: Evaluate) -> QuantityScalar<U> {
        -self.get_or_compute_derivative(PartialDerivative::Third(DT), evaluate)
    }

    /// Pressure: $p=-\left(\frac{\partial A}{\partial V}\right)_{T,N_i}$
    pub fn pressure(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.evaluate_property(Self::pressure_, contributions, true)
    }

    /// Compressibility factor: $Z=\frac{pV}{NRT}$
    pub fn compressibility(&self, contributions: Contributions) -> f64 {
        (self.pressure(contributions) / (self.density * self.temperature * U::gas_constant()))
            .into_value()
            .unwrap()
    }

    /// Partial derivative of pressure w.r.t. volume: $\left(\frac{\partial p}{\partial V}\right)_{T,N_i}$
    pub fn dp_dv(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.evaluate_property(Self::dp_dv_, contributions, true)
    }

    /// Partial derivative of pressure w.r.t. density: $\left(\frac{\partial p}{\partial \rho}\right)_{T,N_i}$
    pub fn dp_drho(&self, contributions: Contributions) -> QuantityScalar<U> {
        -self.volume / self.density * self.dp_dv(contributions)
    }

    /// Partial derivative of pressure w.r.t. temperature: $\left(\frac{\partial p}{\partial T}\right)_{V,N_i}$
    pub fn dp_dt(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.evaluate_property(Self::dp_dt_, contributions, true)
    }

    /// Partial derivative of pressure w.r.t. moles: $\left(\frac{\partial p}{\partial N_i}\right)_{T,V,N_j}$
    pub fn dp_dni(&self, contributions: Contributions) -> QuantityArray1<U> {
        self.evaluate_property(Self::dp_dni_, contributions, true)
    }

    /// Second partial derivative of pressure w.r.t. volume: $\left(\frac{\partial^2 p}{\partial V^2}\right)_{T,N_j}$
    pub fn d2p_dv2(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.evaluate_property(Self::d2p_dv2_, contributions, true)
    }

    /// Second partial derivative of pressure w.r.t. density: $\left(\frac{\partial^2 p}{\partial \rho^2}\right)_{T,N_j}$
    pub fn d2p_drho2(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.volume / (self.density * self.density)
            * (self.volume * self.d2p_dv2(contributions) + 2.0 * self.dp_dv(contributions))
    }

    /// Partial molar volume: $v_i=\left(\frac{\partial V}{\partial N_i}\right)_{T,p,N_j}$
    pub fn molar_volume(&self, contributions: Contributions) -> QuantityArray1<U> {
        let func = |s: &Self, evaluate: Evaluate| -s.dp_dni_(evaluate) / s.dp_dv_(evaluate);
        self.evaluate_property(func, contributions, false)
    }

    /// Chemical potential: $\mu_i=\left(\frac{\partial A}{\partial N_i}\right)_{T,V,N_j}$
    pub fn chemical_potential(&self, contributions: Contributions) -> QuantityArray1<U> {
        self.evaluate_property(Self::chemical_potential_, contributions, true)
    }

    /// Partial derivative of chemical potential w.r.t. temperature: $\left(\frac{\partial\mu_i}{\partial T}\right)_{V,N_i}$
    pub fn dmu_dt(&self, contributions: Contributions) -> QuantityArray1<U> {
        self.evaluate_property(Self::dmu_dt_, contributions, true)
    }

    /// Partial derivative of chemical potential w.r.t. moles: $\left(\frac{\partial\mu_i}{\partial \mu_j}\right)_{T,V,N_k}$
    pub fn dmu_dni(&self, contributions: Contributions) -> QuantityArray2<U> {
        self.evaluate_property(Self::dmu_dni_, contributions, true)
    }

    /// Logarithm of the fugacity coefficient: $\ln\varphi_i=\beta\mu_i^\mathrm{res}\left(T,p,\lbrace N_i\rbrace\right)$
    pub fn ln_phi(&self) -> Array1<f64> {
        (self.chemical_potential(Contributions::ResidualNpt)
            / (U::gas_constant() * self.temperature))
            .into_value()
            .unwrap()
    }

    /// Logarithm of the fugacity coefficient of all components treated as pure substance at mixture temperature and pressure.
    pub fn ln_phi_pure(&self) -> EosResult<Array1<f64>> {
        let pressure = self.pressure(Contributions::Total);
        (0..self.eos.components())
            .map(|i| {
                let eos = Rc::new(self.eos.subset(&[i]));
                let state = Self::new_npt(
                    &eos,
                    self.temperature,
                    pressure,
                    &(arr1(&[1.0]) * U::reference_moles()),
                    crate::DensityInitialization::Liquid,
                )?;
                Ok(state.ln_phi()[0])
            })
            .collect()
    }

    /// Activity coefficient $\ln \gamma_i = \ln \varphi_i(T, p, \mathbf{N}) - \ln \varphi_i(T, p)$
    pub fn ln_symmetric_activity_coefficient(&self) -> EosResult<Array1<f64>> {
        match self.eos.components() {
            1 => Ok(arr1(&[0.0])),
            _ => Ok(self.ln_phi() - &self.ln_phi_pure()?),
        }
    }

    /// Partial derivative of the logarithm of the fugacity coefficient w.r.t. temperature: $\left(\frac{\partial\ln\varphi_i}{\partial T}\right)_{p,N_i}$
    pub fn dln_phi_dt(&self) -> QuantityArray1<U> {
        let func = |s: &Self, evaluate: Evaluate| {
            (s.dmu_dt_(evaluate) + s.dp_dni_(evaluate) * (s.dp_dt_(evaluate) / s.dp_dv_(evaluate))
                - s.chemical_potential_(evaluate) / self.temperature)
                / (U::gas_constant() * self.temperature)
        };
        self.evaluate_property(func, Contributions::ResidualNpt, false)
    }

    /// Partial derivative of the logarithm of the fugacity coefficient w.r.t. pressure: $\left(\frac{\partial\ln\varphi_i}{\partial p}\right)_{T,N_i}$
    pub fn dln_phi_dp(&self) -> QuantityArray1<U> {
        self.molar_volume(Contributions::ResidualNpt) / (U::gas_constant() * self.temperature)
    }

    /// Partial derivative of the logarithm of the fugacity coefficient w.r.t. moles: $\left(\frac{\partial\ln\varphi_i}{\partial N_j}\right)_{T,p,N_k}$
    pub fn dln_phi_dnj(&self) -> QuantityArray2<U> {
        let n = self.eos.components();
        let dmu_dni = self.dmu_dni(Contributions::ResidualNvt);
        let dp_dni = self.dp_dni(Contributions::Total);
        let dp_dv = self.dp_dv(Contributions::Total);
        let dp_dn_2 = QuantityArray::from_shape_fn((n, n), |(i, j)| dp_dni.get(i) * dp_dni.get(j));
        (dmu_dni + dp_dn_2 / dp_dv) / (U::gas_constant() * self.temperature)
            + 1.0 / self.total_moles
    }

    /// Thermodynamic factor: $\Gamma_{ij}=\delta_{ij}+x_i\left(\frac{\partial\ln\varphi_i}{\partial x_j}\right)_{T,p,\Sigma}$
    pub fn thermodynamic_factor(&self) -> Array2<f64> {
        let dln_phi_dnj = self
            .dln_phi_dnj()
            .to_reduced(U::reference_moles().powi(-1))
            .unwrap();
        let moles = self.moles.to_reduced(U::reference_moles()).unwrap();
        let n = self.eos.components() - 1;
        Array2::from_shape_fn((n, n), |(i, j)| {
            moles[i] * (dln_phi_dnj[[i, j]] - dln_phi_dnj[[i, n]]) + if i == j { 1.0 } else { 0.0 }
        })
    }

    /// Molar isochoric heat capacity: $c_v=\left(\frac{\partial u}{\partial T}\right)_{V,N_i}$
    pub fn c_v(&self, contributions: Contributions) -> QuantityScalar<U> {
        let func =
            |s: &Self, evaluate: Evaluate| s.temperature * s.ds_dt_(evaluate) / s.total_moles;
        self.evaluate_property(func, contributions, true)
    }

    /// Partial derivative of the molar isochoric heat capacity w.r.t. temperature: $\left(\frac{\partial c_V}{\partial T}\right)_{V,N_i}$
    pub fn dc_v_dt(&self, contributions: Contributions) -> QuantityScalar<U> {
        let func = |s: &Self, evaluate: Evaluate| {
            (s.temperature * s.d2s_dt2_(evaluate) + s.ds_dt_(evaluate)) / s.total_moles
        };
        self.evaluate_property(func, contributions, true)
    }

    /// Molar isobaric heat capacity: $c_p=\left(\frac{\partial h}{\partial T}\right)_{p,N_i}$
    pub fn c_p(&self, contributions: Contributions) -> QuantityScalar<U> {
        let func = |s: &Self, evaluate: Evaluate| {
            s.temperature / s.total_moles
                * (s.ds_dt_(evaluate)
                    - s.dp_dt_(evaluate) * s.dp_dt_(evaluate) / s.dp_dv_(evaluate))
        };
        self.evaluate_property(func, contributions, false)
    }

    /// Entropy: $S=-\left(\frac{\partial A}{\partial T}\right)_{V,N_i}$
    pub fn entropy(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.evaluate_property(Self::entropy_, contributions, true)
    }

    /// Partial derivative of the entropy w.r.t. temperature: $\left(\frac{\partial S}{\partial T}\right)_{V,N_i}$
    pub fn ds_dt(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.evaluate_property(Self::ds_dt_, contributions, true)
    }

    /// molar entropy: $s=\frac{S}{N}$
    pub fn molar_entropy(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.entropy(contributions) / self.total_moles
    }

    /// Enthalpy: $H=A+TS+pV$
    pub fn enthalpy(&self, contributions: Contributions) -> QuantityScalar<U> {
        let func = |s: &Self, evaluate: Evaluate| {
            s.temperature * s.entropy_(evaluate)
                + s.helmholtz_energy_(evaluate)
                + s.pressure_(evaluate) * s.volume
        };
        self.evaluate_property(func, contributions, true)
    }

    /// molar enthalpy: $h=\frac{H}{N}$
    pub fn molar_enthalpy(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.enthalpy(contributions) / self.total_moles
    }

    /// Helmholtz energy: $A$
    pub fn helmholtz_energy(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.evaluate_property(Self::helmholtz_energy_, contributions, true)
    }

    /// molar Helmholtz energy: $a=\frac{A}{N}$
    pub fn molar_helmholtz_energy(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.helmholtz_energy(contributions) / self.total_moles
    }

    /// Internal energy: $U=A+TS$
    pub fn internal_energy(&self, contributions: Contributions) -> QuantityScalar<U> {
        let func = |s: &Self, evaluate: Evaluate| {
            s.temperature * s.entropy_(evaluate) + s.helmholtz_energy_(evaluate)
        };
        self.evaluate_property(func, contributions, true)
    }

    /// Molar internal energy: $u=\frac{U}{N}$
    pub fn molar_internal_energy(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.internal_energy(contributions) / self.total_moles
    }

    /// Gibbs energy: $G=A+pV$
    pub fn gibbs_energy(&self, contributions: Contributions) -> QuantityScalar<U> {
        let func = |s: &Self, evaluate: Evaluate| {
            s.pressure_(evaluate) * s.volume + s.helmholtz_energy_(evaluate)
        };
        self.evaluate_property(func, contributions, true)
    }

    /// Molar Gibbs energy: $g=\frac{G}{N}$
    pub fn molar_gibbs_energy(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.gibbs_energy(contributions) / self.total_moles
    }

    /// Partial molar entropy: $s_i=\left(\frac{\partial S}{\partial N_i}\right)_{T,p,N_j}$
    pub fn partial_molar_entropy(&self, contributions: Contributions) -> QuantityArray1<U> {
        let func = |s: &Self, evaluate: Evaluate| {
            -(s.dmu_dt_(evaluate) + s.dp_dni_(evaluate) * (s.dp_dt_(evaluate) / s.dp_dv_(evaluate)))
        };
        self.evaluate_property(func, contributions, false)
    }

    /// Partial molar enthalpy: $h_i=\left(\frac{\partial H}{\partial N_i}\right)_{T,p,N_j}$
    pub fn partial_molar_enthalpy(&self, contributions: Contributions) -> QuantityArray1<U> {
        let s = self.partial_molar_entropy(contributions);
        let mu = self.chemical_potential(contributions);
        s * self.temperature + mu
    }

    /// Joule Thomson coefficient: $\mu_{JT}=\left(\frac{\partial T}{\partial p}\right)_{H,N_i}$
    pub fn joule_thomson(&self) -> QuantityScalar<U> {
        let c = Contributions::Total;
        -(self.volume + self.temperature * self.dp_dt(c) / self.dp_dv(c))
            / (self.total_moles * self.c_p(c))
    }

    /// Isentropic compressibility: $\kappa_s=-\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_{S,N_i}$
    pub fn isentropic_compressibility(&self) -> QuantityScalar<U> {
        let c = Contributions::Total;
        -self.c_v(c) / (self.c_p(c) * self.dp_dv(c) * self.volume)
    }

    /// Isothermal compressibility: $\kappa_T=-\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_{T,N_i}$
    pub fn isothermal_compressibility(&self) -> QuantityScalar<U> {
        let c = Contributions::Total;
        -1.0 / (self.dp_dv(c) * self.volume)
    }

    /// Structure factor: $S(0)=k_BT\left(\frac{\partial\rho}{\partial p}\right)_{T,N_i}$
    pub fn structure_factor(&self) -> f64 {
        -(U::gas_constant() * self.temperature * self.density)
            .to_reduced(self.volume * self.dp_dv(Contributions::Total))
            .unwrap()
    }

    /// Helmholtz energy $A$ evaluated for each contribution of the equation of state.
    pub fn helmholtz_energy_contributions(&self) -> Vec<(String, QuantityScalar<U>)> {
        let new_state = self.derive0();
        let contributions = self.eos.evaluate_residual_contributions(&new_state);
        let mut res = Vec::with_capacity(contributions.len() + 1);
        let ig = self.eos.ideal_gas();
        res.push((
            ig.to_string(),
            ig.evaluate(&new_state) * new_state.temperature * U::reference_energy(),
        ));
        for (s, v) in contributions {
            res.push((s, v * new_state.temperature * U::reference_energy()));
        }
        res
    }

    /// Pressure $p$ evaluated for each contribution of the equation of state.
    pub fn pressure_contributions(&self) -> Vec<(String, QuantityScalar<U>)> {
        let new_state = self.derive1(DV);
        let contributions = self.eos.evaluate_residual_contributions(&new_state);
        let mut res = Vec::with_capacity(contributions.len() + 1);
        let ig = self.eos.ideal_gas();
        res.push((
            ig.to_string(),
            -(ig.evaluate(&new_state) * new_state.temperature).eps[0] * U::reference_pressure(),
        ));
        for (s, v) in contributions {
            res.push((
                s,
                -(v * new_state.temperature).eps[0] * U::reference_pressure(),
            ));
        }
        res
    }

    /// Chemical potential $\mu_i$ evaluated for each contribution of the equation of state.
    pub fn chemical_potential_contributions(
        &self,
        component: usize,
    ) -> Vec<(String, QuantityScalar<U>)> {
        let new_state = self.derive1(DN(component));
        let contributions = self.eos.evaluate_residual_contributions(&new_state);
        let mut res = Vec::with_capacity(contributions.len() + 1);
        let ig = self.eos.ideal_gas();
        res.push((
            ig.to_string(),
            (ig.evaluate(&new_state) * new_state.temperature).eps[0] * U::reference_molar_energy(),
        ));
        for (s, v) in contributions {
            res.push((
                s,
                (v * new_state.temperature).eps[0] * U::reference_molar_energy(),
            ));
        }
        res
    }
}

/// # Mass specific state properties
///
/// These properties are available for equations of state
/// that implement the [MolarWeight] trait.
impl<U: EosUnit, E: EquationOfState + MolarWeight<U>> State<U, E> {
    /// Total molar weight: $MW=\sum_ix_iMW_i$
    pub fn total_molar_weight(&self) -> QuantityScalar<U> {
        (self.eos.molar_weight() * &self.molefracs).sum()
    }

    /// Mass of each component: $m_i=n_iMW_i$
    pub fn mass(&self) -> QuantityArray1<U> {
        self.moles.clone() * self.eos.molar_weight()
    }

    /// Total mass: $m=\sum_im_i=nMW$
    pub fn total_mass(&self) -> QuantityScalar<U> {
        self.total_moles * self.total_molar_weight()
    }

    /// Mass density: $\rho^{(m)}=\frac{m}{V}$
    pub fn mass_density(&self) -> QuantityScalar<U> {
        self.density * self.total_molar_weight()
    }

    /// Mass fractions: $w_i=\frac{m_i}{m}$
    pub fn massfracs(&self) -> Array1<f64> {
        self.mass().to_reduced(self.total_mass()).unwrap()
    }

    /// Specific entropy: $s^{(m)}=\frac{S}{m}$
    pub fn specific_entropy(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.molar_entropy(contributions) / self.total_molar_weight()
    }

    /// Specific enthalpy: $h^{(m)}=\frac{H}{m}$
    pub fn specific_enthalpy(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.molar_enthalpy(contributions) / self.total_molar_weight()
    }

    /// Specific Helmholtz energy: $a^{(m)}=\frac{A}{m}$
    pub fn specific_helmholtz_energy(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.molar_helmholtz_energy(contributions) / self.total_molar_weight()
    }

    /// Specific internal energy: $u^{(m)}=\frac{U}{m}$
    pub fn specific_internal_energy(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.molar_internal_energy(contributions) / self.total_molar_weight()
    }

    /// Specific Gibbs energy: $g^{(m)}=\frac{G}{m}$
    pub fn specific_gibbs_energy(&self, contributions: Contributions) -> QuantityScalar<U> {
        self.molar_gibbs_energy(contributions) / self.total_molar_weight()
    }

    /// Speed of sound: $c=\sqrt{\left(\frac{\partial p}{\partial\rho}\right)_{S,N_i}}$
    pub fn speed_of_sound(&self) -> QuantityScalar<U> {
        (1.0 / (self.density * self.total_molar_weight() * self.isentropic_compressibility()))
            .sqrt()
            .unwrap()
    }
}

impl<U: EosUnit, E: EquationOfState> State<U, E> {
    // This function is designed specifically for use in density iterations
    pub(crate) fn p_dpdrho(&self) -> (QuantityScalar<U>, QuantityScalar<U>) {
        let dp_dv = self.dp_dv(Contributions::Total);
        (
            self.pressure(Contributions::Total),
            (-self.volume * dp_dv / self.density),
        )
    }

    // This function is designed specifically for use in spinodal iterations
    pub(crate) fn d2pdrho2(&self) -> (QuantityScalar<U>, QuantityScalar<U>, QuantityScalar<U>) {
        let d2p_dv2 = self.d2p_dv2(Contributions::Total);
        let dp_dv = self.dp_dv(Contributions::Total);
        (
            self.pressure(Contributions::Total),
            (-self.volume * dp_dv / self.density),
            (self.volume / (self.density * self.density) * (2.0 * dp_dv + self.volume * d2p_dv2)),
        )
    }
}

/// # Transport properties
///
/// These properties are available for equations of state
/// that implement the [EntropyScaling] trait.
impl<U: EosUnit, E: EquationOfState + EntropyScaling<U>> State<U, E> {
    /// Return the viscosity via entropy scaling.
    pub fn viscosity(&self) -> EosResult<QuantityScalar<U>> {
        let s = self
            .molar_entropy(Contributions::ResidualNvt)
            .to_reduced(U::reference_molar_entropy())?;
        Ok(self
            .eos
            .viscosity_reference(self.temperature, self.volume, &self.moles)?
            * self.eos.viscosity_correlation(s, &self.molefracs)?.exp())
    }

    /// Return the logarithm of the reduced viscosity.
    ///
    /// This term equals the viscosity correlation function
    /// that is used for entropy scaling.
    pub fn ln_viscosity_reduced(&self) -> EosResult<f64> {
        let s = self
            .molar_entropy(Contributions::ResidualNvt)
            .to_reduced(U::reference_molar_entropy())?;
        self.eos.viscosity_correlation(s, &self.molefracs)
    }

    /// Return the viscosity reference as used in entropy scaling.
    pub fn viscosity_reference(&self) -> EosResult<QuantityScalar<U>> {
        self.eos
            .viscosity_reference(self.temperature, self.volume, &self.moles)
    }

    /// Return the diffusion via entropy scaling.
    pub fn diffusion(&self) -> EosResult<QuantityScalar<U>> {
        let s = self
            .molar_entropy(Contributions::ResidualNvt)
            .to_reduced(U::reference_molar_entropy())?;
        Ok(self
            .eos
            .diffusion_reference(self.temperature, self.volume, &self.moles)?
            * self.eos.diffusion_correlation(s, &self.molefracs)?.exp())
    }

    /// Return the logarithm of the reduced diffusion.
    ///
    /// This term equals the diffusion correlation function
    /// that is used for entropy scaling.
    pub fn ln_diffusion_reduced(&self) -> EosResult<f64> {
        let s = self
            .molar_entropy(Contributions::ResidualNvt)
            .to_reduced(U::reference_molar_entropy())?;
        self.eos.diffusion_correlation(s, &self.molefracs)
    }

    /// Return the diffusion reference as used in entropy scaling.
    pub fn diffusion_reference(&self) -> EosResult<QuantityScalar<U>> {
        self.eos
            .diffusion_reference(self.temperature, self.volume, &self.moles)
    }

    /// Return the thermal conductivity via entropy scaling.
    pub fn thermal_conductivity(&self) -> EosResult<QuantityScalar<U>> {
        let s = self
            .molar_entropy(Contributions::ResidualNvt)
            .to_reduced(U::reference_molar_entropy())?;
        Ok(self
            .eos
            .thermal_conductivity_reference(self.temperature, self.volume, &self.moles)?
            * self
                .eos
                .thermal_conductivity_correlation(s, &self.molefracs)?
                .exp())
    }

    /// Return the logarithm of the reduced thermal conductivity.
    ///
    /// This term equals the thermal conductivity correlation function
    /// that is used for entropy scaling.
    pub fn ln_thermal_conductivity_reduced(&self) -> EosResult<f64> {
        let s = self
            .molar_entropy(Contributions::ResidualNvt)
            .to_reduced(U::reference_molar_entropy())?;
        self.eos
            .thermal_conductivity_correlation(s, &self.molefracs)
    }

    /// Return the thermal conductivity reference as used in entropy scaling.
    pub fn thermal_conductivity_reference(&self) -> EosResult<QuantityScalar<U>> {
        self.eos
            .thermal_conductivity_reference(self.temperature, self.volume, &self.moles)
    }
}

/// A list of states for a simple access to properties
/// of multiple states.
pub struct StateVec<'a, U, E>(pub Vec<&'a State<U, E>>);

impl<'a, U, E> FromIterator<&'a State<U, E>> for StateVec<'a, U, E> {
    fn from_iter<I: IntoIterator<Item = &'a State<U, E>>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<'a, U, E> IntoIterator for StateVec<'a, U, E> {
    type Item = &'a State<U, E>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, U, E> Deref for StateVec<'a, U, E> {
    type Target = Vec<&'a State<U, E>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, U: EosUnit, E: EquationOfState> StateVec<'a, U, E> {
    pub fn temperature(&self) -> QuantityArray1<U> {
        QuantityArray1::from_shape_fn(self.0.len(), |i| self.0[i].temperature)
    }

    pub fn pressure(&self) -> QuantityArray1<U> {
        QuantityArray1::from_shape_fn(self.0.len(), |i| self.0[i].pressure(Contributions::Total))
    }

    pub fn compressibility(&self) -> Array1<f64> {
        Array1::from_shape_fn(self.0.len(), |i| {
            self.0[i].compressibility(Contributions::Total)
        })
    }

    pub fn density(&self) -> QuantityArray1<U> {
        QuantityArray1::from_shape_fn(self.0.len(), |i| self.0[i].density)
    }

    pub fn molefracs(&self) -> Array2<f64> {
        Array2::from_shape_fn((self.0.len(), self.0[0].eos.components()), |(i, j)| {
            self.0[i].molefracs[j]
        })
    }

    pub fn molar_enthalpy(&self) -> QuantityArray1<U> {
        QuantityArray1::from_shape_fn(self.0.len(), |i| {
            self.0[i].molar_enthalpy(Contributions::Total)
        })
    }

    pub fn molar_entropy(&self) -> QuantityArray1<U> {
        QuantityArray1::from_shape_fn(self.0.len(), |i| {
            self.0[i].molar_entropy(Contributions::Total)
        })
    }
}

impl<'a, U: EosUnit, E: EquationOfState + MolarWeight<U>> StateVec<'a, U, E> {
    pub fn mass_density(&self) -> QuantityArray1<U> {
        QuantityArray1::from_shape_fn(self.0.len(), |i| self.0[i].mass_density())
    }

    pub fn massfracs(&self) -> Array2<f64> {
        Array2::from_shape_fn((self.0.len(), self.0[0].eos.components()), |(i, j)| {
            self.0[i].massfracs()[j]
        })
    }

    pub fn specific_enthalpy(&self) -> QuantityArray1<U> {
        QuantityArray1::from_shape_fn(self.0.len(), |i| {
            self.0[i].specific_enthalpy(Contributions::Total)
        })
    }

    pub fn specific_entropy(&self) -> QuantityArray1<U> {
        QuantityArray1::from_shape_fn(self.0.len(), |i| {
            self.0[i].specific_entropy(Contributions::Total)
        })
    }
}
