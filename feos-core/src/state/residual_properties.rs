use super::{Contributions, Derivative::*, PartialDerivative, State};
use crate::equation_of_state::{EntropyScaling, Residual};
use crate::errors::EosResult;
use crate::EosUnit;
use ndarray::{arr1, Array1, Array2};
use quantity::si::*;
use std::sync::Arc;

/// # State properties
impl<E: Residual> State<E> {
    pub(super) fn get_or_compute_derivative_residual(
        &self,
        derivative: PartialDerivative,
    ) -> SINumber {
        let mut cache = self.cache.lock().unwrap();

        match derivative {
            PartialDerivative::Zeroth => {
                let new_state = self.derive0();
                let computation = || self.eos.evaluate_residual(&new_state) * new_state.temperature;
                cache.get_or_insert_with_f64(computation) * SIUnit::reference_energy()
            }
            PartialDerivative::First(v) => {
                let new_state = self.derive1(v);
                let computation = || self.eos.evaluate_residual(&new_state) * new_state.temperature;
                cache.get_or_insert_with_d64(v, computation) * SIUnit::reference_energy()
                    / v.reference()
            }
            PartialDerivative::Second(v) => {
                let new_state = self.derive2(v);
                let computation = || self.eos.evaluate_residual(&new_state) * new_state.temperature;
                cache.get_or_insert_with_d2_64(v, computation) * SIUnit::reference_energy()
                    / (v.reference() * v.reference())
            }
            PartialDerivative::SecondMixed(v1, v2) => {
                let new_state = self.derive2_mixed(v1, v2);
                let computation = || self.eos.evaluate_residual(&new_state) * new_state.temperature;
                cache.get_or_insert_with_hd64(v1, v2, computation) * SIUnit::reference_energy()
                    / (v1.reference() * v2.reference())
            }
            PartialDerivative::Third(v) => {
                let new_state = self.derive3(v);
                let computation = || self.eos.evaluate_residual(&new_state) * new_state.temperature;
                cache.get_or_insert_with_hd364(v, computation) * SIUnit::reference_energy()
                    / (v.reference() * v.reference() * v.reference())
            }
        }
    }
}

impl<E: Residual> State<E> {
    fn contributions(
        ideal_gas: SINumber,
        residual: SINumber,
        contributions: Contributions,
    ) -> SINumber {
        match contributions {
            Contributions::IdealGas => ideal_gas,
            Contributions::Total => ideal_gas + residual,
            Contributions::Residual => residual,
        }
    }

    /// Residual Helmholtz energy $A^\text{res}$
    pub fn residual_helmholtz_energy(&self) -> SINumber {
        self.get_or_compute_derivative_residual(PartialDerivative::Zeroth)
    }

    /// Residual molar Helmholtz energy $a^\text{res}$
    pub fn residual_molar_helmholtz_energy(&self) -> SINumber {
        self.residual_helmholtz_energy() / self.total_moles
    }

    /// Residual Helmholtz energy $A^\text{res}$ evaluated for each contribution of the equation of state.
    pub fn residual_helmholtz_energy_contributions(
        &self,
    ) -> Vec<(String, SINumber)> {
        let new_state = self.derive0();
        let residual_contributions = self.eos.evaluate_residual_contributions(&new_state);
        let mut res = Vec::with_capacity(residual_contributions.len());
        for (s, v) in residual_contributions {
            res.push((s, v * new_state.temperature * SIUnit::reference_energy()));
        }
        res
    }

    /// Residual entropy $S^\text{res}=\left(\frac{\partial A^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn residual_entropy(&self) -> SINumber {
        -self.get_or_compute_derivative_residual(PartialDerivative::First(DT))
    }

    /// Residual entropy $s^\text{res}=\left(\frac{\partial a^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn residual_molar_entropy(&self) -> SINumber {
        -self.residual_entropy() / self.total_moles
    }

    /// Pressure: $p=-\left(\frac{\partial A}{\partial V}\right)_{T,N_i}$
    pub fn pressure(&self, contributions: Contributions) -> SINumber {
        let ideal_gas = self.density * SIUnit::gas_constant() * self.temperature;
        let residual = -self.get_or_compute_derivative_residual(PartialDerivative::First(DV));
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Residual chemical potential: $\mu_i^\text{res}=\left(\frac{\partial A^\text{res}}{\partial N_i}\right)_{T,V,N_j}$
    pub fn residual_chemical_potential(&self) -> SIArray1 {
        SIArray::from_shape_fn(self.eos.components(), |i| {
            self.get_or_compute_derivative_residual(PartialDerivative::First(DN(i)))
        })
    }

    /// Chemical potential $\mu_i^\text{res}$ evaluated for each contribution of the equation of state.
    pub fn residual_chemical_potential_contributions(&self, component: usize) -> Vec<(String, SINumber)> {
        let new_state = self.derive1(DN(component));
        let contributions = self.eos.evaluate_residual_contributions(&new_state);
        let mut res = Vec::with_capacity(contributions.len());
        for (s, v) in contributions {
            res.push((
                s,
                (v * new_state.temperature).eps * SIUnit::reference_molar_energy(),
            ));
        }
        res
    }

    /// Compressibility factor: $Z=\frac{pV}{NRT}$
    pub fn compressibility(&self, contributions: Contributions) -> f64 {
        (self.pressure(contributions) / (self.density * self.temperature * SIUnit::gas_constant()))
            .into_value()
            .unwrap()
    }

    // pressure derivatives

    /// Partial derivative of pressure w.r.t. volume: $\left(\frac{\partial p}{\partial V}\right)_{T,N_i}$
    pub fn dp_dv(&self, contributions: Contributions) -> SINumber {
        let ideal_gas = -self.density * SIUnit::gas_constant() * self.temperature / self.volume;
        let residual = -self.get_or_compute_derivative_residual(PartialDerivative::Second(DV));
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Partial derivative of pressure w.r.t. density: $\left(\frac{\partial p}{\partial \rho}\right)_{T,N_i}$
    pub fn dp_drho(&self, contributions: Contributions) -> SINumber {
        -self.volume / self.density * self.dp_dv(contributions)
    }

    /// Partial derivative of pressure w.r.t. temperature: $\left(\frac{\partial p}{\partial T}\right)_{V,N_i}$
    pub fn dp_dt(&self, contributions: Contributions) -> SINumber {
        let ideal_gas = self.density * SIUnit::gas_constant();
        let residual =
            -self.get_or_compute_derivative_residual(PartialDerivative::SecondMixed(DV, DT));
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Partial derivative of pressure w.r.t. moles: $\left(\frac{\partial p}{\partial N_i}\right)_{T,V,N_j}$
    pub fn dp_dni(&self, contributions: Contributions) -> SIArray1 {
        match contributions {
            Contributions::IdealGas => {
                SIArray::from_vec(vec![
                    SIUnit::gas_constant() * self.temperature / self.volume;
                    self.eos.components()
                ])
            }
            Contributions::Residual => SIArray::from_shape_fn(self.eos.components(), |i| {
                -self.get_or_compute_derivative_residual(PartialDerivative::SecondMixed(DV, DN(i)))
            }),
            Contributions::Total => SIArray::from_shape_fn(self.eos.components(), |i| {
                -self.get_or_compute_derivative_residual(PartialDerivative::SecondMixed(DV, DN(i)))
                    + SIUnit::gas_constant() * self.temperature / self.volume
            }),
        }
    }

    /// Second partial derivative of pressure w.r.t. volume: $\left(\frac{\partial^2 p}{\partial V^2}\right)_{T,N_j}$
    pub fn d2p_dv2(&self, contributions: Contributions) -> SINumber {
        let ideal_gas = 2.0 * self.density * SIUnit::gas_constant() * self.temperature
            / (self.volume * self.volume);
        let residual = -self.get_or_compute_derivative_residual(PartialDerivative::Third(DV));
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Second partial derivative of pressure w.r.t. density: $\left(\frac{\partial^2 p}{\partial \rho^2}\right)_{T,N_j}$
    pub fn d2p_drho2(&self, contributions: Contributions) -> SINumber {
        self.volume / (self.density * self.density)
            * (self.volume * self.d2p_dv2(contributions) + 2.0 * self.dp_dv(contributions))
    }

    /// Structure factor: $S(0)=k_BT\left(\frac{\partial\rho}{\partial p}\right)_{T,N_i}$
    pub fn structure_factor(&self) -> f64 {
        -(SIUnit::gas_constant() * self.temperature * self.density)
            .to_reduced(self.volume * self.dp_dv(Contributions::Total))
            .unwrap()
    }

    // This function is designed specifically for use in density iterations
    pub(crate) fn p_dpdrho(&self) -> (SINumber, SINumber) {
        let dp_dv = self.dp_dv(Contributions::Total);
        (
            self.pressure(Contributions::Total),
            (-self.volume * dp_dv / self.density),
        )
    }

    /// Partial molar volume: $v_i=\left(\frac{\partial V}{\partial N_i}\right)_{T,p,N_j}$
    pub fn partial_molar_volume(&self) -> SIArray1 {
        -self.dp_dni(Contributions::Total) / self.dp_dv(Contributions::Total)
    }

    /// Partial derivative of chemical potential w.r.t. moles: $\left(\frac{\partial\mu_i}{\partial N_j}\right)_{T,V,N_k}$
    pub fn dmu_dni(&self, contributions: Contributions) -> SIArray2 {
        let n = self.eos.components();
        SIArray::from_shape_fn((n, n), |(i, j)| {
            let ideal_gas = if i == j {
                SIUnit::gas_constant() * self.temperature / self.moles.get(i)
            } else {
                0.0 * SIUnit::reference_molar_energy() / SIUnit::reference_moles()
            };
            let residual = self
                .get_or_compute_derivative_residual(PartialDerivative::SecondMixed(DN(i), DN(j)));
            Self::contributions(ideal_gas, residual, contributions)
        })
    }

    // This function is designed specifically for use in spinodal iterations
    pub(crate) fn d2pdrho2(&self) -> (SINumber, SINumber, SINumber) {
        let d2p_dv2 = self.d2p_dv2(Contributions::Total);
        let dp_dv = self.dp_dv(Contributions::Total);
        (
            self.pressure(Contributions::Total),
            (-self.volume * dp_dv / self.density),
            (self.volume / (self.density * self.density) * (2.0 * dp_dv + self.volume * d2p_dv2)),
        )
    }

    /// Isothermal compressibility: $\kappa_T=-\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_{T,N_i}$
    pub fn isothermal_compressibility(&self) -> SINumber {
        -1.0 / (self.dp_dv(Contributions::Total) * self.volume)
    }

    /// Pressure $p$ evaluated for each contribution of the equation of state.
    pub fn pressure_contributions(&self) -> Vec<(String, SINumber)> {
        let new_state = self.derive1(DV);
        let contributions = self.eos.evaluate_residual_contributions(&new_state);
        let mut res = Vec::with_capacity(contributions.len() + 1);
        res.push((
            "Ideal gas".into(),
            self.density * SIUnit::gas_constant() * self.temperature,
        ));
        for (s, v) in contributions {
            res.push((
                s,
                -(v * new_state.temperature).eps * SIUnit::reference_pressure(),
            ));
        }
        res
    }

    // entropy derivatives

    /// Partial derivative of the residual entropy w.r.t. temperature: $\left(\frac{\partial S^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn ds_res_dt(&self) -> SINumber {
        -self.get_or_compute_derivative_residual(PartialDerivative::Second(DT))
    }

    /// Second partial derivative of the residual entropy w.r.t. temperature: $\left(\frac{\partial^2S^\text{res}}{\partial T^2}\right)_{V,N_i}$
    pub fn d2s_res_dt2(&self) -> SINumber {
        -self.get_or_compute_derivative_residual(PartialDerivative::Third(DT))
    }

    /// Partial derivative of chemical potential w.r.t. temperature: $\left(\frac{\partial\mu_i}{\partial T}\right)_{V,N_i}$
    pub fn dmu_res_dt(&self) -> SIArray1 {
        SIArray::from_shape_fn(self.eos.components(), |i| {
            self.get_or_compute_derivative_residual(PartialDerivative::SecondMixed(DT, DN(i)))
        })
    }

    /// Logarithm of the fugacity coefficient: $\ln\varphi_i=\beta\mu_i^\mathrm{res}\left(T,p,\lbrace N_i\rbrace\right)$
    pub fn ln_phi(&self) -> Array1<f64> {
        (self.residual_chemical_potential() / (SIUnit::gas_constant() * self.temperature))
            .into_value()
            .unwrap()
            - self.compressibility(Contributions::Total).ln()
    }

    /// Logarithm of the fugacity coefficient of all components treated as pure substance at mixture temperature and pressure.
    pub fn ln_phi_pure_liquid(&self) -> EosResult<Array1<f64>> {
        let pressure = self.pressure(Contributions::Total);
        (0..self.eos.components())
            .map(|i| {
                let eos = Arc::new(self.eos.subset(&[i]));
                let state = Self::new_npt(
                    &eos,
                    self.temperature,
                    pressure,
                    &(arr1(&[1.0]) * SIUnit::reference_moles()),
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
            _ => Ok(self.ln_phi() - &self.ln_phi_pure_liquid()?),
        }
    }

    /// Partial derivative of the logarithm of the fugacity coefficient w.r.t. temperature: $\left(\frac{\partial\ln\varphi_i}{\partial T}\right)_{p,N_i}$
    pub fn dln_phi_dt(&self) -> SIArray1 {
        let vi = self.partial_molar_volume();
        (self.dmu_res_dt()
            - self.residual_chemical_potential() / self.temperature
            - vi * self.dp_dt(Contributions::Total))
            / (SIUnit::gas_constant() * self.temperature)
            + 1.0 / self.temperature
    }

    /// Partial derivative of the logarithm of the fugacity coefficient w.r.t. pressure: $\left(\frac{\partial\ln\varphi_i}{\partial p}\right)_{T,N_i}$
    pub fn dln_phi_dp(&self) -> SIArray1 {
        self.partial_molar_volume() / (SIUnit::gas_constant() * self.temperature)
            - 1.0 / self.pressure(Contributions::Total)
    }

    /// Partial derivative of the logarithm of the fugacity coefficient w.r.t. moles: $\left(\frac{\partial\ln\varphi_i}{\partial N_j}\right)_{T,p,N_k}$
    pub fn dln_phi_dnj(&self) -> SIArray2 {
        let n = self.eos.components();
        let dmu_dni = self.dmu_dni(Contributions::Residual);
        let dp_dni = self.dp_dni(Contributions::Total);
        let dp_dv = self.dp_dv(Contributions::Total);
        let dp_dn_2 = SIArray::from_shape_fn((n, n), |(i, j)| dp_dni.get(i) * dp_dni.get(j));
        (dmu_dni + dp_dn_2 / dp_dv) / (SIUnit::gas_constant() * self.temperature)
            + 1.0 / self.total_moles
    }

    /// Thermodynamic factor: $\Gamma_{ij}=\delta_{ij}+x_i\left(\frac{\partial\ln\varphi_i}{\partial x_j}\right)_{T,p,\Sigma}$
    pub fn thermodynamic_factor(&self) -> Array2<f64> {
        let dln_phi_dnj = self
            .dln_phi_dnj()
            .to_reduced(SIUnit::reference_moles().powi(-1))
            .unwrap();
        let moles = self.moles.to_reduced(SIUnit::reference_moles()).unwrap();
        let n = self.eos.components() - 1;
        Array2::from_shape_fn((n, n), |(i, j)| {
            moles[i] * (dln_phi_dnj[[i, j]] - dln_phi_dnj[[i, n]]) + if i == j { 1.0 } else { 0.0 }
        })
    }

    /// Residual molar isochoric heat capacity: $c_v^\text{res}=\left(\frac{\partial u^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn residual_molar_isochoric_heat_capacity(&self) -> SINumber {
        self.temperature * self.ds_res_dt() / self.total_moles
    }

    /// Partial derivative of the residual molar isochoric heat capacity w.r.t. temperature: $\left(\frac{\partial c_V^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn dc_v_res_dt(&self) -> SINumber {
        (self.temperature * self.d2s_res_dt2() + self.ds_res_dt()) / self.total_moles
    }

    /// Residual molar isobaric heat capacity: $c_p^\text{res}=\left(\frac{\partial h^\text{res}}{\partial T}\right)_{p,N_i}$
    pub fn residual_molar_isobaric_heat_capacity(&self) -> SINumber {
        self.temperature / self.total_moles
            * (self.ds_res_dt()
                - self.dp_dt(Contributions::Total).powi(2) / self.dp_dv(Contributions::Total))
            - SIUnit::gas_constant()
    }

    /// Residual enthalpy: $H^\text{res}(T,p,\mathbf{n})=A^\text{res}+TS^\text{res}+p^\text{res}V$
    pub fn residual_enthalpy(&self) -> SINumber {
        self.temperature * self.residual_entropy()
            + self.residual_helmholtz_energy()
            + self.pressure(Contributions::Residual) * self.volume
    }

    /// Residual molar enthalpy: $h^\text{res}(T,p,\mathbf{n})=a^\text{res}+Ts^\text{res}+p^\text{res}v$
    pub fn residual_molar_enthalpy(&self) -> SINumber {
        self.residual_enthalpy() / self.total_moles
    }

    /// Residual internal energy: $U^\text{res}(T,V,\mathbf{n})=A^\text{res}+TS^\text{res}$
    pub fn residual_internal_energy(&self) -> SINumber {
        self.temperature * self.residual_entropy() + self.residual_helmholtz_energy()
    }

    /// Residual molar internal energy: $u^\text{res}(T,V,\mathbf{n})=a^\text{res}+Ts^\text{res}$
    pub fn residual_molar_internal_energy(&self) -> SINumber {
        self.residual_internal_energy() / self.total_moles
    }

    /// Residual Gibbs energy: $G^\text{res}(T,p,\mathbf{n})=A^\text{res}+p^\text{res}V-NRT \ln Z$
    pub fn residual_gibbs_energy(&self) -> SINumber {
        self.pressure(Contributions::Residual) * self.volume + self.residual_helmholtz_energy()
            - self.total_moles
                * SIUnit::gas_constant()
                * self.temperature
                * self.compressibility(Contributions::Total).ln()
    }

    /// Residual Gibbs energy: $g^\text{res}(T,p,\mathbf{n})=a^\text{res}+p^\text{res}v-RT \ln Z$
    pub fn residual_molar_gibbs_energy(&self) -> SINumber {
        self.residual_gibbs_energy() / self.total_moles
    }

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

/// # Transport properties
///
/// These properties are available for equations of state
/// that implement the [EntropyScaling] trait.
impl<E: Residual + EntropyScaling> State<E> {
    /// Return the viscosity via entropy scaling.
    pub fn viscosity(&self) -> EosResult<SINumber> {
        let s = self.residual_molar_entropy()
            .to_reduced(SIUnit::reference_molar_entropy())?;
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
        let s = self.residual_molar_entropy()
            .to_reduced(SIUnit::reference_molar_entropy())?;
        self.eos.viscosity_correlation(s, &self.molefracs)
    }

    /// Return the viscosity reference as used in entropy scaling.
    pub fn viscosity_reference(&self) -> EosResult<SINumber> {
        self.eos
            .viscosity_reference(self.temperature, self.volume, &self.moles)
    }

    /// Return the diffusion via entropy scaling.
    pub fn diffusion(&self) -> EosResult<SINumber> {
        let s = self.residual_molar_entropy()
            .to_reduced(SIUnit::reference_molar_entropy())?;
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
        let s = self.residual_molar_entropy()
            .to_reduced(SIUnit::reference_molar_entropy())?;
        self.eos.diffusion_correlation(s, &self.molefracs)
    }

    /// Return the diffusion reference as used in entropy scaling.
    pub fn diffusion_reference(&self) -> EosResult<SINumber> {
        self.eos
            .diffusion_reference(self.temperature, self.volume, &self.moles)
    }

    /// Return the thermal conductivity via entropy scaling.
    pub fn thermal_conductivity(&self) -> EosResult<SINumber> {
        let s = self.residual_molar_entropy()
            .to_reduced(SIUnit::reference_molar_entropy())?;
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
        let s = self.residual_molar_entropy()
            .to_reduced(SIUnit::reference_molar_entropy())?;
        self.eos
            .thermal_conductivity_correlation(s, &self.molefracs)
    }

    /// Return the thermal conductivity reference as used in entropy scaling.
    pub fn thermal_conductivity_reference(&self) -> EosResult<SINumber> {
        self.eos
            .thermal_conductivity_reference(self.temperature, self.volume, &self.moles)
    }
}
