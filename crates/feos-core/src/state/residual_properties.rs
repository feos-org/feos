use super::{Contributions, Derivative::*, PartialDerivative, State};
use crate::ReferenceSystem;
use crate::equation_of_state::{EntropyScaling, Molarweight, Residual};
use crate::errors::FeosResult;
use crate::phase_equilibria::PhaseEquilibrium;
use ndarray::{Array1, Array2, arr1};
use quantity::*;
use std::ops::{Add, Div};
use std::sync::Arc;
use typenum::P2;

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

impl<E: Residual> State<E> {
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

    /// Residual Helmholtz energy $A^\text{res}$
    pub fn residual_helmholtz_energy(&self) -> Energy {
        Energy::from_reduced(self.get_or_compute_derivative_residual(PartialDerivative::Zeroth))
    }

    /// Residual molar Helmholtz energy $a^\text{res}$
    pub fn residual_molar_helmholtz_energy(&self) -> MolarEnergy {
        self.residual_helmholtz_energy() / self.total_moles
    }

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

    /// Residual entropy $S^\text{res}=\left(\frac{\partial A^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn residual_entropy(&self) -> Entropy {
        Entropy::from_reduced(
            -self.get_or_compute_derivative_residual(PartialDerivative::First(DT)),
        )
    }

    /// Residual entropy $s^\text{res}=\left(\frac{\partial a^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn residual_molar_entropy(&self) -> MolarEntropy {
        self.residual_entropy() / self.total_moles
    }

    /// Pressure: $p=-\left(\frac{\partial A}{\partial V}\right)_{T,N_i}$
    pub fn pressure(&self, contributions: Contributions) -> Pressure {
        let ideal_gas = self.density * RGAS * self.temperature;
        let residual = Pressure::from_reduced(
            -self.get_or_compute_derivative_residual(PartialDerivative::First(DV)),
        );
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Residual chemical potential: $\mu_i^\text{res}=\left(\frac{\partial A^\text{res}}{\partial N_i}\right)_{T,V,N_j}$
    pub fn residual_chemical_potential(&self) -> MolarEnergy<Array1<f64>> {
        MolarEnergy::from_reduced(Array1::from_shape_fn(self.eos.components(), |i| {
            self.get_or_compute_derivative_residual(PartialDerivative::First(DN(i)))
        }))
    }

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

    /// Compressibility factor: $Z=\frac{pV}{NRT}$
    pub fn compressibility(&self, contributions: Contributions) -> f64 {
        (self.pressure(contributions) / (self.density * self.temperature * RGAS)).into_value()
    }

    // pressure derivatives

    /// Partial derivative of pressure w.r.t. volume: $\left(\frac{\partial p}{\partial V}\right)_{T,N_i}$
    pub fn dp_dv(&self, contributions: Contributions) -> <Pressure as Div<Volume>>::Output {
        let ideal_gas = -self.density * RGAS * self.temperature / self.volume;
        let residual = Quantity::from_reduced(
            -self.get_or_compute_derivative_residual(PartialDerivative::Second(DV)),
        );
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Partial derivative of pressure w.r.t. density: $\left(\frac{\partial p}{\partial \rho}\right)_{T,N_i}$
    pub fn dp_drho(&self, contributions: Contributions) -> <Pressure as Div<Density>>::Output {
        -self.volume / self.density * self.dp_dv(contributions)
    }

    /// Partial derivative of pressure w.r.t. temperature: $\left(\frac{\partial p}{\partial T}\right)_{V,N_i}$
    pub fn dp_dt(&self, contributions: Contributions) -> <Pressure as Div<Temperature>>::Output {
        let ideal_gas = self.density * RGAS;
        let residual = Quantity::from_reduced(
            -self.get_or_compute_derivative_residual(PartialDerivative::SecondMixed(DV, DT)),
        );
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Partial derivative of pressure w.r.t. moles: $\left(\frac{\partial p}{\partial N_i}\right)_{T,V,N_j}$
    pub fn dp_dni(
        &self,
        contributions: Contributions,
    ) -> <Pressure as Div<Moles<Array1<f64>>>>::Output {
        let ideal_gas = Quantity::from_vec(vec![
            RGAS * self.temperature / self.volume;
            self.eos.components()
        ]);
        let residual = Quantity::from_reduced(Array1::from_shape_fn(self.eos.components(), |i| {
            -self.get_or_compute_derivative_residual(PartialDerivative::SecondMixed(DV, DN(i)))
        }));
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Second partial derivative of pressure w.r.t. volume: $\left(\frac{\partial^2 p}{\partial V^2}\right)_{T,N_j}$
    pub fn d2p_dv2(
        &self,
        contributions: Contributions,
    ) -> <<Pressure as Div<Volume>>::Output as Div<Volume>>::Output {
        let ideal_gas = 2.0 * self.density * RGAS * self.temperature / (self.volume * self.volume);
        let residual = Quantity::from_reduced(
            -self.get_or_compute_derivative_residual(PartialDerivative::Third(DV)),
        );
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Second partial derivative of pressure w.r.t. density: $\left(\frac{\partial^2 p}{\partial \rho^2}\right)_{T,N_j}$
    pub fn d2p_drho2(
        &self,
        contributions: Contributions,
    ) -> <<Pressure as Div<Density>>::Output as Div<Density>>::Output {
        self.volume / (self.density * self.density)
            * (self.volume * self.d2p_dv2(contributions) + 2.0 * self.dp_dv(contributions))
    }

    /// Structure factor: $S(0)=k_BT\left(\frac{\partial\rho}{\partial p}\right)_{T,N_i}$
    pub fn structure_factor(&self) -> f64 {
        -(RGAS * self.temperature * self.density / (self.volume * self.dp_dv(Contributions::Total)))
            .into_value()
    }

    // This function is designed specifically for use in density iterations
    pub(crate) fn p_dpdrho(&self) -> (Pressure, <Pressure as Div<Density>>::Output) {
        let dp_dv = self.dp_dv(Contributions::Total);
        (
            self.pressure(Contributions::Total),
            (-self.volume * dp_dv / self.density),
        )
    }

    /// Partial molar volume: $v_i=\left(\frac{\partial V}{\partial N_i}\right)_{T,p,N_j}$
    pub fn partial_molar_volume(&self) -> MolarVolume<Array1<f64>> {
        -self.dp_dni(Contributions::Total) / self.dp_dv(Contributions::Total)
    }

    /// Partial derivative of chemical potential w.r.t. moles: $\left(\frac{\partial\mu_i}{\partial N_j}\right)_{T,V,N_k}$
    pub fn dmu_dni(
        &self,
        contributions: Contributions,
    ) -> <MolarEnergy<Array2<f64>> as Div<Moles>>::Output {
        let n = self.eos.components();
        Quantity::from_shape_fn((n, n), |(i, j)| {
            let ideal_gas = if i == j {
                RGAS * self.temperature / self.moles.get(i)
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

    // This function is designed specifically for use in spinodal iterations
    #[expect(clippy::type_complexity)]
    pub(crate) fn d2pdrho2(
        &self,
    ) -> (
        Pressure,
        <Pressure as Div<Density>>::Output,
        <<Pressure as Div<Density>>::Output as Div<Density>>::Output,
    ) {
        let d2p_dv2 = self.d2p_dv2(Contributions::Total);
        let dp_dv = self.dp_dv(Contributions::Total);
        (
            self.pressure(Contributions::Total),
            (-self.volume * dp_dv / self.density),
            (self.volume / (self.density * self.density) * (2.0 * dp_dv + self.volume * d2p_dv2)),
        )
    }

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

    // entropy derivatives

    /// Partial derivative of the residual entropy w.r.t. temperature: $\left(\frac{\partial S^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn ds_res_dt(&self) -> <Entropy as Div<Temperature>>::Output {
        Quantity::from_reduced(
            -self.get_or_compute_derivative_residual(PartialDerivative::Second(DT)),
        )
    }

    /// Second partial derivative of the residual entropy w.r.t. temperature: $\left(\frac{\partial^2S^\text{res}}{\partial T^2}\right)_{V,N_i}$
    pub fn d2s_res_dt2(
        &self,
    ) -> <<Entropy as Div<Temperature>>::Output as Div<Temperature>>::Output {
        Quantity::from_reduced(
            -self.get_or_compute_derivative_residual(PartialDerivative::Third(DT)),
        )
    }

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
        let dp_dni = self.dp_dni(Contributions::Total);
        let dp_dv = self.dp_dv(Contributions::Total);
        let dp_dn_2 = Quantity::from_shape_fn((n, n), |(i, j)| dp_dni.get(i) * dp_dni.get(j));
        (dmu_dni + dp_dn_2 / dp_dv) / (RGAS * self.temperature) + 1.0 / self.total_moles
    }

    /// Thermodynamic factor: $\Gamma_{ij}=\delta_{ij}+x_i\left(\frac{\partial\ln\varphi_i}{\partial x_j}\right)_{T,p,\Sigma}$
    pub fn thermodynamic_factor(&self) -> Array2<f64> {
        let dln_phi_dnj = (self.dln_phi_dnj() * Moles::from_reduced(1.0)).into_value();
        let moles = self.moles.to_reduced();
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

    /// Residual Gibbs energy: $G^\text{res}(T,p,\mathbf{n})=A^\text{res}+p^\text{res}V-NRT \ln Z$
    pub fn residual_gibbs_energy(&self) -> Energy {
        self.pressure(Contributions::Residual) * self.volume + self.residual_helmholtz_energy()
            - self.total_moles
                * RGAS
                * self.temperature
                * self.compressibility(Contributions::Total).ln()
    }

    /// Residual Gibbs energy: $g^\text{res}(T,p,\mathbf{n})=a^\text{res}+p^\text{res}v-RT \ln Z$
    pub fn residual_molar_gibbs_energy(&self) -> MolarEnergy {
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
        self.moles.clone() * self.eos.molar_weight()
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

/// # Transport properties
///
/// These properties are available for equations of state
/// that implement the [EntropyScaling] trait.
impl<E: Residual + EntropyScaling> State<E> {
    /// Return the viscosity via entropy scaling.
    pub fn viscosity(&self) -> FeosResult<Viscosity> {
        let s = self.residual_molar_entropy().to_reduced();
        Ok(self
            .eos
            .viscosity_reference(self.temperature, self.volume, &self.moles)?
            * self.eos.viscosity_correlation(s, &self.molefracs)?.exp())
    }

    /// Return the logarithm of the reduced viscosity.
    ///
    /// This term equals the viscosity correlation function
    /// that is used for entropy scaling.
    pub fn ln_viscosity_reduced(&self) -> FeosResult<f64> {
        let s = self.residual_molar_entropy().to_reduced();
        self.eos.viscosity_correlation(s, &self.molefracs)
    }

    /// Return the viscosity reference as used in entropy scaling.
    pub fn viscosity_reference(&self) -> FeosResult<Viscosity> {
        self.eos
            .viscosity_reference(self.temperature, self.volume, &self.moles)
    }

    /// Return the diffusion via entropy scaling.
    pub fn diffusion(&self) -> FeosResult<Diffusivity> {
        let s = self.residual_molar_entropy().to_reduced();
        Ok(self
            .eos
            .diffusion_reference(self.temperature, self.volume, &self.moles)?
            * self.eos.diffusion_correlation(s, &self.molefracs)?.exp())
    }

    /// Return the logarithm of the reduced diffusion.
    ///
    /// This term equals the diffusion correlation function
    /// that is used for entropy scaling.
    pub fn ln_diffusion_reduced(&self) -> FeosResult<f64> {
        let s = self.residual_molar_entropy().to_reduced();
        self.eos.diffusion_correlation(s, &self.molefracs)
    }

    /// Return the diffusion reference as used in entropy scaling.
    pub fn diffusion_reference(&self) -> FeosResult<Diffusivity> {
        self.eos
            .diffusion_reference(self.temperature, self.volume, &self.moles)
    }

    /// Return the thermal conductivity via entropy scaling.
    pub fn thermal_conductivity(&self) -> FeosResult<ThermalConductivity> {
        let s = self.residual_molar_entropy().to_reduced();
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
    pub fn ln_thermal_conductivity_reduced(&self) -> FeosResult<f64> {
        let s = self.residual_molar_entropy().to_reduced();
        self.eos
            .thermal_conductivity_correlation(s, &self.molefracs)
    }

    /// Return the thermal conductivity reference as used in entropy scaling.
    pub fn thermal_conductivity_reference(&self) -> FeosResult<ThermalConductivity> {
        self.eos
            .thermal_conductivity_reference(self.temperature, self.volume, &self.moles)
    }
}
