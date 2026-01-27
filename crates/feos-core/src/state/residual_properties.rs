use super::{Contributions, State};
use crate::equation_of_state::{EntropyScaling, Molarweight, Residual, Subset};
use crate::{FeosResult, PhaseEquilibrium, ReferenceSystem};
use nalgebra::allocator::Allocator;
use nalgebra::{DMatrix, DVector, DefaultAllocator, OMatrix, OVector, dvector};
use num_dual::{Dual, DualNum, Gradients, partial, partial2};
use quantity::*;
use std::ops::{Add, Div, Neg, Sub};

type InvT<T> = Quantity<T, <_Temperature as Neg>::Output>;
type InvP<T> = Quantity<T, <_Pressure as Neg>::Output>;
type POverT<T> = Quantity<T, <_Pressure as Sub<_Temperature>>::Output>;

/// # State properties
impl<E: Residual<N, D>, N: Gradients, D: DualNum<f64> + Copy> State<E, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    pub(super) fn contributions<
        T: Add<T, Output = T>,
        U,
        I: FnOnce() -> Quantity<T, U>,
        R: FnOnce() -> Quantity<T, U>,
    >(
        ideal_gas: I,
        residual: R,
        contributions: Contributions,
    ) -> Quantity<T, U> {
        match contributions {
            Contributions::IdealGas => ideal_gas(),
            Contributions::Total => ideal_gas() + residual(),
            Contributions::Residual => residual(),
        }
    }

    /// Residual Helmholtz energy $A^\text{res}$
    pub fn residual_helmholtz_energy(&self) -> Energy<D> {
        self.residual_molar_helmholtz_energy() * self.total_moles()
    }

    /// Residual molar Helmholtz energy $a^\text{res}$
    pub fn residual_molar_helmholtz_energy(&self) -> MolarEnergy<D> {
        *self.cache.a.get_or_init(|| {
            self.eos.residual_molar_helmholtz_energy(
                self.temperature,
                self.molar_volume,
                &self.molefracs,
            )
        })
    }

    /// Residual entropy $S^\text{res}=\left(\frac{\partial A^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn residual_entropy(&self) -> Entropy<D> {
        self.residual_molar_entropy() * self.total_moles()
    }

    /// Residual molar entropy $s^\text{res}=\left(\frac{\partial a^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn residual_molar_entropy(&self) -> MolarEntropy<D> {
        -*self.cache.da_dt.get_or_init(|| {
            let (a, da_dt) = quantity::ad::first_derivative(
                partial2(
                    |t, &v, n| self.eos.lift().residual_molar_helmholtz_energy(t, v, n),
                    &self.molar_volume,
                    &self.molefracs,
                ),
                self.temperature,
            );
            let _ = self.cache.a.set(a);
            da_dt
        })
    }

    /// Pressure: $p=-\left(\frac{\partial A}{\partial V}\right)_{T,N_i}$
    pub fn pressure(&self, contributions: Contributions) -> Pressure<D> {
        let ideal_gas = || self.density * RGAS * self.temperature;
        let residual = || {
            -*self.cache.da_dv.get_or_init(|| {
                let (a, da_dv) = quantity::ad::first_derivative(
                    partial2(
                        |v, &t, n| self.eos.lift().residual_molar_helmholtz_energy(t, v, n),
                        &self.temperature,
                        &self.molefracs,
                    ),
                    self.molar_volume,
                );
                let _ = self.cache.a.set(a);
                da_dv
            })
        };
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Residual chemical potential: $\mu_i^\text{res}=\left(\frac{\partial A^\text{res}}{\partial N_i}\right)_{T,V,N_j}$
    pub fn residual_chemical_potential(&self) -> MolarEnergy<OVector<D, N>> {
        self.cache
            .da_dn
            .get_or_init(|| {
                let (a, mu) = quantity::ad::gradient_copy(
                    partial2(
                        |n: Dimensionless<_>, &t, &v| {
                            self.eos.lift().residual_molar_helmholtz_energy(t, v, &n)
                        },
                        &self.temperature,
                        &self.molar_volume,
                    ),
                    &Dimensionless::new(self.molefracs.clone()),
                );
                let _ = self.cache.a.set(a);
                mu
            })
            .clone()
    }

    /// Compressibility factor: $Z=\frac{pV}{NRT}$
    pub fn compressibility(&self, contributions: Contributions) -> D {
        (self.pressure(contributions) / (self.density * self.temperature * RGAS)).into_value()
    }

    // pressure derivatives

    /// Partial derivative of pressure w.r.t. molar volume: $\left(\frac{\partial p}{\partial v}\right)_{T,N_i}$
    pub fn dp_dv(
        &self,
        contributions: Contributions,
    ) -> <Pressure<D> as Div<MolarVolume<D>>>::Output {
        let ideal_gas = || -self.density * RGAS * self.temperature / self.molar_volume;
        let residual = || {
            -*self.cache.d2a_dv2.get_or_init(|| {
                let (a, da_dv, d2a_dv2) = quantity::ad::second_derivative(
                    partial2(
                        |v, &t, n| self.eos.lift().residual_molar_helmholtz_energy(t, v, n),
                        &self.temperature,
                        &self.molefracs,
                    ),
                    self.molar_volume,
                );
                let _ = self.cache.a.set(a);
                let _ = self.cache.da_dv.set(da_dv);
                d2a_dv2
            })
        };
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Partial derivative of pressure w.r.t. density: $\left(\frac{\partial p}{\partial \rho}\right)_{T,N_i}$
    pub fn dp_drho(
        &self,
        contributions: Contributions,
    ) -> <Pressure<D> as Div<Density<D>>>::Output {
        -self.molar_volume / self.density * self.dp_dv(contributions)
    }

    /// Partial derivative of pressure w.r.t. temperature: $\left(\frac{\partial p}{\partial T}\right)_{V,N_i}$
    pub fn dp_dt(&self, contributions: Contributions) -> POverT<D> {
        let ideal_gas = || self.density * RGAS;
        let residual = || {
            -*self.cache.d2a_dtdv.get_or_init(|| {
                let (a, da_dt, da_dv, d2a_dtdv) = quantity::ad::second_partial_derivative(
                    partial(
                        |(t, v), n| self.eos.lift().residual_molar_helmholtz_energy(t, v, n),
                        &self.molefracs,
                    ),
                    (self.temperature, self.molar_volume),
                );
                let _ = self.cache.a.set(a);
                let _ = self.cache.da_dt.set(da_dt);
                let _ = self.cache.da_dv.set(da_dv);
                d2a_dtdv
            })
        };
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Partial derivative of pressure w.r.t. moles: $N\left(\frac{\partial p}{\partial N_i}\right)_{T,V,N_j}$
    pub fn n_dp_dni(&self, contributions: Contributions) -> Pressure<OVector<D, N>> {
        let residual = -self
            .cache
            .d2a_dndv
            .get_or_init(|| {
                let (a, da_dn, da_dv, dmu_dv) = quantity::ad::partial_hessian_copy(
                    partial(
                        |(n, v): (Dimensionless<_>, _), &t| {
                            self.eos.lift().residual_molar_helmholtz_energy(t, v, &n)
                        },
                        &self.temperature,
                    ),
                    (
                        &Dimensionless::new(self.molefracs.clone()),
                        self.molar_volume,
                    ),
                );
                let _ = self.cache.a.set(a);
                let _ = self.cache.da_dn.set(da_dn);
                let _ = self.cache.da_dv.set(da_dv);
                dmu_dv
            })
            .clone();
        let (r, c) = residual.shape_generic();
        let ideal_gas = || self.temperature * self.density * RGAS;
        Quantity::from_fn_generic(r, c, |i, _| {
            Self::contributions(ideal_gas, || residual.get(i), contributions)
        })
    }

    /// Second partial derivative of pressure w.r.t. volume: $\left(\frac{\partial^2 p}{\partial V^2}\right)_{T,N_j}$
    pub fn d2p_dv2(
        &self,
        contributions: Contributions,
    ) -> <<Pressure<D> as Div<MolarVolume<D>>>::Output as Div<MolarVolume<D>>>::Output {
        let ideal_gas = || {
            self.density * RGAS * self.temperature / (self.molar_volume * self.molar_volume) * 2.0
        };
        let residual = || {
            -*self.cache.d3a_dv3.get_or_init(|| {
                let (a, da_dv, d2a_dv2, d3a_dv3) = quantity::ad::third_derivative(
                    partial2(
                        |v, &t, n| self.eos.lift().residual_molar_helmholtz_energy(t, v, n),
                        &self.temperature,
                        &self.molefracs,
                    ),
                    self.molar_volume,
                );
                let _ = self.cache.a.set(a);
                let _ = self.cache.da_dv.set(da_dv);
                let _ = self.cache.d2a_dv2.set(d2a_dv2);
                d3a_dv3
            })
        };
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Second partial derivative of pressure w.r.t. density: $\left(\frac{\partial^2 p}{\partial \rho^2}\right)_{T,N_j}$
    pub fn d2p_drho2(
        &self,
        contributions: Contributions,
    ) -> <<Pressure<D> as Div<Density<D>>>::Output as Div<Density<D>>>::Output {
        self.molar_volume.powi::<3>()
            * (self.molar_volume * self.d2p_dv2(contributions) + self.dp_dv(contributions) * 2.0)
    }

    /// Structure factor: $S(0)=k_BT\left(\frac{\partial\rho}{\partial p}\right)_{T,N_i}$
    pub fn structure_factor(&self) -> D {
        -(self.temperature * self.density * RGAS
            / (self.molar_volume * self.dp_dv(Contributions::Total)))
        .into_value()
    }

    /// Partial molar volume: $v_i=\left(\frac{\partial V}{\partial N_i}\right)_{T,p,N_j}$
    pub fn partial_molar_volume(&self) -> MolarVolume<OVector<D, N>> {
        -self.n_dp_dni(Contributions::Total) / self.dp_dv(Contributions::Total)
    }

    /// Partial derivative of chemical potential w.r.t. moles: $N\left(\frac{\partial\mu_i}{\partial N_j}\right)_{T,V,N_k}$
    pub fn n_dmu_dni(&self, contributions: Contributions) -> MolarEnergy<OMatrix<D, N, N>>
    where
        DefaultAllocator: Allocator<N, N>,
    {
        let (a, da_dn, d2a_dn2) = quantity::ad::hessian_copy(
            partial2(
                |n: Dimensionless<_>, &t, &v| {
                    self.eos.lift().residual_molar_helmholtz_energy(t, v, &n)
                },
                &self.temperature,
                &self.molar_volume,
            ),
            &Dimensionless::new(self.molefracs.clone()),
        );
        let _ = self.cache.a.set(a);
        let _ = self.cache.da_dn.set(da_dn);
        let residual = || d2a_dn2;
        let ideal_gas = || {
            Dimensionless::new(OMatrix::from_diagonal(&self.molefracs.map(|x| x.recip())))
                * (self.temperature * RGAS)
        };
        Self::contributions(ideal_gas, residual, contributions)
    }

    /// Isothermal compressibility: $\kappa_T=-\frac{1}{V}\left(\frac{\partial V}{\partial p}\right)_{T,N_i}$
    pub fn isothermal_compressibility(&self) -> InvP<D> {
        (self.dp_dv(Contributions::Total) * self.molar_volume).inv()
    }

    // entropy derivatives

    /// Partial derivative of the residual molar entropy w.r.t. temperature: $\left(\frac{\partial s^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn ds_res_dt(&self) -> <MolarEntropy<D> as Div<Temperature<D>>>::Output {
        -*self.cache.d2a_dt2.get_or_init(|| {
            let (a, da_dt, d2a_dt2) = quantity::ad::second_derivative(
                partial2(
                    |t, &v, n| self.eos.lift().residual_molar_helmholtz_energy(t, v, n),
                    &self.molar_volume,
                    &self.molefracs,
                ),
                self.temperature,
            );
            let _ = self.cache.a.set(a);
            let _ = self.cache.da_dt.set(da_dt);
            d2a_dt2
        })
    }

    /// Second partial derivative of the residual molar entropy w.r.t. temperature: $\left(\frac{\partial^2s^\text{res}}{\partial T^2}\right)_{V,N_i}$
    pub fn d2s_res_dt2(
        &self,
    ) -> <<MolarEntropy<D> as Div<Temperature<D>>>::Output as Div<Temperature<D>>>::Output {
        -*self.cache.d3a_dt3.get_or_init(|| {
            let (a, da_dt, d2a_dt2, d3a_dt3) = quantity::ad::third_derivative(
                partial2(
                    |t, &v, n| self.eos.lift().residual_molar_helmholtz_energy(t, v, n),
                    &self.molar_volume,
                    &self.molefracs,
                ),
                self.temperature,
            );
            let _ = self.cache.a.set(a);
            let _ = self.cache.da_dt.set(da_dt);
            let _ = self.cache.d2a_dt2.set(d2a_dt2);
            d3a_dt3
        })
    }

    /// Partial derivative of chemical potential w.r.t. temperature: $\left(\frac{\partial\mu_i}{\partial T}\right)_{V,N_i}$
    pub fn dmu_res_dt(&self) -> MolarEntropy<OVector<D, N>> {
        self.cache
            .d2a_dndt
            .get_or_init(|| {
                let (a, da_dn, da_dt, d2a_dndt) = quantity::ad::partial_hessian_copy(
                    partial(
                        |(n, t): (Dimensionless<_>, _), &v| {
                            self.eos.lift().residual_molar_helmholtz_energy(t, v, &n)
                        },
                        &self.molar_volume,
                    ),
                    (
                        &Dimensionless::new(self.molefracs.clone()),
                        self.temperature,
                    ),
                );
                let _ = self.cache.a.set(a);
                let _ = self.cache.da_dn.set(da_dn);
                let _ = self.cache.da_dt.set(da_dt);
                d2a_dndt
            })
            .clone()
    }

    /// Logarithm of the fugacity coefficient: $\ln\varphi_i=\beta\mu_i^\mathrm{res}\left(T,p,\lbrace N_i\rbrace\right)$
    pub fn ln_phi(&self) -> OVector<D, N> {
        let mu_res = self.residual_chemical_potential();
        let ln_z = self.compressibility(Contributions::Total).ln();
        (mu_res / (self.temperature * RGAS))
            .into_value()
            .map(|mu| mu - ln_z)
    }

    /// Partial derivative of the logarithm of the fugacity coefficient w.r.t. temperature: $\left(\frac{\partial\ln\varphi_i}{\partial T}\right)_{p,N_i}$
    pub fn dln_phi_dt(&self) -> InvT<OVector<D, N>> {
        let vi = self.partial_molar_volume();
        ((self.dmu_res_dt()
            - self.residual_chemical_potential() / self.temperature
            - vi * self.dp_dt(Contributions::Total))
            / (self.temperature * RGAS))
            .add_scalar(self.temperature.inv())
    }

    /// Partial derivative of the logarithm of the fugacity coefficient w.r.t. pressure: $\left(\frac{\partial\ln\varphi_i}{\partial p}\right)_{T,N_i}$
    pub fn dln_phi_dp(&self) -> InvP<OVector<D, N>> {
        (self.partial_molar_volume() / (self.temperature * RGAS))
            .add_scalar(-self.pressure(Contributions::Total).inv())
    }

    /// Partial derivative of the logarithm of the fugacity coefficient w.r.t. moles: $N\left(\frac{\partial\ln\varphi_i}{\partial N_j}\right)_{T,p,N_k}$
    pub fn n_dln_phi_dnj(&self) -> OMatrix<D, N, N>
    where
        DefaultAllocator: Allocator<N, N>,
    {
        let dmu_dni = self.n_dmu_dni(Contributions::Residual);
        let dp_dni = self.n_dp_dni(Contributions::Total);
        let dp_dv = self.dp_dv(Contributions::Total);
        let (r, c) = dmu_dni.shape_generic();
        let dp_dn_2 = Quantity::from_fn_generic(r, c, |i, j| dp_dni.get(i) * dp_dni.get(j));
        ((dmu_dni + dp_dn_2 / dp_dv) / (self.temperature * RGAS))
            .into_value()
            .add_scalar(D::from(1.0))
    }
}

impl<E: Residual + Subset> State<E> {
    /// Logarithm of the fugacity coefficient of all components treated as pure substance at mixture temperature and pressure.
    pub fn ln_phi_pure_liquid(&self) -> FeosResult<DVector<f64>> {
        let pressure = self.pressure(Contributions::Total);
        (0..self.eos.components())
            .map(|i| {
                let eos = self.eos.subset(&[i]);
                let state = State::new_npt(
                    &eos,
                    self.temperature,
                    pressure,
                    dvector![1.0],
                    Some(crate::DensityInitialization::Liquid),
                )?;
                Ok(state.ln_phi()[0])
            })
            .collect::<FeosResult<Vec<_>>>()
            .map(DVector::from)
    }

    /// Activity coefficient $\ln \gamma_i = \ln \varphi_i(T, p, \mathbf{N}) - \ln \varphi_i^\mathrm{pure}(T, p)$
    pub fn ln_symmetric_activity_coefficient(&self) -> FeosResult<DVector<f64>> {
        Ok(match self.eos.components() {
            1 => dvector![0.0],
            _ => self.ln_phi() - &self.ln_phi_pure_liquid()?,
        })
    }

    /// Henry's law constant $H_{i,s}=\lim_{x_i\to 0}\frac{y_ip}{x_i}=p_s^\mathrm{sat}\frac{\varphi_i^{\infty,\mathrm{L}}}{\varphi_i^{\infty,\mathrm{V}}}$
    ///
    /// The composition of the (possibly mixed) solvent is determined by the molefracs. All components for which the composition is 0 are treated as solutes.
    ///
    /// For some reason the compiler is overwhelmed if returning a quantity array, therefore it is returned as list.
    pub fn henrys_law_constant(
        eos: &E,
        temperature: Temperature,
        molefracs: &DVector<f64>,
    ) -> FeosResult<Vec<Pressure>> {
        // Calculate the phase equilibrium (bubble point) of the solvent only
        let (solvent_comps, solvent_molefracs): (Vec<_>, Vec<_>) = molefracs
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| (x != 0.0).then_some((i, x)))
            .unzip();
        let solvent_molefracs = DVector::from_vec(solvent_molefracs);
        let solvent = eos.subset(&solvent_comps);
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
        let liquid = State::new(eos, temperature, vle.liquid().density, molefracs.clone())?;

        // Calculate the vapor state including the Henry components
        let mut molefracs_vapor = molefracs.clone();
        solvent_comps
            .into_iter()
            .zip(&vle.vapor().molefracs)
            .for_each(|(i, &y)| molefracs_vapor[i] = y);
        let vapor = State::new(eos, temperature, vle.vapor().density, molefracs.clone())?;

        // Determine the Henry's law coefficients and return only those of the Henry components
        let p = vle.vapor().pressure(Contributions::Total).into_reduced();
        let h = (liquid.ln_phi() - vapor.ln_phi()).map(f64::exp) * p;
        Ok(h.into_iter()
            .zip(molefracs)
            .filter_map(|(h, &x)| (x == 0.0).then_some(h))
            .map(|&h| Pressure::from_reduced(h))
            .collect())
    }

    /// Henry's law constant $H_{i,s}=\lim_{x_i\to 0}\frac{y_ip}{x_i}=p_s^\mathrm{sat}\frac{\varphi_i^{\infty,\mathrm{L}}}{\varphi_i^{\infty,\mathrm{V}}}$ for a binary system
    ///
    /// The solute (i) is the first component and the solvent (s) the second component.
    pub fn henrys_law_constant_binary(eos: &E, temperature: Temperature) -> FeosResult<Pressure> {
        Ok(Self::henrys_law_constant(eos, temperature, &dvector![0.0, 1.0])?[0])
    }
}

impl<E: Residual> State<E> {
    /// Thermodynamic factor: $\Gamma_{ij}=\delta_{ij}+x_i\left(\frac{\partial\ln\varphi_i}{\partial x_j}\right)_{T,p,\Sigma}$
    pub fn thermodynamic_factor(&self) -> DMatrix<f64> {
        let dln_phi_dnj = self.n_dln_phi_dnj();
        let n = self.eos.components() - 1;
        DMatrix::from_fn(n, n, |i, j| {
            dln_phi_dnj[(i, j)] - dln_phi_dnj[(i, n)] + if i == j { 1.0 } else { 0.0 }
        })
    }
}

impl<E: Residual<N, D>, N: Gradients, D: DualNum<f64> + Copy> State<E, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    /// Residual molar isochoric heat capacity: $c_v^\text{res}=\left(\frac{\partial u^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn residual_molar_isochoric_heat_capacity(&self) -> MolarEntropy<D> {
        self.ds_res_dt() * self.temperature
    }

    /// Partial derivative of the residual molar isochoric heat capacity w.r.t. temperature: $\left(\frac{\partial c_V^\text{res}}{\partial T}\right)_{V,N_i}$
    pub fn dc_v_res_dt(&self) -> <MolarEntropy<D> as Div<Temperature<D>>>::Output {
        self.temperature * self.d2s_res_dt2() + self.ds_res_dt()
    }

    /// Residual molar isobaric heat capacity: $c_p^\text{res}=\left(\frac{\partial h^\text{res}}{\partial T}\right)_{p,N_i}$
    pub fn residual_molar_isobaric_heat_capacity(&self) -> MolarEntropy<D> {
        let dp_dt = self.dp_dt(Contributions::Total);
        self.temperature * (self.ds_res_dt() - dp_dt * dp_dt / self.dp_dv(Contributions::Total))
            - RGAS
    }

    /// Residual enthalpy: $H^\text{res}(T,p,\mathbf{n})=A^\text{res}+TS^\text{res}+p^\text{res}V$
    pub fn residual_enthalpy(&self) -> Energy<D> {
        self.residual_molar_enthalpy() * self.total_moles()
    }

    /// Residual molar enthalpy: $h^\text{res}(T,p,\mathbf{n})=a^\text{res}+Ts^\text{res}+p^\text{res}v$
    pub fn residual_molar_enthalpy(&self) -> MolarEnergy<D> {
        self.temperature * self.residual_molar_entropy()
            + self.residual_molar_helmholtz_energy()
            + self.pressure(Contributions::Residual) * self.molar_volume
    }

    /// Residual internal energy: $U^\text{res}(T,V,\mathbf{n})=A^\text{res}+TS^\text{res}$
    pub fn residual_internal_energy(&self) -> Energy<D> {
        self.residual_molar_internal_energy() * self.total_moles()
    }

    /// Residual molar internal energy: $u^\text{res}(T,V,\mathbf{n})=a^\text{res}+Ts^\text{res}$
    pub fn residual_molar_internal_energy(&self) -> MolarEnergy<D> {
        self.temperature * self.residual_molar_entropy() + self.residual_molar_helmholtz_energy()
    }

    /// Residual Gibbs energy: $G^\text{res}(T,p,\mathbf{n})=A^\text{res}+p^\text{res}V-NRT \ln Z$
    pub fn residual_gibbs_energy(&self) -> Energy<D> {
        self.residual_molar_gibbs_energy() * self.total_moles()
    }

    /// Residual Gibbs energy: $g^\text{res}(T,p,\mathbf{n})=a^\text{res}+p^\text{res}v-RT \ln Z$
    pub fn residual_molar_gibbs_energy(&self) -> MolarEnergy<D> {
        self.pressure(Contributions::Residual) * self.molar_volume
            + self.residual_molar_helmholtz_energy()
            - self.temperature
                * RGAS
                * Dimensionless::new(self.compressibility(Contributions::Total).ln())
    }

    /// Molar Helmholtz energy $a^\text{res}$ evaluated for each residual contribution of the equation of state.
    pub fn residual_molar_helmholtz_energy_contributions(
        &self,
    ) -> Vec<(&'static str, MolarEnergy<D>)> {
        let residual_contributions = self.eos.helmholtz_energy_contributions(
            self.temperature.into_reduced(),
            self.density.into_reduced().recip(),
            &self.molefracs,
        );
        let mut res = Vec::with_capacity(residual_contributions.len());
        for (s, v) in residual_contributions {
            res.push((s, MolarEnergy::from_reduced(v)));
        }
        res
    }

    /// Chemical potential $\mu_i^\text{res}$ evaluated for each residual contribution of the equation of state.
    pub fn residual_chemical_potential_contributions(
        &self,
        component: usize,
    ) -> Vec<(&'static str, MolarEnergy<D>)> {
        let t = Dual::from_re(self.temperature.into_reduced());
        let v = Dual::from_re(self.temperature.into_reduced());
        let mut x = self.molefracs.map(Dual::from_re);
        x[component].eps = D::one();
        let contributions = self.eos.lift().helmholtz_energy_contributions(t, v, &x);
        let mut res = Vec::with_capacity(contributions.len());
        for (s, v) in contributions {
            res.push((s, MolarEnergy::from_reduced(v.eps)));
        }
        res
    }

    /// Pressure $p$ evaluated for each contribution of the equation of state.
    pub fn pressure_contributions(&self) -> Vec<(&'static str, Pressure<D>)> {
        let t = Dual::from_re(self.temperature.into_reduced());
        let v = Dual::from_re(self.density.into_reduced().recip()).derivative();
        let x = self.molefracs.map(Dual::from_re);
        let contributions = self.eos.lift().helmholtz_energy_contributions(t, v, &x);
        let mut res = Vec::with_capacity(contributions.len() + 1);
        res.push(("Ideal gas", self.density * RGAS * self.temperature));
        for (s, v) in contributions {
            res.push((s, Pressure::from_reduced(-v.eps)));
        }
        res
    }
}

impl<E: Residual<N, D> + Molarweight<N, D>, N: Gradients, D: DualNum<f64> + Copy> State<E, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    /// Total molar weight: $MW=\sum_ix_iMW_i$
    pub fn total_molar_weight(&self) -> MolarWeight<D> {
        self.eos
            .molar_weight()
            .dot(&Dimensionless::new(self.molefracs.clone()))
    }

    /// Mass of each component: $m_i=n_iMW_i$
    pub fn mass(&self) -> Mass<OVector<D, N>> {
        self.eos
            .molar_weight()
            .component_mul(&Dimensionless::new(self.molefracs.clone()))
            * self.total_moles()
    }

    /// Total mass: $m=\sum_im_i=nMW$
    pub fn total_mass(&self) -> Mass<D> {
        self.total_molar_weight() * self.total_moles()
    }

    /// Mass density: $\rho^{(m)}=\frac{m}{V}$
    pub fn mass_density(&self) -> MassDensity<D> {
        self.density * self.total_molar_weight()
    }

    /// Mass fractions: $w_i=\frac{m_i}{m}$
    pub fn massfracs(&self) -> OVector<D, N> {
        self.eos
            .molar_weight()
            .convert_into(self.total_molar_weight())
            .component_mul(&self.molefracs)
    }
}

/// # Transport properties
///
/// These properties are available for equations of state
/// that implement the [EntropyScaling] trait.
impl<E: Residual<N, D> + EntropyScaling<N, D>, N: Gradients, D: DualNum<f64> + Copy> State<E, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    /// Return the viscosity via entropy scaling.
    pub fn viscosity(&self) -> Viscosity<D> {
        let s = self.residual_molar_entropy().into_reduced();
        self.eos
            .viscosity_reference(self.temperature, self.molar_volume, &self.molefracs)
            * Dimensionless::new(self.eos.viscosity_correlation(s, &self.molefracs).exp())
    }

    /// Return the logarithm of the reduced viscosity.
    ///
    /// This term equals the viscosity correlation function
    /// that is used for entropy scaling.
    pub fn ln_viscosity_reduced(&self) -> D {
        let s = self.residual_molar_entropy().into_reduced();
        self.eos.viscosity_correlation(s, &self.molefracs)
    }

    /// Return the viscosity reference as used in entropy scaling.
    pub fn viscosity_reference(&self) -> Viscosity<D> {
        self.eos
            .viscosity_reference(self.temperature, self.molar_volume, &self.molefracs)
    }

    /// Return the diffusion via entropy scaling.
    pub fn diffusion(&self) -> Diffusivity<D> {
        let s = self.residual_molar_entropy().into_reduced();
        self.eos
            .diffusion_reference(self.temperature, self.molar_volume, &self.molefracs)
            * Dimensionless::new(self.eos.diffusion_correlation(s, &self.molefracs).exp())
    }

    /// Return the logarithm of the reduced diffusion.
    ///
    /// This term equals the diffusion correlation function
    /// that is used for entropy scaling.
    pub fn ln_diffusion_reduced(&self) -> D {
        let s = self.residual_molar_entropy().into_reduced();
        self.eos.diffusion_correlation(s, &self.molefracs)
    }

    /// Return the diffusion reference as used in entropy scaling.
    pub fn diffusion_reference(&self) -> Diffusivity<D> {
        self.eos
            .diffusion_reference(self.temperature, self.molar_volume, &self.molefracs)
    }

    /// Return the thermal conductivity via entropy scaling.
    pub fn thermal_conductivity(&self) -> ThermalConductivity<D> {
        let s = self.residual_molar_entropy().into_reduced();
        self.eos.thermal_conductivity_reference(
            self.temperature,
            self.molar_volume,
            &self.molefracs,
        ) * Dimensionless::new(
            self.eos
                .thermal_conductivity_correlation(s, &self.molefracs)
                .exp(),
        )
    }

    /// Return the logarithm of the reduced thermal conductivity.
    ///
    /// This term equals the thermal conductivity correlation function
    /// that is used for entropy scaling.
    pub fn ln_thermal_conductivity_reduced(&self) -> D {
        let s = self.residual_molar_entropy().into_reduced();
        self.eos
            .thermal_conductivity_correlation(s, &self.molefracs)
    }

    /// Return the thermal conductivity reference as used in entropy scaling.
    pub fn thermal_conductivity_reference(&self) -> ThermalConductivity<D> {
        self.eos.thermal_conductivity_reference(
            self.temperature,
            self.molar_volume,
            &self.molefracs,
        )
    }
}
