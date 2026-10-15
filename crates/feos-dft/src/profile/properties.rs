#![allow(type_alias_bounds)]
use super::{DFTProfile, DFTSpecification};
use crate::association::AssociationFunctional;
use crate::functional::HelmholtzEnergyFunctional;
use crate::functional_contribution::FunctionalContribution;
use crate::solver::{DFTSolverLog, Evaluate, Tape, solve_linearized};
use feos_core::{Contributions, FeosResult, ReferenceSystem, State, Total};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array, Array1, Axis, Dimension, RemoveAxis};
use num_dual::{Dual64, DualNumCopy};
use quantity::{
    Density, Energy, Entropy, EntropyDensity, MolarEnergy, Moles, Pressure, Quantity, Temperature,
};
use rustdct::DctNum;
use std::ops::{AddAssign, Div};

type DrhoDmu<D: Dimension> =
    <Density<Array<f64, <D::Larger as Dimension>::Larger>> as Div<MolarEnergy>>::Output;
type DnDmu = <Moles<DMatrix<f64>> as Div<MolarEnergy>>::Output;
type DrhoDp<D: Dimension> = <Density<Array<f64, D::Larger>> as Div<Pressure>>::Output;
type DnDp = <Moles<DVector<f64>> as Div<Pressure>>::Output;
type DrhoDT<D: Dimension> = <Density<Array<f64, D::Larger>> as Div<Temperature>>::Output;
type DnDT = <Moles<DVector<f64>> as Div<Temperature>>::Output;

impl<D: Dimension, F: HelmholtzEnergyFunctional> DFTProfile<D, F>
where
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    /// Calculate the grand potential density $\omega$ for a converged density profile.
    pub fn grand_potential_density(&self) -> FeosResult<Pressure<Array<f64, D>>> {
        // Calculate residual Helmholtz energy density and functional derivative
        let t = self.temperature.to_reduced();
        let rho = self.density.to_reduced();
        let (mut f, dfdrho) = self.functional_derivative()?;

        // Calculate the grand potential density
        for ((rho, dfdrho), &m) in rho
            .outer_iter()
            .zip(dfdrho.outer_iter())
            .zip(self.functional.m().iter())
        {
            f -= &((&dfdrho + m) * &rho);
        }

        let bond_lengths = self.functional.bond_lengths(t);
        for segment in bond_lengths.node_indices() {
            let n = bond_lengths.neighbors(segment).count();
            f += &(&rho.index_axis(Axis(0), segment.index()) * (0.5 * n as f64));
        }

        Ok(Pressure::from_reduced(f * t))
    }

    /// Calculate the grand potential $\Omega$.
    pub fn grand_potential(&self) -> FeosResult<Energy> {
        Ok(self.integrate(&self.grand_potential_density()?))
    }

    /// Calculate the reduced residual Helmholtz energy density $\beta\hat f^\mathrm{res}$ and
    /// functional derivative $\beta\hat F_{\rho_\alpha}^\mathrm{res}(\mathbf{r})$.
    #[expect(clippy::type_complexity)]
    pub fn functional_derivative(&self) -> FeosResult<(Array<f64, D>, Array<f64, D::Larger>)> {
        let density = self.density.to_reduced();
        let temperature = self.temperature.into_reduced();

        // partial derivatives of the functional contributions
        let weighted_densities = self.convolver.weighted_densities(&density);
        let mut helmholtz_energy_density = Array::zeros(density.raw_dim().remove_axis(Axis(0)));
        let mut partial_derivatives = Vec::new();
        for (c, wd) in self.functional.contributions().zip(weighted_densities) {
            let (phi, pd) = Evaluate.local_derivatives(temperature, &wd, &c)?;
            helmholtz_energy_density += &phi.index_axis_move(Axis(0), 0);
            partial_derivatives.push(pd);
        }
        let mut functional_derivative = self.convolver.functional_derivative(&partial_derivatives);

        // association at fixed X
        if let (Some(a), Some(convolver)) =
            (self.functional.association(), &self.association_convolver)
        {
            let association = a.evaluate(
                temperature,
                &density,
                &self.non_bonded_fractions,
                convolver.as_ref(),
                self.association_kernel_convolver.as_deref(),
            );
            helmholtz_energy_density += &association.helmholtz_energy_density;
            functional_derivative += &association.functional_derivative;
        }

        Ok((helmholtz_energy_density, functional_derivative))
    }
}

impl<D: Dimension + RemoveAxis + 'static, F: HelmholtzEnergyFunctional> DFTProfile<D, F>
where
    D::Larger: Dimension<Smaller = D>,
    D::Smaller: Dimension<Larger = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    fn residual_helmholtz_energy_density<N: DualNumCopy<Primitive = f64> + DctNum>(
        &self,
        temperature: N,
    ) -> FeosResult<Array<N, D>> {
        let (convolver, association_convolver, kernel_convolver) =
            self.functional.convolvers::<D, _>(&self.grid, temperature);
        let density = self.density.to_reduced().mapv(N::from);
        let weighted_densities = convolver.weighted_densities(&density);
        let mut helmholtz_energy_density = self
            .functional
            .ideal_chain_contribution()
            .helmholtz_energy_density(&density)?;
        for (c, wd) in self.functional.contributions().zip(weighted_densities) {
            let nwd = wd.shape()[0];
            let ngrid = wd.len() / nwd;
            helmholtz_energy_density
                .view_mut()
                .into_shape_with_order(ngrid)
                .unwrap()
                .add_assign(&c.helmholtz_energy_density(
                    temperature,
                    wd.into_shape_with_order((nwd, ngrid)).unwrap().view(),
                )?);
        }
        if let (Some(a), Some(association_convolver)) =
            (self.functional.association(), association_convolver)
        {
            let association = a.evaluate(
                temperature,
                &density,
                &self.non_bonded_fractions.mapv(N::from),
                association_convolver.as_ref(),
                kernel_convolver.as_deref(),
            );
            helmholtz_energy_density += &association.helmholtz_energy_density;
        }
        Ok(helmholtz_energy_density.mapv(|a| a * temperature))
    }

    /// Calculate the residual entropy density $s^\mathrm{res}(\mathbf{r})$.
    ///
    /// Untested with heterosegmented functionals.
    pub fn residual_entropy_density(&self) -> FeosResult<EntropyDensity<Array<f64, D>>> {
        let temperature = Dual64::from(self.temperature.to_reduced()).derivative();
        let helmholtz_energy_density = self.residual_helmholtz_energy_density(temperature)?;
        Ok(EntropyDensity::from_reduced(
            helmholtz_energy_density.mapv(|f| -f.eps),
        ))
    }
}

impl<D: Dimension + RemoveAxis + 'static, F: HelmholtzEnergyFunctional + Total> DFTProfile<D, F>
where
    D::Larger: Dimension<Smaller = D>,
    D::Smaller: Dimension<Larger = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    fn ideal_gas_contribution<N: DualNumCopy<Primitive = f64, Inner = f64>>(
        &self,
        temperature: N,
    ) -> Array<N, D> {
        let density = self.density.to_reduced();
        let lambda = self.functional.lift_total().ln_lambda3(temperature);
        let mut phi = Array::zeros(density.raw_dim().remove_axis(Axis(0)));
        for (i, rhoi) in density.outer_iter().enumerate() {
            phi += &rhoi.mapv(|rhoi| (lambda[i] + rhoi.ln() - 1.0) * rhoi);
        }
        phi.mapv(|p| p * temperature)
    }

    fn helmholtz_energy_density<N: DualNumCopy<Primitive = f64, Inner = f64> + DctNum>(
        &self,
        temperature: N,
        contributions: Contributions,
    ) -> FeosResult<Array<N, D>> {
        let mut helmholtz_energy_density = self.residual_helmholtz_energy_density(temperature)?;
        match contributions {
            Contributions::Total => {
                helmholtz_energy_density += &self.ideal_gas_contribution(temperature);
            }
            Contributions::IdealGas => panic!(
                "Only Contributions::Residual or Contributions::Total are supported for temperature derivatives"
            ),
            Contributions::Residual => (),
        }
        Ok(helmholtz_energy_density)
    }

    /// Calculate the entropy density $s(\mathbf{r})$.
    ///
    /// Untested with heterosegmented functionals.
    pub fn entropy_density(
        &self,
        contributions: Contributions,
    ) -> FeosResult<EntropyDensity<Array<f64, D>>> {
        let temperature = Dual64::from(self.temperature.into_reduced()).derivative();
        let helmholtz_energy_density = self.helmholtz_energy_density(temperature, contributions)?;
        Ok(EntropyDensity::from_reduced(
            helmholtz_energy_density.mapv(|f| -f.eps),
        ))
    }

    /// Calculate the entropy $S$.
    ///
    /// Untested with heterosegmented functionals.
    pub fn entropy(&self, contributions: Contributions) -> FeosResult<Entropy> {
        Ok(self.integrate(&self.entropy_density(contributions)?))
    }

    /// Calculate the internal energy density $u(\mathbf{r})$.
    ///
    /// Untested with heterosegmented functionals.
    pub fn internal_energy_density(
        &self,
        contributions: Contributions,
    ) -> FeosResult<Pressure<Array<f64, D>>> {
        let temperature = self.temperature.into_reduced();
        let temperature_dual = Dual64::from(temperature).derivative();
        let density = self.density.to_reduced();
        let helmholtz_energy_density =
            self.helmholtz_energy_density(temperature_dual, contributions)?;
        let internal_energy_density = helmholtz_energy_density.mapv(|f| f.re - f.eps * temperature)
            + (&self.external_potential * density).sum_axis(Axis(0)) * temperature;
        Ok(Pressure::from_reduced(internal_energy_density))
    }

    /// Calculate the internal energy $U$.
    ///
    /// Untested with heterosegmented functionals.
    pub fn internal_energy(&self, contributions: Contributions) -> FeosResult<Energy> {
        Ok(self.integrate(&self.internal_energy_density(contributions)?))
    }
}

impl<D: Dimension + RemoveAxis + 'static, F: HelmholtzEnergyFunctional> DFTProfile<D, F>
where
    D::Larger: Dimension<Smaller = D>,
    D::Smaller: Dimension<Larger = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    /// Derivatives of the density profiles with respect to a parameter $\theta$ from the
    /// linearized Euler-Lagrange equation and mass action equations,
    /// $J\frac{\mathrm{d}x}{\mathrm{d}\theta}=-\frac{\partial r}{\partial\theta}$.
    ///
    /// `dr_dtheta` returns the partial derivatives of the residual (stacked like the unknowns
    /// $x=(\ln\rho,\ln X)$). It is evaluated, and the system is linearized, at constant chemical
    /// potential: the specification is temporarily replaced by fixed fugacities (their values do
    /// not enter the derivatives).
    fn density_derivative(
        &mut self,
        dr_dtheta: impl FnOnce(&Self) -> FeosResult<Array<f64, D::Larger>>,
    ) -> FeosResult<Array<f64, D::Larger>> {
        let rho = self.density.to_reduced();
        let specification = std::mem::replace(
            &mut self.specification,
            DFTSpecification::ChemicalPotential(Array1::zeros(rho.shape()[0])),
        );
        let delta = (|| {
            let mut lhs = dr_dtheta(self)?;
            lhs.mapv_inplace(|l| if l.is_finite() { l } else { 0.0 });
            // the density rows come first in the stacked array
            self.mask_external_potential(&mut lhs);
            let x = Self::stack_variables(&rho, &self.non_bonded_fractions);
            solve_linearized(&*self, &x, &lhs, 200, 1e-13)
        })();
        self.specification = specification;
        Ok(&self.density_rows(&delta?) * &rho)
    }

    /// Zero array with the shape of the stacked unknowns $x=(\ln\rho,\ln X)$, with the density
    /// rows set to `value(segment)`.
    fn density_rows_from(&self, value: impl Fn(usize) -> f64) -> Array<f64, D::Larger> {
        let mut dim = self.density.raw_dim();
        dim[0] += self.non_bonded_fractions.shape()[0];
        let mut a = Array::zeros(dim);
        for (i, mut a) in a.outer_iter_mut().take(self.density.shape()[0]).enumerate() {
            a.fill(value(i));
        }
        a
    }

    /// Return the partial derivatives of the density profiles w.r.t. the chemical potentials $\left(\frac{\partial\rho_i(\mathbf{r})}{\partial\mu_k}\right)_T$
    pub fn drho_dmu(&mut self) -> FeosResult<DrhoDmu<D>> {
        let shape: Vec<_> = std::iter::once(&self.functional.components())
            .chain(self.density.shape())
            .copied()
            .collect();
        let mut drho_dmu = Array::zeros(shape).into_dimensionality().unwrap();
        // d ln(f_i) / d(beta mu_k) = delta_ik / m (see drho_dt)
        let component_index = self.functional.component_index().into_owned();
        let m = self.functional.m().into_owned();
        for (k, mut d) in drho_dmu.outer_iter_mut().enumerate() {
            let dr_dmu = self.density_rows_from(|i| {
                if component_index[i] == k {
                    1.0 / m[i]
                } else {
                    0.0
                }
            });
            d.assign(&self.density_derivative(|_| Ok(dr_dmu))?);
        }
        Ok(Quantity::from_reduced(
            drho_dmu / self.temperature.to_reduced(),
        ))
    }

    /// Return the partial derivatives of the number of moles w.r.t. the chemical potentials $\left(\frac{\partial N_i}{\partial\mu_k}\right)_T$
    pub fn dn_dmu(&mut self) -> FeosResult<DnDmu> {
        let drho_dmu = self.drho_dmu()?.into_reduced();
        let n = drho_dmu.shape()[0];
        let mut dn_dmu = DMatrix::zeros(n, n);
        dn_dmu
            .column_iter_mut()
            .zip(drho_dmu.outer_iter())
            .for_each(|(mut dn, drho)| dn.add_assign(&self.integrate_reduced_segments(&drho)));
        Ok(DnDmu::from_reduced(dn_dmu))
    }

    /// Return the partial derivatives of the density profiles w.r.t. the bulk pressure at constant temperature and bulk composition $\left(\frac{\partial\rho_i(\mathbf{r})}{\partial p}\right)_{T,\mathbf{x}}$
    pub fn drho_dp(&mut self, bulk: &State<F>) -> FeosResult<DrhoDp<D>> {
        // d ln(f_i) / d(beta p) = v_i / m (see drho_dt)
        let v = bulk.partial_molar_volume().to_reduced();
        let component_index = self.functional.component_index().into_owned();
        let m = self.functional.m().into_owned();
        let dr_dp = self.density_rows_from(|i| v[component_index[i]] / m[i]);
        Ok(Quantity::from_reduced(
            self.density_derivative(|_| Ok(dr_dp))? / self.temperature.to_reduced(),
        ))
    }

    /// Return the partial derivatives of the number of moles w.r.t. the bulk pressure at constant temperature and bulk composition $\left(\frac{\partial N_i}{\partial p}\right)_{T,\mathbf{x}}$
    pub fn dn_dp(&mut self, bulk: &State<F>) -> FeosResult<DnDp> {
        let drho_dp = self.drho_dp(bulk)?;
        Ok(self.integrate_segments(&drho_dp))
    }

    /// Return the partial derivatives of the density profiles w.r.t. the temperature at constant bulk pressure and composition $\left(\frac{\partial\rho_i(\mathbf{r})}{\partial T}\right)_{p,\mathbf{x}}$
    pub fn drho_dt(&mut self, bulk: &State<F>) -> FeosResult<DrhoDT<D>> {
        let t = self.temperature.to_reduced();

        // temperature derivative of the bulk fugacities ln(f_i) = ln(x_i p phi_i / kT) / m at
        // constant pressure and composition
        let dln_f_dt = bulk.dln_phi_dt().to_reduced().map(|d| d - 1.0 / t);
        let component_index = self.functional.component_index().into_owned();
        let m = self.functional.m().into_owned();
        let df_dt = self.density_rows_from(|i| dln_f_dt[component_index[i]] / m[i]);

        // temperature derivative of the residual at fixed density, X, and fugacities, from the
        // Euler-Lagrange equation in dual numbers
        let drho_dt = self.density_derivative(|profile| {
            let t_dual = Dual64::from(t).derivative();
            let (convolver, association_convolver, kernel_convolver) =
                profile.functional.convolvers::<D, _>(&profile.grid, t_dual);
            let rho = profile.density.to_reduced();
            let x = Self::stack_variables(&rho, &profile.non_bonded_fractions).mapv(Dual64::from);
            let (residual, _) = profile.euler_lagrange_dual(
                &mut Evaluate,
                t_dual,
                convolver.as_ref(),
                association_convolver.as_deref(),
                kernel_convolver.as_deref(),
                &x,
                &mut DFTSolverLog::new(),
            )?;
            Ok(residual.mapv(|r| r.eps) + df_dt)
        })?;
        Ok(Quantity::from_reduced(drho_dt))
    }

    /// Return the partial derivatives of the number of moles w.r.t. the temperature at constant bulk pressure and composition $\left(\frac{\partial N_i}{\partial T}\right)_{p,\mathbf{x}}$
    pub fn dn_dt(&mut self, bulk: &State<F>) -> FeosResult<DnDT> {
        let drho_dt = self.drho_dt(bulk)?;
        Ok(self.integrate_segments(&drho_dt))
    }
}
