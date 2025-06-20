#![allow(type_alias_bounds)]
use super::DFTProfile;
use crate::convolver::{BulkConvolver, Convolver};
use crate::functional_contribution::FunctionalContribution;
use crate::{ConvolverFFT, DFTSolverLog, HelmholtzEnergyFunctional, WeightFunctionInfo};
use feos_core::{Contributions, FeosResult, IdealGas, ReferenceSystem, Verbosity};
use ndarray::{Array, Array1, Array2, Axis, Dimension, RemoveAxis, ScalarOperand};
use num_dual::{Dual64, DualNum};
use quantity::{
    Density, Energy, Entropy, EntropyDensity, MolarEnergy, Moles, Pressure, Quantity, Temperature,
};
use std::ops::{AddAssign, Div};
use std::sync::Arc;

type DrhoDmu<D: Dimension> =
    <Density<Array<f64, <D::Larger as Dimension>::Larger>> as Div<MolarEnergy>>::Output;
type DnDmu = <Moles<Array2<f64>> as Div<MolarEnergy>>::Output;
type DrhoDp<D: Dimension> = <Density<Array<f64, D::Larger>> as Div<Pressure>>::Output;
type DnDp = <Moles<Array1<f64>> as Div<Pressure>>::Output;
type DrhoDT<D: Dimension> = <Density<Array<f64, D::Larger>> as Div<Temperature>>::Output;
type DnDT = <Moles<Array1<f64>> as Div<Temperature>>::Output;

impl<D: Dimension, F: HelmholtzEnergyFunctional> DFTProfile<D, F>
where
    D::Larger: Dimension<Smaller = D>,
{
    /// Calculate the grand potential density $\omega$.
    pub fn grand_potential_density(&self) -> FeosResult<Pressure<Array<f64, D>>> {
        // Calculate residual Helmholtz energy density and functional derivative
        let t = self.temperature.to_reduced();
        let rho = self.density.to_reduced();
        let (mut f, dfdrho) = self.dft.functional_derivative(t, &rho, &self.convolver)?;

        // Calculate the grand potential density
        for ((rho, dfdrho), &m) in rho
            .outer_iter()
            .zip(dfdrho.outer_iter())
            .zip(self.dft.m().iter())
        {
            f -= &((&dfdrho + m) * &rho);
        }

        let bond_lengths = self.dft.bond_lengths(t);
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

    /// Calculate the (residual) intrinsic functional derivative $\frac{\delta\mathcal{F}}{\delta\rho_i(\mathbf{r})}$.
    pub fn functional_derivative(&self) -> FeosResult<Array<f64, D::Larger>> {
        let (_, dfdrho) = self.dft.functional_derivative(
            self.temperature.to_reduced(),
            &self.density.to_reduced(),
            &self.convolver,
        )?;
        Ok(dfdrho)
    }
}

impl<D: Dimension + RemoveAxis + 'static, F: HelmholtzEnergyFunctional> DFTProfile<D, F>
where
    D::Larger: Dimension<Smaller = D>,
    D::Smaller: Dimension<Larger = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    fn intrinsic_helmholtz_energy_density<N>(
        &self,
        temperature: N,
        density: &Array<f64, D::Larger>,
        convolver: &Arc<dyn Convolver<N, D>>,
    ) -> FeosResult<Array<N, D>>
    where
        N: DualNum<f64> + Copy + ScalarOperand,
    {
        let density_dual = density.mapv(N::from);
        let weighted_densities = convolver.weighted_densities(&density_dual);
        let functional_contributions = self.dft.contributions();
        let mut helmholtz_energy_density: Array<N, D> = self
            .dft
            .ideal_chain_contribution()
            .helmholtz_energy_density(&density.mapv(N::from))?;
        for (c, wd) in functional_contributions.into_iter().zip(weighted_densities) {
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
        Ok(helmholtz_energy_density * temperature)
    }

    /// Calculate the residual entropy density $s^\mathrm{res}(\mathbf{r})$.
    ///
    /// Untested with heterosegmented functionals.
    pub fn residual_entropy_density(&self) -> FeosResult<EntropyDensity<Array<f64, D>>> {
        // initialize convolver
        let temperature = self.temperature.to_reduced();
        let temperature_dual = Dual64::from(temperature).derivative();
        let functional_contributions = self.dft.contributions();
        let weight_functions: Vec<WeightFunctionInfo<Dual64>> = functional_contributions
            .into_iter()
            .map(|c| c.weight_functions(temperature_dual))
            .collect();
        let convolver = ConvolverFFT::plan(&self.grid, &weight_functions, self.lanczos);

        let density = self.density.to_reduced();

        let helmholtz_energy_density =
            self.intrinsic_helmholtz_energy_density(temperature_dual, &density, &convolver)?;
        Ok(EntropyDensity::from_reduced(
            helmholtz_energy_density.mapv(|f| -f.eps),
        ))
    }

    /// Calculate the individual contributions to the entropy density.
    ///
    /// Untested with heterosegmented functionals.
    pub fn entropy_density_contributions(
        &self,
        temperature: f64,
        density: &Array<f64, D::Larger>,
        convolver: &Arc<dyn Convolver<Dual64, D>>,
    ) -> FeosResult<Vec<Array<f64, D>>> {
        let density_dual = density.mapv(Dual64::from);
        let temperature_dual = Dual64::from(temperature).derivative();
        let weighted_densities = convolver.weighted_densities(&density_dual);
        let functional_contributions = self.dft.contributions();
        let mut helmholtz_energy_density: Vec<Array<Dual64, _>> = Vec::new();
        helmholtz_energy_density.push(
            self.dft
                .ideal_chain_contribution()
                .helmholtz_energy_density(&density.mapv(Dual64::from))?,
        );

        for (c, wd) in functional_contributions.into_iter().zip(weighted_densities) {
            let nwd = wd.shape()[0];
            let ngrid = wd.len() / nwd;
            helmholtz_energy_density.push(
                c.helmholtz_energy_density(
                    temperature_dual,
                    wd.into_shape_with_order((nwd, ngrid)).unwrap().view(),
                )?
                .into_shape_with_order(density.raw_dim().remove_axis(Axis(0)))
                .unwrap(),
            );
        }
        Ok(helmholtz_energy_density
            .iter()
            .map(|v| v.mapv(|f| -(f * temperature_dual).eps))
            .collect())
    }
}

impl<D: Dimension + RemoveAxis + 'static, F: HelmholtzEnergyFunctional + IdealGas> DFTProfile<D, F>
where
    D::Larger: Dimension<Smaller = D>,
    D::Smaller: Dimension<Larger = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    fn ideal_gas_contribution_dual(
        &self,
        temperature: Dual64,
        density: &Array<f64, D::Larger>,
    ) -> Array<Dual64, D> {
        let lambda = self.dft.ln_lambda3(temperature);
        let mut phi = Array::zeros(density.raw_dim().remove_axis(Axis(0)));
        for (i, rhoi) in density.outer_iter().enumerate() {
            phi += &rhoi.mapv(|rhoi| (lambda[i] + rhoi.ln() - 1.0) * rhoi);
        }
        phi * temperature
    }

    /// Calculate the entropy density $s(\mathbf{r})$.
    ///
    /// Untested with heterosegmented functionals.
    pub fn entropy_density(
        &self,
        contributions: Contributions,
    ) -> FeosResult<EntropyDensity<Array<f64, D>>> {
        // initialize convolver
        let temperature = self.temperature.to_reduced();
        let temperature_dual = Dual64::from(temperature).derivative();
        let functional_contributions = self.dft.contributions();
        let weight_functions: Vec<WeightFunctionInfo<Dual64>> = functional_contributions
            .into_iter()
            .map(|c| c.weight_functions(temperature_dual))
            .collect();
        let convolver = ConvolverFFT::plan(&self.grid, &weight_functions, self.lanczos);

        let density = self.density.to_reduced();

        let mut helmholtz_energy_density =
            self.intrinsic_helmholtz_energy_density(temperature_dual, &density, &convolver)?;
        match contributions {
            Contributions::Total => {
                helmholtz_energy_density +=
                    &self.ideal_gas_contribution_dual(temperature_dual, &density);
            }
            Contributions::IdealGas => panic!(
                "Entropy density can only be calculated for Contributions::Residual or Contributions::Total"
            ),
            Contributions::Residual => (),
        }
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
    ) -> FeosResult<Pressure<Array<f64, D>>>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
    {
        // initialize convolver
        let temperature = self.temperature.to_reduced();
        let temperature_dual = Dual64::from(temperature).derivative();
        let functional_contributions = self.dft.contributions();
        let weight_functions: Vec<WeightFunctionInfo<Dual64>> = functional_contributions
            .into_iter()
            .map(|c| c.weight_functions(temperature_dual))
            .collect();
        let convolver = ConvolverFFT::plan(&self.grid, &weight_functions, self.lanczos);

        let density = self.density.to_reduced();

        let mut helmholtz_energy_density_dual =
            self.intrinsic_helmholtz_energy_density(temperature_dual, &density, &convolver)?;
        match contributions {
            Contributions::Total => {
                helmholtz_energy_density_dual +=
                    &self.ideal_gas_contribution_dual(temperature_dual, &density);
            }
            Contributions::IdealGas => panic!(
                "Internal energy density can only be calculated for Contributions::Residual or Contributions::Total"
            ),
            Contributions::Residual => (),
        }
        let helmholtz_energy_density = helmholtz_energy_density_dual
            .mapv(|f| f.re - f.eps * temperature)
            + (&self.external_potential * density).sum_axis(Axis(0)) * temperature;
        Ok(Pressure::from_reduced(helmholtz_energy_density))
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
    fn density_derivative(&self, lhs: &Array<f64, D::Larger>) -> FeosResult<Array<f64, D::Larger>> {
        let rho = self.density.to_reduced();
        let partial_density = self.bulk.partial_density.to_reduced();
        let rho_bulk = self.dft.component_index().mapv(|i| partial_density[i]);

        let second_partial_derivatives = self.second_partial_derivatives(&rho)?;
        let (_, _, _, exp_dfdrho, _) = self.euler_lagrange_equation(&rho, &rho_bulk, false)?;

        let rhs = |x: &_| {
            let delta_functional_derivative =
                self.delta_functional_derivative(x, &second_partial_derivatives);
            let mut xm = x.clone();
            xm.outer_iter_mut()
                .zip(self.dft.m().iter())
                .for_each(|(mut x, &m)| x *= m);
            let delta_i = self.delta_bond_integrals(&exp_dfdrho, &delta_functional_derivative);
            xm + (delta_functional_derivative - delta_i) * &rho
        };
        let mut log = DFTSolverLog::new(Verbosity::None);
        Self::gmres(rhs, lhs, 200, 1e-13, &mut log)
    }

    /// Return the partial derivatives of the density profiles w.r.t. the chemical potentials $\left(\frac{\partial\rho_i(\mathbf{r})}{\partial\mu_k}\right)_T$
    pub fn drho_dmu(&self) -> FeosResult<DrhoDmu<D>> {
        let shape: Vec<_> = std::iter::once(&self.dft.components())
            .chain(self.density.shape())
            .copied()
            .collect();
        let mut drho_dmu = Array::zeros(shape).into_dimensionality().unwrap();
        let component_index = self.dft.component_index();
        for (k, mut d) in drho_dmu.outer_iter_mut().enumerate() {
            let mut lhs = self.density.to_reduced();
            for (i, mut l) in lhs.outer_iter_mut().enumerate() {
                if component_index[i] != k {
                    l.fill(0.0);
                }
            }
            d.assign(&self.density_derivative(&lhs)?);
        }
        Ok(Quantity::from_reduced(
            drho_dmu / self.temperature.to_reduced(),
        ))
    }

    /// Return the partial derivatives of the number of moles w.r.t. the chemical potentials $\left(\frac{\partial N_i}{\partial\mu_k}\right)_T$
    pub fn dn_dmu(&self) -> FeosResult<DnDmu> {
        let drho_dmu = self.drho_dmu()?.into_reduced();
        let n = drho_dmu.shape()[0];
        let mut dn_dmu = Array2::zeros([n; 2]);
        dn_dmu
            .outer_iter_mut()
            .zip(drho_dmu.outer_iter())
            .for_each(|(mut dn, drho)| dn.assign(&self.integrate_reduced_segments(&drho)));
        Ok(DnDmu::from_reduced(dn_dmu))
    }

    /// Return the partial derivatives of the density profiles w.r.t. the bulk pressure at constant temperature and bulk composition $\left(\frac{\partial\rho_i(\mathbf{r})}{\partial p}\right)_{T,\mathbf{x}}$
    pub fn drho_dp(&self) -> FeosResult<DrhoDp<D>> {
        let mut lhs = self.density.to_reduced();
        let v = self.bulk.partial_molar_volume().to_reduced();
        for (mut l, &c) in lhs.outer_iter_mut().zip(self.dft.component_index().iter()) {
            l *= v[c];
        }
        Ok(Quantity::from_reduced(
            self.density_derivative(&lhs)? / self.temperature.to_reduced(),
        ))
    }

    /// Return the partial derivatives of the number of moles w.r.t. the bulk pressure at constant temperature and bulk composition $\left(\frac{\partial N_i}{\partial p}\right)_{T,\mathbf{x}}$
    pub fn dn_dp(&self) -> FeosResult<DnDp> {
        Ok(self.integrate_segments(&self.drho_dp()?))
    }

    /// Return the partial derivatives of the density profiles w.r.t. the temperature at constant bulk pressure and composition $\left(\frac{\partial\rho_i(\mathbf{r})}{\partial T}\right)_{p,\mathbf{x}}$
    pub fn drho_dt(&self) -> FeosResult<DrhoDT<D>> {
        let rho = self.density.to_reduced();
        let t = self.temperature.to_reduced();
        let rho_dual = rho.mapv(Dual64::from);
        let t_dual = Dual64::from(t).derivative();

        // calculate intrinsic functional derivative
        let functional_contributions = self.dft.contributions();
        let weight_functions: Vec<WeightFunctionInfo<Dual64>> = functional_contributions
            .into_iter()
            .map(|c| c.weight_functions(t_dual))
            .collect();
        let convolver: Arc<dyn Convolver<_, D>> =
            ConvolverFFT::plan(&self.grid, &weight_functions, self.lanczos);
        let (_, mut dfdrho) = self
            .dft
            .functional_derivative(t_dual, &rho_dual, &convolver)?;

        // calculate total functional derivative
        dfdrho += &((&self.external_potential * t).mapv(Dual64::from) / t_dual);

        // calculate bulk functional derivative
        let partial_density = self.bulk.partial_density.to_reduced();
        let rho_bulk = self.dft.component_index().mapv(|i| partial_density[i]);
        let rho_bulk_dual = rho_bulk.mapv(Dual64::from);
        let bulk_convolver = BulkConvolver::new(weight_functions);
        let (_, dfdrho_bulk) =
            self.dft
                .functional_derivative(t_dual, &rho_bulk_dual, &bulk_convolver)?;
        dfdrho
            .outer_iter_mut()
            .zip(dfdrho_bulk)
            .zip(self.dft.m().iter())
            .for_each(|((mut df, df_b), &m)| {
                df -= df_b;
                df /= Dual64::from(m)
            });

        // calculate bond integrals
        let exp_dfdrho = dfdrho.mapv(|x| (-x).exp());
        let bonds = self.dft.bond_integrals(t_dual, &exp_dfdrho, &convolver);

        // solve for drho_dt
        let mut lhs = ((exp_dfdrho * bonds).mapv(|x| -x.ln()) * t_dual).mapv(|d| d.eps);
        let x =
            (self.bulk.partial_molar_volume() * self.bulk.dp_dt(Contributions::Total)).to_reduced();
        let x = self.dft.component_index().mapv(|i| x[i]);
        lhs.outer_iter_mut()
            .zip(rho.outer_iter())
            .zip(rho_bulk)
            .zip(self.dft.m().iter())
            .zip(x)
            .for_each(|((((mut lhs, rho), rho_b), &m), x)| {
                lhs += &(&rho / rho_b).mapv(f64::ln);
                lhs *= m;
                lhs += x;
            });

        lhs *= &(-&rho / t);
        lhs.iter_mut().for_each(|l| {
            if !l.is_finite() {
                *l = 0.0
            }
        });
        Ok(Quantity::from_reduced(self.density_derivative(&lhs)?))
    }

    /// Return the partial derivatives of the number of moles w.r.t. the temperature at constant bulk pressure and composition $\left(\frac{\partial N_i}{\partial T}\right)_{p,\mathbf{x}}$
    pub fn dn_dt(&self) -> FeosResult<DnDT> {
        Ok(self.integrate_segments(&self.drho_dt()?))
    }
}
