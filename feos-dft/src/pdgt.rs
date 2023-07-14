use super::functional::{HelmholtzEnergyFunctional, DFT};
use super::functional_contribution::FunctionalContribution;
use super::weight_functions::WeightFunctionInfo;
use feos_core::{Components, Contributions, EosResult, EosUnit, PhaseEquilibrium};
use ndarray::*;
use num_dual::Dual2_64;
use quantity::si::{SIArray1, SIArray2, SINumber, SIUnit};
use std::ops::AddAssign;

impl WeightFunctionInfo<Dual2_64> {
    fn pdgt_weight_constants(&self) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let k = Dual2_64::from(0.0).derivative();
        let w = self.weight_constants(k, 1);
        (w.mapv(|w| w.re), w.mapv(|w| -w.v1), w.mapv(|w| -0.5 * w.v2))
    }
}

impl dyn FunctionalContribution {
    pub fn pdgt_properties(
        &self,
        temperature: f64,
        density: &Array2<f64>,
        helmholtz_energy_density: &mut Array1<f64>,
        first_partial_derivatives: Option<&mut Array2<f64>>,
        second_partial_derivatives: Option<&mut Array3<f64>>,
        influence_diagonal: Option<&mut Array2<f64>>,
        influence_matrix: Option<&mut Array3<f64>>,
    ) -> EosResult<()> {
        // calculate weighted densities
        let weight_functions = self.weight_functions_pdgt(Dual2_64::from(temperature));
        let (w0, w1, w2) = weight_functions.pdgt_weight_constants();
        let weighted_densities = w0.dot(density);

        // calculate Helmholtz energy and derivatives
        let w = weighted_densities.shape()[0]; // # of weighted densities
        let s = density.shape()[0]; // # of segments
        let n = density.shape()[1]; // # of grid points
        let mut df = Array::zeros((w, n));
        let mut d2f = Array::zeros((w, w, n));
        self.second_partial_derivatives(
            temperature,
            weighted_densities.view(),
            helmholtz_energy_density.view_mut(),
            df.view_mut(),
            d2f.view_mut(),
        )?;

        // calculate partial derivatives w.r.t. density
        if let Some(df_drho) = first_partial_derivatives {
            df_drho.assign(&df.t().dot(&w0));
        }

        // calculate second partial derivatives w.r.t. density
        if let Some(d2f_drho2) = second_partial_derivatives {
            for i in 0..s {
                for j in 0..s {
                    for alpha in 0..w {
                        for beta in 0..w {
                            d2f_drho2
                                .index_axis_mut(Axis(0), i)
                                .index_axis_mut(Axis(0), j)
                                .add_assign(
                                    &(&d2f.index_axis(Axis(0), alpha).index_axis(Axis(0), beta)
                                        * w0[(alpha, i)]
                                        * w0[(beta, j)]),
                                );
                        }
                    }
                }
            }
        }

        // calculate influence diagonal
        if let Some(c) = influence_diagonal {
            for i in 0..s {
                for alpha in 0..w {
                    for beta in 0..w {
                        c.index_axis_mut(Axis(0), i).add_assign(
                            &(&d2f.index_axis(Axis(0), alpha).index_axis(Axis(0), beta)
                                * (w1[(alpha, i)] * w1[(beta, i)]
                                    - w0[(alpha, i)] * w2[(beta, i)]
                                    - w2[(alpha, i)] * w0[(beta, i)])),
                        );
                    }
                }
            }
        }

        // calculate influence matrix
        if let Some(c) = influence_matrix {
            for i in 0..s {
                for j in 0..s {
                    for alpha in 0..w {
                        for beta in 0..w {
                            c.index_axis_mut(Axis(0), i)
                                .index_axis_mut(Axis(0), j)
                                .add_assign(
                                    &(&d2f.index_axis(Axis(0), alpha).index_axis(Axis(0), beta)
                                        * (w1[(alpha, i)] * w1[(beta, j)]
                                            - w0[(alpha, i)] * w2[(beta, j)]
                                            - w2[(alpha, i)] * w0[(beta, j)])),
                                );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn influence_diagonal(
        &self,
        temperature: SINumber,
        density: &SIArray2,
    ) -> EosResult<(SIArray1, SIArray2)> {
        let t = temperature.to_reduced(SIUnit::reference_temperature())?;
        let n = density.shape()[1];
        let mut f = Array::zeros(n);
        let mut c = Array::zeros(density.raw_dim());
        self.pdgt_properties(
            t,
            &density.to_reduced(SIUnit::reference_density())?,
            &mut f,
            None,
            None,
            Some(&mut c),
            None,
        )?;
        Ok((
            f * t * SIUnit::reference_pressure(),
            c * t * SIUnit::reference_influence_parameter(),
        ))
    }
}

impl<T: HelmholtzEnergyFunctional> DFT<T> {
    pub fn solve_pdgt(
        &self,
        vle: &PhaseEquilibrium<Self, 2>,
        n_grid: usize,
        reference_component: usize,
        z: Option<(&mut SIArray1, &mut SINumber)>,
    ) -> EosResult<(SIArray2, SINumber)> {
        // calculate density profile
        let density = if self.components() == 1 {
            let delta_rho = (vle.vapor().density - vle.liquid().density) / (n_grid + 1) as f64;
            SIArray1::linspace(
                vle.liquid().density + delta_rho,
                vle.vapor().density - delta_rho,
                n_grid,
            )?
            .insert_axis(Axis(0))
        } else {
            self.pdgt_density_profile_mix(vle, n_grid, reference_component)?
        };

        // calculate Helmholtz energy density and influence parameter
        let mut delta_omega = Array::zeros(n_grid) * SIUnit::reference_pressure();
        let mut influence_diagonal =
            Array::zeros(density.raw_dim()) * SIUnit::reference_influence_parameter();
        for contribution in self.contributions() {
            let (f, c) = contribution.influence_diagonal(vle.vapor().temperature, &density)?;
            delta_omega += &f;
            influence_diagonal += &c;
        }
        delta_omega += &self
            .ideal_chain_contribution()
            .helmholtz_energy_density::<Ix1>(vle.vapor().temperature, &density)?;

        // calculate excess grand potential density
        let mu_res = vle.vapor().residual_chemical_potential();
        for i in 0..self.components() {
            let rhoi = density.index_axis(Axis(0), i).to_owned();
            let rhoi_b = vle.vapor().partial_density.get(i);
            let mui_res = mu_res.get(i);
            let kt = SIUnit::gas_constant() * vle.vapor().temperature;
            delta_omega +=
                &(&rhoi * (kt * (rhoi.to_reduced(rhoi_b)?.mapv(f64::ln) - 1.0) - mui_res));
        }
        delta_omega += vle.vapor().pressure(Contributions::Total);

        // calculate density gradients w.r.t. reference density
        let dx = density.get((reference_component, 0)) - density.get((reference_component, 1));
        let drho = density.gradient(
            -dx,
            &vle.liquid().partial_density,
            &vle.vapor().partial_density,
        );

        // calculate integrand
        let gamma_int =
            ((influence_diagonal * delta_omega.clone() * 2.0).sqrt()? * drho).sum_axis(Axis(0));

        // calculate z-axis
        if let Some((z, w)) = z {
            let z_int = gamma_int.clone() / (delta_omega * 2.0);
            *z = z_int.integrate_trapezoidal_cumulative(dx);

            // calculate equimolar surface
            let rho_v = density.index_axis(Axis(1), 0).sum();
            let rho_l = density.index_axis(Axis(1), n_grid - 1).sum();
            let rho_r = (density.sum_axis(Axis(0)) - rho_v) / (rho_l - rho_v);
            let ze = (rho_r.clone() * z_int.clone()).integrate_trapezoidal(dx);

            // calculate interfacial width
            *w = (rho_r * z.clone() * z_int).integrate_trapezoidal(dx);
            *w = (24.0 * (*w - 0.5 * ze.powi(2))).sqrt()?;

            // shift density profile
            *z -= ze;
        }

        // integration weights (First and last points are 0)
        let mut weights = Array::ones(n_grid);
        weights[0] = 7.0 / 6.0;
        weights[1] = 23.0 / 24.0;
        weights[n_grid - 2] = 23.0 / 24.0;
        weights[n_grid - 1] = 7.0 / 6.0;
        let weights = weights * dx;

        // calculate surface tension
        Ok((density, gamma_int.integrate(&[weights])))
    }

    fn pdgt_density_profile_mix(
        &self,
        _vle: &PhaseEquilibrium<Self, 2>,
        _n_grid: usize,
        _reference_component: usize,
    ) -> EosResult<SIArray2> {
        unimplemented!()
    }
}
