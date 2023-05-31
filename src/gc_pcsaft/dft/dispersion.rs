use super::parameter::GcPcSaftFunctionalParameters;
use crate::gc_pcsaft::eos::dispersion::{A0, A1, A2, B0, B1, B2};
use crate::hard_sphere::HardSphereProperties;
use feos_core::EosError;
use feos_dft::{
    FunctionalContributionDual, WeightFunction, WeightFunctionInfo, WeightFunctionShape,
};
use ndarray::*;
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_6, PI};
use std::fmt;
use std::sync::Arc;

#[derive(Clone)]
pub struct AttractiveFunctional {
    parameters: Arc<GcPcSaftFunctionalParameters>,
}

impl AttractiveFunctional {
    pub fn new(parameters: &Arc<GcPcSaftFunctionalParameters>) -> Self {
        Self {
            parameters: parameters.clone(),
        }
    }
}

impl<N: DualNum<f64> + Copy + ScalarOperand> FunctionalContributionDual<N>
    for AttractiveFunctional
{
    fn weight_functions(&self, temperature: N) -> WeightFunctionInfo<N> {
        let p = &self.parameters;

        let d = p.hs_diameter(temperature);
        WeightFunctionInfo::new(p.component_index.clone(), false).add(
            WeightFunction::new_scaled(d * &p.psi_dft, WeightFunctionShape::Theta),
            false,
        )
    }

    fn calculate_helmholtz_energy_density(
        &self,
        temperature: N,
        density: ArrayView2<N>,
    ) -> Result<Array1<N>, EosError> {
        // auxiliary variables
        let p = &self.parameters;
        let n = p.m.len();

        // temperature dependent segment diameter
        let d = p.hs_diameter(temperature);

        // packing fraction
        let eta = density
            .outer_iter()
            .zip((&d * &d * &d * &p.m * FRAC_PI_6).into_iter())
            .map(|(rho, d3m)| &rho * d3m)
            .reduce(|a, b| a + b)
            .unwrap();

        // mean segment number
        let mut m_i: Array1<f64> = Array::zeros(p.component_index[n - 1] + 1);
        let mut s_i: Array1<f64> = Array::zeros(p.component_index[n - 1] + 1);
        for (&c, &m) in p.component_index.iter().zip(p.m.iter()) {
            m_i[c] += m;
            s_i[c] += 1.0;
        }
        let mut rhog: Array1<N> = Array::zeros(eta.raw_dim());
        let mut m_bar: Array1<N> = Array::zeros(eta.raw_dim());
        for (rho, &c) in density.outer_iter().zip(p.component_index.iter()) {
            m_bar += &(&rho * m_i[c] / s_i[c]);
            rhog += &(&rho / s_i[c]);
        }
        m_bar.iter_mut().zip(rhog.iter()).for_each(|(m, &r)| {
            if r.re() > f64::EPSILON {
                *m /= r
            } else {
                *m = N::one()
            }
        });

        // mixture densities, crosswise interactions of all segments on all chains
        let mut rho1mix: Array1<N> = Array::zeros(eta.raw_dim());
        let mut rho2mix: Array1<N> = Array::zeros(eta.raw_dim());
        for i in 0..n {
            for j in 0..n {
                let eps_ij = temperature.recip() * p.epsilon_k_ij[(i, j)];
                let sigma_ij = p.sigma_ij[(i, j)].powi(3);
                rho1mix = rho1mix
                    + (&density.index_axis(Axis(0), i) * &density.index_axis(Axis(0), j))
                        .mapv(|x| x * (eps_ij * sigma_ij * p.m[i] * p.m[j]));
                rho2mix = rho2mix
                    + (&density.index_axis(Axis(0), i) * &density.index_axis(Axis(0), j))
                        .mapv(|x| x * (eps_ij * eps_ij * sigma_ij * p.m[i] * p.m[j]));
            }
        }

        // I1, I2 and C1
        let mut i1: Array1<N> = Array::zeros(eta.raw_dim());
        let mut i2: Array1<N> = Array::zeros(eta.raw_dim());
        let mut eta_i: Array1<N> = Array::ones(eta.raw_dim());
        let m1 = (m_bar.clone() - 1.0) / &m_bar;
        let m2 = (m_bar.clone() - 2.0) / &m_bar * &m1;
        for i in 0..=6 {
            i1 = i1 + (&m2 * A2[i] + &m1 * A1[i] + A0[i]) * &eta_i;
            i2 = i2 + (&m2 * B2[i] + &m1 * B1[i] + B0[i]) * &eta_i;
            eta_i = &eta_i * &eta;
        }
        let c1 = Zip::from(&eta).and(&m_bar).map_collect(|&eta, &m| {
            (m * (eta * 8.0 - eta.powi(2) * 2.0) / (eta - 1.0).powi(4)
                + (eta * (eta * (eta * (eta * 2.0 - 12.0) + 27.0) - 20.0))
                    / ((eta - 1.0) * (eta - 2.0)).powi(2)
                    * (m - 1.0)
                + 1.0)
                .recip()
        });

        // Helmholtz energy density
        Ok((-rho1mix * i1 * 2.0 - rho2mix * m_bar * c1 * i2) * PI)
    }
}

impl fmt::Display for AttractiveFunctional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Attractive functional (GC)")
    }
}
