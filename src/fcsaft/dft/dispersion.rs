use super::FcSaftParameters;
use crate::fcsaft::heterosegmented::dispersion::{Dispersion, A0, A1, A2, B0, B1, B2};
use crate::hard_sphere::HardSphereProperties;
use feos_core::EosError;
use feos_dft::{FunctionalContribution, WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use ndarray::*;
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_6, PI};
use std::fmt;
use std::sync::Arc;

pub struct DispersionFunctional {
    parameters: Arc<FcSaftParameters>,
}

impl DispersionFunctional {
    pub fn new(parameters: &Arc<FcSaftParameters>) -> Self {
        Self {
            parameters: parameters.clone(),
        }
    }
}

impl FunctionalContribution for DispersionFunctional {
    fn weight_functions<N: DualNum<f64> + Copy>(&self, temperature: N) -> WeightFunctionInfo<N> {
        let p = &self.parameters;

        let d = p.hs_diameter(temperature);
        WeightFunctionInfo::new(p.component_index.clone(), false).add(
            WeightFunction::new_scaled(d * &p.psi_dft, WeightFunctionShape::Theta),
            false,
        )
    }

    fn helmholtz_energy_density<N: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        temperature: N,
        density: ArrayView2<N>,
    ) -> Result<Array1<N>, EosError> {
        // auxiliary variables
        let p = &self.parameters;
        let n = p.sigma.len();

        // temperature dependent segment diameter
        let d = p.hs_diameter(temperature);
        let [a, v] = p.geometry_coefficients(&d);

        // packing fraction
        let eta = density
            .outer_iter()
            .zip(&d * &d * &d * &v * FRAC_PI_6)
            .map(|(rho, d3m)| &rho * d3m)
            .reduce(|a, b| a + b)
            .unwrap();

        // mean segment number
        let mut m_i: Array1<N> = Array::zeros(p.component_index[n - 1] + 1);
        let mut m1_i: Array1<N> = Array::zeros(p.component_index[n - 1] + 1);
        let mut s_i: Array1<f64> = Array::zeros(p.component_index[n - 1] + 1);
        for ((&c, &a), &v) in p.component_index.iter().zip(a.iter()).zip(v.iter()) {
            m_i[c] += a;
            m1_i[c] += v;
            s_i[c] += 1.0;
        }
        let mut rhog: Array1<N> = Array::zeros(eta.raw_dim());
        let mut m_bar: Array1<N> = Array::zeros(eta.raw_dim());
        let mut m1_bar: Array1<N> = Array::zeros(eta.raw_dim());
        let mut s_bar: Array1<N> = Array::zeros(eta.raw_dim());
        for (rho, &c) in density.outer_iter().zip(p.component_index.iter()) {
            m_bar += &(&rho * m_i[c] / s_i[c]);
            m1_bar += &(&rho * m1_i[c] / s_i[c]);
            s_bar += &rho;
            rhog += &(&rho / s_i[c]);
        }
        let save_div = |(x, &r): (&mut N, &N)| {
            if x.re() > f64::EPSILON {
                *x /= r
            } else {
                *x = N::one() * 1.1
            }
        };
        m_bar.iter_mut().zip(rhog.iter()).for_each(save_div);
        m1_bar.iter_mut().zip(rhog.iter()).for_each(save_div);
        s_bar.iter_mut().zip(rhog.iter()).for_each(save_div);

        // mixture densities, crosswise interactions of all segments on all chains
        let mut rho1mix: Array1<N> = Array::zeros(eta.raw_dim());
        let mut rho2mix: Array1<N> = Array::zeros(eta.raw_dim());
        for i in 0..n {
            for j in 0..n {
                let eps_ij = temperature.recip() * p.epsilon_k_ij[(i, j)];
                let sigma_ij = p.sigma_ij[(i, j)].powi(3);
                let rho_ij = &density.index_axis(Axis(0), i) * &density.index_axis(Axis(0), j);
                rho1mix += &rho_ij.mapv(|x| x * (eps_ij * sigma_ij * a[i] * a[j]));
                rho2mix += &rho_ij.mapv(|x| x * (eps_ij * eps_ij * sigma_ij * a[i] * a[j]));
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
        let c1 = if n == 1 {
            eta.mapv(Dispersion::compressibility_term_monomer)
        } else {
            Zip::from(&s_bar)
                .and(&m_bar)
                .and(&m1_bar)
                .and(&eta)
                .map_collect(|&s, &m, &m1, &eta| {
                    Dispersion::compressibility_term_chain(s, m, m1, eta)
                })
        };
        // println!("{s_bar} {m_bar} {m1_bar} {c1}");

        // Helmholtz energy density
        Ok((-rho1mix * i1 * 2.0 - rho2mix * m_bar * c1 * i2) * PI)
    }
}

impl fmt::Display for DispersionFunctional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Attractive functional (GC)")
    }
}
