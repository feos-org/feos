use crate::fcsaft::eos::dispersion::{A0, A1, A2, B0, B1, B2};
use crate::fcsaft::parameters::FcSaftPars;
use crate::fcsaft::reference::ReferenceFluid;
use crate::hard_sphere::HardSphereProperties;
use feos_core::FeosResult;
use feos_dft::{FunctionalContribution, WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use nalgebra::DVector;
use ndarray::*;
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_6, PI};

#[derive(Clone)]
pub struct DispersionFunctional<'a> {
    parameters: &'a FcSaftPars,
}

impl<'a> DispersionFunctional<'a> {
    pub fn new(parameters: &'a FcSaftPars) -> Self {
        Self { parameters }
    }
}

impl<'a> FunctionalContribution for DispersionFunctional<'a> {
    fn name(&self) -> &'static str {
        "Dispersion functional"
    }

    fn weight_functions<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
    ) -> WeightFunctionInfo<N> {
        let p = &self.parameters;
        let d = p.hs_diameter(temperature);
        WeightFunctionInfo::new(p.component_index.clone(), false).add(
            WeightFunction::new_scaled(
                d.zip_map(&p.psi_dft, |d, psi| d * psi),
                WeightFunctionShape::Theta,
            ),
            false,
        )
    }

    fn helmholtz_energy_density<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
        density: ArrayView2<N>,
    ) -> FeosResult<Array1<N>> {
        // auxiliary variables
        let p = &self.parameters;
        let n = p.sigma.len();

        // temperature dependent segment diameter
        let d = p.hs_diameter(temperature);
        let av = p.fused_sphere_coefficients(&d);
        let [a, v] = &av;

        // packing fraction
        let eta = density
            .outer_iter()
            .zip(d.iter().zip(v.iter()))
            .map(|(rho, (&d, &v))| &rho * (d.powi(3) * v * FRAC_PI_6))
            .reduce(|a, b| a + b)
            .unwrap();

        // local densities of the molecules of every component
        let n_comp = p.component_index[n - 1] + 1;
        let mut m_i: DVector<N> = DVector::zeros(n_comp);
        let mut s_i: Vec<f64> = vec![0.0; n_comp];
        for (&c, &a) in p.component_index.iter().zip(a.iter()) {
            m_i[c] += a;
            s_i[c] += 1.0;
        }
        let mut rho_comp: Array2<N> = Array::zeros((n_comp, eta.len()));
        for (rho, &c) in density.outer_iter().zip(p.component_index.iter()) {
            let mut rho_c = rho_comp.index_axis_mut(Axis(0), c);
            rho_c += &rho.mapv(|r| r / s_i[c]);
        }

        // mean segment number and compressibility term of the local reference fluid
        // (in the limit of vanishing density m_bar = 1 and C1 = 1)
        let reference = ReferenceFluid::new(&p.component_index, &p.bonds, &d, &av);
        let mut m_bar: Array1<N> = Array::ones(eta.raw_dim());
        let mut c1: Array1<N> = Array::ones(eta.raw_dim());
        let mut x = DVector::zeros(n_comp);
        for (k, rho) in rho_comp.axis_iter(Axis(1)).enumerate() {
            let density = rho.sum();
            if density.re() > f64::EPSILON {
                x.iter_mut().zip(rho).for_each(|(x, &r)| *x = r / density);
                m_bar[k] = x.dot(&m_i);
                c1[k] = reference.helmholtz_energy_density(density, &x).1;
            }
        }

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

        // I1 and I2
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

        // Helmholtz energy density
        Ok((-rho1mix * i1 * 2.0 - rho2mix * m_bar * c1 * i2) * PI)
    }
}
