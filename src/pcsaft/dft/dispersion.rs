use super::polar::calculate_helmholtz_energy_density_polar;
use super::PcSaftParameters;
use crate::pcsaft::eos::dispersion::{A0, A1, A2, B0, B1, B2};
use feos_core::EosError;
use feos_dft::{
    FunctionalContributionDual, WeightFunction, WeightFunctionInfo, WeightFunctionShape,
};
use ndarray::*;
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_3, PI};
use std::fmt;
use std::rc::Rc;

/// psi Parameter for DFT (Sauer2017)
const PSI_DFT: f64 = 1.3862;
/// psi Parameter for pDGT (Rehner2018)
const PSI_PDGT: f64 = 1.3286;

#[derive(Clone)]
pub struct AttractiveFunctional {
    parameters: Rc<PcSaftParameters>,
}

impl AttractiveFunctional {
    pub fn new(parameters: Rc<PcSaftParameters>) -> Self {
        Self { parameters }
    }
}

fn att_weight_functions<N: DualNum<f64> + ScalarOperand>(
    p: &PcSaftParameters,
    psi: f64,
    temperature: N,
) -> WeightFunctionInfo<N> {
    let d = p.hs_diameter(temperature);
    WeightFunctionInfo::new(Array1::from_shape_fn(d.len(), |i| i), false).add(
        WeightFunction::new_scaled(d * psi, WeightFunctionShape::Theta),
        false,
    )
}

impl<N: DualNum<f64> + ScalarOperand> FunctionalContributionDual<N> for AttractiveFunctional {
    fn weight_functions(&self, temperature: N) -> WeightFunctionInfo<N> {
        att_weight_functions(&self.parameters, PSI_DFT, temperature)
    }

    fn weight_functions_pdgt(&self, temperature: N) -> WeightFunctionInfo<N> {
        att_weight_functions(&self.parameters, PSI_PDGT, temperature)
    }

    fn calculate_helmholtz_energy_density(
        &self,
        temperature: N,
        density: ArrayView2<N>,
    ) -> Result<Array1<N>, EosError> {
        // auxiliary variables
        let p = &self.parameters;
        let n = p.m.len();

        // temperature dependent segment radius
        let r = p.hs_diameter(temperature) * 0.5;

        // packing fraction
        let eta = density
            .outer_iter()
            .zip((&r * &r * &r * &p.m * 4.0 * FRAC_PI_3).into_iter())
            .fold(
                Array::zeros(density.raw_dim().remove_axis(Axis(0))),
                |acc: Array1<N>, (rho, r3m)| acc + &rho * r3m,
            );

        // mean segment number
        let mut rhog = Array::zeros(eta.raw_dim());
        let mut m_bar = Array::zeros(eta.raw_dim());
        for (rhoi, &mi) in density.axis_iter(Axis(0)).zip(p.m.iter()) {
            m_bar += &(&rhoi * mi);
            rhog += &rhoi;
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
                let eps_ij_t = temperature.recip() * p.epsilon_k_ij[(i, j)];
                let sigma_ij_3 = p.sigma_ij[(i, j)].powi(3);
                rho1mix = rho1mix
                    + (&density.index_axis(Axis(0), i) * &density.index_axis(Axis(0), j))
                        .mapv(|x| x * (eps_ij_t * sigma_ij_3 * p.m[i] * p.m[j]));
                rho2mix = rho2mix
                    + (&density.index_axis(Axis(0), i) * &density.index_axis(Axis(0), j))
                        .mapv(|x| x * (eps_ij_t * eps_ij_t * sigma_ij_3 * p.m[i] * p.m[j]));
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
        let phi_polar = calculate_helmholtz_energy_density_polar(p, temperature, density)?;
        Ok((-rho1mix * i1 * 2.0 - rho2mix * m_bar * c1 * i2) * PI + phi_polar)
    }
}

impl fmt::Display for AttractiveFunctional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Attractive functional")
    }
}
