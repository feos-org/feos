use enum_dispatch::enum_dispatch;
use feos_core::StateHD;
use ndarray::ScalarOperand;
use num_dual::DualNum;

use super::{Cubic, alpha::AlphaFunction};

/// Parameters of cubics
pub struct MixtureParameters<D> {
    /// attractive contribution divided by R
    pub a: D,
    /// repulsive contribution
    pub b: D,
    /// volume translation
    pub c: D,
}

#[enum_dispatch]
pub trait MixingRuleFunction {
    fn apply<D: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        cubic: &Cubic,
        state: &StateHD<D>,
    ) -> MixtureParameters<D>;
}

/// Quadratic summation over a and b.
#[derive(Debug, Clone)]
pub struct Quadratic;

impl MixingRuleFunction for Quadratic {
    fn apply<D: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        cubic: &Cubic,
        state: &StateHD<D>,
    ) -> MixtureParameters<D> {
        let p = &cubic.parameters;
        let pc = &cubic.critical_parameters;
        let n = p.tc.len();
        let tr = p.tc.mapv(|tc| state.temperature / tc);
        let at = cubic.options.alpha.alpha(&p.acentric_factor, &tr) * &pc.ac;
        let mut a = D::zero();
        let mut b = D::zero();
        for i in 0..n {
            let xi = state.molefracs[i];
            let ai = at[i];
            let bi = pc.bc[i];
            a += xi * xi * ai;
            b += xi * xi * bi;
            for j in i + 1..n {
                a += xi * state.molefracs[j] * (ai * at[j]).sqrt() * (1.0 - p.k_ij[[i, j]]) * 2.0;
                b += xi * state.molefracs[j] * (bi + pc.bc[j]) * 0.5 * (1.0 - p.l_ij[[i, j]]) * 2.0;
            }
        }
        MixtureParameters { a, b, c: D::zero() }
    }
}

#[enum_dispatch(MixingRuleFunction)]
#[derive(Debug, Clone)]
pub enum MixingRule {
    Quadratic,
}
