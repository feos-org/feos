use crate::hard_sphere::HardSphere;
use nalgebra::{DVector, Matrix4xX};
use num_dual::{Dual2, DualNum, second_derivative};
use petgraph::graph::UnGraph;
use petgraph::visit::EdgeRef;
use std::f64::consts::FRAC_PI_6;

/// The reference fluid of FC-SAFT: fused hard spheres (BMCSL) and the
/// fused-sphere chain contribution.
///
/// All temperature dependent quantities are evaluated on construction, so that
/// [ReferenceFluid::helmholtz_energy_density] only has to evaluate a scalar
/// expression in the density. This expression is evaluated with a second-order
/// dual number, which yields the compressibility term
/// $$C_1=\left(1+\rho\frac{\partial^2\phi^\mathrm{ref}}{\partial\rho^2}\right)^{-1}$$
/// (at constant temperature and composition) of the dispersion contribution
/// together with the Helmholtz energy densities.
pub(crate) struct ReferenceFluid<D> {
    /// $\frac{\pi}{6}\sum_{\alpha\in i}C_{k,\alpha}d_\alpha^k$ for every component $i$
    zeta: Matrix4xX<D>,
    /// component index, geometry factor $b$, and number of every (type of) bond
    bonds: Vec<(usize, D, f64)>,
}

impl<D: DualNum<Primitive = f64> + Copy> ReferenceFluid<D> {
    /// The reference fluid of the heterosegmented model.
    ///
    /// Requires the component index and the temperature dependent diameter
    /// and fused-sphere coefficients $a$ and $v$ of every segment, as well as
    /// the bond lengths.
    pub(crate) fn new(
        component_index: &DVector<usize>,
        bonds: &UnGraph<(), f64>,
        diameter: &DVector<D>,
        [a, v]: &[DVector<D>; 2],
    ) -> Self {
        let n = component_index.iter().max().map_or(0, |&c| c + 1);
        let mut zeta = Matrix4xX::zeros(n);
        for (i, &c) in component_index.iter().enumerate() {
            let d = diameter[i];
            zeta[(0, c)] += D::from(FRAC_PI_6);
            zeta[(1, c)] += a[i] * d * FRAC_PI_6;
            zeta[(2, c)] += a[i] * d * d * FRAC_PI_6;
            zeta[(3, c)] += v[i] * d * d * d * FRAC_PI_6;
        }
        let bonds = bonds
            .edge_references()
            .map(|e| {
                let (i, j) = (e.source().index(), e.target().index());
                let l = *e.weight();
                let (di, dj) = (diameter[i], diameter[j]);
                let b = -((di - dj).powi(2) - 4.0 * l.powi(2)) / (4.0 * l);
                (component_index[i], b, 1.0)
            })
            .collect();
        Self { zeta, bonds }
    }

    /// The reference fluid of the homosegmented model.
    ///
    /// Requires the number of segments `s` and bond lengths `l` of every
    /// component, as well as the temperature dependent diameter and the
    /// reduced surface $m$ and volume $m^*$.
    pub(crate) fn new_homosegmented(
        s: &DVector<f64>,
        l: &DVector<f64>,
        diameter: &DVector<D>,
        [m, m_star]: &[DVector<D>; 2],
    ) -> Self {
        let zeta = Matrix4xX::from_fn(s.len(), |k, i| {
            let c = [D::from(s[i]), m[i], m[i], m_star[i]];
            c[k] * diameter[i].powi(k as i32) * FRAC_PI_6
        });
        let bonds = (0..s.len())
            .filter(|&i| s[i] > 1.0)
            .map(|i| (i, D::from(l[i]), s[i] - 1.0))
            .collect();
        Self { zeta, bonds }
    }

    /// The Helmholtz energy densities of the hard-sphere and the chain
    /// contribution, and the compressibility term $C_1$.
    pub(crate) fn helmholtz_energy_density(
        &self,
        density: D,
        molefracs: &DVector<D>,
    ) -> ([D; 2], D) {
        let zeta_hat: [D; 4] = (&self.zeta * molefracs).into();
        let zeta_23 = Dual2::from_re(zeta_hat[2] / zeta_hat[3]);

        // reference fluid and its second derivative w.r.t. the density, which
        // yields the compressibility term C1
        let ((hs, _, d2_hs), (chain, _, d2_chain)) = second_derivative(
            |rho: Dual2<D>| {
                // the packing fractions are linear in the density
                let zeta = zeta_hat.map(|z| rho.scale(&z));

                // hard spheres
                let hs = HardSphere::bmcsl_helmholtz_energy_density(zeta, zeta_23);

                // fused-sphere chains
                let frac_1mz3 = -(zeta[3] - 1.0).recip();
                let ln_y = self
                    .bonds
                    .iter()
                    .map(|&(c, b, count)| {
                        let z2b = zeta[2].scale(&b);
                        let y =
                            z2b * frac_1mz3 * frac_1mz3 * (z2b * frac_1mz3 * 0.5 + 1.5) + frac_1mz3;
                        y.ln().scale(&(molefracs[c] * count))
                    })
                    .fold(Dual2::from_re(D::zero()), |acc, x| acc + x);
                (hs, -rho * ln_y)
            },
            density,
        );

        let c1 = ((d2_hs + d2_chain) * density + 1.0).recip();
        ([hs, chain], c1)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::dvector;

    /// C1 of PC-SAFT, i.e., for tangent chains with `m` segments.
    fn c1_pcsaft(m: f64, eta: f64) -> f64 {
        (m * (eta * 8.0 - eta.powi(2) * 2.0) / (eta - 1.0).powi(4)
            - (m - 1.0)
                * (eta * 20.0 - eta.powi(2) * 27.0 + eta.powi(3) * 12.0 - eta.powi(4) * 2.0)
                / ((eta - 1.0) * (eta - 2.0)).powi(2)
            + 1.0)
            .recip()
    }

    #[test]
    fn test_compressibility_term_tangent_chains() {
        // for tangent spheres (l = d) the reduced surface and volume are equal to
        // the number of segments and C1 reduces to the expression of PC-SAFT
        let eta = 0.3;
        for s in [1, 2, 5] {
            let m = s as f64;
            let density = eta / (m * FRAC_PI_6);
            let c1 = c1_pcsaft(m, eta);

            let d = dvector![1.0];
            let geometry = [dvector![m], dvector![m]];
            let homo =
                ReferenceFluid::new_homosegmented(&dvector![m], &dvector![1.0], &d, &geometry);
            let (_, c1_homo) = homo.helmholtz_energy_density(density, &dvector![1.0]);
            assert_relative_eq!(c1_homo, c1, max_relative = 1e-14);

            let component_index = DVector::zeros(s);
            let bonds = UnGraph::from_edges((1..s as u32).map(|i| (i - 1, i, 1.0)));
            let d = DVector::from_element(s, 1.0);
            let geometry = [DVector::from_element(s, 1.0), DVector::from_element(s, 1.0)];
            let hetero = ReferenceFluid::new(&component_index, &bonds, &d, &geometry);
            let (_, c1_hetero) = hetero.helmholtz_energy_density(density, &dvector![1.0]);
            assert_relative_eq!(c1_hetero, c1, max_relative = 1e-14);
        }
    }
}
