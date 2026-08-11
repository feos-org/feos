//! Generic implementation of the hard-sphere contribution
//! that can be used across models.
use feos_core::StateHD;
use nalgebra::DVector;
use num_dual::DualNum;
use std::borrow::Cow;
use std::f64::consts::FRAC_PI_6;

#[cfg(feature = "dft")]
mod dft;
#[cfg(feature = "dft")]
pub use dft::{FMTContribution, FMTFunctional, FMTVersion, HardSphereParameters};

/// Different monomer shapes for FMT and BMCSL.
pub enum MonomerShape<'a, D> {
    /// For spherical monomers, the number of components.
    Spherical(usize),
    /// For non-spherical molecules in a homosegmented approach, the
    /// chain length parameter $m$.
    NonSpherical(&'a DVector<f64>),
    /// For non-spherical molecules in a heterosegmented approach,
    /// the geometry factors for every segment and the component
    /// index for every segment.
    Heterosegmented([DVector<D>; 4], &'a DVector<usize>),
}

/// Properties of (generalized) hard sphere systems.
pub trait HardSphereProperties {
    /// The [MonomerShape] used in the model.
    fn monomer_shape<D: DualNum<Primitive = f64> + Copy>(&self, temperature: D) -> MonomerShape<'_, D>;

    /// The temperature dependent hard-sphere diameters of every segment.
    fn hs_diameter<D: DualNum<Primitive = f64> + Copy>(&self, temperature: D) -> DVector<D>;

    /// For every segment, the index of the component that it is on.
    fn component_index(&self) -> Cow<'_, [usize]> {
        match self.monomer_shape(1.0) {
            MonomerShape::Spherical(n) => Cow::Owned((0..n).collect()),
            MonomerShape::NonSpherical(m) => Cow::Owned((0..m.len()).collect()),
            MonomerShape::Heterosegmented(_, component_index) => {
                Cow::Borrowed(component_index.as_slice())
            }
        }
    }

    /// The geometry coefficients $C_{k,\alpha}$ for every segment.
    fn geometry_coefficients<D: DualNum<Primitive = f64> + Copy>(&self, temperature: D) -> [DVector<D>; 4] {
        match self.monomer_shape(temperature) {
            MonomerShape::Spherical(n) => {
                let m = DVector::from_element(n, D::from(1.0));
                [m.clone(), m.clone(), m.clone(), m]
            }
            MonomerShape::NonSpherical(m) => {
                let m = m.map(D::from);
                [m.clone(), m.clone(), m.clone(), m]
            }
            MonomerShape::Heterosegmented(g, _) => g,
        }
    }

    /// The packing fractions $\zeta_k$.
    fn zeta<D: DualNum<Primitive = f64> + Copy, const N: usize>(
        &self,
        temperature: D,
        partial_density: &DVector<D>,
        k: [i32; N],
    ) -> [D; N] {
        let diameter = self.hs_diameter(temperature);
        weighted_zeta(
            self.monomer_shape(temperature),
            &diameter,
            partial_density,
            k,
        )
    }
}

/// $\frac{\pi}{6}\sum_\alpha C_{k,\alpha} w_{i(\alpha)}d_\alpha^k$ for every requested $k$,
/// where the weights $w_i$ are either partial densities or mole fractions.
///
/// Matches on the [MonomerShape] directly instead of materializing
/// [HardSphereProperties::component_index] and
/// [HardSphereProperties::geometry_coefficients], which would allocate.
#[inline]
fn weighted_zeta<D: DualNum<Primitive = f64> + Copy, const N: usize>(
    monomer_shape: MonomerShape<'_, D>,
    diameter: &DVector<D>,
    weights: &DVector<D>,
    k: [i32; N],
) -> [D; N] {
    let mut zeta = [D::zero(); N];
    match monomer_shape {
        MonomerShape::Spherical(_) => {
            for i in 0..diameter.len() {
                for (z, &k) in zeta.iter_mut().zip(k.iter()) {
                    *z += weights[i] * diameter[i].powi(k) * FRAC_PI_6;
                }
            }
        }
        MonomerShape::NonSpherical(m) => {
            for i in 0..diameter.len() {
                for (z, &k) in zeta.iter_mut().zip(k.iter()) {
                    *z += weights[i] * diameter[i].powi(k) * (m[i] * FRAC_PI_6);
                }
            }
        }
        MonomerShape::Heterosegmented(g, component_index) => {
            for i in 0..diameter.len() {
                for (z, &k) in zeta.iter_mut().zip(k.iter()) {
                    *z += weights[component_index[i]]
                        * diameter[i].powi(k)
                        * (g[k as usize][i] * FRAC_PI_6);
                }
            }
        }
    }
    zeta
}

/// Implementation of the BMCSL equation of state for hard-sphere mixtures.
///
/// This structure provides an implementation of the Boublík-Mansoori-Carnahan-Starling-Leland (BMCSL) equation of state ([Boublík, 1970](https://doi.org/10.1063/1.1673824), [Mansoori et al., 1971](https://doi.org/10.1063/1.1675048)) that is often used as reference contribution in SAFT equations of state. The implementation is generalized to allow the description of non-sperical or fused-sphere reference fluids.
///
/// The reduced Helmholtz energy is calculated according to
/// $$\frac{\beta A}{V}=\frac{6}{\pi}\left(\frac{3\zeta_1\zeta_2}{1-\zeta_3}+\frac{\zeta_2^3}{\zeta_3\left(1-\zeta_3\right)^2}+\left(\frac{\zeta_2^3}{\zeta_3^2}-\zeta_0\right)\ln\left(1-\zeta_3\right)\right)$$
/// with the packing fractions
/// $$\zeta_k=\frac{\pi}{6}\sum_\alpha C_{k,\alpha}\rho_\alpha d_\alpha^k,~~~~~~~~k=0\ldots 3.$$
///
/// The geometry coefficients $C_{k,\alpha}$ and the segment diameters $d_\alpha$ are specified via the [HardSphereProperties] trait.
pub struct HardSphere;

impl HardSphere {
    /// Returns the Helmholtz energy, packing fractions, and temperature dependent diameters without redundant calculations.
    #[inline]
    pub fn helmholtz_energy_density_and_properties<
        D: DualNum<Primitive = f64> + Copy,
        P: HardSphereProperties,
    >(
        &self,
        parameters: &P,
        state: &StateHD<D>,
    ) -> (D, [D; 4], DVector<D>) {
        let p = parameters;
        let diameter = p.hs_diameter(state.temperature);
        let mut zeta = weighted_zeta(
            p.monomer_shape(state.temperature),
            &diameter,
            &state.molefracs,
            [0, 1, 2, 3],
        );
        let zeta_23 = zeta[2] / zeta[3];
        let density = state.partial_density.sum();
        zeta.iter_mut().for_each(|z| *z *= density);
        let frac_1mz3 = -(zeta[3] - 1.0).recip();
        let a = (zeta[1] * zeta[2] * frac_1mz3 * 3.0
            + zeta[2].powi(2) * frac_1mz3.powi(2) * zeta_23
            + (zeta[2] * zeta_23.powi(2) - zeta[0]) * (-zeta[3]).ln_1p())
            / std::f64::consts::FRAC_PI_6;
        (a, zeta, diameter)
    }

    #[inline]
    pub fn helmholtz_energy_density<D: DualNum<Primitive = f64> + Copy, P: HardSphereProperties>(
        &self,
        parameters: &P,
        state: &StateHD<D>,
    ) -> D {
        self.helmholtz_energy_density_and_properties(parameters, state)
            .0
    }
}
