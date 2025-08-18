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
    NonSpherical(DVector<D>),
    /// For non-spherical molecules in a heterosegmented approach,
    /// the geometry factors for every segment and the component
    /// index for every segment.
    Heterosegmented([DVector<D>; 4], &'a DVector<usize>),
}

/// Properties of (generalized) hard sphere systems.
pub trait HardSphereProperties {
    /// The [MonomerShape] used in the model.
    fn monomer_shape<D: DualNum<f64> + Copy>(&self, temperature: D) -> MonomerShape<D>;

    /// The temperature dependent hard-sphere diameters of every segment.
    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> DVector<D>;

    /// For every segment, the index of the component that it is on.
    fn component_index(&self) -> Cow<DVector<usize>> {
        match self.monomer_shape(1.0) {
            MonomerShape::Spherical(n) => Cow::Owned(DVector::from_fn(n, |i, _| i)),
            MonomerShape::NonSpherical(m) => Cow::Owned(DVector::from_fn(m.len(), |i, _| i)),
            MonomerShape::Heterosegmented(_, component_index) => Cow::Borrowed(component_index),
        }
    }

    /// The geometry coefficients $C_{k,\alpha}$ for every segment.
    fn geometry_coefficients<D: DualNum<f64> + Copy>(&self, temperature: D) -> [DVector<D>; 4] {
        match self.monomer_shape(temperature) {
            MonomerShape::Spherical(n) => {
                let m = DVector::from_element(n, D::from(1.0));
                [m.clone(), m.clone(), m.clone(), m]
            }
            MonomerShape::NonSpherical(m) => [m.clone(), m.clone(), m.clone(), m],
            MonomerShape::Heterosegmented(g, _) => g,
        }
    }

    /// The packing fractions $\zeta_k$.
    fn zeta<D: DualNum<f64> + Copy, const N: usize>(
        &self,
        temperature: D,
        partial_density: &DVector<D>,
        k: [i32; N],
    ) -> [D; N] {
        let component_index = self.component_index();
        let geometry_coefficients = self.geometry_coefficients(temperature);
        let diameter = self.hs_diameter(temperature);
        let mut zeta = [D::zero(); N];
        for i in 0..diameter.len() {
            for (z, &k) in zeta.iter_mut().zip(k.iter()) {
                *z += partial_density[component_index[i]]
                    * diameter[i].powi(k)
                    * (geometry_coefficients[k as usize][i] * FRAC_PI_6);
            }
        }

        zeta
    }
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
        D: DualNum<f64> + Copy,
        P: HardSphereProperties,
    >(
        &self,
        parameters: &P,
        state: &StateHD<D>,
    ) -> (D, [D; 4], DVector<D>) {
        let p = parameters;
        let diameter = p.hs_diameter(state.temperature);

        let component_index = p.component_index();
        let geometry_coefficients = p.geometry_coefficients(state.temperature);
        let mut zeta = [D::zero(); 4];
        for i in 0..diameter.len() {
            for (z, &k) in zeta.iter_mut().zip([0, 1, 2, 3].iter()) {
                *z += state.molefracs[component_index[i]]
                    * diameter[i].powi(k)
                    * (geometry_coefficients[k as usize][i] * FRAC_PI_6);
            }
        }
        let zeta_23 = zeta[2] / zeta[3];
        let density = state.partial_density.sum();
        zeta.iter_mut().for_each(|z| *z *= density);
        let frac_1mz3 = -(zeta[3] - 1.0).recip();
        let a = (zeta[1] * zeta[2] * frac_1mz3 * 3.0
            + zeta[2].powi(2) * frac_1mz3.powi(2) * zeta_23
            + (zeta[2] * zeta_23.powi(2) - zeta[0]) * (zeta[3] * (-1.0)).ln_1p())
            / std::f64::consts::FRAC_PI_6;
        (a, zeta, diameter)
    }

    #[inline]
    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy, P: HardSphereProperties>(
        &self,
        parameters: &P,
        state: &StateHD<D>,
    ) -> D {
        self.helmholtz_energy_density_and_properties(parameters, state)
            .0
    }
}
