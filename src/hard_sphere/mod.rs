//! Generic implementation of the hard-sphere contribution
//! that can be used across models.
use feos_core::{HelmholtzEnergyDual, StateHD};
use ndarray::*;
use num_dual::DualNum;
use std::f64::consts::FRAC_PI_6;
use std::fmt;
use std::{borrow::Cow, sync::Arc};

#[cfg(feature = "dft")]
mod dft;
#[cfg(feature = "dft")]
pub use dft::{FMTContribution, FMTFunctional, FMTVersion};

/// Different monomer shapes for FMT and BMCSL.
pub enum MonomerShape<'a, D> {
    /// For spherical monomers, the number of components.
    Spherical(usize),
    /// For non-spherical molecules in a homosegmented approach, the
    /// chain length parameter $m$.
    NonSpherical(Array1<D>),
    /// For non-spherical molecules in a heterosegmented approach,
    /// the geometry factors for every segment and the component
    /// index for every segment.
    Heterosegmented([Array1<D>; 4], &'a Array1<usize>),
}

/// Properties of (generalized) hard sphere systems.
pub trait HardSphereProperties {
    /// The [MonomerShape] used in the model.
    fn monomer_shape<D: DualNum<f64>>(&self, temperature: D) -> MonomerShape<D>;

    /// The temperature dependent hard-sphere diameters of every segment.
    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D>;

    /// For every segment, the index of the component that it is on.
    fn component_index(&self) -> Cow<Array1<usize>> {
        match self.monomer_shape(1.0) {
            MonomerShape::Spherical(n) => Cow::Owned(Array1::from_shape_fn(n, |i| i)),
            MonomerShape::NonSpherical(m) => Cow::Owned(Array1::from_shape_fn(m.len(), |i| i)),
            MonomerShape::Heterosegmented(_, component_index) => Cow::Borrowed(component_index),
        }
    }

    /// The geometry coefficients $C_{k,\alpha}$ for every segment.
    fn geometry_coefficients<D: DualNum<f64>>(&self, temperature: D) -> [Array1<D>; 4] {
        match self.monomer_shape(temperature) {
            MonomerShape::Spherical(n) => {
                let m = Array1::ones(n);
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
        partial_density: &Array1<D>,
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

    /// The fraction $\frac{\zeta_2}{\zeta_3}$ evaluated in a way to avoid a division by 0 when the density is 0.
    fn zeta_23<D: DualNum<f64> + Copy>(&self, temperature: D, molefracs: &Array1<D>) -> D {
        let component_index = self.component_index();
        let geometry_coefficients = self.geometry_coefficients(temperature);
        let diameter = self.hs_diameter(temperature);
        let mut zeta: [D; 2] = [D::zero(); 2];
        for i in 0..diameter.len() {
            for (k, z) in zeta.iter_mut().enumerate() {
                *z += molefracs[component_index[i]]
                    * diameter[i].powi((k + 2) as i32)
                    * (geometry_coefficients[k + 2][i] * FRAC_PI_6);
            }
        }

        zeta[0] / zeta[1]
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
pub struct HardSphere<P> {
    parameters: Arc<P>,
}

impl<P> HardSphere<P> {
    pub fn new(parameters: &Arc<P>) -> Self {
        Self {
            parameters: parameters.clone(),
        }
    }
}

impl<D: DualNum<f64> + Copy, P: HardSphereProperties> HelmholtzEnergyDual<D> for HardSphere<P> {
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let p = &self.parameters;
        let zeta = p.zeta(state.temperature, &state.partial_density, [0, 1, 2, 3]);
        let frac_1mz3 = -(zeta[3] - 1.0).recip();
        let zeta_23 = p.zeta_23(state.temperature, &state.molefracs);
        state.volume * 6.0 / std::f64::consts::PI
            * (zeta[1] * zeta[2] * frac_1mz3 * 3.0
                + zeta[2].powi(2) * frac_1mz3.powi(2) * zeta_23
                + (zeta[2] * zeta_23.powi(2) - zeta[0]) * (zeta[3] * (-1.0)).ln_1p())
    }
}

impl<P> fmt::Display for HardSphere<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hard Sphere")
    }
}
