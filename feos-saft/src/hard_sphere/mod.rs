use feos_core::{HelmholtzEnergyDual, StateHD};
use ndarray::*;
use num_dual::DualNum;
use std::borrow::Cow;
use std::f64::consts::FRAC_PI_6;
use std::fmt;
use std::rc::Rc;

#[cfg(feature = "dft")]
mod dft;
#[cfg(feature = "dft")]
pub use dft::{FMTContribution, FMTFunctional, FMTVersion};

/// Different monomer shapes for FMT.
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
    fn monomer_shape<D: DualNum<f64>>(&self, temperature: D) -> MonomerShape<D>;
    fn hs_diameter<D: DualNum<f64>>(&self, temperature: D) -> Array1<D>;

    fn component_index(&self) -> Cow<Array1<usize>> {
        match self.monomer_shape(1.0) {
            MonomerShape::Spherical(n) => Cow::Owned(Array1::from_shape_fn(n, |i| i)),
            MonomerShape::NonSpherical(m) => Cow::Owned(Array1::from_shape_fn(m.len(), |i| i)),
            MonomerShape::Heterosegmented(_, component_index) => Cow::Borrowed(component_index),
        }
    }

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

    fn zeta<D: DualNum<f64>, const N: usize>(
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

    fn zeta_23<D: DualNum<f64>>(&self, temperature: D, molefracs: &Array1<D>) -> D {
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

pub struct HardSphere<P> {
    parameters: Rc<P>,
}

impl<P> HardSphere<P> {
    pub fn new(parameters: &Rc<P>) -> Self {
        Self {
            parameters: parameters.clone(),
        }
    }
}

impl<D: DualNum<f64>, P: HardSphereProperties> HelmholtzEnergyDual<D> for HardSphere<P> {
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

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::pcsaft::parameters::utils::{
//         butane_parameters, propane_butane_parameters, propane_parameters,
//     };
//     use approx::assert_relative_eq;
//     use ndarray::arr1;

//     #[test]
//     fn helmholtz_energy() {
//         let hs = HardSphere {
//             parameters: propane_parameters(),
//         };
//         let t = 250.0;
//         let v = 1000.0;
//         let n = 1.0;
//         let s = StateHD::new(t, v, arr1(&[n]));
//         let a_rust = hs.helmholtz_energy(&s);
//         assert_relative_eq!(a_rust, 0.410610492598808, epsilon = 1e-10);
//     }

//     #[test]
//     fn mix() {
//         let c1 = HardSphere {
//             parameters: propane_parameters(),
//         };
//         let c2 = HardSphere {
//             parameters: butane_parameters(),
//         };
//         let c12 = HardSphere {
//             parameters: propane_butane_parameters(),
//         };
//         let t = 250.0;
//         let v = 2.5e28;
//         let n = 1.0;
//         let s = StateHD::new(t, v, arr1(&[n]));
//         let a1 = c1.helmholtz_energy(&s);
//         let a2 = c2.helmholtz_energy(&s);
//         let s1m = StateHD::new(t, v, arr1(&[n, 0.0]));
//         let a1m = c12.helmholtz_energy(&s1m);
//         let s2m = StateHD::new(t, v, arr1(&[0.0, n]));
//         let a2m = c12.helmholtz_energy(&s2m);
//         assert_relative_eq!(a1, a1m, epsilon = 1e-14);
//         assert_relative_eq!(a2, a2m, epsilon = 1e-14);
//     }
// }
