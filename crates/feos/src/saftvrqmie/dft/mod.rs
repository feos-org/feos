use super::SaftVRQMie;
use crate::hard_sphere::{FMTContribution, FMTVersion, HardSphereProperties, MonomerShape};
use crate::saftvrqmie::eos::SaftVRQMieOptions;
use crate::saftvrqmie::parameters::{SaftVRQMieParameters, SaftVRQMiePars};
use dispersion::AttractiveFunctional;
use feos_core::FeosResult;
use feos_derive::FunctionalContribution;
use feos_dft::adsorption::FluidParameters;
use feos_dft::solvation::PairPotential;
use feos_dft::{FunctionalContribution, HelmholtzEnergyFunctionalDyn, MoleculeShape};
use nalgebra::DVector;
use ndarray::{Array, Array1, Array2};
use non_additive_hs::NonAddHardSphereFunctional;
use num_dual::DualNum;

mod dispersion;
mod non_additive_hs;

impl SaftVRQMie {
    /// SAFT-VRQ Mie model with default options and provided FMT version.
    pub fn with_fmt_version(
        parameters: SaftVRQMieParameters,
        fmt_version: FMTVersion,
    ) -> FeosResult<Self> {
        let options = SaftVRQMieOptions {
            fmt_version,
            ..Default::default()
        };
        Self::with_options(parameters, options)
    }
}

impl HelmholtzEnergyFunctionalDyn for SaftVRQMie {
    type Contribution<'a> = SaftVRQMieFunctionalContribution<'a>;

    fn contributions<'a>(&'a self) -> impl Iterator<Item = SaftVRQMieFunctionalContribution<'a>> {
        let mut contributions = Vec::with_capacity(3);

        // Hard sphere contribution
        let hs = FMTContribution::new(&self.params, self.options.fmt_version);
        contributions.push(hs.into());

        // Non-additive hard-sphere contribution
        if self.options.inc_nonadd_term {
            let non_add_hs = NonAddHardSphereFunctional::new(&self.params);
            contributions.push(non_add_hs.into());
        }

        // Dispersion
        let att = AttractiveFunctional::new(&self.params);
        contributions.push(att.into());

        contributions.into_iter()
    }

    fn molecule_shape(&self) -> MoleculeShape<'_> {
        MoleculeShape::NonSpherical(&self.params.m)
    }
}

impl HardSphereProperties for SaftVRQMiePars {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<'_, N> {
        MonomerShape::Spherical(self.m.len())
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> DVector<D> {
        self.hs_diameter(temperature)
    }
}

impl FluidParameters for SaftVRQMie {
    fn epsilon_k_ff(&self) -> DVector<f64> {
        self.params.epsilon_k.clone()
    }

    fn sigma_ff(&self) -> DVector<f64> {
        self.params.sigma.clone()
    }
}

impl PairPotential for SaftVRQMie {
    fn pair_potential(&self, i: usize, r: &Array1<f64>, temperature: f64) -> Array2<f64> {
        Array::from_shape_fn((self.params.m.len(), r.len()), |(j, k)| {
            self.params.qmie_potential_ij(i, j, r[k], temperature)[0]
        })
    }
}

/// Individual contributions for the SAFT-VRQ Mie Helmholtz energy functional.
#[derive(FunctionalContribution)]
pub enum SaftVRQMieFunctionalContribution<'a> {
    Fmt(FMTContribution<'a, SaftVRQMiePars>),
    NonAddHardSphere(NonAddHardSphereFunctional<'a>),
    Attractive(AttractiveFunctional<'a>),
}
