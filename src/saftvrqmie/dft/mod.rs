use crate::hard_sphere::{FMTContribution, FMTVersion, HardSphereProperties, MonomerShape};
use crate::saftvrqmie::eos::SaftVRQMieOptions;
use crate::saftvrqmie::parameters::SaftVRQMieParameters;
use dispersion::AttractiveFunctional;
use feos_core::parameter::Parameter;
use feos_core::{Components, EosResult, Molarweight};
use feos_derive::FunctionalContribution;
use feos_dft::adsorption::FluidParameters;
use feos_dft::solvation::PairPotential;
use feos_dft::{
    FunctionalContribution, HelmholtzEnergyFunctional, MoleculeShape, WeightFunctionInfo, DFT,
};
use ndarray::{Array, Array1, Array2, ArrayView2, ScalarOperand};
use non_additive_hs::NonAddHardSphereFunctional;
use num_dual::DualNum;
use quantity::{MolarWeight, GRAM, MOL};
use std::f64::consts::FRAC_PI_6;
use std::sync::Arc;

mod dispersion;
mod non_additive_hs;

/// SAFT-VRQ Mie Helmholtz energy functional.
pub struct SaftVRQMieFunctional {
    pub parameters: Arc<SaftVRQMieParameters>,
    fmt_version: FMTVersion,
    options: SaftVRQMieOptions,
}

impl SaftVRQMieFunctional {
    pub fn new(parameters: Arc<SaftVRQMieParameters>) -> DFT<Self> {
        Self::with_options(
            parameters,
            FMTVersion::WhiteBear,
            SaftVRQMieOptions::default(),
        )
    }

    pub fn new_full(parameters: Arc<SaftVRQMieParameters>, fmt_version: FMTVersion) -> DFT<Self> {
        Self::with_options(parameters, fmt_version, SaftVRQMieOptions::default())
    }

    pub fn with_options(
        parameters: Arc<SaftVRQMieParameters>,
        fmt_version: FMTVersion,
        saft_options: SaftVRQMieOptions,
    ) -> DFT<Self> {
        DFT(Self {
            parameters,
            fmt_version,
            options: saft_options,
        })
    }
}

impl Components for SaftVRQMieFunctional {
    fn components(&self) -> usize {
        self.parameters.pure_records.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            Arc::new(self.parameters.subset(component_list)),
            self.fmt_version,
            self.options,
        )
        .0
    }
}

impl HelmholtzEnergyFunctional for SaftVRQMieFunctional {
    type Contribution = SaftVRQMieFunctionalContribution;

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * &self.parameters.m * self.parameters.sigma.mapv(|v| v.powi(3)) * moles)
                .sum()
    }

    fn contributions(&self) -> Box<(dyn Iterator<Item = SaftVRQMieFunctionalContribution>)> {
        let mut contributions = Vec::with_capacity(3);

        // Hard sphere contribution
        let hs = FMTContribution::new(&self.parameters, self.fmt_version);
        contributions.push(hs.into());

        // Non-additive hard-sphere contribution
        if self.options.inc_nonadd_term {
            let non_add_hs = NonAddHardSphereFunctional::new(self.parameters.clone());
            contributions.push(non_add_hs.into());
        }

        // Dispersion
        let att = AttractiveFunctional::new(self.parameters.clone());
        contributions.push(att.into());

        Box::new(contributions.into_iter())
    }

    fn molecule_shape(&self) -> MoleculeShape {
        MoleculeShape::NonSpherical(&self.parameters.m)
    }
}

impl Molarweight for SaftVRQMieFunctional {
    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}

impl HardSphereProperties for SaftVRQMieParameters {
    fn monomer_shape<N: DualNum<f64>>(&self, _: N) -> MonomerShape<N> {
        MonomerShape::Spherical(self.m.len())
    }

    fn hs_diameter<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D> {
        self.hs_diameter(temperature)
    }
}

impl FluidParameters for SaftVRQMieFunctional {
    fn epsilon_k_ff(&self) -> Array1<f64> {
        self.parameters.epsilon_k.clone()
    }

    fn sigma_ff(&self) -> &Array1<f64> {
        &self.parameters.sigma
    }
}

impl PairPotential for SaftVRQMieFunctional {
    fn pair_potential(&self, i: usize, r: &Array1<f64>, temperature: f64) -> Array2<f64> {
        Array::from_shape_fn((self.parameters.m.len(), r.len()), |(j, k)| {
            self.parameters.qmie_potential_ij(i, j, r[k], temperature)[0]
        })
    }
}

/// Individual contributions for the SAFT-VRQ Mie Helmholtz energy functional.
#[derive(FunctionalContribution)]
pub enum SaftVRQMieFunctionalContribution {
    Fmt(FMTContribution<SaftVRQMieParameters>),
    NonAddHardSphere(NonAddHardSphereFunctional),
    Attractive(AttractiveFunctional),
}
