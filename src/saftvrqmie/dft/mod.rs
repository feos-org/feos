use crate::hard_sphere::{FMTContribution, FMTVersion, HardSphereProperties, MonomerShape};
use crate::saftvrqmie::eos::SaftVRQMieOptions;
use crate::saftvrqmie::parameters::SaftVRQMieParameters;
use dispersion::AttractiveFunctional;
use feos_core::parameter::Parameter;
use feos_core::si::{MolarWeight, GRAM, MOL};
use feos_core::Components;
use feos_dft::adsorption::FluidParameters;
use feos_dft::solvation::PairPotential;
use feos_dft::{FunctionalContribution, HelmholtzEnergyFunctional, MoleculeShape, DFT};
use ndarray::{Array, Array1, Array2};
use non_additive_hs::NonAddHardSphereFunctional;
use num_dual::DualNum;
use std::f64::consts::FRAC_PI_6;
use std::sync::Arc;

mod dispersion;
mod non_additive_hs;

/// SAFT-VRQ Mie Helmholtz energy functional.
pub struct SaftVRQMieFunctional {
    pub parameters: Arc<SaftVRQMieParameters>,
    fmt_version: FMTVersion,
    options: SaftVRQMieOptions,
    contributions: Vec<Box<dyn FunctionalContribution>>,
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
        let mut contributions: Vec<Box<dyn FunctionalContribution>> = Vec::with_capacity(3);

        // Hard sphere contribution
        let hs = FMTContribution::new(&parameters, fmt_version);
        contributions.push(Box::new(hs));

        // Non-additive hard-sphere contribution
        if saft_options.inc_nonadd_term {
            let non_add_hs = NonAddHardSphereFunctional::new(parameters.clone());
            contributions.push(Box::new(non_add_hs));
        }

        // Dispersion
        let att = AttractiveFunctional::new(parameters.clone());
        contributions.push(Box::new(att));

        DFT(Self {
            parameters,
            fmt_version,
            options: saft_options,
            contributions,
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
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * &self.parameters.m * self.parameters.sigma.mapv(|v| v.powi(3)) * moles)
                .sum()
    }

    fn contributions(&self) -> &[Box<dyn FunctionalContribution>] {
        &self.contributions
    }

    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molarweight.clone() * GRAM / MOL
    }

    fn molecule_shape(&self) -> MoleculeShape {
        MoleculeShape::NonSpherical(&self.parameters.m)
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
