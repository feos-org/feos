use super::FcSaftOptions;
use super::parameters::FcSaftPars;
use super::record::FcSaftParameters;
use crate::association::{Association, YuWuAssociationFunctional};
use crate::hard_sphere::{FMTContribution, FMTVersion};
use feos_core::{FeosResult, Molarweight, ResidualDyn, StateHD, Subset};
use feos_derive::FunctionalContribution;
use feos_dft::adsorption::FluidParameters;
use feos_dft::{
    FunctionalContribution, HelmholtzEnergyFunctional, HelmholtzEnergyFunctionalDyn, MoleculeShape,
};
use nalgebra::DVector;
use ndarray::Array1;
use num_dual::DualNum;
use petgraph::graph::UnGraph;
use quantity::MolarWeight;
use std::f64::consts::FRAC_PI_6;

mod dispersion;
mod fused_chain;
use dispersion::DispersionFunctional;
use fused_chain::FusedChainFunctional;

/// Heterosegmented FC-SAFT Helmholtz energy functional.
pub struct FcSaftFunctional {
    pub parameters: FcSaftParameters,
    pub params: FcSaftPars,
    association: Option<Association>,
    fmt_version: FMTVersion,
    options: FcSaftOptions,
}

impl FcSaftFunctional {
    pub fn new(parameters: FcSaftParameters) -> Self {
        Self::with_options(parameters, FMTVersion::WhiteBear, FcSaftOptions::default())
    }

    pub fn with_options(
        parameters: FcSaftParameters,
        fmt_version: FMTVersion,
        options: FcSaftOptions,
    ) -> Self {
        let params = FcSaftPars::new(&parameters);
        let association = (!parameters.association.is_empty())
            .then(|| Association::new(options.max_iter_cross_assoc, options.tol_cross_assoc));
        Self {
            parameters,
            params,
            association,
            fmt_version,
            options,
        }
    }
}

impl Subset for FcSaftFunctional {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            self.parameters.subset(component_list),
            self.fmt_version,
            self.options,
        )
    }
}

impl ResidualDyn for FcSaftFunctional {
    fn components(&self) -> usize {
        self.parameters.molar_weight.len()
    }

    fn compute_max_density<D: DualNum<Primitive = f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        let p = &self.params;
        let [_, v] = p.fused_sphere_coefficients(&p.sigma);
        let segment_volume: D = (0..p.sigma.len())
            .map(|i| molefracs[p.component_index[i]] * p.sigma[i].powi(3) * v[i])
            .sum();
        (segment_volume * FRAC_PI_6).recip() * self.options.max_eta
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<Primitive = f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        self.evaluate_bulk(state)
    }
}

impl HelmholtzEnergyFunctionalDyn for FcSaftFunctional {
    type Contribution<'a> = FcSaftFunctionalContribution<'a>;

    fn molecule_shape(&self) -> MoleculeShape<'_> {
        MoleculeShape::Heterosegmented(&self.params.component_index)
    }

    fn contributions<'a>(&'a self) -> impl Iterator<Item = FcSaftFunctionalContribution<'a>> {
        let mut contributions = Vec::with_capacity(4);

        // Hard sphere contribution
        let hs = FMTContribution::new(&self.params, self.fmt_version);
        contributions.push(hs.into());

        // Fused hard chains
        let chain = FusedChainFunctional::new(&self.params);
        contributions.push(chain.into());

        // Dispersion
        let disp = DispersionFunctional::new(&self.params);
        contributions.push(disp.into());

        // Association
        if let Some(assoc) =
            YuWuAssociationFunctional::new(&self.params, &self.parameters, self.association)
        {
            contributions.push(assoc.into());
        }

        contributions.into_iter()
    }

    fn bond_lengths<N: DualNum<Primitive = f64> + Copy>(&self, _: N) -> UnGraph<(), N> {
        self.params.bonds.map(|_, _| (), |_, &l| N::from(l))
    }
}

impl Molarweight for FcSaftFunctional {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

impl FluidParameters for FcSaftFunctional {
    fn epsilon_k_ff(&self) -> DVector<f64> {
        self.params.epsilon_k.clone()
    }

    fn sigma_ff(&self) -> DVector<f64> {
        self.params.sigma.clone()
    }
}

/// Individual contributions for the FC-SAFT Helmholtz energy functional.
#[derive(FunctionalContribution)]
pub enum FcSaftFunctionalContribution<'a> {
    Fmt(FMTContribution<'a, FcSaftPars>),
    FusedChain(FusedChainFunctional<'a>),
    Dispersion(DispersionFunctional<'a>),
    Association(YuWuAssociationFunctional<'a, FcSaftPars>),
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::fcsaft::{FcSaft, FcSaftAssociationRecord, FcSaftBondRecord, FcSaftRecord};
    use approx::assert_relative_eq;
    use feos_core::parameter::{
        AssociationRecord, BinarySegmentRecord, ChemicalRecord, SegmentRecord,
    };
    use feos_core::{FeosResult, ReferenceSystem, State};
    use nalgebra::dvector;
    use quantity::{KELVIN, METER, MOL};

    fn binary_mixture() -> FeosResult<FcSaftParameters> {
        let association = vec![AssociationRecord::new(
            Some(FcSaftAssociationRecord::new(0.03, 2500.0)),
            1.0,
            1.0,
            0.0,
        )];
        let segments = [
            SegmentRecord::new("A".into(), 15.0, FcSaftRecord::new(3.6, 180.0, None)),
            SegmentRecord::new("B".into(), 14.0, FcSaftRecord::new(3.2, 220.0, None)),
            SegmentRecord::with_association(
                "C".into(),
                17.0,
                FcSaftRecord::new(2.9, 300.0, None),
                association,
            ),
        ];
        let bond = |id1: &str, id2: &str, l| {
            BinarySegmentRecord::new(id1.into(), id2.into(), Some(FcSaftBondRecord::new(l)))
        };
        let bonds = [
            bond("A", "B", 2.2),
            bond("B", "B", 2.0),
            bond("B", "C", 2.4),
        ];
        let chemical_records = vec![
            ChemicalRecord::new(
                Default::default(),
                vec!["A".into(), "B".into(), "C".into()],
                None,
            ),
            ChemicalRecord::new(
                Default::default(),
                vec!["A".into(), "B".into(), "B".into(), "A".into()],
                None,
            ),
        ];
        let binary = [BinarySegmentRecord::new(
            "A".into(),
            "C".into(),
            Some(crate::fcsaft::FcSaftBinaryRecord::new(0.05)),
        )];
        FcSaftParameters::from_segments_with_bonds(
            chemical_records,
            &segments,
            Some(&binary),
            &bonds,
        )
    }

    #[test]
    fn test_bulk_implementation() -> FeosResult<()> {
        let eos = FcSaft::new(binary_mixture()?);
        let func = FcSaftFunctional::new(binary_mixture()?);
        let t = 300.0 * KELVIN;
        let v = METER.powi::<3>();
        let n = dvector![300.0, 200.0] * MOL;
        let p_eos = State::new_nvt(&&eos, t, v, &n)?.pressure_contributions();
        let p_func = State::new_nvt(&&func, t, v, &n)?.pressure_contributions();
        // the functional additionally reports the (vanishing) ideal chain contribution
        assert_eq!(p_eos.len() + 1, p_func.len());
        for ((name_eos, p_eos), (name_func, p_func)) in p_eos.iter().zip(&p_func) {
            println!("{name_eos:20} {p_eos:13.8} | {name_func:30} {p_func:13.8}");
            assert_relative_eq!(p_eos, p_func, max_relative = 1e-10);
        }
        Ok(())
    }

    #[test]
    fn test_bulk_implementation_monomers() -> FeosResult<()> {
        // mixtures that contain monomers, including a mixture of only monomers
        let segment_records = [
            SegmentRecord::new("A".into(), 16.0, FcSaftRecord::new(3.7, 150.0, None)),
            SegmentRecord::new("B".into(), 40.0, FcSaftRecord::new(3.4, 120.0, None)),
        ];
        let bond_records = [BinarySegmentRecord::new(
            "B".into(),
            "B".into(),
            Some(FcSaftBondRecord::new(2.0)),
        )];
        for segments in [1, 2] {
            let chemical_records = vec![
                ChemicalRecord::new(Default::default(), vec!["A".into()], None),
                ChemicalRecord::new(Default::default(), vec!["B".into(); segments], None),
            ];
            let params = || {
                FcSaftParameters::from_segments_with_bonds(
                    chemical_records.clone(),
                    &segment_records,
                    None,
                    &bond_records,
                )
            };
            let eos = FcSaft::new(params()?);
            let func = FcSaftFunctional::new(params()?);
            let t = 300.0 * KELVIN;
            let v = METER.powi::<3>();
            let n = dvector![300.0, 200.0] * MOL;
            let p_eos = State::new_nvt(&&eos, t, v, &n)?.pressure_contributions();
            let p_func = State::new_nvt(&&func, t, v, &n)?.pressure_contributions();
            for ((name, p_eos), (_, p_func)) in p_eos.iter().zip(&p_func) {
                println!("{segments} {name:20} {p_eos:13.8} {p_func:13.8}");
                assert!(p_func.into_reduced().is_finite());
                assert_relative_eq!(p_eos, p_func, max_relative = 1e-10);
            }
        }
        Ok(())
    }
}
