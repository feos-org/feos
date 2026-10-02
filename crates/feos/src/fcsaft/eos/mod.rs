use super::FcSaftOptions;
use super::parameters::FcSaftPars;
use super::record::FcSaftParameters;
use super::reference::ReferenceFluid;
use crate::association::Association;
use crate::hard_sphere::HardSphereProperties;
use feos_core::{Molarweight, ResidualDyn, StateHD, Subset};
use nalgebra::DVector;
use num_dual::DualNum;
use quantity::MolarWeight;
use std::f64::consts::FRAC_PI_6;

pub(crate) mod dispersion;
use dispersion::Dispersion;
pub use dispersion::DispersionConstants;

/// Heterosegmented FC-SAFT equation of state.
pub struct FcSaft {
    pub parameters: FcSaftParameters,
    pub params: FcSaftPars,
    options: FcSaftOptions,
    dispersion: Dispersion,
    association: Option<Association>,
    model_constants: Option<DispersionConstants>,
}

impl FcSaft {
    pub fn new(parameters: FcSaftParameters) -> Self {
        Self::with_options(parameters, FcSaftOptions::default(), None)
    }

    pub fn with_options(
        parameters: FcSaftParameters,
        options: FcSaftOptions,
        model_constants: Option<DispersionConstants>,
    ) -> Self {
        let params = FcSaftPars::new(&parameters);
        let dispersion = Dispersion::new(
            &params.component_index,
            &params.sigma_ij,
            &params.epsilon_k_ij,
            model_constants,
        );
        let association = (!parameters.association.is_empty())
            .then(|| Association::new(options.max_iter_cross_assoc, options.tol_cross_assoc));
        Self {
            parameters,
            params,
            options,
            dispersion,
            association,
            model_constants,
        }
    }
}

impl ResidualDyn for FcSaft {
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
        let mut v = Vec::with_capacity(4);
        let p = &self.params;
        let d = p.hs_diameter(state.temperature);
        let av = p.fused_sphere_coefficients(&d);

        let density = state.partial_density.sum();
        let reference = ReferenceFluid::new(&p.component_index, &p.bonds, &d, &av);
        let ([hs, chain], c1) = reference.helmholtz_energy_density(density, &state.molefracs);
        v.push(("Hard Sphere", hs));
        v.push(("Fused-sphere chain", chain));
        v.push((
            "Dispersion",
            self.dispersion.helmholtz_energy_density(state, &d, &av, c1),
        ));
        if let Some(association) = self.association.as_ref() {
            v.push((
                "Association",
                association.helmholtz_energy_density(
                    &self.params,
                    &self.parameters.association,
                    state,
                    &d,
                ),
            ))
        }
        v
    }
}

impl Subset for FcSaft {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            self.parameters.subset(component_list),
            self.options,
            self.model_constants,
        )
    }
}

impl Molarweight for FcSaft {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::fcsaft::new_pure_homosegmented;
    use crate::fcsaft::{FcSaftBondRecord, FcSaftRecord};
    use approx::assert_relative_eq;
    use feos_core::parameter::{BinarySegmentRecord, ChemicalRecord, Identifier, SegmentRecord};
    use feos_core::{FeosResult, ReferenceSystem, Residual, State};
    use nalgebra::dvector;
    use num_dual::Dual64;
    use quantity::{KELVIN, METER, MOL};

    #[test]
    fn test_dispersion() -> FeosResult<()> {
        let params = new_pure_homosegmented(Identifier::default(), 2, 3.0, 150.0, 1.5, 20.0, None)?;
        let mut model_constants = [[0.0; 7]; 4];
        model_constants[0][0] = 1e-8;
        let eos = FcSaft::with_options(params, Default::default(), Some(model_constants));
        let state = StateHD::new(100.0f64, 1000.0, &dvector![1.0]);
        let d = eos.params.hs_diameter(state.temperature);
        let av = eos.params.fused_sphere_coefficients(&d);
        let reference =
            ReferenceFluid::new(&eos.params.component_index, &eos.params.bonds, &d, &av);
        let (_, c1) = reference.helmholtz_energy_density(1e-3, &DVector::from_element(1, 1.0));
        let a = eos.dispersion.helmholtz_energy_density(&state, &d, &av, c1);
        assert!(a.is_finite());
        let state = State::new_nvt(&&eos, 100.0 * KELVIN, 10.0 * METER.powi::<3>(), MOL)?;
        assert_eq!(state.total_molar_weight(), 20.0 * quantity::GRAM / MOL);
        Ok(())
    }

    /// Monomer (component 1) and a fused trimer (component 2).
    fn monomer_trimer() -> FeosResult<FcSaft> {
        let segment_records = [
            SegmentRecord::new("A".into(), 16.0, FcSaftRecord::new(3.7, 150.0, None)),
            SegmentRecord::new("B".into(), 15.0, FcSaftRecord::new(3.5, 180.0, None)),
        ];
        let chemical_records = vec![
            ChemicalRecord::new(Default::default(), vec!["A".into()], None),
            ChemicalRecord::new(Default::default(), vec!["B".into(); 3], None),
        ];
        let bond_records = [BinarySegmentRecord::new(
            "B".into(),
            "B".into(),
            Some(FcSaftBondRecord::new(2.5)),
        )];
        let params = FcSaftParameters::from_segments_with_bonds(
            chemical_records,
            &segment_records,
            None,
            &bond_records,
        )?;
        Ok(FcSaft::new(params))
    }

    #[test]
    fn test_infinite_dilution() -> FeosResult<()> {
        // the derivative of the Helmholtz energy w.r.t. the density of the trimer
        // has to be continuous for a vanishing amount of trimer
        let eos = monomer_trimer()?;
        let mu = |rho2: f64| {
            let state = StateHD::new_density(
                Dual64::from(300.0),
                &dvector![Dual64::from(0.01), Dual64::from(rho2).derivative()],
            );
            eos.reduced_helmholtz_energy_density_contributions(&state)
                .into_iter()
                .map(|(name, a)| (name, a.eps))
                .collect::<Vec<_>>()
        };
        for ((name, mu_0), (_, mu_dilute)) in mu(0.0).into_iter().zip(mu(1e-12)) {
            println!("{name:20} {mu_0:.12} {mu_dilute:.12}");
            assert_relative_eq!(mu_0, mu_dilute, max_relative = 1e-8);
        }
        Ok(())
    }

    #[test]
    fn test_second_virial_coefficient() -> FeosResult<()> {
        // C1 has to be well-defined at zero density
        let eos = monomer_trimer()?;
        let b = (&eos).second_virial_coefficient(300.0 * KELVIN, dvector![0.4, 0.6])?;
        assert!(b.into_reduced().is_finite());
        Ok(())
    }
}
