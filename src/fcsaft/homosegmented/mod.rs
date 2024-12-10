use super::FcSaftOptions;
use crate::association::Association;
use crate::hard_sphere::{HardSphere, HardSphereProperties};
use feos_core::parameter::Parameter;
use feos_core::parameter::PureRecord;
use feos_core::{Components, Residual};
use feos_core::{EosResult, Molarweight};
use ndarray::Array1;
use quantity::*;
use std::f64::consts::FRAC_PI_6;
use std::sync::Arc;

mod dispersion;
mod fused_chain;
pub(super) mod parameters;
mod polar;
use dispersion::Dispersion;
use fused_chain::HardChain;
use parameters::{FcSaftHomoParameters, FcSaftHomoRecord};
use polar::Dipole;

pub struct FcSaftHomo {
    parameters: Arc<FcSaftHomoParameters>,
    options: FcSaftOptions,
    hard_sphere: HardSphere<FcSaftHomoParameters>,
    hard_chain: HardChain,
    dispersion: Dispersion,
    dipole: Option<Dipole>,
    association: Option<Association<FcSaftHomoParameters>>,
}

impl FcSaftHomo {
    #[expect(clippy::too_many_arguments)]
    pub fn new_pure_homosegmented(
        molarweight: f64,
        s: usize,
        l: f64,
        sigma: f64,
        epsilon_k: f64,
        mu: Option<f64>,
        kappa_ab: Option<f64>,
        epsilon_k_ab: Option<f64>,
        na: Option<f64>,
        nb: Option<f64>,
        nc: Option<f64>,
    ) -> EosResult<Self> {
        let model_record = FcSaftHomoRecord::new(
            s,
            l,
            sigma,
            epsilon_k,
            mu,
            kappa_ab,
            epsilon_k_ab,
            na,
            nb,
            nc,
        );
        let pure_record = PureRecord::new(Default::default(), molarweight, model_record);
        let params_homo = FcSaftHomoParameters::new_pure(pure_record)?;
        Ok(Self::new(Arc::new(params_homo)))
    }

    pub fn new(parameters: Arc<FcSaftHomoParameters>) -> Self {
        Self::with_options(parameters, FcSaftOptions::default())
    }

    pub fn with_options(parameters: Arc<FcSaftHomoParameters>, options: FcSaftOptions) -> Self {
        let hard_sphere = HardSphere::new(&parameters);
        let hard_chain = HardChain::new(&parameters);
        let dispersion = Dispersion::new(&parameters);
        let dipole = (parameters.ndipole > 0).then(|| Dipole::new(&parameters));
        let association = (!parameters.association.is_empty()).then(|| {
            Association::new(
                &parameters,
                &parameters.association,
                options.max_iter_cross_assoc,
                options.tol_cross_assoc,
            )
        });

        Self {
            parameters,
            options,
            hard_sphere,
            hard_chain,
            dispersion,
            dipole,
            association,
        }
    }
}

impl Components for FcSaftHomo {
    fn components(&self) -> usize {
        self.parameters.pure_records.len()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(
            Arc::new(self.parameters.subset(component_list)),
            self.options,
        )
    }
}

impl Residual for FcSaftHomo {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        let [_, m_star] = self.parameters.m_values(&self.parameters.sigma);
        self.options.max_eta * moles.sum()
            / (FRAC_PI_6 * m_star * self.parameters.sigma.mapv(|v| v.powi(3)) * moles).sum()
    }

    fn residual_helmholtz_energy_contributions<D: num_dual::DualNum<f64> + Copy>(
        &self,
        state: &feos_core::StateHD<D>,
    ) -> Vec<(String, D)> {
        let mut v = Vec::with_capacity(7);
        let d = self.parameters.hs_diameter(state.temperature);

        v.push((
            self.hard_sphere.to_string(),
            self.hard_sphere.helmholtz_energy(state),
        ));
        v.push((
            self.hard_chain.to_string(),
            self.hard_chain.helmholtz_energy(state),
        ));
        v.push((
            self.dispersion.to_string(),
            self.dispersion.helmholtz_energy(state),
        ));
        if let Some(dipole) = self.dipole.as_ref() {
            v.push((dipole.to_string(), dipole.helmholtz_energy(state)))
        }
        if let Some(association) = self.association.as_ref() {
            v.push((
                association.to_string(),
                association.helmholtz_energy(state, &d),
            ))
        }
        v
    }
}

impl Molarweight for FcSaftHomo {
    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.parameters.molarweight.clone() * GRAM / MOL
    }
}

#[cfg(test)]
mod test {
    use super::parameters::FcSaftHomoRecord;
    use super::*;
    use crate::fcsaft::{FcSaft, FcSaftParameters, FcSaftRecord};
    use approx::assert_relative_eq;
    use feos_core::parameter::{BinaryRecord, ChemicalRecord, SegmentRecord};
    use feos_core::EosError;
    use feos_core::State;
    use quantity::{METER, MOL};
    use typenum::P3;

    #[test]
    fn test_homo() -> Result<(), EosError> {
        let s = 3;
        let l = 2.5;
        let sigma = 3.5;
        let epsilon_k = 250.0;
        let kappa_ab = 0.03;
        let epsilon_k_ab = 2500.;
        let na = 1.0;
        let nb = 1.0;

        let params_homo = FcSaftHomoParameters::from_model_records(vec![FcSaftHomoRecord::new(
            s,
            l,
            sigma,
            epsilon_k,
            None,
            Some(kappa_ab),
            Some(epsilon_k_ab),
            Some(na),
            Some(nb),
            None,
        )])?;
        let fcsaft_homo = Arc::new(FcSaftHomo::new(Arc::new(params_homo)));

        let mut segments = vec!["S".into(); s - 1];
        segments.push("SA".into());
        let cr = ChemicalRecord::new(Default::default(), segments, None);
        let sr = SegmentRecord::new(
            "S".into(),
            0.0,
            FcSaftRecord::new(sigma, epsilon_k, None, None, None, None, None, None),
        );
        let sr_a = SegmentRecord::new(
            "SA".into(),
            0.0,
            FcSaftRecord::new(
                sigma,
                epsilon_k,
                Some(kappa_ab),
                Some(epsilon_k_ab),
                Some(na),
                Some(nb),
                None,
                None,
            ),
        );
        let br = vec![
            BinaryRecord::new("S".into(), "S".into(), l),
            BinaryRecord::new("S".into(), "SA".into(), l),
        ];
        let params = FcSaftParameters::from_segments(vec![cr], vec![sr, sr_a], br, None)?;
        let fcsaft = Arc::new(FcSaft::new(Arc::new(params)));

        let t = 300.0 * KELVIN;
        let rho = 500.0 * MOL / METER.powi::<P3>();
        let c_homo = State::new_pure(&fcsaft_homo, t, rho)?.pressure_contributions();
        let c = State::new_pure(&fcsaft, t, rho)?.pressure_contributions();
        for i in 0..5 {
            println!("{:20} {:13.8} {:13.8}", c[i].0, c[i].1, c_homo[i].1);
        }
        for i in 0..5 {
            assert_relative_eq!(c[i].1, c_homo[i].1, max_relative = 1e-10);
        }

        Ok(())
    }
}
