//! Adsorption profiles and isotherms.
use super::functional::{HelmholtzEnergyFunctional, DFT};
use super::solver::DFTSolver;
use feos_core::si::{Energy, MolarEnergy, Moles, Pressure, Temperature};
use feos_core::{
    Components, Contributions, DensityInitialization, EosError, EosResult, Residual, SolverOptions,
    State, StateBuilder,
};
use ndarray::{Array1, Array2, Dimension, Ix1, Ix3, RemoveAxis};
use std::iter;
use std::sync::Arc;

mod external_potential;
#[cfg(feature = "rayon")]
mod fea_potential;
mod pore;
mod pore2d;
pub use external_potential::{ExternalPotential, FluidParameters};
pub use pore::{Pore1D, PoreProfile, PoreProfile1D, PoreSpecification};
pub use pore2d::{Pore2D, PoreProfile2D};

#[cfg(feature = "rayon")]
mod pore3d;
#[cfg(feature = "rayon")]
pub use pore3d::{Pore3D, PoreProfile3D};

const MAX_ITER_ADSORPTION_EQUILIBRIUM: usize = 50;
const TOL_ADSORPTION_EQUILIBRIUM: f64 = 1e-8;

/// Container structure for the calculation of adsorption isotherms.
pub struct Adsorption<D: Dimension, F> {
    components: usize,
    pub profiles: Vec<EosResult<PoreProfile<D, F>>>,
}

/// Container structure for adsorption isotherms in 1D pores.
pub type Adsorption1D<F> = Adsorption<Ix1, F>;
/// Container structure for adsorption isotherms in 3D pores.
pub type Adsorption3D<F> = Adsorption<Ix3, F>;

impl<D: Dimension + RemoveAxis + 'static, F: HelmholtzEnergyFunctional + FluidParameters>
    Adsorption<D, F>
where
    D::Larger: Dimension<Smaller = D>,
    D::Smaller: Dimension<Larger = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    fn new(functional: &Arc<DFT<F>>, profiles: Vec<EosResult<PoreProfile<D, F>>>) -> Self {
        Self {
            components: functional.components(),
            profiles,
        }
    }

    /// Calculate an adsorption isotherm (starting at low pressure)
    pub fn adsorption_isotherm<S: PoreSpecification<D>>(
        functional: &Arc<DFT<F>>,
        temperature: Temperature,
        pressure: &Pressure<Array1<f64>>,
        pore: &S,
        molefracs: Option<&Array1<f64>>,
        solver: Option<&DFTSolver>,
    ) -> EosResult<Adsorption<D, F>> {
        Self::isotherm(
            functional,
            temperature,
            pressure,
            pore,
            molefracs,
            DensityInitialization::Vapor,
            solver,
        )
    }

    /// Calculate an desorption isotherm (starting at high pressure)
    pub fn desorption_isotherm<S: PoreSpecification<D>>(
        functional: &Arc<DFT<F>>,
        temperature: Temperature,
        pressure: &Pressure<Array1<f64>>,
        pore: &S,
        molefracs: Option<&Array1<f64>>,
        solver: Option<&DFTSolver>,
    ) -> EosResult<Adsorption<D, F>> {
        let pressure = pressure.into_iter().rev().collect();
        let isotherm = Self::isotherm(
            functional,
            temperature,
            &pressure,
            pore,
            molefracs,
            DensityInitialization::Liquid,
            solver,
        )?;
        Ok(Adsorption::new(
            functional,
            isotherm.profiles.into_iter().rev().collect(),
        ))
    }

    /// Calculate an equilibrium isotherm
    pub fn equilibrium_isotherm<S: PoreSpecification<D>>(
        functional: &Arc<DFT<F>>,
        temperature: Temperature,
        pressure: &Pressure<Array1<f64>>,
        pore: &S,
        molefracs: Option<&Array1<f64>>,
        solver: Option<&DFTSolver>,
    ) -> EosResult<Adsorption<D, F>> {
        let (p_min, p_max) = (pressure.get(0), pressure.get(pressure.len() - 1));
        let equilibrium = Self::phase_equilibrium(
            functional,
            temperature,
            p_min,
            p_max,
            pore,
            molefracs,
            solver,
            SolverOptions::default(),
        );
        if let Ok(equilibrium) = equilibrium {
            let p_eq = equilibrium.pressure().get(0);
            let p_ads = pressure
                .into_iter()
                .filter(|&p| p <= p_eq)
                .chain(iter::once(p_eq))
                .collect();
            let p_des = iter::once(p_eq)
                .chain(pressure.into_iter().filter(|&p| p > p_eq))
                .collect();
            let adsorption = Self::adsorption_isotherm(
                functional,
                temperature,
                &p_ads,
                pore,
                molefracs,
                solver,
            )?
            .profiles;
            let desorption = Self::desorption_isotherm(
                functional,
                temperature,
                &p_des,
                pore,
                molefracs,
                solver,
            )?
            .profiles;
            Ok(Adsorption {
                profiles: adsorption
                    .into_iter()
                    .chain(desorption.into_iter())
                    .collect(),
                components: functional.components(),
            })
        } else {
            let adsorption = Self::adsorption_isotherm(
                functional,
                temperature,
                pressure,
                pore,
                molefracs,
                solver,
            )?;
            let desorption = Self::desorption_isotherm(
                functional,
                temperature,
                pressure,
                pore,
                molefracs,
                solver,
            )?;
            let omega_a = adsorption.grand_potential();
            let omega_d = desorption.grand_potential();
            let is_ads = Array1::from_shape_fn(adsorption.profiles.len(), |i| {
                omega_d.get(i).is_nan() || omega_a.get(i) < omega_d.get(i)
            });
            let profiles = is_ads
                .into_iter()
                .zip(adsorption.profiles.into_iter())
                .zip(desorption.profiles.into_iter())
                .map(|((is_ads, a), d)| if is_ads { a } else { d })
                .collect();
            Ok(Adsorption::new(functional, profiles))
        }
    }

    fn isotherm<S: PoreSpecification<D>>(
        functional: &Arc<DFT<F>>,
        temperature: Temperature,
        pressure: &Pressure<Array1<f64>>,
        pore: &S,
        molefracs: Option<&Array1<f64>>,
        density_initialization: DensityInitialization,
        solver: Option<&DFTSolver>,
    ) -> EosResult<Adsorption<D, F>> {
        let moles = functional
            .validate_moles(molefracs.map(|x| Moles::from_reduced(x.clone())).as_ref())?;
        let mut profiles: Vec<EosResult<PoreProfile<D, F>>> = Vec::with_capacity(pressure.len());

        // On the first iteration, initialize the density profile according to the direction
        // and calculate the external potential once.
        let mut bulk = State::new_npt(
            functional,
            temperature,
            pressure.get(0),
            &moles,
            density_initialization,
        )?;
        if functional.components() > 1 && !bulk.is_stable(SolverOptions::default())? {
            let vle = bulk.tp_flash(None, SolverOptions::default(), None)?;
            bulk = match density_initialization {
                DensityInitialization::Liquid => vle.liquid().clone(),
                DensityInitialization::Vapor => vle.vapor().clone(),
                _ => unreachable!(),
            };
        }
        let profile = pore.initialize(&bulk, None, None)?.solve(solver)?.profile;
        let external_potential = Some(&profile.external_potential);
        let mut old_density = Some(&profile.density);

        for i in 0..pressure.len() {
            let mut bulk = StateBuilder::new(functional)
                .temperature(temperature)
                .pressure(pressure.get(i))
                .moles(&moles)
                .build()?;
            if functional.components() > 1 && !bulk.is_stable(SolverOptions::default())? {
                bulk = bulk
                    .tp_flash(None, SolverOptions::default(), None)?
                    .vapor()
                    .clone();
            }

            let p = pore.initialize(&bulk, old_density, external_potential)?;
            let p2 = pore.initialize(&bulk, None, external_potential)?;
            profiles.push(p.solve(solver).or_else(|_| p2.solve(solver)));

            old_density = if let Some(Ok(l)) = profiles.last() {
                Some(&l.profile.density)
            } else {
                None
            };
        }

        Ok(Adsorption::new(functional, profiles))
    }

    /// Calculate the phase transition from an empty to a filled pore.
    pub fn phase_equilibrium<S: PoreSpecification<D>>(
        functional: &Arc<DFT<F>>,
        temperature: Temperature,
        p_min: Pressure,
        p_max: Pressure,
        pore: &S,
        molefracs: Option<&Array1<f64>>,
        solver: Option<&DFTSolver>,
        options: SolverOptions,
    ) -> EosResult<Adsorption<D, F>> {
        let moles = functional
            .validate_moles(molefracs.map(|x| Moles::from_reduced(x.clone())).as_ref())?;

        // calculate density profiles for the minimum and maximum pressure
        let vapor_bulk = StateBuilder::new(functional)
            .temperature(temperature)
            .pressure(p_min)
            .moles(&moles)
            .vapor()
            .build()?;
        let bulk_init = StateBuilder::new(functional)
            .temperature(temperature)
            .pressure(p_max)
            .moles(&moles)
            .liquid()
            .build()?;
        let liquid_bulk = StateBuilder::new(functional)
            .temperature(temperature)
            .pressure(p_max)
            .moles(&moles)
            .vapor()
            .build()?;

        let mut vapor = pore.initialize(&vapor_bulk, None, None)?.solve(solver)?;
        let mut liquid = pore.initialize(&bulk_init, None, None)?.solve(solver)?;

        // calculate initial value for bulk density
        let n_dp_drho_v = (vapor.profile.moles() * vapor_bulk.dp_drho(Contributions::Total)).sum();
        let n_dp_drho_l =
            (liquid.profile.moles() * liquid_bulk.dp_drho(Contributions::Total)).sum();
        let mut rho = (vapor.grand_potential.unwrap() + n_dp_drho_v
            - (liquid.grand_potential.unwrap() + n_dp_drho_l))
            / (n_dp_drho_v / vapor_bulk.density - n_dp_drho_l / liquid_bulk.density);

        // update filled pore with limited step size
        let mut bulk = StateBuilder::new(functional)
            .temperature(temperature)
            .pressure(p_max)
            .moles(&moles)
            .vapor()
            .build()?;
        let rho0 = liquid_bulk.density;
        let steps = (10.0 * (rho - rho0) / rho0).into_value().abs().ceil() as usize;
        let delta_rho = (rho - rho0) / steps as f64;
        for i in 1..=steps {
            let rho_i = rho0 + i as f64 * delta_rho;
            bulk = State::new_nvt(functional, temperature, moles.sum() / rho_i, &moles)?;
            liquid = liquid.update_bulk(&bulk).solve(solver)?;
        }

        for _ in 0..options.max_iter.unwrap_or(MAX_ITER_ADSORPTION_EQUILIBRIUM) {
            // update empty pore
            vapor = vapor.update_bulk(&bulk).solve(solver)?;

            // update filled pore
            liquid = liquid.update_bulk(&bulk).solve(solver)?;

            // calculate moles
            let n_dp_drho = ((liquid.profile.moles() - vapor.profile.moles())
                * bulk.dp_drho(Contributions::Total))
            .sum();

            // Newton step
            let delta_rho = (liquid.grand_potential.unwrap() - vapor.grand_potential.unwrap())
                / n_dp_drho
                * bulk.density;
            if delta_rho.to_reduced().abs() < options.tol.unwrap_or(TOL_ADSORPTION_EQUILIBRIUM) {
                return Ok(Adsorption::new(functional, vec![Ok(vapor), Ok(liquid)]));
            }
            rho += delta_rho;

            // update bulk phase
            bulk = State::new_nvt(functional, temperature, moles.sum() / rho, &moles)?;
        }
        Err(EosError::NotConverged(
            "Adsorption::phase_equilibrium".into(),
        ))
    }

    pub fn pressure(&self) -> Pressure<Array1<f64>> {
        Pressure::from_shape_fn(self.profiles.len(), |i| match &self.profiles[i] {
            Ok(p) => {
                if p.profile.bulk.eos.components() > 1
                    && !p.profile.bulk.is_stable(SolverOptions::default()).unwrap()
                {
                    p.profile
                        .bulk
                        .tp_flash(None, SolverOptions::default(), None)
                        .unwrap()
                        .vapor()
                        .pressure(Contributions::Total)
                } else {
                    p.profile.bulk.pressure(Contributions::Total)
                }
            }
            Err(_) => Pressure::from_reduced(f64::NAN),
        })
    }

    pub fn adsorption(&self) -> Moles<Array2<f64>> {
        Moles::from_shape_fn(
            (self.components, self.profiles.len()),
            |(j, i)| match &self.profiles[i] {
                Ok(p) => p.profile.moles().get(j),
                Err(_) => Moles::from_reduced(f64::NAN),
            },
        )
    }

    pub fn total_adsorption(&self) -> Moles<Array1<f64>> {
        Moles::from_shape_fn(self.profiles.len(), |i| match &self.profiles[i] {
            Ok(p) => p.profile.total_moles(),
            Err(_) => Moles::from_reduced(f64::NAN),
        })
    }

    pub fn grand_potential(&self) -> Energy<Array1<f64>> {
        Energy::from_shape_fn(self.profiles.len(), |i| match &self.profiles[i] {
            Ok(p) => p.grand_potential.unwrap(),
            Err(_) => Energy::from_reduced(f64::NAN),
        })
    }

    pub fn partial_molar_enthalpy_of_adsorption(&self) -> MolarEnergy<Array2<f64>> {
        let h_ads: Vec<_> = self
            .profiles
            .iter()
            .map(|p| {
                match p
                    .as_ref()
                    .ok()
                    .and_then(|p| p.partial_molar_enthalpy_of_adsorption().ok())
                {
                    Some(p) => p,
                    None => MolarEnergy::from_reduced(f64::NAN * Array1::ones(self.components)),
                }
            })
            .collect();
        MolarEnergy::from_shape_fn((self.components, self.profiles.len()), |(j, i)| {
            h_ads[i].get(j)
        })
    }

    pub fn enthalpy_of_adsorption(&self) -> MolarEnergy<Array1<f64>> {
        MolarEnergy::from_shape_fn(self.profiles.len(), |i| {
            match self.profiles[i]
                .as_ref()
                .ok()
                .and_then(|p| p.enthalpy_of_adsorption().ok())
            {
                Some(p) => p,
                None => MolarEnergy::from_reduced(f64::NAN),
            }
        })
    }
}
