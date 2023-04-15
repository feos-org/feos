//! Adsorption profiles and isotherms.
use super::functional::{HelmholtzEnergyFunctional, DFT};
use super::solver::DFTSolver;
use feos_core::{
    Contributions, DensityInitialization, EosError, EosResult, EosUnit, EquationOfState,
    SolverOptions, State, StateBuilder,
};
use ndarray::{Array1, Dimension, Ix1, Ix3, RemoveAxis};
use quantity::si::{SIArray1, SIArray2, SINumber, SIUnit};
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
    dimension: i32,
    pub profiles: Vec<EosResult<PoreProfile<D, F>>>,
}

/// Container structure for adsorption isotherms in 1D pores.
pub type Adsorption1D<F> = Adsorption<Ix1, F>;
/// Container structure for adsorption isotherms in 3D pores.
pub type Adsorption3D<F> = Adsorption<Ix3, F>;

impl<D: Dimension + RemoveAxis + 'static, F: HelmholtzEnergyFunctional + FluidParameters>
    Adsorption<D, F>
where
    SINumber: std::fmt::Display,
    D::Larger: Dimension<Smaller = D>,
    D::Smaller: Dimension<Larger = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    fn new<S: PoreSpecification<D>>(
        functional: &Arc<DFT<F>>,
        pore: &S,
        profiles: Vec<EosResult<PoreProfile<D, F>>>,
    ) -> Self {
        Self {
            components: functional.components(),
            dimension: pore.dimension(),
            profiles,
        }
    }

    /// Calculate an adsorption isotherm (starting at low pressure)
    pub fn adsorption_isotherm<S: PoreSpecification<D>>(
        functional: &Arc<DFT<F>>,
        temperature: SINumber,
        pressure: &SIArray1,
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
        temperature: SINumber,
        pressure: &SIArray1,
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
            pore,
            isotherm.profiles.into_iter().rev().collect(),
        ))
    }

    /// Calculate an equilibrium isotherm
    pub fn equilibrium_isotherm<S: PoreSpecification<D>>(
        functional: &Arc<DFT<F>>,
        temperature: SINumber,
        pressure: &SIArray1,
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
                dimension: pore.dimension(),
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
            Ok(Adsorption::new(functional, pore, profiles))
        }
    }

    fn isotherm<S: PoreSpecification<D>>(
        functional: &Arc<DFT<F>>,
        temperature: SINumber,
        pressure: &SIArray1,
        pore: &S,
        molefracs: Option<&Array1<f64>>,
        density_initialization: DensityInitialization,
        solver: Option<&DFTSolver>,
    ) -> EosResult<Adsorption<D, F>> {
        let moles =
            functional.validate_moles(molefracs.map(|x| x * SIUnit::reference_moles()).as_ref())?;
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

        Ok(Adsorption::new(functional, pore, profiles))
    }

    /// Calculate the phase transition from an empty to a filled pore.
    pub fn phase_equilibrium<S: PoreSpecification<D>>(
        functional: &Arc<DFT<F>>,
        temperature: SINumber,
        p_min: SINumber,
        p_max: SINumber,
        pore: &S,
        molefracs: Option<&Array1<f64>>,
        solver: Option<&DFTSolver>,
        options: SolverOptions,
    ) -> EosResult<Adsorption<D, F>> {
        let moles =
            functional.validate_moles(molefracs.map(|x| x * SIUnit::reference_moles()).as_ref())?;

        // calculate density profiles for the minimum and maximum pressure
        let vapor_bulk = StateBuilder::new(functional)
            .temperature(temperature)
            .pressure(p_min)
            .moles(&moles)
            .vapor()
            .build()?;
        let liquid_bulk = StateBuilder::new(functional)
            .temperature(temperature)
            .pressure(p_max)
            .moles(&moles)
            .liquid()
            .build()?;

        let mut vapor = pore.initialize(&vapor_bulk, None, None)?.solve(None)?;
        let mut liquid = pore.initialize(&liquid_bulk, None, None)?.solve(solver)?;

        // calculate initial value for the molar gibbs energy
        let nv = vapor.profile.bulk.density
            * (vapor.profile.moles()
                * vapor
                    .profile
                    .bulk
                    .partial_molar_volume(Contributions::Total))
            .sum();
        let nl = liquid.profile.bulk.density
            * (liquid.profile.moles()
                * liquid
                    .profile
                    .bulk
                    .partial_molar_volume(Contributions::Total))
            .sum();
        let f = |s: &PoreProfile<D, F>, n: SINumber| -> EosResult<_> {
            Ok(s.grand_potential.unwrap()
                + s.profile.bulk.molar_gibbs_energy(Contributions::Total) * n)
        };
        let mut g = (f(&liquid, nl)? - f(&vapor, nv)?) / (nl - nv);

        // update filled pore with limited step size
        let mut bulk = StateBuilder::new(functional)
            .temperature(temperature)
            .pressure(p_max)
            .moles(&moles)
            .vapor()
            .build()?;
        let g_liquid = liquid.profile.bulk.molar_gibbs_energy(Contributions::Total);
        let steps = (10.0 * (g - g_liquid)).to_reduced(g_liquid)?.abs().ceil() as usize;
        let delta_g = (g - g_liquid) / steps as f64;
        for i in 1..=steps {
            let g_i = g_liquid + i as f64 * delta_g;
            bulk = bulk.update_gibbs_energy(g_i)?;
            liquid = liquid.update_bulk(&bulk).solve(solver)?;
        }

        for _ in 0..options.max_iter.unwrap_or(MAX_ITER_ADSORPTION_EQUILIBRIUM) {
            // update empty pore
            vapor = vapor.update_bulk(&bulk).solve(None)?;

            // update filled pore
            liquid = liquid.update_bulk(&bulk).solve(solver)?;

            // calculate moles
            let nv = vapor.profile.bulk.density
                * (vapor.profile.moles()
                    * vapor
                        .profile
                        .bulk
                        .partial_molar_volume(Contributions::Total))
                .sum();
            let nl = liquid.profile.bulk.density
                * (liquid.profile.moles()
                    * liquid
                        .profile
                        .bulk
                        .partial_molar_volume(Contributions::Total))
                .sum();

            // check for a trivial solution
            if nl.to_reduced(nv)? - 1.0 < 1e-5 {
                return Err(EosError::TrivialSolution);
            }

            // Newton step
            let delta_g =
                (vapor.grand_potential.unwrap() - liquid.grand_potential.unwrap()) / (nv - nl);
            if delta_g.to_reduced(SIUnit::reference_molar_energy())?.abs()
                < options.tol.unwrap_or(TOL_ADSORPTION_EQUILIBRIUM)
            {
                return Ok(Adsorption::new(
                    functional,
                    pore,
                    vec![Ok(vapor), Ok(liquid)],
                ));
            }
            g += delta_g;

            // update bulk phase
            bulk = bulk.update_gibbs_energy(g)?;
        }
        Err(EosError::NotConverged(
            "Adsorption::phase_equilibrium".into(),
        ))
    }

    pub fn pressure(&self) -> SIArray1 {
        SIArray1::from_shape_fn(self.profiles.len(), |i| match &self.profiles[i] {
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
            Err(_) => f64::NAN * SIUnit::reference_pressure(),
        })
    }

    pub fn molar_gibbs_energy(&self) -> SIArray1 {
        SIArray1::from_shape_fn(self.profiles.len(), |i| match &self.profiles[i] {
            Ok(p) => {
                if p.profile.bulk.eos.components() > 1
                    && !p.profile.bulk.is_stable(SolverOptions::default()).unwrap()
                {
                    p.profile
                        .bulk
                        .tp_flash(None, SolverOptions::default(), None)
                        .unwrap()
                        .vapor()
                        .molar_gibbs_energy(Contributions::Total)
                } else {
                    p.profile.bulk.molar_gibbs_energy(Contributions::Total)
                }
            }
            Err(_) => f64::NAN * SIUnit::reference_molar_energy(),
        })
    }

    pub fn adsorption(&self) -> SIArray2 {
        SIArray2::from_shape_fn((self.components, self.profiles.len()), |(j, i)| match &self
            .profiles[i]
        {
            Ok(p) => p.profile.moles().get(j),
            Err(_) => {
                f64::NAN
                    * SIUnit::reference_density()
                    * SIUnit::reference_length().powi(self.dimension)
            }
        })
    }

    pub fn total_adsorption(&self) -> SIArray1 {
        SIArray1::from_shape_fn(self.profiles.len(), |i| match &self.profiles[i] {
            Ok(p) => p.profile.total_moles(),
            Err(_) => {
                f64::NAN
                    * SIUnit::reference_density()
                    * SIUnit::reference_length().powi(self.dimension)
            }
        })
    }

    pub fn grand_potential(&self) -> SIArray1 {
        SIArray1::from_shape_fn(self.profiles.len(), |i| match &self.profiles[i] {
            Ok(p) => p.grand_potential.unwrap(),
            Err(_) => {
                f64::NAN
                    * SIUnit::reference_pressure()
                    * SIUnit::reference_length().powi(self.dimension)
            }
        })
    }

    pub fn partial_molar_enthalpy_of_adsorption(&self) -> SIArray2 {
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
                    None => {
                        f64::NAN * Array1::ones(self.components) * SIUnit::reference_molar_energy()
                    }
                }
            })
            .collect();
        SIArray2::from_shape_fn((self.components, self.profiles.len()), |(j, i)| {
            h_ads[i].get(j)
        })
    }

    pub fn enthalpy_of_adsorption(&self) -> SIArray1 {
        SIArray1::from_shape_fn(self.profiles.len(), |i| {
            match self.profiles[i]
                .as_ref()
                .ok()
                .and_then(|p| p.enthalpy_of_adsorption().ok())
            {
                Some(p) => p,
                None => f64::NAN * SIUnit::reference_molar_energy(),
            }
        })
    }
}
