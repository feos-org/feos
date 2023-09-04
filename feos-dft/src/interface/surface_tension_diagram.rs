use super::PlanarInterface;
use crate::functional::{HelmholtzEnergyFunctional, DFT};
use crate::solver::DFTSolver;
use feos_core::si::{Length, Moles, SurfaceTension, Temperature};
use feos_core::{PhaseEquilibrium, StateVec};
use ndarray::{Array1, Array2};

const DEFAULT_GRID_POINTS: usize = 2048;

/// Container structure for the efficient calculation of surface tension diagrams.
pub struct SurfaceTensionDiagram<F: HelmholtzEnergyFunctional> {
    pub profiles: Vec<PlanarInterface<F>>,
}

#[allow(clippy::ptr_arg)]
impl<F: HelmholtzEnergyFunctional> SurfaceTensionDiagram<F> {
    pub fn new(
        dia: &Vec<PhaseEquilibrium<DFT<F>, 2>>,
        init_densities: Option<bool>,
        n_grid: Option<usize>,
        l_grid: Option<Length>,
        critical_temperature: Option<Temperature>,
        fix_equimolar_surface: Option<bool>,
        solver: Option<&DFTSolver>,
    ) -> Self {
        let n_grid = n_grid.unwrap_or(DEFAULT_GRID_POINTS);
        let mut profiles: Vec<PlanarInterface<F>> = Vec::with_capacity(dia.len());
        for vle in dia.iter() {
            // check for a critical point
            let profile = if PhaseEquilibrium::is_trivial_solution(vle.vapor(), vle.liquid()) {
                Ok(PlanarInterface::from_tanh(
                    vle,
                    10,
                    Length::from_reduced(100.0),
                    Temperature::from_reduced(500.0),
                    fix_equimolar_surface.unwrap_or(false),
                ))
            } else {
                // initialize with pDGT for single segments and tanh for mixtures and segment DFT
                if vle.vapor().eos.component_index().len() == 1 {
                    PlanarInterface::from_pdgt(vle, n_grid, false)
                } else {
                    Ok(PlanarInterface::from_tanh(
                        vle,
                        n_grid,
                        l_grid.unwrap_or(Length::from_reduced(100.0)),
                        critical_temperature.unwrap_or(Temperature::from_reduced(500.0)),
                        fix_equimolar_surface.unwrap_or(false),
                    ))
                }
                .map(|mut profile| {
                    if let Some(init) = profiles.last() {
                        if init.profile.density.shape() == profile.profile.density.shape() {
                            if let Some(scale) = init_densities {
                                profile.set_density_inplace(&init.profile.density, scale)
                            }
                        }
                    }
                    profile
                })
            }
            .and_then(|profile| profile.solve(solver));
            if let Ok(profile) = profile {
                profiles.push(profile);
            }
        }
        Self { profiles }
    }

    pub fn vapor(&self) -> StateVec<'_, DFT<F>> {
        self.profiles.iter().map(|p| p.vle.vapor()).collect()
    }

    pub fn liquid(&self) -> StateVec<'_, DFT<F>> {
        self.profiles.iter().map(|p| p.vle.liquid()).collect()
    }

    pub fn surface_tension(&mut self) -> SurfaceTension<Array1<f64>> {
        SurfaceTension::from_shape_fn(self.profiles.len(), |i| {
            self.profiles[i].surface_tension.unwrap()
        })
    }

    pub fn relative_adsorption(&self) -> Vec<Moles<Array2<f64>>> {
        self.profiles
            .iter()
            .map(|planar_interf| planar_interf.relative_adsorption())
            .collect()
    }

    pub fn interfacial_enrichment(&self) -> Vec<Array1<f64>> {
        self.profiles
            .iter()
            .map(|planar_interf| planar_interf.interfacial_enrichment())
            .collect()
    }

    pub fn interfacial_thickness(&self) -> Length<Array1<f64>> {
        self.profiles
            .iter()
            .map(|planar_interf| planar_interf.interfacial_thickness().unwrap())
            .collect()
    }
}
