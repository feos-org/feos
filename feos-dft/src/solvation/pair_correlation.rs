//! Functionalities for the calculation of pair correlation functions.
use crate::convolver::ConvolverFFT;
use crate::functional::{HelmholtzEnergyFunctional, DFT};
use crate::profile::MAX_POTENTIAL;
use crate::solver::DFTSolver;
use crate::{Axis, DFTProfile, Grid};
use feos_core::si::{Energy, Length};
use feos_core::{Contributions, EosResult, State};
use ndarray::prelude::*;

/// The underlying pair potential, that the Helmholtz energy functional
/// models.
pub trait PairPotential {
    /// Return the pair potential of particle i with all other particles.
    fn pair_potential(&self, i: usize, r: &Array1<f64>, temperature: f64) -> Array2<f64>;
}

/// Density profile and properties of a test particle system.
pub struct PairCorrelation<F> {
    pub profile: DFTProfile<Ix1, F>,
    pub pair_correlation_function: Option<Array2<f64>>,
    pub self_solvation_free_energy: Option<Energy>,
    pub structure_factor: Option<f64>,
}

impl<F> Clone for PairCorrelation<F> {
    fn clone(&self) -> Self {
        Self {
            profile: self.profile.clone(),
            pair_correlation_function: self.pair_correlation_function.clone(),
            self_solvation_free_energy: self.self_solvation_free_energy,
            structure_factor: self.structure_factor,
        }
    }
}

impl<F: HelmholtzEnergyFunctional + PairPotential> PairCorrelation<F> {
    pub fn new(bulk: &State<DFT<F>>, test_particle: usize, n_grid: usize, width: Length) -> Self {
        let dft = &bulk.eos;

        // generate grid
        let axis = Axis::new_spherical(n_grid, width);

        // calculate external potential
        let t = bulk.temperature.to_reduced();
        let mut external_potential = dft.pair_potential(test_particle, &axis.grid, t) / t;
        external_potential.map_inplace(|x| {
            if *x > MAX_POTENTIAL {
                *x = MAX_POTENTIAL
            }
        });

        // initialize convolver
        let grid = Grid::Spherical(axis);
        let weight_functions = dft.weight_functions(t);
        let convolver = ConvolverFFT::plan(&grid, &weight_functions, Some(1));

        Self {
            profile: DFTProfile::new(grid, convolver, bulk, Some(external_potential), None),
            pair_correlation_function: None,
            self_solvation_free_energy: None,
            structure_factor: None,
        }
    }

    pub fn solve_inplace(&mut self, solver: Option<&DFTSolver>, debug: bool) -> EosResult<()> {
        // Solve the profile
        self.profile.solve(solver, debug)?;

        // calculate pair correlation function
        self.pair_correlation_function = Some(Array::from_shape_fn(
            self.profile.density.raw_dim(),
            |(i, j)| {
                (self.profile.density.get((i, j)) / self.profile.bulk.partial_density.get(i))
                    .into_value()
            },
        ));

        // calculate self solvation free energy
        self.self_solvation_free_energy = Some(self.profile.integrate(
            &(self.profile.grand_potential_density()?
                + self.profile.bulk.pressure(Contributions::Total)),
        ));

        // calculate structure factor
        self.structure_factor = Some(
            (self.profile.total_moles() - self.profile.bulk.density * self.profile.volume())
                .to_reduced()
                + 1.0,
        );

        Ok(())
    }

    pub fn solve(mut self, solver: Option<&DFTSolver>) -> EosResult<Self> {
        self.solve_inplace(solver, false)?;
        Ok(self)
    }
}
