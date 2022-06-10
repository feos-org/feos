//! Solvation free energies and pair correlaion functions.
use crate::adsorption::FluidParameters;
use crate::convolver::ConvolverFFT;
use crate::functional::{HelmholtzEnergyFunctional, DFT};
use crate::geometry::{Axis, Grid};
use crate::profile::{DFTProfile, CUTOFF_RADIUS, MAX_POTENTIAL};
use crate::solver::DFTSolver;
use feos_core::{Contributions, EosResult, EosUnit, State};
use ndarray::prelude::*;
use ndarray::Zip;
use quantity::{QuantityArray2, QuantityScalar};

mod pair_correlation;
pub use pair_correlation::{PairCorrelation, PairPotential};

/// Density profile and properties of a solute in a inhomogeneous bulk fluid.
pub struct SolvationProfile<U: EosUnit, F: HelmholtzEnergyFunctional> {
    pub profile: DFTProfile<U, Ix3, F>,
    pub grand_potential: Option<QuantityScalar<U>>,
    pub solvation_free_energy: Option<QuantityScalar<U>>,
}

impl<U: EosUnit, F: HelmholtzEnergyFunctional> Clone for SolvationProfile<U, F> {
    fn clone(&self) -> Self {
        Self {
            profile: self.profile.clone(),
            grand_potential: self.grand_potential,
            solvation_free_energy: self.solvation_free_energy,
        }
    }
}

impl<U: EosUnit, F: HelmholtzEnergyFunctional> SolvationProfile<U, F> {
    pub fn solve_inplace(&mut self, solver: Option<&DFTSolver>, debug: bool) -> EosResult<()> {
        // Solve the profile
        self.profile.solve(solver, debug)?;

        // calculate grand potential density
        let omega = self.profile.grand_potential()?;
        self.grand_potential = Some(omega);

        // calculate solvation free energy
        self.solvation_free_energy = Some(
            (omega + self.profile.bulk.pressure(Contributions::Total) * self.profile.volume())
                / U::reference_moles(),
        );

        Ok(())
    }

    pub fn solve(mut self, solver: Option<&DFTSolver>) -> EosResult<Self> {
        self.solve_inplace(solver, false)?;
        Ok(self)
    }
}

impl<U: EosUnit, F: HelmholtzEnergyFunctional + FluidParameters> SolvationProfile<U, F> {
    pub fn new(
        bulk: &State<U, DFT<F>>,
        n_grid: [usize; 3],
        coordinates: QuantityArray2<U>,
        sigma_ss: Array1<f64>,
        epsilon_ss: Array1<f64>,
        system_size: Option<[QuantityScalar<U>; 3]>,
        cutoff_radius: Option<QuantityScalar<U>>,
        potential_cutoff: Option<f64>,
    ) -> EosResult<Self> {
        let dft: &F = &bulk.eos;

        let system_size = system_size.unwrap_or([40.0 * U::reference_length(); 3]);

        // generate grid
        let x = Axis::new_cartesian(n_grid[0], system_size[0], None)?;
        let y = Axis::new_cartesian(n_grid[1], system_size[1], None)?;
        let z = Axis::new_cartesian(n_grid[2], system_size[2], None)?;

        // move center of geometry of solute to box center
        let mut coordinates = Array2::from_shape_fn(coordinates.raw_dim(), |(i, j)| {
            (coordinates.get((i, j)))
                .to_reduced(U::reference_length())
                .unwrap()
        });

        let center = [
            system_size[0].to_reduced(U::reference_length())? / 2.0,
            system_size[1].to_reduced(U::reference_length())? / 2.0,
            system_size[2].to_reduced(U::reference_length())? / 2.0,
        ];

        let shift: Array2<f64> = Array2::from_shape_fn((3, 1), |(i, _)| {
            center[i] - coordinates.row(i).sum() / coordinates.ncols() as f64
        });

        coordinates = coordinates + shift;

        // temperature
        let t = bulk.temperature.to_reduced(U::reference_temperature())?;

        // calculate external potential
        let external_potential = external_potential_3d(
            dft,
            [&x, &y, &z],
            coordinates,
            sigma_ss,
            epsilon_ss,
            cutoff_radius,
            potential_cutoff,
            t,
        )?;

        // initialize convolver
        let grid = Grid::Cartesian3(x, y, z);
        let weight_functions = dft.weight_functions(t);
        let convolver = ConvolverFFT::plan(&grid, &weight_functions, Some(1));

        Ok(Self {
            profile: DFTProfile::new(grid, convolver, bulk, Some(external_potential), None)?,
            grand_potential: None,
            solvation_free_energy: None,
        })
    }
}

fn external_potential_3d<U: EosUnit, F: FluidParameters>(
    functional: &F,
    axis: [&Axis; 3],
    coordinates: Array2<f64>,
    sigma_ss: Array1<f64>,
    epsilon_ss: Array1<f64>,
    cutoff_radius: Option<QuantityScalar<U>>,
    potential_cutoff: Option<f64>,
    reduced_temperature: f64,
) -> EosResult<Array4<f64>> {
    // allocate external potential
    let m = functional.m();
    let mut external_potential = Array4::zeros((
        m.len(),
        axis[0].grid.len(),
        axis[1].grid.len(),
        axis[2].grid.len(),
    ));

    let cutoff_radius = cutoff_radius
        .unwrap_or(CUTOFF_RADIUS * U::reference_length())
        .to_reduced(U::reference_length())?;

    // square cut-off radius
    let cutoff_radius2 = cutoff_radius.powi(2);

    // calculate external potential
    let sigma_ff = functional.sigma_ff();
    let epsilon_k_ff = functional.epsilon_k_ff();

    Zip::indexed(&mut external_potential).par_for_each(|(i, ix, iy, iz), u| {
        let distance2 = calculate_distance2(
            [&axis[0].grid[ix], &axis[1].grid[iy], &axis[2].grid[iz]],
            &coordinates,
        );
        let sigma_sf = sigma_ss.mapv(|s| (s + sigma_ff[i]) / 2.0);
        let epsilon_sf = epsilon_ss.mapv(|e| (e * epsilon_k_ff[i]).sqrt());
        *u = (0..sigma_ss.len())
            .map(|alpha| {
                m[i] * evaluate(
                    distance2[alpha],
                    sigma_sf[alpha],
                    epsilon_sf[alpha],
                    cutoff_radius2,
                )
            })
            .sum::<f64>()
            / reduced_temperature
    });

    let potential_cutoff = potential_cutoff.unwrap_or(MAX_POTENTIAL);
    external_potential.map_inplace(|x| {
        if *x > potential_cutoff {
            *x = potential_cutoff
        }
    });

    Ok(external_potential)
}

/// Evaluate LJ12-6 potential between solid site "alpha" and fluid segment
fn evaluate(distance2: f64, sigma: f64, epsilon: f64, cutoff_radius2: f64) -> f64 {
    let sigma_r = sigma.powi(2) / distance2;

    let potential: f64 = if distance2 > cutoff_radius2 {
        0.0
    } else if distance2 == 0.0 {
        f64::INFINITY
    } else {
        4.0 * epsilon * (sigma_r.powi(6) - sigma_r.powi(3))
    };

    potential
}

/// Evaluate the squared euclidian distance between a point and the coordinates of all solid atoms.
fn calculate_distance2(point: [&f64; 3], coordinates: &Array2<f64>) -> Array1<f64> {
    Array1::from_shape_fn(coordinates.ncols(), |i| {
        let rx = coordinates[[0, i]] - point[0];
        let ry = coordinates[[1, i]] - point[1];
        let rz = coordinates[[2, i]] - point[2];

        rx.powi(2) + ry.powi(2) + rz.powi(2)
    })
}
