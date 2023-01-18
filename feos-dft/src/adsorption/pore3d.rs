use super::pore::{PoreProfile, PoreSpecification};
use crate::adsorption::FluidParameters;
use crate::convolver::ConvolverFFT;
use crate::functional::{HelmholtzEnergyFunctional, DFT};
use crate::geometry::{Axis, Grid};
use crate::profile::{DFTProfile, CUTOFF_RADIUS, MAX_POTENTIAL};
use feos_core::{EosResult, EosUnit, State};
use ndarray::prelude::*;
use ndarray::Zip;
use quantity::si::{SIArray2, SIArray4, SINumber, SIUnit};

/// Parameters required to specify a 3D pore.
pub struct Pore3D {
    system_size: [SINumber; 3],
    n_grid: [usize; 3],
    coordinates: SIArray2,
    sigma_ss: Array1<f64>,
    epsilon_k_ss: Array1<f64>,
    potential_cutoff: Option<f64>,
    cutoff_radius: Option<SINumber>,
}

impl Pore3D {
    pub fn new(
        system_size: [SINumber; 3],
        n_grid: [usize; 3],
        coordinates: SIArray2,
        sigma_ss: Array1<f64>,
        epsilon_k_ss: Array1<f64>,
        potential_cutoff: Option<f64>,
        cutoff_radius: Option<SINumber>,
    ) -> Self {
        Self {
            system_size,
            n_grid,
            coordinates,
            sigma_ss,
            epsilon_k_ss,
            potential_cutoff,
            cutoff_radius,
        }
    }
}

/// Density profile and properties of a 3D confined system.
pub type PoreProfile3D<F> = PoreProfile<Ix3, F>;

impl PoreSpecification<Ix3> for Pore3D {
    fn initialize<F: HelmholtzEnergyFunctional + FluidParameters>(
        &self,
        bulk: &State<DFT<F>>,
        density: Option<&SIArray4>,
        external_potential: Option<&Array4<f64>>,
    ) -> EosResult<PoreProfile3D<F>> {
        let dft: &F = &bulk.eos;

        // generate grid
        let x = Axis::new_cartesian(self.n_grid[0], self.system_size[0], None)?;
        let y = Axis::new_cartesian(self.n_grid[1], self.system_size[1], None)?;
        let z = Axis::new_cartesian(self.n_grid[2], self.system_size[2], None)?;

        // move center of geometry of solute to box center
        let coordinates = Array2::from_shape_fn(self.coordinates.raw_dim(), |(i, j)| {
            (self.coordinates.get((i, j)))
                .to_reduced(SIUnit::reference_length())
                .unwrap()
        });

        // temperature
        let t = bulk
            .temperature
            .to_reduced(SIUnit::reference_temperature())?;

        // calculate external potential
        let external_potential = external_potential.map_or_else(
            || {
                external_potential_3d(
                    dft,
                    [&x, &y, &z],
                    self.system_size,
                    coordinates,
                    &self.sigma_ss,
                    &self.epsilon_k_ss,
                    self.cutoff_radius,
                    self.potential_cutoff,
                    t,
                )
            },
            |e| Ok(e.clone()),
        )?;

        // initialize convolver
        let grid = Grid::Periodical3(x, y, z);
        let weight_functions = dft.weight_functions(t);
        let convolver = ConvolverFFT::plan(&grid, &weight_functions, Some(1));

        Ok(PoreProfile {
            profile: DFTProfile::new(grid, convolver, bulk, Some(external_potential), density)?,
            grand_potential: None,
            interfacial_tension: None,
        })
    }

    fn dimension(&self) -> i32 {
        3
    }
}

pub fn external_potential_3d<F: FluidParameters>(
    functional: &F,
    axis: [&Axis; 3],
    system_size: [SINumber; 3],
    coordinates: Array2<f64>,
    sigma_ss: &Array1<f64>,
    epsilon_ss: &Array1<f64>,
    cutoff_radius: Option<SINumber>,
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

    let system_size = [
        system_size[0].to_reduced(SIUnit::reference_length())?,
        system_size[1].to_reduced(SIUnit::reference_length())?,
        system_size[2].to_reduced(SIUnit::reference_length())?,
    ];

    let cutoff_radius = cutoff_radius
        .unwrap_or(CUTOFF_RADIUS * SIUnit::reference_length())
        .to_reduced(SIUnit::reference_length())?;

    // square cut-off radius
    let cutoff_radius2 = cutoff_radius.powi(2);

    // calculate external potential
    let sigma_ff = functional.sigma_ff();
    let epsilon_k_ff = functional.epsilon_k_ff();

    Zip::indexed(&mut external_potential).par_for_each(|(i, ix, iy, iz), u| {
        let distance2 = calculate_distance2(
            [axis[0].grid[ix], axis[1].grid[iy], axis[2].grid[iz]],
            &coordinates,
            system_size,
        );
        let sigma_sf = sigma_ss.mapv(|s| (s + sigma_ff[i]) / 2.0);
        let epsilon_sf = epsilon_ss.mapv(|e| (e * epsilon_k_ff[i]).sqrt());
        *u = (0..sigma_ss.len())
            .map(|alpha| {
                m[i] * evaluate_lj_potential(
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
pub(super) fn evaluate_lj_potential(
    distance2: f64,
    sigma: f64,
    epsilon: f64,
    cutoff_radius2: f64,
) -> f64 {
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
pub(super) fn calculate_distance2(
    point: [f64; 3],
    coordinates: &Array2<f64>,
    system_size: [f64; 3],
) -> Array1<f64> {
    Array1::from_shape_fn(coordinates.ncols(), |i| {
        let mut rx = coordinates[[0, i]] - point[0];
        let mut ry = coordinates[[1, i]] - point[1];
        let mut rz = coordinates[[2, i]] - point[2];

        rx -= system_size[0] * (rx / system_size[0]).round();
        ry -= system_size[1] * (ry / system_size[1]).round();
        rz -= system_size[2] * (rz / system_size[2]).round();

        rx.powi(2) + ry.powi(2) + rz.powi(2)
    })
}
