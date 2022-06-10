use crate::adsorption::{ExternalPotential, FluidParameters};
use crate::convolver::ConvolverFFT;
use crate::functional::{HelmholtzEnergyFunctional, MoleculeShape, DFT};
use crate::functional_contribution::FunctionalContribution;
use crate::geometry::{Axis, Geometry, Grid};
use crate::profile::{DFTProfile, CUTOFF_RADIUS, MAX_POTENTIAL};
use crate::solver::DFTSolver;
use feos_core::{Contributions, EosResult, EosUnit, State, StateBuilder};
use ndarray::prelude::*;
use ndarray::Axis as Axis_nd;
use ndarray::{RemoveAxis, Zip};
use ndarray_stats::QuantileExt;
use quantity::{QuantityArray, QuantityArray2, QuantityArray4, QuantityScalar};
use std::rc::Rc;

const POTENTIAL_OFFSET: f64 = 2.0;
const DEFAULT_GRID_POINTS: usize = 2048;

/// Parameters required to specify a 1D pore.
pub struct Pore1D<U> {
    pub geometry: Geometry,
    pub pore_size: QuantityScalar<U>,
    pub potential: ExternalPotential<U>,
    pub n_grid: Option<usize>,
    pub potential_cutoff: Option<f64>,
}

impl<U: EosUnit> Pore1D<U> {
    pub fn new(
        geometry: Geometry,
        pore_size: QuantityScalar<U>,
        potential: ExternalPotential<U>,
        n_grid: Option<usize>,
        potential_cutoff: Option<f64>,
    ) -> Self {
        Self {
            geometry,
            pore_size,
            potential,
            n_grid,
            potential_cutoff,
        }
    }
}

/// Parameters required to specify a 3D pore.
pub struct Pore3D<U> {
    system_size: [QuantityScalar<U>; 3],
    n_grid: [usize; 3],
    coordinates: QuantityArray2<U>,
    sigma_ss: Array1<f64>,
    epsilon_k_ss: Array1<f64>,
    potential_cutoff: Option<f64>,
    cutoff_radius: Option<QuantityScalar<U>>,
}

impl<U> Pore3D<U> {
    pub fn new(
        system_size: [QuantityScalar<U>; 3],
        n_grid: [usize; 3],
        coordinates: QuantityArray2<U>,
        sigma_ss: Array1<f64>,
        epsilon_k_ss: Array1<f64>,
        potential_cutoff: Option<f64>,
        cutoff_radius: Option<QuantityScalar<U>>,
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

/// Trait for the generic implementation of adsorption applications.
pub trait PoreSpecification<U: EosUnit, D: Dimension> {
    /// Initialize a new single pore.
    fn initialize<F: HelmholtzEnergyFunctional + FluidParameters>(
        &self,
        bulk: &State<U, DFT<F>>,
        density: Option<&QuantityArray<U, D::Larger>>,
        external_potential: Option<&Array<f64, D::Larger>>,
    ) -> EosResult<PoreProfile<U, D, F>>;

    /// Return the number of spatial dimensions of the pore.
    fn dimension(&self) -> i32;

    /// Return the pore volume using Helium at 298 K as reference.
    fn pore_volume(&self) -> EosResult<QuantityScalar<U>>
    where
        D::Larger: Dimension<Smaller = D>,
    {
        let bulk = StateBuilder::new(&Rc::new(Helium::new()))
            .temperature(298.0 * U::reference_temperature())
            .density(U::reference_density())
            .build()?;
        let pore = self.initialize(&bulk, None, None)?;
        let pot = pore
            .profile
            .external_potential
            .index_axis(Axis(0), 0)
            .mapv(|v| (-v).exp())
            * U::reference_temperature()
            / U::reference_temperature();
        Ok(pore.profile.integrate(&pot))
    }
}

/// Density profile and properties of a confined system in arbitrary dimensions.
pub struct PoreProfile<U, D: Dimension, F> {
    pub profile: DFTProfile<U, D, F>,
    pub grand_potential: Option<QuantityScalar<U>>,
    pub interfacial_tension: Option<QuantityScalar<U>>,
}

/// Density profile and properties of a 1D confined system.
pub type PoreProfile1D<U, F> = PoreProfile<U, Ix1, F>;
/// Density profile and properties of a 3D confined system.
pub type PoreProfile3D<U, F> = PoreProfile<U, Ix3, F>;

impl<U: Copy, D: Dimension, F> Clone for PoreProfile<U, D, F> {
    fn clone(&self) -> Self {
        Self {
            profile: self.profile.clone(),
            grand_potential: self.grand_potential,
            interfacial_tension: self.interfacial_tension,
        }
    }
}

impl<U: EosUnit, D: Dimension + RemoveAxis + 'static, F: HelmholtzEnergyFunctional>
    PoreProfile<U, D, F>
where
    D::Larger: Dimension<Smaller = D>,
    D::Smaller: Dimension<Larger = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    pub fn solve_inplace(&mut self, solver: Option<&DFTSolver>, debug: bool) -> EosResult<()> {
        // Solve the profile
        self.profile.solve(solver, debug)?;

        // calculate grand potential density
        let omega = self.profile.grand_potential()?;
        self.grand_potential = Some(omega);

        // calculate interfacial tension
        self.interfacial_tension =
            Some(omega + self.profile.bulk.pressure(Contributions::Total) * self.profile.volume());

        Ok(())
    }

    pub fn solve(mut self, solver: Option<&DFTSolver>) -> EosResult<Self> {
        self.solve_inplace(solver, false)?;
        Ok(self)
    }

    pub fn update_bulk(mut self, bulk: &State<U, DFT<F>>) -> Self {
        self.profile.bulk = bulk.clone();
        self.grand_potential = None;
        self.interfacial_tension = None;
        self
    }
}

impl<U: EosUnit> PoreSpecification<U, Ix1> for Pore1D<U> {
    fn initialize<F: HelmholtzEnergyFunctional + FluidParameters>(
        &self,
        bulk: &State<U, DFT<F>>,
        density: Option<&QuantityArray2<U>>,
        external_potential: Option<&Array2<f64>>,
    ) -> EosResult<PoreProfile1D<U, F>> {
        let dft: &F = &bulk.eos;
        let n_grid = self.n_grid.unwrap_or(DEFAULT_GRID_POINTS);

        let axis = match self.geometry {
            Geometry::Cartesian => {
                let potential_offset = POTENTIAL_OFFSET * bulk.eos.sigma_ff().max().unwrap();
                Axis::new_cartesian(n_grid, 0.5 * self.pore_size, Some(potential_offset))?
            }
            Geometry::Cylindrical => Axis::new_polar(n_grid, self.pore_size)?,
            Geometry::Spherical => Axis::new_spherical(n_grid, self.pore_size)?,
        };

        // calculate external potential
        let external_potential = external_potential.map_or_else(
            || {
                external_potential_1d(
                    self.pore_size,
                    bulk.temperature,
                    &self.potential,
                    dft,
                    &axis,
                    self.potential_cutoff,
                )
            },
            |e| Ok(e.clone()),
        )?;

        // initialize convolver
        let grid = Grid::new_1d(axis);
        let t = bulk.temperature.to_reduced(U::reference_temperature())?;
        let weight_functions = dft.weight_functions(t);
        let convolver = ConvolverFFT::plan(&grid, &weight_functions, Some(1));

        Ok(PoreProfile {
            profile: DFTProfile::new(grid, convolver, bulk, Some(external_potential), density)?,
            grand_potential: None,
            interfacial_tension: None,
        })
    }

    fn dimension(&self) -> i32 {
        self.geometry.dimension()
    }
}

impl<U: EosUnit> PoreSpecification<U, Ix3> for Pore3D<U> {
    fn initialize<F: HelmholtzEnergyFunctional + FluidParameters>(
        &self,
        bulk: &State<U, DFT<F>>,
        density: Option<&QuantityArray4<U>>,
        external_potential: Option<&Array4<f64>>,
    ) -> EosResult<PoreProfile3D<U, F>> {
        let dft: &F = &bulk.eos;

        // generate grid
        let x = Axis::new_cartesian(self.n_grid[0], self.system_size[0], None)?;
        let y = Axis::new_cartesian(self.n_grid[1], self.system_size[1], None)?;
        let z = Axis::new_cartesian(self.n_grid[2], self.system_size[2], None)?;

        // move center of geometry of solute to box center
        let coordinates = Array2::from_shape_fn(self.coordinates.raw_dim(), |(i, j)| {
            (self.coordinates.get((i, j)))
                .to_reduced(U::reference_length())
                .unwrap()
        });

        // temperature
        let t = bulk.temperature.to_reduced(U::reference_temperature())?;

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

fn external_potential_1d<U: EosUnit, P: FluidParameters>(
    pore_width: QuantityScalar<U>,
    temperature: QuantityScalar<U>,
    potential: &ExternalPotential<U>,
    fluid_parameters: &P,
    axis: &Axis,
    potential_cutoff: Option<f64>,
) -> EosResult<Array2<f64>> {
    let potential_cutoff = potential_cutoff.unwrap_or(MAX_POTENTIAL);
    let effective_pore_size = match axis.geometry {
        Geometry::Spherical => pore_width.to_reduced(U::reference_length())?,
        Geometry::Cylindrical => pore_width.to_reduced(U::reference_length())?,
        Geometry::Cartesian => 0.5 * pore_width.to_reduced(U::reference_length())?,
    };
    let t = temperature.to_reduced(U::reference_temperature())?;
    let mut external_potential = match &axis.geometry {
        Geometry::Cartesian => {
            potential.calculate_cartesian_potential(
                &(effective_pore_size + &axis.grid),
                fluid_parameters,
                t,
            ) + &potential.calculate_cartesian_potential(
                &(effective_pore_size - &axis.grid),
                fluid_parameters,
                t,
            )
        }
        Geometry::Spherical => potential.calculate_spherical_potential(
            &axis.grid,
            effective_pore_size,
            fluid_parameters,
            t,
        ),
        Geometry::Cylindrical => potential.calculate_cylindrical_potential(
            &axis.grid,
            effective_pore_size,
            fluid_parameters,
            t,
        ),
    } / t;

    for (i, &z) in axis.grid.iter().enumerate() {
        if z > effective_pore_size {
            external_potential
                .index_axis_mut(Axis_nd(1), i)
                .fill(potential_cutoff);
        }
    }
    external_potential.map_inplace(|x| {
        if *x > potential_cutoff {
            *x = potential_cutoff
        }
    });
    Ok(external_potential)
}

pub fn external_potential_3d<U: EosUnit, F: FluidParameters>(
    functional: &F,
    axis: [&Axis; 3],
    system_size: [QuantityScalar<U>; 3],
    coordinates: Array2<f64>,
    sigma_ss: &Array1<f64>,
    epsilon_ss: &Array1<f64>,
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

    let system_size = [
        system_size[0].to_reduced(U::reference_length())?,
        system_size[1].to_reduced(U::reference_length())?,
        system_size[2].to_reduced(U::reference_length())?,
    ];

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
            system_size,
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
fn calculate_distance2(
    point: [&f64; 3],
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

const EPSILON_HE: f64 = 10.9;
const SIGMA_HE: f64 = 2.64;

struct Helium {
    epsilon: Array1<f64>,
    sigma: Array1<f64>,
}

impl Helium {
    fn new() -> DFT<Self> {
        let epsilon = arr1(&[EPSILON_HE]);
        let sigma = arr1(&[SIGMA_HE]);
        (Self { epsilon, sigma }).into()
    }
}

impl HelmholtzEnergyFunctional for Helium {
    fn contributions(&self) -> &[Box<dyn FunctionalContribution>] {
        &[]
    }

    fn subset(&self, _: &[usize]) -> DFT<Self> {
        Self::new()
    }

    fn compute_max_density(&self, _: &Array1<f64>) -> f64 {
        1.0
    }

    fn molecule_shape(&self) -> MoleculeShape {
        MoleculeShape::Spherical(1)
    }
}

impl FluidParameters for Helium {
    fn epsilon_k_ff(&self) -> Array1<f64> {
        self.epsilon.clone()
    }

    fn sigma_ff(&self) -> &Array1<f64> {
        &self.sigma
    }
}
