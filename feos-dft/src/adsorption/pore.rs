use crate::adsorption::{ExternalPotential, FluidParameters};
use crate::convolver::ConvolverFFT;
use crate::functional::{HelmholtzEnergyFunctional, MoleculeShape, DFT};
use crate::functional_contribution::FunctionalContribution;
use crate::geometry::{Axis, Geometry, Grid};
use crate::profile::{DFTProfile, MAX_POTENTIAL};
use crate::solver::DFTSolver;
use feos_core::si::{
    Density, Dimensionless, Energy, Length, MolarEnergy, MolarWeight, Temperature, Volume, KELVIN,
};
use feos_core::{Components, Contributions, EosResult, State, StateBuilder};
use ndarray::prelude::*;
use ndarray::Axis as Axis_nd;
use ndarray::RemoveAxis;
use num_dual::linalg::LU;
use std::sync::Arc;

const POTENTIAL_OFFSET: f64 = 2.0;
const DEFAULT_GRID_POINTS: usize = 2048;

/// Parameters required to specify a 1D pore.
pub struct Pore1D {
    pub geometry: Geometry,
    pub pore_size: Length,
    pub potential: ExternalPotential,
    pub n_grid: Option<usize>,
    pub potential_cutoff: Option<f64>,
}

impl Pore1D {
    pub fn new(
        geometry: Geometry,
        pore_size: Length,
        potential: ExternalPotential,
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

/// Trait for the generic implementation of adsorption applications.
pub trait PoreSpecification<D: Dimension> {
    /// Initialize a new single pore.
    fn initialize<F: HelmholtzEnergyFunctional + FluidParameters>(
        &self,
        bulk: &State<DFT<F>>,
        density: Option<&Density<Array<f64, D::Larger>>>,
        external_potential: Option<&Array<f64, D::Larger>>,
    ) -> EosResult<PoreProfile<D, F>>;

    /// Return the pore volume using Helium at 298 K as reference.
    fn pore_volume(&self) -> EosResult<Volume>
    where
        D::Larger: Dimension<Smaller = D>,
    {
        let bulk = StateBuilder::new(&Arc::new(Helium::new()))
            .temperature(298.0 * KELVIN)
            .density(Density::from_reduced(1.0))
            .build()?;
        let pore = self.initialize(&bulk, None, None)?;
        let pot = Dimensionless::from_reduced(
            pore.profile
                .external_potential
                .index_axis(Axis(0), 0)
                .mapv(|v| (-v).exp()),
        );
        Ok(pore.profile.integrate(&pot))
    }
}

/// Density profile and properties of a confined system in arbitrary dimensions.
pub struct PoreProfile<D: Dimension, F> {
    pub profile: DFTProfile<D, F>,
    pub grand_potential: Option<Energy>,
    pub interfacial_tension: Option<Energy>,
}

/// Density profile and properties of a 1D confined system.
pub type PoreProfile1D<F> = PoreProfile<Ix1, F>;

impl<D: Dimension, F> Clone for PoreProfile<D, F> {
    fn clone(&self) -> Self {
        Self {
            profile: self.profile.clone(),
            grand_potential: self.grand_potential,
            interfacial_tension: self.interfacial_tension,
        }
    }
}

impl<D: Dimension + RemoveAxis + 'static, F: HelmholtzEnergyFunctional> PoreProfile<D, F>
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

    pub fn update_bulk(mut self, bulk: &State<DFT<F>>) -> Self {
        self.profile.bulk = bulk.clone();
        self.grand_potential = None;
        self.interfacial_tension = None;
        self
    }

    pub fn partial_molar_enthalpy_of_adsorption(&self) -> EosResult<MolarEnergy<Array1<f64>>> {
        let a = self.profile.dn_dmu()?;
        let a_unit = a.get((0, 0));
        let b = -self.profile.temperature * self.profile.dn_dt()?;
        let b_unit = b.get(0);

        let h_ads = LU::new((a / a_unit).into_value())?.solve(&(b / b_unit).into_value());
        Ok(&h_ads * b_unit / a_unit)
    }

    pub fn enthalpy_of_adsorption(&self) -> EosResult<MolarEnergy> {
        Ok((self.partial_molar_enthalpy_of_adsorption()?
            * Dimensionless::from(&self.profile.bulk.molefracs))
        .sum())
    }
}

impl PoreSpecification<Ix1> for Pore1D {
    fn initialize<F: HelmholtzEnergyFunctional + FluidParameters>(
        &self,
        bulk: &State<DFT<F>>,
        density: Option<&Density<Array2<f64>>>,
        external_potential: Option<&Array2<f64>>,
    ) -> EosResult<PoreProfile1D<F>> {
        let dft: &F = &bulk.eos;
        let n_grid = self.n_grid.unwrap_or(DEFAULT_GRID_POINTS);

        let axis = match self.geometry {
            Geometry::Cartesian => {
                let potential_offset = POTENTIAL_OFFSET
                    * bulk
                        .eos
                        .sigma_ff()
                        .iter()
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap();
                Axis::new_cartesian(n_grid, 0.5 * self.pore_size, Some(potential_offset))
            }
            Geometry::Cylindrical => Axis::new_polar(n_grid, self.pore_size),
            Geometry::Spherical => Axis::new_spherical(n_grid, self.pore_size),
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
            |e| e.clone(),
        );

        // initialize convolver
        let grid = Grid::new_1d(axis);
        let t = bulk.temperature.to_reduced();
        let weight_functions = dft.weight_functions(t);
        let convolver = ConvolverFFT::plan(&grid, &weight_functions, Some(1));

        Ok(PoreProfile {
            profile: DFTProfile::new(grid, convolver, bulk, Some(external_potential), density),
            grand_potential: None,
            interfacial_tension: None,
        })
    }
}

fn external_potential_1d<P: FluidParameters>(
    pore_width: Length,
    temperature: Temperature,
    potential: &ExternalPotential,
    fluid_parameters: &P,
    axis: &Axis,
    potential_cutoff: Option<f64>,
) -> Array2<f64> {
    let potential_cutoff = potential_cutoff.unwrap_or(MAX_POTENTIAL);
    let effective_pore_size = match axis.geometry {
        Geometry::Spherical => pore_width.to_reduced(),
        Geometry::Cylindrical => pore_width.to_reduced(),
        Geometry::Cartesian => 0.5 * pore_width.to_reduced(),
    };
    let t = temperature.to_reduced();
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
    external_potential
}

const EPSILON_HE: f64 = 10.9;
const SIGMA_HE: f64 = 2.64;

#[derive(Clone)]
struct Helium {
    epsilon: Array1<f64>,
    sigma: Array1<f64>,
}

impl Helium {
    fn new() -> DFT<Self> {
        let epsilon = arr1(&[EPSILON_HE]);
        let sigma = arr1(&[SIGMA_HE]);
        DFT(Self { epsilon, sigma })
    }
}

impl Components for Helium {
    fn components(&self) -> usize {
        1
    }

    fn subset(&self, _: &[usize]) -> Self {
        self.clone()
    }
}

impl HelmholtzEnergyFunctional for Helium {
    fn contributions(&self) -> &[Box<dyn FunctionalContribution>] {
        &[]
    }

    fn compute_max_density(&self, _: &Array1<f64>) -> f64 {
        1.0
    }

    fn molecule_shape(&self) -> MoleculeShape {
        MoleculeShape::Spherical(1)
    }

    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        unreachable!()
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
