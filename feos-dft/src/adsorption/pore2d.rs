use super::{FluidParameters, PoreProfile, PoreSpecification};
use crate::{Axis, DFTProfile, Grid, HelmholtzEnergyFunctional, DFT};
use feos_core::{EosResult, State};
use ndarray::{Array3, Ix2};
use quantity::{Angle, Density, Length};

pub struct Pore2D {
    system_size: [Length<f64>; 2],
    angle: Angle,
    n_grid: [usize; 2],
}

pub type PoreProfile2D<F> = PoreProfile<Ix2, F>;

impl Pore2D {
    pub fn new(system_size: [Length<f64>; 2], angle: Angle, n_grid: [usize; 2]) -> Self {
        Self {
            system_size,
            angle,
            n_grid,
        }
    }
}

impl PoreSpecification<Ix2> for Pore2D {
    fn initialize<F: HelmholtzEnergyFunctional + FluidParameters>(
        &self,
        bulk: &State<DFT<F>>,
        density: Option<&Density<Array3<f64>>>,
        external_potential: Option<&Array3<f64>>,
    ) -> EosResult<PoreProfile<Ix2, F>> {
        // generate grid
        let x = Axis::new_cartesian(self.n_grid[0], self.system_size[0], None);
        let y = Axis::new_cartesian(self.n_grid[1], self.system_size[1], None);
        let grid = Grid::Periodical2(x, y, self.angle);

        Ok(PoreProfile {
            profile: DFTProfile::new(grid, bulk, external_potential.cloned(), density, Some(1)),
            grand_potential: None,
            interfacial_tension: None,
        })
    }
}
