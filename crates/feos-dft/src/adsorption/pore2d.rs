use super::{FluidParameters, Pore, PoreProfile};
use crate::{Axis, Grid, HelmholtzEnergyFunctional, adsorption::pore::PoreSpecification};
use feos_core::{FeosResult, State};
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

impl Pore<Ix2> for Pore2D {
    fn initialize<F: HelmholtzEnergyFunctional + FluidParameters>(
        &self,
        bulk: &State<F>,
        density: Option<&Density<Array3<f64>>>,
        external_potential: Option<&Array3<f64>>,
        specification: PoreSpecification,
    ) -> FeosResult<PoreProfile<Ix2, F>> {
        // generate grid
        let x = Axis::new_cartesian(self.n_grid[0], self.system_size[0], None);
        let y = Axis::new_cartesian(self.n_grid[1], self.system_size[1], None);
        let grid = Grid::Periodical2(x, y, self.angle);

        Ok(PoreProfile::new(
            grid,
            bulk,
            external_potential.cloned(),
            density,
            specification,
        ))
    }
}
