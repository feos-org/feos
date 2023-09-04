use super::{FluidParameters, PoreProfile, PoreSpecification};
use crate::{Axis, ConvolverFFT, DFTProfile, Grid, HelmholtzEnergyFunctional, DFT};
use ang::Angle;
use feos_core::si::{Density, Length};
use feos_core::{EosResult, State};
use ndarray::{Array3, Ix2};

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
        let dft: &F = &bulk.eos;

        // generate grid
        let x = Axis::new_cartesian(self.n_grid[0], self.system_size[0], None);
        let y = Axis::new_cartesian(self.n_grid[1], self.system_size[1], None);

        // temperature
        let t = bulk.temperature.to_reduced();

        // initialize convolver
        let grid = Grid::Periodical2(x, y, self.angle);
        let weight_functions = dft.weight_functions(t);
        let convolver = ConvolverFFT::plan(&grid, &weight_functions, Some(1));

        Ok(PoreProfile {
            profile: DFTProfile::new(grid, convolver, bulk, external_potential.cloned(), density),
            grand_potential: None,
            interfacial_tension: None,
        })
    }
}
